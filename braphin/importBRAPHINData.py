import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import InputConfig
from .exceptions import BRAPHINInputError
from .io.nifti import get_nifti_metadata, load_nifti_file, validate_fmri_nifti
from .io.tabular import load_tabular_file

logger = logging.getLogger(__name__)


@dataclass
class BRAPHINInputBundle:
    """
    Internal data structure representing the loaded input data.

    Fields
    ------
    fmri_path : str or None
        Path to the main fMRI file.
    fmri_image : object or None
        Loaded NIfTI image object.
    fmri_metadata : dict or None
        Basic metadata extracted from the NIfTI image.
    auxiliary_files : dict
        Dictionary mapping filename to loaded content for each auxiliary file.
    """

    fmri_path: str | None = None
    fmri_image: object | None = None
    fmri_metadata: dict[str, object] | None = None
    auxiliary_files: dict[str, object] = field(default_factory=dict)


class InputfMRIData:
    """
    Stage 1 of the BRAPHIN fMRI pipeline: fMRI data loading.

    Accepts a path to the main fMRI file and an optional list of auxiliary
    files (BIDS-format TSV confound matrices, JSON sidecars, CSV, or NumPy
    arrays). Validates extensions, loads all data, and returns an
    ``BRAPHINInputBundle``.
    """

    def __init__(
        self,
        fmri_path: str | Path,
        auxiliary_paths: list[str | Path] | None = None,
        config: InputConfig | None = None,
    ):
        self.fmri_path = Path(fmri_path)
        self.auxiliary_paths = [Path(p) for p in auxiliary_paths] if auxiliary_paths else []
        self.config = config if config is not None else InputConfig()

    def load(self) -> BRAPHINInputBundle:
        """
        Execute the full input loading phase.

        Returns
        -------
        BRAPHINInputBundle
            Bundle containing the loaded NIfTI image, its metadata, and all
            auxiliary files.
        """
        self._validate_main_input()

        fmri_image = load_nifti_file(self.fmri_path)
        validate_fmri_nifti(fmri_image)
        fmri_metadata = get_nifti_metadata(fmri_image)

        # Issue #8 — auto-extract TR from NIfTI header (pixdim[4])
        try:
            zooms = fmri_image.header.get_zooms()
            tr_from_header = float(zooms[3]) if len(zooms) >= 4 else None
        except Exception:
            tr_from_header = None

        if tr_from_header is not None and 0.1 <= tr_from_header <= 20.0:
            fmri_metadata["tr"] = tr_from_header
        else:
            fmri_metadata["tr"] = None  # User must supply via config

        auxiliary_data = self._load_auxiliary_files()

        # Issue #6 — parse BIDS JSON sidecar for RepetitionTime and SliceTiming
        for aux_path in self.auxiliary_paths:
            if str(aux_path).lower().endswith(".json"):
                try:
                    with open(aux_path, encoding="utf-8") as f:
                        sidecar = json.load(f)
                    # BIDS sidecar RepetitionTime overrides the header TR
                    if "RepetitionTime" in sidecar:
                        tr_sidecar = float(sidecar["RepetitionTime"])
                        if 0.1 <= tr_sidecar <= 20.0:
                            fmri_metadata["tr"] = tr_sidecar
                    # Per-slice acquisition time offsets
                    if "SliceTiming" in sidecar:
                        fmri_metadata["slice_timing_offsets"] = list(sidecar["SliceTiming"])
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Could not parse BIDS JSON sidecar %s: %s", aux_path, e)

        bundle = BRAPHINInputBundle(
            fmri_path=str(self.fmri_path),
            fmri_image=fmri_image,
            fmri_metadata=fmri_metadata,
            auxiliary_files=auxiliary_data,
        )

        return bundle

    def _validate_main_input(self) -> None:
        """Validate the main fMRI file path and extension."""
        if not self.fmri_path.exists():
            raise BRAPHINInputError(f"fMRI file not found: {self.fmri_path}")

        # Use endswith() rather than Path.suffixes to avoid false failures on
        # filenames that contain extra dots (e.g. sub-01.2_bold.nii.gz → suffixes
        # would give ['.2', '.nii', '.gz'] instead of ['.nii', '.gz']).
        name_lower = self.fmri_path.name.lower()
        suffix = next(
            (
                ext
                for ext in self.config.allowed_fmri_extensions
                if name_lower.endswith(ext.lower())
            ),
            None,
        )

        if suffix is None:
            raise BRAPHINInputError(
                f"Unsupported fMRI file extension: {self.fmri_path.name}. "
                f"Supported extensions: {', '.join(self.config.allowed_fmri_extensions)}"
            )

    def _load_auxiliary_files(self) -> dict[str, object]:
        """Load and return all auxiliary files."""
        loaded_aux: dict[str, object] = {}

        for aux_path in self.auxiliary_paths:
            if not aux_path.exists():
                raise BRAPHINInputError(f"Auxiliary file not found: {aux_path}")

            suffix = aux_path.suffix.lower()

            if suffix not in self.config.allowed_aux_extensions:
                raise BRAPHINInputError(
                    f"Unsupported auxiliary file extension: {aux_path.name}. "
                    f"Supported extensions: {', '.join(self.config.allowed_aux_extensions)}"
                )

            # JSON: loaded as raw text; parse downstream if structured access is needed.
            if suffix == ".json":
                loaded_aux[aux_path.name] = aux_path.read_text(encoding="utf-8")
                continue

            # CSV / TSV / NPY: generic tabular load.
            if suffix in {".csv", ".tsv", ".npy"}:
                loaded_aux[aux_path.name] = load_tabular_file(aux_path)
                continue

        return loaded_aux

    def display_info(self, bundle: BRAPHINInputBundle) -> None:
        """
        Log a summary of the loaded input data.

        Useful for debugging and pipeline transparency.
        """
        logger.info("[BRAPHIN] Input loaded successfully")
        logger.info("  fMRI path: %s", bundle.fmri_path)

        if bundle.fmri_metadata is not None:
            logger.info("  fMRI shape: %s", bundle.fmri_metadata.get("shape"))
            logger.info("  Number of dimensions: %s", bundle.fmri_metadata.get("ndim"))
            tr = bundle.fmri_metadata.get("tr")
            if tr is not None:
                logger.info("  TR (from header): %s s", tr)
            else:
                logger.info("  TR: not found in header — set via PreprocessConfig/DenoiseConfig")

        if bundle.auxiliary_files:
            logger.info("  Auxiliary files detected:")
            for file_name in bundle.auxiliary_files:
                logger.info("    - %s", file_name)
        else:
            logger.info("  No auxiliary files provided.")


# Backward-compatibility alias — will be removed in a future major release.
InputBRAPHINData = InputfMRIData
