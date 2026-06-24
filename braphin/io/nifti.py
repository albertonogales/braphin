"""
NIfTI I/O helpers for BRAPHIN.

Provides three thin wrappers around nibabel:
- :func:`load_nifti_file`    — load a ``.nii`` / ``.nii.gz`` file.
- :func:`get_nifti_metadata` — extract shape, affine and zoom metadata.
- :func:`validate_fmri_nifti` — assert that the image is a 4-D fMRI volume.

This module does NOT apply preprocessing, atlas parcellation, or ROI
extraction — it only loads and validates NIfTI images.
"""

from pathlib import Path

from ..exceptions import BRAPHINFormatError, BRAPHINInputError


def load_nifti_file(file_path: str | Path):
    """
    Load a NIfTI file using nibabel.

    Parameters
    ----------
    file_path : str or Path
        Path to a ``.nii`` or ``.nii.gz`` file.

    Returns
    -------
    nibabel image object

    Raises
    ------
    BRAPHINInputError
        If the file does not exist or has an unsupported extension.
    BRAPHINFormatError
        If nibabel cannot load the file.
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise BRAPHINInputError(
            "nibabel is not installed. Install it to work with NIfTI files."
        ) from exc

    file_path = Path(file_path)

    if not file_path.exists():
        raise BRAPHINInputError(f"NIfTI file not found: {file_path}")

    # Check by filename ending rather than Path.suffixes to avoid false
    # failures on filenames with extra dots (e.g. sub-01.2_bold.nii.gz).
    name_lower = file_path.name.lower()
    if not (name_lower.endswith(".nii") or name_lower.endswith(".nii.gz")):
        raise BRAPHINInputError(f"Unsupported extension for NIfTI file: {file_path.name}")

    try:
        image = nib.load(str(file_path))
    except Exception as exc:
        raise BRAPHINFormatError(f"Failed to load NIfTI file: {file_path}") from exc

    return image


def get_nifti_metadata(image) -> dict[str, object]:
    """
    Extract basic metadata from a loaded NIfTI image.

    Returns
    -------
    dict with keys:
        ``shape``, ``ndim``, ``affine``, ``zooms``, ``axis_codes``
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise BRAPHINInputError(
            "nibabel is not installed. Install it to work with NIfTI files."
        ) from exc

    shape = image.shape
    ndim = len(shape)
    affine = image.affine
    zooms = tuple(float(z) for z in image.header.get_zooms()[:ndim])
    axis_codes = tuple(nib.aff2axcodes(affine))

    return {
        "shape": shape,
        "ndim": ndim,
        "affine": affine,
        "zooms": zooms,
        "axis_codes": axis_codes,
    }


def validate_fmri_nifti(image) -> None:
    """
    Assert that a NIfTI image has the expected 4-D fMRI shape (X, Y, Z, T).

    Raises
    ------
    BRAPHINFormatError
        If the image is not 4-D.
    """
    metadata = get_nifti_metadata(image)

    if metadata["ndim"] != 4:
        raise BRAPHINFormatError(
            f"Expected a 4-D fMRI volume, but received an image with shape {metadata['shape']}."
        )
