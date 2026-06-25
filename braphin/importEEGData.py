"""
Stage 1 of the BRAPHIN EEG pipeline: EEG data loading.

Wraps EEGraph's InputData (MNE-Python) and exposes a bundle interface
consistent with InputfMRIData / BRAPHINInputBundle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EEGInputBundle:
    """
    Output bundle of the EEG input loading stage.

    Fields
    ------
    eeg_path : str or None
        Path to the EEG file.
    raw_data : object or None
        Loaded MNE Raw object.
    eeg_metadata : dict or None
        Basic metadata: n_channels, sample_rate, duration, ch_names.
    ch_names : list of str
        Channel / electrode names (after optional montage remapping).
    auxiliary_files : dict
        Reserved for auxiliary files (currently unused for EEG).
    """

    eeg_path: str | None = None
    raw_data: object | None = None
    eeg_metadata: dict[str, object] | None = None
    ch_names: list[str] = field(default_factory=list)
    auxiliary_files: dict[str, object] = field(default_factory=dict)


class InputEEGData:
    """
    Stage 1 of the BRAPHIN EEG pipeline: EEG data loading.

    Wraps EEGraph's InputData (MNE-Python) and returns an EEGInputBundle,
    providing the same input-stage interface as InputfMRIData.

    Supported formats: .edf .gdf .vhdr .bdf .cnt .egi .mff .nxe
    (via MNE-Python — must be installed for EEG support).

    Parameters
    ----------
    eeg_path : str
        Path to the EEG recording file.
    exclude : list, optional
        Channel names to exclude. Default [None] (exclude nothing).
    electrode_montage_path : str, optional
        Path to a custom electrode montage CSV/TSV for channel renaming.
    """

    def __init__(
        self,
        eeg_path: str,
        exclude: list | None = None,
        electrode_montage_path: str | None = None,
    ):
        self.eeg_path = str(eeg_path)
        self.exclude = exclude if exclude is not None else [None]
        self.electrode_montage_path = electrode_montage_path
        self._backend: Any = None

    def load(self) -> EEGInputBundle:
        """
        Load the EEG file via MNE-Python.

        Returns
        -------
        EEGInputBundle
            Bundle containing the MNE Raw object, metadata, and channel names.
        """
        from eegraph.importData import InputData

        self._backend = InputData(self.eeg_path, self.exclude)
        raw = self._backend.load()

        ch_names = list(raw.ch_names)
        if self.electrode_montage_path:
            ch_names = self._backend.set_montage(self.electrode_montage_path)

        eeg_metadata = {
            "n_channels": int(raw.info["nchan"]),
            "sample_rate": float(raw.info["sfreq"]),
            "duration_sec": round(float(raw.times.max()), 3),
            "ch_names": ch_names,
        }

        bundle = EEGInputBundle(
            eeg_path=self.eeg_path,
            raw_data=raw,
            eeg_metadata=eeg_metadata,
            ch_names=ch_names,
        )

        return bundle

    def display_info(self, bundle: EEGInputBundle) -> None:
        """Log a summary of the loaded EEG data."""
        logger.info("[BRAPHIN] EEG input loaded")
        logger.info("  EEG path:    %s", bundle.eeg_path)
        if bundle.eeg_metadata:
            m = bundle.eeg_metadata
            logger.info("  Channels:    %d", m["n_channels"])
            logger.info("  Sample rate: %.1f Hz", m["sample_rate"])
            logger.info("  Duration:    %.3f s", m["duration_sec"])
            logger.info("  Channel names: %s", m["ch_names"])
