"""
fMRI frequency sub-band decomposition for BRAPHIN.

Applies Butterworth bandpass filters to ROI time series before connectivity
computation, enabling band-specific functional connectivity analysis.

fMRI BOLD sub-bands  (Zuo et al. 2010)
---------------------------------------
slow5      :  0.010–0.027 Hz  (infra-slow oscillations)
slow4      :  0.027–0.073 Hz  (canonical resting-state networks)
slow3      :  0.073–0.167 Hz  (only accessible at TR ≤ 3 s)
broadband  :  0.010–0.100 Hz  (standard Biswal 1995 resting-state range)

Note
----
These are NOT EEG frequency bands (delta / theta / alpha / beta / gamma).
EEG bands span 1–45 Hz; fMRI BOLD oscillations are entirely below 0.2 Hz.

References
----------
Zuo et al. (2010). The oscillating brain: complex and reliable.
    NeuroImage, 49(2), 1432-1445.
Biswal et al. (1995). Functional connectivity in the motor cortex of resting
    human brain using echo-planar MRI. Magn. Reson. Med., 34(4), 537-541.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal

from .exceptions import ConnectivityError
from .strategy import get_connectivity_strategy
from .tools import validate_roi_time_series

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Band definitions
# ─────────────────────────────────────────────────────────────────────────────

#: Standard fMRI BOLD frequency bands.  Each value is ``(fmin_Hz, fmax_Hz)``.
FMRI_BANDS: dict[str, tuple[float, float]] = {
    "slow5": (0.010, 0.027),  # infra-slow
    "slow4": (0.027, 0.073),  # canonical resting-state
    "slow3": (0.073, 0.167),  # near-Nyquist; requires TR ≤ 3 s
    "broadband": (0.010, 0.100),  # Biswal 1995 standard
}


# ─────────────────────────────────────────────────────────────────────────────
# Bandpass filtering
# ─────────────────────────────────────────────────────────────────────────────


def bandpass_roi_time_series(
    roi_time_series: np.ndarray,
    tr: float,
    fmin: float,
    fmax: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to each ROI time series.

    Parameters
    ----------
    roi_time_series : ndarray (N, T)
        ROI × time matrix.
    tr : float
        Repetition time in seconds (1 / sampling_frequency).
    fmin : float
        Lower cut-off frequency in Hz.
    fmax : float
        Upper cut-off frequency in Hz.
    order : int
        Butterworth filter order (default 4).

    Returns
    -------
    ndarray (N, T)
        Bandpass-filtered time series (float32).

    Raises
    ------
    ConnectivityError
        If the frequency range is incompatible with the sampling rate.
    """
    validate_roi_time_series(roi_time_series)

    if tr is None or tr <= 0:
        raise ConnectivityError(
            "A positive TR (repetition time, seconds) is required for bandpass filtering."
        )

    fs = 1.0 / tr
    nyquist = fs / 2.0

    if fmin <= 0 or fmax <= fmin:
        raise ConnectivityError(
            f"Invalid frequency range: fmin={fmin} Hz, fmax={fmax} Hz. Require 0 < fmin < fmax."
        )
    if fmax >= nyquist:
        raise ConnectivityError(
            f"fmax={fmax} Hz >= Nyquist frequency {nyquist:.4f} Hz "
            f"(TR={tr}s). Reduce fmax or use a shorter TR."
        )

    low = fmin / nyquist
    high = fmax / nyquist

    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(
        sos,
        roi_time_series.astype(np.float64),
        axis=1,
    )

    return np.asarray(filtered, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Band-specific connectivity
# ─────────────────────────────────────────────────────────────────────────────


def compute_band_connectivity(
    roi_time_series: np.ndarray,
    tr: float,
    band: str = "broadband",
    method: str = "pearson_correlation",
    model_order: int = 1,
    filter_order: int = 4,
    fmin: float | None = None,
    fmax: float | None = None,
) -> np.ndarray:
    """
    Compute a connectivity matrix within a specific fMRI frequency sub-band.

    The ROI time series are bandpass-filtered to the requested sub-band and
    then the connectivity strategy is applied to the filtered signals.

    Parameters
    ----------
    roi_time_series : ndarray (N, T)
        Unfiltered ROI × time matrix.
    tr : float
        Repetition time in seconds.
    band : str
        Sub-band name.  One of ``"slow5"``, ``"slow4"``, ``"slow3"``,
        ``"broadband"``.  Ignored when *fmin* and *fmax* are provided.
    method : str
        Connectivity method name (same aliases as ``ConnectivityConfig.method``).
    model_order : int
        AR model order (only used by Granger causality and PDC).
    filter_order : int
        Butterworth filter order (default 4).
    fmin : float or None
        Custom lower cut-off in Hz.  Overrides *band* when provided.
    fmax : float or None
        Custom upper cut-off in Hz.  Overrides *band* when provided.

    Returns
    -------
    ndarray (N, N)
        Band-specific connectivity matrix.
    """
    validate_roi_time_series(roi_time_series)

    if fmin is None or fmax is None:
        if band not in FMRI_BANDS:
            raise ConnectivityError(
                f"Unknown fMRI band '{band}'. "
                f"Available bands: {list(FMRI_BANDS)}. "
                "Alternatively, set fmin and fmax explicitly."
            )
        fmin, fmax = FMRI_BANDS[band]

    filtered_ts = bandpass_roi_time_series(
        roi_time_series, tr=tr, fmin=fmin, fmax=fmax, order=filter_order
    )

    strategy = get_connectivity_strategy(method, tr=tr, model_order=model_order)
    return strategy.compute(filtered_ts)


def compute_all_bands_connectivity(
    roi_time_series: np.ndarray,
    tr: float,
    method: str = "pearson_correlation",
    model_order: int = 1,
    filter_order: int = 4,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute connectivity matrices for every fMRI frequency sub-band.

    Bands incompatible with the scan TR (i.e. whose upper edge exceeds the
    Nyquist frequency) are skipped with a warning.

    Parameters
    ----------
    roi_time_series : ndarray (N, T)
        Unfiltered ROI × time matrix.
    tr : float
        Repetition time in seconds.
    method : str
        Connectivity method name.
    model_order : int
        AR model order (only used by Granger causality and PDC).
    filter_order : int
        Butterworth filter order (default 4).
    bands : dict or None
        Custom band definitions ``{name: (fmin_Hz, fmax_Hz)}``.
        Defaults to :data:`FMRI_BANDS`.

    Returns
    -------
    dict {band_name: ndarray (N, N)}
        One connectivity matrix per successfully processed band.

    Notes
    -----
    ``slow3`` (0.073–0.167 Hz) requires TR ≤ 3 s.  With TR = 2 s the
    Nyquist is 0.25 Hz so slow3 is included.  With TR ≥ 3 s the Nyquist
    is ≤ 0.167 Hz and slow3 will be skipped with a warning.
    """
    validate_roi_time_series(roi_time_series)

    if bands is None:
        bands = FMRI_BANDS

    results: dict[str, np.ndarray] = {}

    for band_name, (fmin, fmax) in bands.items():
        try:
            mat = compute_band_connectivity(
                roi_time_series,
                tr=tr,
                method=method,
                model_order=model_order,
                filter_order=filter_order,
                fmin=fmin,
                fmax=fmax,
            )
            results[band_name] = mat
            logger.info(
                "[BRAPHIN] Band connectivity: %s (%.3f–%.3f Hz) → shape %s",
                band_name,
                fmin,
                fmax,
                mat.shape,
            )
        except ConnectivityError as exc:
            logger.warning(
                "[BRAPHIN] Skipping band '%s' (%.3f–%.3f Hz): %s",
                band_name,
                fmin,
                fmax,
                exc,
            )

    return results
