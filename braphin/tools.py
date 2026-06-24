from typing import Dict, List, Optional
import numpy as np
from scipy import signal, stats
from scipy.spatial.distance import cdist
from .exceptions import ConnectivityError


# ============================================================
# Registry of supported connectivity measures.
# ============================================================
CONNECTIVITY_MEASURES: Dict[str, str] = {
    # ── Undirected, no TR ────────────────────────────────────────────────────
    "pearson_correlation":   "Pearson correlation",
    "cross_correlation":     "Cross-correlation",
    "corr_cross_correlation": "Corrected cross-correlation",
    "partial_correlation":   "Partial correlation",
    "plv":                   "Phase Locking Value",
    "pli":                   "Phase Lag Index",
    "wpli":                  "Weighted Phase Lag Index",
    "aec":                   "Amplitude Envelope Correlation",
    "aec_orth":              "Orthogonalized Amplitude Envelope Correlation",
    "dwpli":                 "Debiased Weighted Phase Lag Index",
    "ppc":                   "Pairwise Phase Consistency",
    "mutual_information":    "Mutual Information",
    "sync_likelihood":       "Synchronisation Likelihood",
    # ── Undirected, TR required ──────────────────────────────────────────────
    "coherence":             "Magnitude-squared coherence",
    "imag_coherence":        "Imaginary coherence",
    "lagged_coherence":      "Lagged coherence",
    # ── Directed, no TR ─────────────────────────────────────────────────────
    "granger_causality":     "Granger Causality",
    "transfer_entropy":      "Transfer Entropy",
    # ── Directed, TR required ────────────────────────────────────────────────
    "pdc":                   "Partial Directed Coherence",
    "psi":                   "Phase Slope Index",
}

# Measures that require a TR (repetition time / sample period).
SPECTRAL_MEASURES = {"coherence", "imag_coherence", "lagged_coherence", "pdc", "psi"}

# Measures that produce an asymmetric (directed) matrix.
DIRECTED_MEASURES = {"granger_causality", "transfer_entropy", "pdc", "psi",
                     "cross_correlation", "corr_cross_correlation"}

# Allowed aliases. A dictionary to parse the different names that a user may provide.
CONNECTIVITY_ALIASES: Dict[str, str] = {
    "pearson": "pearson_correlation",
    "pearson_correlation": "pearson_correlation",
    "pearson correlation": "pearson_correlation",

    "cross_correlation": "cross_correlation",
    "cross-correlation": "cross_correlation",
    "cross correlation": "cross_correlation",

    "corr_cross_correlation": "corr_cross_correlation",
    "corrected_cross_correlation": "corr_cross_correlation",
    "corrected cross correlation": "corr_cross_correlation",
    "corrected cross-correlation": "corr_cross_correlation",

    "partial_correlation": "partial_correlation",
    "partial correlation": "partial_correlation",
    "partial corr": "partial_correlation",

    "plv": "plv",
    "phase locking value": "plv",
    "phase_locking_value": "plv",

    "pli": "pli",
    "phase lag index": "pli",
    "phase_lag_index": "pli",

    "wpli": "wpli",
    "weighted_phase_lag_index": "wpli",
    "weighted phase lag index": "wpli",

    "coherence": "coherence",
    "squared_coherence": "coherence",
    "squared coherence": "coherence",

    "imag_coherence": "imag_coherence",
    "imaginary_coherence": "imag_coherence",
    "imaginary coherence": "imag_coherence",

    "lagged_coherence": "lagged_coherence",
    "lagged coherence": "lagged_coherence",

    "aec": "aec",
    "amplitude_envelope_correlation": "aec",
    "amplitude envelope correlation": "aec",

    "aec_orth": "aec_orth",
    "aec_c": "aec_orth",
    "orthogonalized_aec": "aec_orth",
    "orthogonalized aec": "aec_orth",

    "dwpli": "dwpli",
    "debiased_wpli": "dwpli",
    "debiased wpli": "dwpli",

    "ppc": "ppc",
    "pairwise_phase_consistency": "ppc",
    "pairwise phase consistency": "ppc",

    "mutual_information": "mutual_information",
    "mutual information": "mutual_information",
    "mi": "mutual_information",

    "sync_likelihood": "sync_likelihood",
    "synchronisation_likelihood": "sync_likelihood",
    "synchronization_likelihood": "sync_likelihood",
    "synchronisation likelihood": "sync_likelihood",

    "granger_causality": "granger_causality",
    "granger causality": "granger_causality",
    "gc": "granger_causality",

    "transfer_entropy": "transfer_entropy",
    "transfer entropy": "transfer_entropy",
    "te": "transfer_entropy",

    "pdc": "pdc",
    "partial_directed_coherence": "pdc",
    "partial directed coherence": "pdc",

    "psi": "psi",
    "phase_slope_index": "psi",
    "phase slope index": "psi",
}


def list_connectivity_measures() -> List[str]:
    """
    Return the list of supported canonical method names.
    """
    return list(CONNECTIVITY_MEASURES.keys())


def validate_connectivity_method(method: str) -> str:
    """
    Validate and normalise the connectivity method name.
    """
    if method is None:
        raise ConnectivityError("The connectivity method cannot be None.")

    normalized_method = method.lower().strip()
    normalized_method = CONNECTIVITY_ALIASES.get(normalized_method, normalized_method)

    if normalized_method not in CONNECTIVITY_MEASURES:
        raise ConnectivityError(
            f"Connectivity method '{method}' is not supported. "
            f"Available methods: {', '.join(list_connectivity_measures())}"
        )

    return normalized_method


def validate_roi_time_series(roi_time_series: np.ndarray) -> None:
    """
    Validate that the input has a shape compatible with connectivity computation.

    Expected shape:
    - 2-D matrix with shape (num_rois, num_timepoints)
    """
    if not isinstance(roi_time_series, np.ndarray):
        raise ConnectivityError(
            "ROI time series must be a NumPy ndarray."
        )

    if roi_time_series.ndim != 2:
        raise ConnectivityError(
            f"Expected a 2-D ROI × time matrix, but received "
            f"a structure with shape {roi_time_series.shape}."
        )

    num_rois, num_timepoints = roi_time_series.shape

    if num_rois < 2:
        raise ConnectivityError(
            "At least 2 ROIs are required to compute connectivity."
        )

    if num_timepoints < 2:
        raise ConnectivityError(
            "At least 2 timepoints are required to compute connectivity."
        )


def compute_pearson_correlation(roi_time_series: np.ndarray) -> np.ndarray:
    """
    Compute the connectivity matrix using Pearson correlation.

    Zero-variance ROI time series produce NaN correlation values, which are
    replaced with 0 (off-diagonal) and 1 (diagonal) via nan_to_num. A
    RuntimeWarning is emitted when this substitution occurs.
    """
    validate_roi_time_series(roi_time_series)

    try:
        connectivity_matrix = np.corrcoef(roi_time_series, rowvar=True)
    except Exception as exc:
        raise ConnectivityError(
            "Failed to compute Pearson correlation."
        ) from exc

    connectivity_matrix = np.asarray(connectivity_matrix, dtype=np.float32)

    # Replace NaN/inf values: a zero-variance ROI produces NaN correlations.
    # Off-diagonal NaNs are set to 0; the diagonal is restored to 1 below.
    if np.any(np.isnan(connectivity_matrix)):
        import warnings
        warnings.warn(
            "One or more ROI time series had zero variance; "
            "NaN correlations replaced with 0 (off-diagonal) and 1 (diagonal).",
            RuntimeWarning,
            stacklevel=2,
        )

    connectivity_matrix = np.nan_to_num(
        connectivity_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    np.fill_diagonal(connectivity_matrix, 1.0)

    return connectivity_matrix


def _normalized_cross_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the normalised cross-correlation used by EEGraph:

        Rxy_norm = (1 / sqrt(Rxx_0 * Ryy_0)) * Rxy
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if x.ndim != 1 or y.ndim != 1:
        raise ConnectivityError(
            "Individual signals for cross-correlation must be 1-D."
        )

    if len(x) != len(y):
        raise ConnectivityError(
            "Signals being compared must have the same number of timepoints."
        )

    try:
        Rxy = signal.correlate(x, y, mode="full")
        Rxx = signal.correlate(x, x, mode="full")
        Ryy = signal.correlate(y, y, mode="full")
    except Exception as exc:
        raise ConnectivityError(
            "Failed to compute cross-correlation."
        ) from exc

    lags = np.arange(-len(x) + 1, len(x))
    lag_0 = int(np.where(lags == 0)[0][0])

    Rxx_0 = Rxx[lag_0]
    Ryy_0 = Ryy[lag_0]

    if Rxx_0 == 0 or Ryy_0 == 0:
        return np.zeros_like(Rxy, dtype=np.float32)

    Rxy_norm = (1.0 / np.sqrt(Rxx_0 * Ryy_0)) * Rxy
    Rxy_norm = np.asarray(Rxy_norm, dtype=np.float32)

    Rxy_norm = np.nan_to_num(
        Rxy_norm,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return Rxy_norm


def _cross_correlation_coef(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the scalar cross-correlation coefficient, reproducing the
    Cross_correlation_Estimator logic from EEGraph.

    The function returns the **mean** of normalised cross-correlation values
    from lag 0 to lag = floor(0.1 · T), where T is the number of timepoints.
    This is a scalar summary adapted from the EEGraph library; it is not the
    peak cross-correlation.
    """
    Rxy_norm = _normalized_cross_correlation(x, y)

    lags = np.arange(-len(x) + 1, len(x))
    lag_0 = int(np.where(lags == 0)[0][0])

    disp = max(1, round(len(x) * 0.10))
    fragment = Rxy_norm[lag_0: lag_0 + disp]

    if fragment.size == 0:
        return 0.0

    return float(np.mean(fragment))


def _corr_cross_correlation_coef(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the scalar corrected cross-correlation coefficient, reproducing the
    Corr_cross_correlation_Estimator logic from EEGraph.

    This computes corCC = mean(R_xy(+lags)) − mean(R_xy(−lags)) over lags 1
    to floor(0.1 · T). Specifically:

    - Rxy_norm is split into negative-lag and positive-lag portions.
    - corCC[k] = Rxy_norm(+k) − Rxy_norm(−k) for k = 1 … floor(0.1 · T).
    - The scalar returned is the mean of that fragment.

    This is a scalar adaptation of the Roebroeck et al. (2005) directed
    connectivity measure; the original paper defines a full lag-dependent curve
    rather than this scalar summary.
    """
    Rxy_norm = _normalized_cross_correlation(x, y)

    lags = np.arange(-len(x) + 1, len(x))
    lag_0 = int(np.where(lags == 0)[0][0])

    negative_lag = Rxy_norm[:lag_0]
    positive_lag = Rxy_norm[lag_0 + 1:]

    if negative_lag.size == 0 or positive_lag.size == 0:
        return 0.0

    # corCC[k] = Rxy(+k) - Rxy(-k).
    # negative_lag is ordered from lag -(N-1) to lag -1, so it must be
    # reversed before subtracting so that index 0 aligns lag -1 with lag +1.
    # The original EEGraph code omitted this reversal, producing wrong values.
    corCC = positive_lag - negative_lag[::-1]

    disp = max(1, round(len(x) * 0.10))
    fragment = corCC[:disp]

    if fragment.size == 0:
        return 0.0

    return float(np.mean(fragment))


def compute_cross_correlation(roi_time_series: np.ndarray) -> np.ndarray:
    """
    Compute the ROI × ROI connectivity matrix using the
    Cross_correlation_Estimator logic from EEGraph.

    Important
    ---------
    - This measure can be asymmetric; symmetry is therefore NOT enforced.

    Note
    ----
    This implementation uses nested Python loops over all ROI pairs and calls
    scipy.signal.correlate for each pair. For an atlas with N ROIs the complexity
    is O(N²·T·log T). For large atlases (e.g. Schaefer 400, N=400) this may take
    several minutes. A vectorised FFT-based implementation is planned for a future
    release.
    """
    validate_roi_time_series(roi_time_series)

    num_rois = roi_time_series.shape[0]
    connectivity_matrix = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(num_rois):
            connectivity_matrix[i, j] = _cross_correlation_coef(
                roi_time_series[i],
                roi_time_series[j]
            )

    connectivity_matrix = np.nan_to_num(
        connectivity_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return connectivity_matrix.astype(np.float32)


def compute_corrected_cross_correlation(roi_time_series: np.ndarray) -> np.ndarray:
    """
    Compute the ROI × ROI connectivity matrix using the
    Corr_cross_correlation_Estimator logic from EEGraph.

    Important
    ---------
    - This measure can be asymmetric; symmetry is therefore NOT enforced.

    Note
    ----
    This implementation uses nested Python loops over all ROI pairs and calls
    scipy.signal.correlate for each pair. For an atlas with N ROIs the complexity
    is O(N²·T·log T). For large atlases (e.g. Schaefer 400, N=400) this may take
    several minutes. A vectorised FFT-based implementation is planned for a future
    release.
    """
    validate_roi_time_series(roi_time_series)

    num_rois = roi_time_series.shape[0]
    connectivity_matrix = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(num_rois):
            connectivity_matrix[i, j] = _corr_cross_correlation_coef(
                roi_time_series[i],
                roi_time_series[j]
            )

    connectivity_matrix = np.nan_to_num(
        connectivity_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return connectivity_matrix.astype(np.float32)


# ============================================================
# Partial correlation
# ============================================================

def compute_partial_correlation(roi_time_series: np.ndarray) -> np.ndarray:
    """
    Compute the ROI × ROI connectivity matrix using partial correlation.

    Partial correlation is derived from the precision matrix (inverse of the
    covariance matrix):

        partial_corr[i, j] = -precision[i, j] / sqrt(precision[i, i] * precision[j, j])

    The diagonal is set to 1. Values are clamped to [-1, 1].

    Raises ConnectivityError if the covariance matrix is singular.
    """
    validate_roi_time_series(roi_time_series)

    cov = np.cov(roi_time_series.astype(np.float64))

    try:
        precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError as exc:
        raise ConnectivityError(
            "Covariance matrix is singular; cannot compute partial correlation."
        ) from exc

    diag = np.sqrt(np.diag(precision))
    with np.errstate(invalid="ignore", divide="ignore"):
        partial_corr = -precision / np.outer(diag, diag)

    partial_corr = np.nan_to_num(partial_corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(partial_corr, 1.0)
    partial_corr = np.clip(partial_corr, -1.0, 1.0)

    return partial_corr.astype(np.float32)


# ============================================================
# Phase-based measures (no TR required)
# ============================================================

def _instantaneous_phases(roi_time_series: np.ndarray) -> np.ndarray:
    """Return instantaneous phases (radians) via Hilbert transform for every ROI."""
    analytic = signal.hilbert(roi_time_series.astype(np.float64), axis=1)
    return np.angle(analytic)


def compute_plv(roi_time_series: np.ndarray) -> np.ndarray:
    """
    Compute the ROI × ROI connectivity matrix using Phase Locking Value (PLV).

    PLV[i, j] = |mean_t( exp(j * (phase_i(t) - phase_j(t))) )|

    The matrix is symmetric with 1 on the diagonal.
    """
    validate_roi_time_series(roi_time_series)

    phases = _instantaneous_phases(roi_time_series)
    num_rois = roi_time_series.shape[0]
    connectivity_matrix = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(i, num_rois):
            phase_diff = phases[i] - phases[j]
            plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
            connectivity_matrix[i, j] = plv
            connectivity_matrix[j, i] = plv

    np.fill_diagonal(connectivity_matrix, 1.0)
    return connectivity_matrix


def compute_pli(roi_time_series: np.ndarray) -> np.ndarray:
    """
    Compute the ROI × ROI connectivity matrix using Phase Lag Index (PLI).

    PLI[i, j] = |mean_t( sign(phase_i(t) - phase_j(t)) )|

    The matrix is symmetric. The diagonal is 0 (a signal has no phase lag with itself).
    """
    validate_roi_time_series(roi_time_series)

    phases = _instantaneous_phases(roi_time_series)
    num_rois = roi_time_series.shape[0]
    connectivity_matrix = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(i + 1, num_rois):
            phase_diff = phases[i] - phases[j]
            phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
            pli = float(np.abs(np.mean(np.sign(phase_diff))))
            connectivity_matrix[i, j] = pli
            connectivity_matrix[j, i] = pli

    return connectivity_matrix


def compute_wpli(roi_time_series: np.ndarray) -> np.ndarray:
    """
    Compute the ROI × ROI connectivity matrix using Weighted Phase Lag Index (wPLI).

    wPLI[i, j] = |mean( |imag(C_xy)| * sign(imag(C_xy)) )| / mean( |imag(C_xy)| )

    where C_xy is the element-wise cross-spectrum from the analytic signal.

    The matrix is symmetric. The diagonal is 0.
    """
    validate_roi_time_series(roi_time_series)

    analytic = signal.hilbert(roi_time_series.astype(np.float64), axis=1)
    num_rois = roi_time_series.shape[0]
    connectivity_matrix = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(i + 1, num_rois):
            cross_spectrum = analytic[i] * np.conj(analytic[j])
            imag_cs = np.imag(cross_spectrum)
            denom = float(np.mean(np.abs(imag_cs)))
            if denom == 0.0:
                wpli = 0.0
            else:
                wpli = float(np.abs(np.mean(np.abs(imag_cs) * np.sign(imag_cs))) / denom)
            connectivity_matrix[i, j] = wpli
            connectivity_matrix[j, i] = wpli

    return connectivity_matrix


# ============================================================
# Spectral coherence measures (TR required)
# ============================================================

def compute_coherence(roi_time_series: np.ndarray, tr: float) -> np.ndarray:
    """
    Compute the ROI × ROI connectivity matrix using magnitude-squared coherence.

    The coherence is averaged across all frequency bins (0 to Nyquist). For fMRI
    this covers the full BOLD signal range; apply bandpass denoising upstream to
    restrict the analysis to the 0.008–0.1 Hz band beforehand.

    Parameters
    ----------
    roi_time_series : ndarray (N, T)
    tr : float
        Repetition time in seconds (sample period). Required; must be positive.
    """
    validate_roi_time_series(roi_time_series)
    if tr is None or tr <= 0:
        raise ConnectivityError(
            "A positive TR (repetition time, seconds) is required for coherence computation."
        )

    fs = 1.0 / tr
    num_rois = roi_time_series.shape[0]
    connectivity_matrix = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(i, num_rois):
            _, Cxy = signal.coherence(roi_time_series[i], roi_time_series[j], fs=fs)
            coh = float(np.mean(Cxy))
            connectivity_matrix[i, j] = coh
            connectivity_matrix[j, i] = coh

    connectivity_matrix = np.nan_to_num(connectivity_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(connectivity_matrix, 1.0)
    return connectivity_matrix.astype(np.float32)


def compute_imaginary_coherence(roi_time_series: np.ndarray, tr: float) -> np.ndarray:
    """
    Compute the ROI × ROI connectivity matrix using mean absolute imaginary coherence.

    Imaginary coherence is the imaginary part of the cross-spectral density
    normalised by the geometric mean of the auto-spectra:

        ImCoh[i, j] = mean_f( |imag(P_xy(f))| / sqrt(P_xx(f) * P_yy(f)) )

    Because it is insensitive to zero-lag correlations it is robust to common
    sources of spurious connectivity (e.g. motion artefacts).

    Parameters
    ----------
    roi_time_series : ndarray (N, T)
    tr : float
        Repetition time in seconds. Required; must be positive.
    """
    validate_roi_time_series(roi_time_series)
    if tr is None or tr <= 0:
        raise ConnectivityError(
            "A positive TR (repetition time, seconds) is required for imaginary "
            "coherence computation."
        )

    fs = 1.0 / tr
    num_rois = roi_time_series.shape[0]
    connectivity_matrix = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(i, num_rois):
            _, Pxx = signal.welch(roi_time_series[i], fs=fs)
            _, Pyy = signal.welch(roi_time_series[j], fs=fs)
            _, Pxy = signal.csd(roi_time_series[i], roi_time_series[j], fs=fs)
            with np.errstate(invalid="ignore", divide="ignore"):
                icoh = np.abs(np.imag(Pxy)) / np.sqrt(Pxx * Pyy)
            icoh = np.nan_to_num(icoh, nan=0.0, posinf=0.0, neginf=0.0)
            val = float(np.mean(icoh))
            connectivity_matrix[i, j] = val
            connectivity_matrix[j, i] = val

    np.fill_diagonal(connectivity_matrix, 0.0)
    return connectivity_matrix.astype(np.float32)


# ============================================================
# AEC — Amplitude Envelope Correlation
# ============================================================

def compute_aec(roi_time_series: np.ndarray) -> np.ndarray:
    """
    Compute the ROI × ROI connectivity matrix using Amplitude Envelope
    Correlation (AEC).

    The analytic signal is obtained via the Hilbert transform; the instantaneous
    amplitude (envelope) of each ROI time series is then compared with Pearson
    correlation.

    The matrix is symmetric with 1 on the diagonal.
    """
    validate_roi_time_series(roi_time_series)

    analytic = signal.hilbert(roi_time_series.astype(np.float64), axis=1)
    envelopes = np.abs(analytic)

    conn = np.corrcoef(envelopes, rowvar=True)
    conn = np.nan_to_num(conn, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(conn, 1.0)
    return conn.astype(np.float32)


def compute_aec_orth(roi_time_series: np.ndarray) -> np.ndarray:
    """
    Compute the ROI × ROI connectivity matrix using Orthogonalized AEC (AEC-c).

    The orthogonalization step removes the zero-lag (spurious) component of
    amplitude coupling by projecting each signal onto the direction orthogonal
    to the other signal in the analytic representation:

        x_orth = Im(x_a · ȳ_a) / |y_a|

    The two directed AEC values (x orth. w.r.t. y, and y orth. w.r.t. x) are
    averaged to yield a symmetric, undirected estimate (Hipp et al. 2012).
    """
    validate_roi_time_series(roi_time_series)

    analytic = signal.hilbert(roi_time_series.astype(np.float64), axis=1)
    num_rois = roi_time_series.shape[0]
    conn = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(i, num_rois):
            env_j = np.abs(analytic[j])
            env_i = np.abs(analytic[i])

            # Orthogonalize i w.r.t. j
            orth_ij = np.imag(analytic[i] * np.conj(analytic[j])) / (env_j + 1e-15)
            r1 = float(np.corrcoef(np.abs(orth_ij), env_j)[0, 1])

            # Orthogonalize j w.r.t. i
            orth_ji = np.imag(analytic[j] * np.conj(analytic[i])) / (env_i + 1e-15)
            r2 = float(np.corrcoef(np.abs(orth_ji), env_i)[0, 1])

            val = float(np.nanmean([r1, r2]))
            conn[i, j] = val
            conn[j, i] = val

    conn = np.nan_to_num(conn, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(conn, 1.0)
    return conn


# ============================================================
# dWPLI — Debiased Weighted Phase Lag Index
# ============================================================

def _dwpli_from_cross_spectrum(imag_cs: np.ndarray) -> float:
    """
    Compute the debiased WPLI scalar from a vector of imaginary cross-spectrum
    values (Vinck et al. 2011, NeuroImage).

    dWPLI = (E[Im C]² − E[Im C²] / n) / (E[|Im C|]² − E[Im C²] / n)
    """
    n = len(imag_cs)
    if n < 2:
        return 0.0
    mean_imag = float(np.mean(imag_cs))
    mean_sq_imag = float(np.mean(imag_cs ** 2))
    mean_abs_imag = float(np.mean(np.abs(imag_cs)))

    numer = mean_imag ** 2 - mean_sq_imag / n
    denom = mean_abs_imag ** 2 - mean_sq_imag / n
    if abs(denom) < 1e-15:
        return 0.0
    return float(numer / denom)


def compute_dwpli(roi_time_series: np.ndarray) -> np.ndarray:
    """
    Compute the ROI × ROI connectivity matrix using the Debiased Weighted Phase
    Lag Index (dWPLI; Vinck et al. 2011).

    Unlike WPLI, dWPLI removes the positive bias that arises from finite sample
    sizes, making it a more statistically robust estimator of phase synchrony.

    The matrix is symmetric. The diagonal is 0 (no self-coupling).
    """
    validate_roi_time_series(roi_time_series)

    analytic = signal.hilbert(roi_time_series.astype(np.float64), axis=1)
    num_rois = roi_time_series.shape[0]
    conn = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(i + 1, num_rois):
            cross_spectrum = analytic[i] * np.conj(analytic[j])
            imag_cs = np.imag(cross_spectrum)
            val = _dwpli_from_cross_spectrum(imag_cs)
            conn[i, j] = val
            conn[j, i] = val

    return conn


# ============================================================
# PPC — Pairwise Phase Consistency
# ============================================================

def compute_ppc(roi_time_series: np.ndarray) -> np.ndarray:
    """
    Compute the ROI × ROI connectivity matrix using Pairwise Phase Consistency
    (PPC; Vinck et al. 2010).

    PPC is an unbiased estimator of squared PLV that removes the positive bias
    of PLV at finite sample sizes:

        PPC = (T · PLV² − 1) / (T − 1)

    Values are clipped to [0, 1]. The matrix is symmetric with 1 on the diagonal.
    """
    validate_roi_time_series(roi_time_series)

    phases = _instantaneous_phases(roi_time_series)
    num_rois, T = roi_time_series.shape
    conn = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(i, num_rois):
            phase_diff = phases[i] - phases[j]
            plv2 = float(np.abs(np.mean(np.exp(1j * phase_diff))) ** 2)
            ppc = (T * plv2 - 1.0) / (T - 1.0)
            val = float(np.clip(ppc, 0.0, 1.0))
            conn[i, j] = val
            conn[j, i] = val

    np.fill_diagonal(conn, 1.0)
    return conn


# ============================================================
# Mutual Information
# ============================================================

def _mi_pair(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    Estimate mutual information between two 1-D signals using a joint histogram.

    MI = Σ_{x,y} p(x,y) log[ p(x,y) / (p(x) p(y)) ]
    """
    hist2d, _, _ = np.histogram2d(x, y, bins=n_bins)
    pxy = hist2d / (hist2d.sum() + 1e-15)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    mask = pxy > 0
    mi = float(np.sum(pxy[mask] * np.log(pxy[mask] / (px * py)[mask])))
    return max(0.0, mi)


def compute_mutual_information(
    roi_time_series: np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    """
    Compute the ROI × ROI connectivity matrix using Mutual Information (MI).

    MI captures non-linear statistical dependencies between signals, making it
    more general than Pearson correlation. MI is estimated via a joint histogram
    with ``n_bins`` bins per axis.

    The matrix is symmetric. The diagonal contains MI(x, x) = H(x) (entropy).
    """
    validate_roi_time_series(roi_time_series)

    num_rois = roi_time_series.shape[0]
    conn = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(i, num_rois):
            val = _mi_pair(roi_time_series[i], roi_time_series[j], n_bins)
            conn[i, j] = val
            conn[j, i] = val

    return conn


# ============================================================
# Synchronisation Likelihood
# ============================================================

def _sync_likelihood_pair(
    x: np.ndarray,
    y: np.ndarray,
    m: int = 3,
    tau: int = 1,
    pref: float = 0.05,
) -> float:
    """
    Estimate the Synchronisation Likelihood (SL) between two signals
    (Stam & van Dijk 2002).

    SL measures the probability that two simultaneously embedded time series
    visit nearby states together, beyond what would be expected by chance.

    Parameters
    ----------
    m    : embedding dimension (default 3)
    tau  : time delay for embedding (default 1 sample)
    pref : reference proportion (fraction of nearest neighbours; default 0.05)
    """
    embed_len = len(x) - (m - 1) * tau
    if embed_len < 10:
        return 0.0

    # Phase-space embedding: shape (embed_len, m)
    X_embed = np.column_stack([x[k * tau: k * tau + embed_len] for k in range(m)])
    Y_embed = np.column_stack([y[k * tau: k * tau + embed_len] for k in range(m)])

    Dx = cdist(X_embed, X_embed, metric="chebyshev")
    Dy = cdist(Y_embed, Y_embed, metric="chebyshev")

    # Exclude self-distances
    np.fill_diagonal(Dx, np.inf)
    np.fill_diagonal(Dy, np.inf)

    finite_dx = Dx[np.isfinite(Dx)]
    finite_dy = Dy[np.isfinite(Dy)]
    if finite_dx.size == 0 or finite_dy.size == 0:
        return 0.0

    eps_x = np.percentile(finite_dx, pref * 100)
    eps_y = np.percentile(finite_dy, pref * 100)

    # Recurrence matrices
    Rx = (Dx < eps_x).astype(np.float64)
    Ry = (Dy < eps_y).astype(np.float64)

    n = embed_len
    sl_raw = (Rx * Ry).sum() / (n * (n - 1)) - pref ** 2
    max_sl = pref * (1.0 - pref)
    if max_sl <= 0:
        return 0.0
    return float(np.clip(sl_raw / max_sl, 0.0, 1.0))


def compute_sync_likelihood(
    roi_time_series: np.ndarray,
    m: int = 3,
    tau: int = 1,
    pref: float = 0.05,
) -> np.ndarray:
    """
    Compute the ROI × ROI connectivity matrix using Synchronisation Likelihood
    (SL; Stam & van Dijk 2002).

    SL is sensitive to both linear and non-linear coupling. It measures how
    often two phase-space trajectories are simultaneously in nearby regions.

    Parameters
    ----------
    m    : embedding dimension.
    tau  : embedding delay in samples.
    pref : reference probability (fraction of nearest-neighbour states).

    The matrix is symmetric, values in [0, 1]. The diagonal is 1.
    """
    validate_roi_time_series(roi_time_series)

    num_rois = roi_time_series.shape[0]
    conn = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(i, num_rois):
            val = _sync_likelihood_pair(
                roi_time_series[i], roi_time_series[j], m=m, tau=tau, pref=pref
            )
            conn[i, j] = val
            conn[j, i] = val

    np.fill_diagonal(conn, 1.0)
    return conn


# ============================================================
# Lagged Coherence (TR required)
# ============================================================

def compute_lagged_coherence(roi_time_series: np.ndarray, tr: float) -> np.ndarray:
    """
    Compute the ROI × ROI connectivity matrix using Lagged Coherence (LaC).

    Lagged coherence uses only the imaginary (lagged) part of the cross-spectrum,
    which is insensitive to zero-lag correlations (Pascual-Marqui et al.):

        LaC[i, j] = sqrt( mean_f( Im(P_xy(f))² / (P_xx(f) · P_yy(f)) ) )

    The square root maps the measure to the same scale as ordinary coherence.
    The matrix is symmetric with 0 on the diagonal.

    Parameters
    ----------
    tr : float
        Repetition time in seconds (required).
    """
    validate_roi_time_series(roi_time_series)
    if tr is None or tr <= 0:
        raise ConnectivityError(
            "A positive TR (repetition time, seconds) is required for lagged "
            "coherence computation."
        )

    fs = 1.0 / tr
    num_rois = roi_time_series.shape[0]
    conn = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(i, num_rois):
            _, Pxx = signal.welch(roi_time_series[i], fs=fs)
            _, Pyy = signal.welch(roi_time_series[j], fs=fs)
            _, Pxy = signal.csd(roi_time_series[i], roi_time_series[j], fs=fs)
            with np.errstate(invalid="ignore", divide="ignore"):
                lac = np.imag(Pxy) ** 2 / (Pxx * Pyy)
            lac = np.nan_to_num(lac, nan=0.0, posinf=0.0, neginf=0.0)
            val = float(np.sqrt(np.mean(np.abs(lac))))
            conn[i, j] = val
            conn[j, i] = val

    np.fill_diagonal(conn, 0.0)
    return conn.astype(np.float32)


# ============================================================
# Granger Causality (directed)
# ============================================================

def _ar_residual_var(y: np.ndarray, lag: int) -> float:
    """Fit a univariate AR(lag) model and return the residual variance."""
    T = len(y)
    X = np.column_stack([y[lag - k: T - k] for k in range(1, lag + 1)])
    y_t = y[lag:]
    coeff, _, _, _ = np.linalg.lstsq(X, y_t, rcond=None)
    residuals = y_t - X @ coeff
    return float(np.var(residuals, ddof=lag))


def _arx_residual_var(y: np.ndarray, x: np.ndarray, lag: int) -> float:
    """Fit a bivariate ARX(lag) model (y | x) and return the residual variance."""
    T = len(y)
    X = np.column_stack(
        [y[lag - k: T - k] for k in range(1, lag + 1)]
        + [x[lag - k: T - k] for k in range(1, lag + 1)]
    )
    y_t = y[lag:]
    coeff, _, _, _ = np.linalg.lstsq(X, y_t, rcond=None)
    residuals = y_t - X @ coeff
    return float(np.var(residuals, ddof=2 * lag))


def compute_granger_causality(
    roi_time_series: np.ndarray,
    model_order: int = 1,
) -> np.ndarray:
    """
    Compute the ROI × ROI directed connectivity matrix using bivariate linear
    Granger Causality (GC).

    GC from ROI j to ROI i is:

        GC[i, j] = log( Var(ε_i | AR(i)) / Var(ε_i | ARX(i, j)) )

    A positive value indicates that past values of j improve the linear
    prediction of i beyond i's own past alone.

    The matrix is **asymmetric** (directed). The diagonal is 0.

    Parameters
    ----------
    model_order : int
        Order of the AR / ARX models (number of past lags). Default 1.
    """
    validate_roi_time_series(roi_time_series)

    num_rois = roi_time_series.shape[0]
    conn = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        var_ar = _ar_residual_var(roi_time_series[i], model_order)
        for j in range(num_rois):
            if i == j:
                continue
            var_arx = _arx_residual_var(roi_time_series[i], roi_time_series[j], model_order)
            if var_arx > 0 and var_ar > 0:
                conn[i, j] = float(max(0.0, np.log(var_ar / var_arx)))

    return conn


# ============================================================
# Transfer Entropy (directed)
# ============================================================

def _discretize_signal(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Map signal values onto integer bin indices in [0, n_bins-1]."""
    x_min, x_max = float(x.min()), float(x.max())
    if x_max == x_min:
        return np.zeros(len(x), dtype=np.intp)
    idx = np.floor(n_bins * (x - x_min) / (x_max - x_min)).astype(np.intp)
    return np.clip(idx, 0, n_bins - 1)


def _transfer_entropy_pair(
    x: np.ndarray,
    y: np.ndarray,
    lag: int = 1,
    n_bins: int = 8,
) -> float:
    """
    Estimate Transfer Entropy from Y to X via a joint histogram:

        TE(Y → X) = I(X_future; Y_past | X_past)
    """
    T = len(x)
    if T - lag < 2:
        return 0.0

    xf = _discretize_signal(x[lag:], n_bins)       # X_future
    xp = _discretize_signal(x[: T - lag], n_bins)  # X_past
    yp = _discretize_signal(y[: T - lag], n_bins)  # Y_past
    n = len(xf)

    # Joint counts p(x_future, x_past, y_past)
    joint = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
    np.add.at(joint, (xf, xp, yp), 1.0)
    joint /= n

    p_xf_xp = joint.sum(axis=2)   # p(x_future, x_past)
    p_xp_yp = joint.sum(axis=0)   # p(x_past,   y_past)
    p_xp = p_xf_xp.sum(axis=0)    # p(x_past)

    te = 0.0
    for idx in zip(*np.where(joint > 0)):
        i, j, k = idx
        numer = float(joint[i, j, k]) * float(p_xp[j])
        denom = float(p_xf_xp[i, j]) * float(p_xp_yp[j, k])
        if denom > 0 and numer > 0:
            te += float(joint[i, j, k]) * np.log2(numer / denom)

    return float(max(0.0, te))


def compute_transfer_entropy(
    roi_time_series: np.ndarray,
    lag: int = 1,
    n_bins: int = 8,
) -> np.ndarray:
    """
    Compute the ROI × ROI directed connectivity matrix using Transfer Entropy (TE).

    TE(Y → X) measures the information transferred from the past of Y to the
    future of X, beyond X's own past. It is a model-free, non-linear alternative
    to Granger Causality.

        TE[i, j] = TE(j → i)

    The matrix is **asymmetric** (directed). The diagonal is 0.

    Parameters
    ----------
    lag    : time lag in samples (default 1).
    n_bins : number of histogram bins for density estimation (default 8).
    """
    validate_roi_time_series(roi_time_series)

    num_rois = roi_time_series.shape[0]
    conn = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(num_rois):
            if i == j:
                continue
            # TE from j → i  (j = Y, i = X)
            conn[i, j] = _transfer_entropy_pair(
                roi_time_series[i], roi_time_series[j], lag=lag, n_bins=n_bins
            )

    return conn


# ============================================================
# Partial Directed Coherence (TR and model_order required)
# ============================================================

def _fit_mvar(data: np.ndarray, order: int):
    """
    Fit a Multivariate AutoRegressive (MVAR) model of given order using OLS.

    Parameters
    ----------
    data  : ndarray (N, T) — ROI × time matrix.
    order : model order p.

    Returns
    -------
    A_list : list of p coefficient matrices, each (N, N).
             y(t) ≈ A_list[0] @ y(t-1) + ... + A_list[p-1] @ y(t-p)
    Sigma  : noise covariance matrix (N, N).
    """
    N, T = data.shape
    T_eff = T - order

    Y = data[:, order:]                                # (N, T_eff)
    X = np.vstack([data[:, order - k - 1: T - k - 1]  # (N·p, T_eff)
                   for k in range(order)])

    A_flat, _, _, _ = np.linalg.lstsq(X.T, Y.T, rcond=None)
    A_flat = A_flat.T                                  # (N, N·p)

    A_list = [A_flat[:, k * N: (k + 1) * N] for k in range(order)]

    residuals = Y - A_flat @ X
    Sigma = (residuals @ residuals.T) / T_eff
    return A_list, Sigma


def compute_pdc(
    roi_time_series: np.ndarray,
    tr: float,
    model_order: int = 1,
    nfft: int = 128,
) -> np.ndarray:
    """
    Compute the ROI × ROI directed connectivity matrix using Partial Directed
    Coherence (PDC; Baccalá & Sameshima 2001).

    PDC is derived from the MVAR model transfer matrix A(f):

        PDC[i, j](f) = |A_{ij}(f)| / sqrt( Σ_k |A_{kj}(f)|² )

    where A(f) = I − Σ_k A_k · exp(−j·2π·f·k·TR).

    PDC[i, j] > 0 indicates a direct causal link from ROI j to ROI i.
    Values are averaged over all frequencies 0 – Nyquist.

    The matrix is **asymmetric** (directed). Values in [0, 1].

    Parameters
    ----------
    tr          : repetition time in seconds (required).
    model_order : MVAR model order p (default 1).
    nfft        : number of frequency points (default 128).
    """
    validate_roi_time_series(roi_time_series)
    if tr is None or tr <= 0:
        raise ConnectivityError(
            "A positive TR (repetition time, seconds) is required for PDC computation."
        )

    N = roi_time_series.shape[0]
    A_list, _ = _fit_mvar(roi_time_series.astype(np.float64), model_order)

    freqs = np.fft.rfftfreq(nfft, d=tr)
    n_freqs = len(freqs)

    pdc_accum = np.zeros((N, N), dtype=np.float64)

    for freq in freqs:
        # A(f) = I − Σ_k A_k · exp(−j·2π·f·k·TR)
        Af = np.eye(N, dtype=complex)
        for k, Ak in enumerate(A_list, 1):
            Af -= Ak * np.exp(-2j * np.pi * freq * k * tr)

        # PDC_{ij}(f) = |Af_{ij}| / ||column j of Af||
        col_norms = np.sqrt(np.sum(np.abs(Af) ** 2, axis=0))  # (N,)
        col_norms[col_norms == 0] = 1.0
        pdc_f = np.abs(Af) / col_norms[np.newaxis, :]         # (N, N)
        pdc_accum += pdc_f

    return (pdc_accum / n_freqs).astype(np.float32)


# ============================================================
# Phase Slope Index (TR required, directed)
# ============================================================

def compute_psi(
    roi_time_series: np.ndarray,
    tr: float,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the ROI × ROI directed connectivity matrix using the Phase Slope
    Index (PSI; Nolte et al. 2008).

    PSI[i, j] = Im( Σ_f  C*_{ij}(f) · C_{ij}(f + Δf) )

    where C_{ij}(f) is the complex coherence at frequency f.

    A positive PSI[i, j] means that ROI j **leads** ROI i (j is the driver).

    The matrix is **asymmetric** and antisymmetric: PSI[i,j] = −PSI[j,i].

    Parameters
    ----------
    tr   : repetition time in seconds (required).
    fmin : lower frequency bound in Hz (default 0).
    fmax : upper frequency bound in Hz (default Nyquist).
    """
    validate_roi_time_series(roi_time_series)
    if tr is None or tr <= 0:
        raise ConnectivityError(
            "A positive TR (repetition time, seconds) is required for PSI computation."
        )

    fs = 1.0 / tr
    if fmax is None:
        fmax = fs / 2.0

    num_rois, T = roi_time_series.shape
    nperseg = min(T, 64)
    conn = np.zeros((num_rois, num_rois), dtype=np.float32)

    for i in range(num_rois):
        for j in range(i + 1, num_rois):
            _, Pxx = signal.welch(roi_time_series[i], fs=fs, nperseg=nperseg)
            _, Pyy = signal.welch(roi_time_series[j], fs=fs, nperseg=nperseg)
            f, Pxy = signal.csd(roi_time_series[i], roi_time_series[j], fs=fs, nperseg=nperseg)

            with np.errstate(invalid="ignore", divide="ignore"):
                Cxy = Pxy / np.sqrt(Pxx * Pyy)
            Cxy = np.nan_to_num(Cxy, nan=0.0, posinf=0.0, neginf=0.0)

            band = (f >= fmin) & (f <= fmax)
            Cxy_b = Cxy[band]

            if len(Cxy_b) < 2:
                continue

            # PSI = Im( Σ_f C*(f) · C(f+Δf) )
            psi_val = float(np.imag(np.sum(np.conj(Cxy_b[:-1]) * Cxy_b[1:])))
            conn[i, j] = psi_val
            conn[j, i] = -psi_val  # antisymmetric by construction

    return conn


def apply_connectivity_threshold(
    connectivity_matrix: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Apply an absolute threshold to a connectivity matrix.

    Strategy:
    - values whose absolute value is >= threshold are kept,
    - all others are set to 0,
    - the diagonal is NOT forced to 1, to avoid artificially altering
      measures that do not naturally produce a unit diagonal.
    """
    if threshold < 0:
        raise ConnectivityError(
            "The connectivity threshold cannot be negative."
        )

    thresholded_matrix = np.array(connectivity_matrix, copy=True)
    mask = np.abs(thresholded_matrix) < threshold
    thresholded_matrix[mask] = 0.0

    return thresholded_matrix.astype(np.float32)
