from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .exceptions import ConnectivityError
from .tools import (
    compute_aec,
    compute_aec_orth,
    compute_coherence,
    compute_corrected_cross_correlation,
    compute_cross_correlation,
    compute_granger_causality,
    compute_imaginary_coherence,
    compute_lagged_coherence,
    compute_mutual_information,
    compute_partial_correlation,
    compute_pdc,
    compute_pearson_correlation,
    compute_psi,
    compute_sync_likelihood,
    compute_transfer_entropy,
    validate_connectivity_method,
)


class ConnectivityStrategy(ABC):
    """
    Abstract base class for connectivity strategies.

    Any concrete strategy must implement the compute() method.
    """

    @abstractmethod
    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        """
        Compute a connectivity matrix from an ROI × time series matrix.

        Parameters
        ----------
        roi_time_series : ndarray of shape (N, T)
            ROI time series where N is the number of ROIs and T is the number
            of timepoints.

        Returns
        -------
        ndarray of shape (N, N)
            Connectivity matrix.
        """


class PearsonConnectivityStrategy(ConnectivityStrategy):
    """Connectivity strategy based on Pearson correlation."""

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_pearson_correlation(roi_time_series)

    def __repr__(self) -> str:
        return "PearsonConnectivityStrategy()"


class CrossCorrelationConnectivityStrategy(ConnectivityStrategy):
    """Connectivity strategy based on normalised cross-correlation."""

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_cross_correlation(roi_time_series)

    def __repr__(self) -> str:
        return "CrossCorrelationConnectivityStrategy()"


class CorrectedCrossCorrelationConnectivityStrategy(ConnectivityStrategy):
    """Connectivity strategy based on corrected cross-correlation."""

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_corrected_cross_correlation(roi_time_series)

    def __repr__(self) -> str:
        return "CorrectedCrossCorrelationConnectivityStrategy()"


class PartialCorrelationConnectivityStrategy(ConnectivityStrategy):
    """Connectivity strategy based on partial correlation (precision matrix)."""

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_partial_correlation(roi_time_series)

    def __repr__(self) -> str:
        return "PartialCorrelationConnectivityStrategy()"


class CoherenceConnectivityStrategy(ConnectivityStrategy):
    """Connectivity strategy based on magnitude-squared coherence. Requires TR."""

    def __init__(self, tr: float) -> None:
        self._tr = tr

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_coherence(roi_time_series, self._tr)

    def __repr__(self) -> str:
        return f"CoherenceConnectivityStrategy(tr={self._tr})"


class ImaginaryCoherenceConnectivityStrategy(ConnectivityStrategy):
    """Connectivity strategy based on imaginary coherence. Requires TR."""

    def __init__(self, tr: float) -> None:
        self._tr = tr

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_imaginary_coherence(roi_time_series, self._tr)

    def __repr__(self) -> str:
        return f"ImaginaryCoherenceConnectivityStrategy(tr={self._tr})"


class AECConnectivityStrategy(ConnectivityStrategy):
    """Connectivity strategy based on Amplitude Envelope Correlation."""

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_aec(roi_time_series)

    def __repr__(self) -> str:
        return "AECConnectivityStrategy()"


class AECOrthConnectivityStrategy(ConnectivityStrategy):
    """Connectivity strategy based on Orthogonalized AEC (AEC-c)."""

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_aec_orth(roi_time_series)

    def __repr__(self) -> str:
        return "AECOrthConnectivityStrategy()"


class MutualInformationConnectivityStrategy(ConnectivityStrategy):
    """Connectivity strategy based on histogram-estimated Mutual Information."""

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_mutual_information(roi_time_series)

    def __repr__(self) -> str:
        return "MutualInformationConnectivityStrategy()"


class SyncLikelihoodConnectivityStrategy(ConnectivityStrategy):
    """Connectivity strategy based on Synchronisation Likelihood."""

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_sync_likelihood(roi_time_series)

    def __repr__(self) -> str:
        return "SyncLikelihoodConnectivityStrategy()"


class LaggedCoherenceConnectivityStrategy(ConnectivityStrategy):
    """Connectivity strategy based on Lagged Coherence. Requires TR."""

    def __init__(self, tr: float) -> None:
        self._tr = tr

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_lagged_coherence(roi_time_series, self._tr)

    def __repr__(self) -> str:
        return f"LaggedCoherenceConnectivityStrategy(tr={self._tr})"


class GrangerCausalityConnectivityStrategy(ConnectivityStrategy):
    """Directed connectivity strategy based on bivariate Granger Causality."""

    def __init__(self, model_order: int = 1) -> None:
        self._model_order = model_order

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_granger_causality(roi_time_series, model_order=self._model_order)

    def __repr__(self) -> str:
        return f"GrangerCausalityConnectivityStrategy(model_order={self._model_order})"


class TransferEntropyConnectivityStrategy(ConnectivityStrategy):
    """Directed connectivity strategy based on Transfer Entropy."""

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_transfer_entropy(roi_time_series)

    def __repr__(self) -> str:
        return "TransferEntropyConnectivityStrategy()"


class PDCConnectivityStrategy(ConnectivityStrategy):
    """Directed connectivity strategy based on Partial Directed Coherence. Requires TR."""

    def __init__(self, tr: float, model_order: int = 1) -> None:
        self._tr = tr
        self._model_order = model_order

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_pdc(roi_time_series, tr=self._tr, model_order=self._model_order)

    def __repr__(self) -> str:
        return f"PDCConnectivityStrategy(tr={self._tr}, model_order={self._model_order})"


class PSIConnectivityStrategy(ConnectivityStrategy):
    """Directed connectivity strategy based on Phase Slope Index. Requires TR."""

    def __init__(self, tr: float) -> None:
        self._tr = tr

    def compute(self, roi_time_series: np.ndarray) -> np.ndarray:
        return compute_psi(roi_time_series, tr=self._tr)

    def __repr__(self) -> str:
        return f"PSIConnectivityStrategy(tr={self._tr})"


def get_connectivity_strategy(method: str, tr: Optional[float] = None,
                               model_order: int = 1) -> ConnectivityStrategy:
    """
    Return the connectivity strategy for the requested method name.

    Undirected (no TR):
        pearson_correlation, cross_correlation, corr_cross_correlation,
        partial_correlation, aec, aec_orth, mutual_information, sync_likelihood

    Undirected (TR required):
        coherence, imag_coherence, lagged_coherence

    Directed (no TR):
        granger_causality (uses model_order), transfer_entropy

    Directed (TR required):
        pdc (uses model_order), psi

    Parameters
    ----------
    method      : Connectivity method name; aliases accepted.
    tr          : Repetition time in seconds. Required for spectral methods.
    model_order : AR / MVAR model order for GC and PDC (default 1).

    Raises
    ------
    ConnectivityError
        Unknown method, or TR missing for a spectral method.
    """
    from .tools import SPECTRAL_MEASURES
    normalized = validate_connectivity_method(method)

    if normalized in SPECTRAL_MEASURES and (tr is None or tr <= 0):
        raise ConnectivityError(
            f"Method '{normalized}' requires a positive TR "
            "(repetition time in seconds). Set tr in ConnectivityConfig."
        )

    _map = {
        "pearson_correlation":   PearsonConnectivityStrategy,
        "cross_correlation":     CrossCorrelationConnectivityStrategy,
        "corr_cross_correlation": CorrectedCrossCorrelationConnectivityStrategy,
        "partial_correlation":   PartialCorrelationConnectivityStrategy,
        "aec":                   AECConnectivityStrategy,
        "aec_orth":              AECOrthConnectivityStrategy,
        "mutual_information":    MutualInformationConnectivityStrategy,
        "sync_likelihood":       SyncLikelihoodConnectivityStrategy,
        "transfer_entropy":      TransferEntropyConnectivityStrategy,
    }
    if normalized in _map:
        return _map[normalized]()

    if normalized == "coherence":
        return CoherenceConnectivityStrategy(tr)
    if normalized == "imag_coherence":
        return ImaginaryCoherenceConnectivityStrategy(tr)
    if normalized == "lagged_coherence":
        return LaggedCoherenceConnectivityStrategy(tr)
    if normalized == "granger_causality":
        return GrangerCausalityConnectivityStrategy(model_order=model_order)
    if normalized == "pdc":
        return PDCConnectivityStrategy(tr=tr, model_order=model_order)
    if normalized == "psi":
        return PSIConnectivityStrategy(tr=tr)

    # Safety net — unreachable when validate_connectivity_method is called first.
    raise ConnectivityError(f"Unrecognised connectivity method: '{normalized}'.")
