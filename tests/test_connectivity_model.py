"""
Tests for ModelBRAPHINConnectivityData, _compute_sliding_window_dfc, and
strategy __repr__ methods.

Covers:
- Sliding-window dFC: basic output, error cases (window too small, too large,
  no windows fit)
- ModelBRAPHINConnectivityData: static, windowed, display_info
- Validation errors: None bundle, no ROI time series, wrong ndim
- Strategy __repr__ for every concrete strategy class
"""

import logging

import numpy as np
import pytest

from braphin.config import ConnectivityConfig
from braphin.connectivity import (
    BRAPHINConnectivityBundle,
    ModelBRAPHINConnectivityData,
    _compute_sliding_window_dfc,
)
from braphin.exceptions import ConnectivityError
from braphin.strategy import (
    AECConnectivityStrategy,
    AECOrthConnectivityStrategy,
    CoherenceConnectivityStrategy,
    CorrectedCrossCorrelationConnectivityStrategy,
    CrossCorrelationConnectivityStrategy,
    GrangerCausalityConnectivityStrategy,
    ImaginaryCoherenceConnectivityStrategy,
    LaggedCoherenceConnectivityStrategy,
    MutualInformationConnectivityStrategy,
    PDCConnectivityStrategy,
    PSIConnectivityStrategy,
    PartialCorrelationConnectivityStrategy,
    PearsonConnectivityStrategy,
    SyncLikelihoodConnectivityStrategy,
    TransferEntropyConnectivityStrategy,
)
from braphin.transform import BRAPHINTransformBundle

N = 6
T = 100


@pytest.fixture(scope="module")
def roi_ts():
    rng = np.random.default_rng(0)
    return rng.random((N, T)).astype(np.float32)


@pytest.fixture(scope="module")
def transform_bundle(roi_ts):
    return BRAPHINTransformBundle(
        roi_time_series=roi_ts,
        roi_labels=[f"ROI_{i}" for i in range(N)],
        atlas_name="aal",
    )


# ---------------------------------------------------------------------------
# _compute_sliding_window_dfc
# ---------------------------------------------------------------------------

def test_sliding_window_returns_3d_array(roi_ts):
    strategy = PearsonConnectivityStrategy()
    mats, centers = _compute_sliding_window_dfc(roi_ts, strategy, window_size=20.0, tr=1.0)
    assert mats.ndim == 3
    assert mats.shape[1] == N
    assert mats.shape[2] == N


def test_sliding_window_centers_match_windows(roi_ts):
    strategy = PearsonConnectivityStrategy()
    mats, centers = _compute_sliding_window_dfc(roi_ts, strategy, window_size=20.0, tr=1.0)
    assert len(centers) == mats.shape[0]


def test_sliding_window_custom_step(roi_ts):
    strategy = PearsonConnectivityStrategy()
    mats_default, _ = _compute_sliding_window_dfc(roi_ts, strategy, window_size=20.0, tr=1.0)
    mats_step, _ = _compute_sliding_window_dfc(
        roi_ts, strategy, window_size=20.0, tr=1.0, step_size=10.0
    )
    assert mats_step.shape[0] >= mats_default.shape[0]


def test_sliding_window_too_small_raises(roi_ts):
    strategy = PearsonConnectivityStrategy()
    with pytest.raises(ConnectivityError, match="Window size"):
        _compute_sliding_window_dfc(roi_ts, strategy, window_size=0.5, tr=1.0)


def test_sliding_window_too_large_raises(roi_ts):
    strategy = PearsonConnectivityStrategy()
    with pytest.raises(ConnectivityError, match="time series length"):
        _compute_sliding_window_dfc(roi_ts, strategy, window_size=200.0, tr=1.0)


def test_sliding_window_result_dtype(roi_ts):
    strategy = PearsonConnectivityStrategy()
    mats, _ = _compute_sliding_window_dfc(roi_ts, strategy, window_size=20.0, tr=1.0)
    assert mats.dtype in (np.float32, np.float64)


# ---------------------------------------------------------------------------
# ModelBRAPHINConnectivityData — windowed (dynamic) mode
# ---------------------------------------------------------------------------

def test_model_windowed_returns_bundle(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation", window_size=20.0, tr=1.0)
    bundle = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert isinstance(bundle, BRAPHINConnectivityBundle)


def test_model_windowed_has_dynamic_matrices(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation", window_size=20.0, tr=1.0)
    bundle = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert bundle.dynamic_connectivity_matrices is not None
    assert bundle.dynamic_connectivity_matrices.ndim == 3


def test_model_windowed_applied_steps(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation", window_size=20.0, tr=1.0)
    bundle = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert "windowed_dynamic_connectivity" in bundle.applied_steps
    assert "mean_over_windows" in bundle.applied_steps


def test_model_windowed_window_centers(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation", window_size=20.0, tr=1.0)
    bundle = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert bundle.window_centers_sec is not None
    assert len(bundle.window_centers_sec) == bundle.dynamic_connectivity_matrices.shape[0]


def test_model_windowed_missing_tr_raises(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation", window_size=20.0)
    with pytest.raises(ConnectivityError, match="tr"):
        ModelBRAPHINConnectivityData(transform_bundle, cfg).run()


def test_model_windowed_with_threshold(transform_bundle):
    cfg = ConnectivityConfig(
        method="pearson_correlation", window_size=20.0, tr=1.0, threshold=0.3
    )
    bundle = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert "threshold" in bundle.applied_steps


# ---------------------------------------------------------------------------
# display_info
# ---------------------------------------------------------------------------

def test_display_info_static(transform_bundle, caplog):
    cfg = ConnectivityConfig(method="pearson_correlation")
    model = ModelBRAPHINConnectivityData(transform_bundle, cfg)
    bundle = model.run()
    with caplog.at_level(logging.INFO):
        model.display_info(bundle)


def test_display_info_windowed(transform_bundle, caplog):
    cfg = ConnectivityConfig(method="pearson_correlation", window_size=20.0, tr=1.0)
    model = ModelBRAPHINConnectivityData(transform_bundle, cfg)
    bundle = model.run()
    with caplog.at_level(logging.INFO):
        model.display_info(bundle)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_none_bundle_raises():
    with pytest.raises(ConnectivityError):
        ModelBRAPHINConnectivityData(None).run()


def test_no_roi_time_series_raises():
    tb = BRAPHINTransformBundle(roi_time_series=None)
    with pytest.raises(ConnectivityError):
        ModelBRAPHINConnectivityData(tb).run()


def test_non_ndarray_roi_ts_raises():
    tb = BRAPHINTransformBundle(roi_time_series=[[1, 2], [3, 4]])
    with pytest.raises(ConnectivityError):
        ModelBRAPHINConnectivityData(tb).run()


def test_1d_roi_ts_raises():
    tb = BRAPHINTransformBundle(roi_time_series=np.zeros(10))
    with pytest.raises(ConnectivityError):
        ModelBRAPHINConnectivityData(tb).run()


# ---------------------------------------------------------------------------
# Strategy .compute() wrappers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_roi_ts():
    rng = np.random.default_rng(42)
    return rng.random((4, T)).astype(np.float32)


def test_partial_strategy_compute(small_roi_ts):
    m = PartialCorrelationConnectivityStrategy().compute(small_roi_ts)
    assert m.shape == (4, 4)


def test_coherence_strategy_compute(small_roi_ts):
    m = CoherenceConnectivityStrategy(tr=2.0).compute(small_roi_ts)
    assert m.shape == (4, 4)


def test_imag_coherence_strategy_compute(small_roi_ts):
    m = ImaginaryCoherenceConnectivityStrategy(tr=2.0).compute(small_roi_ts)
    assert m.shape == (4, 4)


def test_aec_strategy_compute(small_roi_ts):
    m = AECConnectivityStrategy().compute(small_roi_ts)
    assert m.shape == (4, 4)


def test_aec_orth_strategy_compute(small_roi_ts):
    m = AECOrthConnectivityStrategy().compute(small_roi_ts)
    assert m.shape == (4, 4)


def test_mutual_information_strategy_compute(small_roi_ts):
    m = MutualInformationConnectivityStrategy().compute(small_roi_ts)
    assert m.shape == (4, 4)


def test_sync_likelihood_strategy_compute(small_roi_ts):
    m = SyncLikelihoodConnectivityStrategy().compute(small_roi_ts)
    assert m.shape == (4, 4)


def test_lagged_coherence_strategy_compute(small_roi_ts):
    m = LaggedCoherenceConnectivityStrategy(tr=2.0).compute(small_roi_ts)
    assert m.shape == (4, 4)


def test_granger_causality_strategy_compute(small_roi_ts):
    m = GrangerCausalityConnectivityStrategy().compute(small_roi_ts)
    assert m.shape == (4, 4)


def test_transfer_entropy_strategy_compute(small_roi_ts):
    m = TransferEntropyConnectivityStrategy().compute(small_roi_ts)
    assert m.shape == (4, 4)


def test_pdc_strategy_compute(small_roi_ts):
    m = PDCConnectivityStrategy(tr=2.0).compute(small_roi_ts)
    assert m.shape == (4, 4)


def test_psi_strategy_compute(small_roi_ts):
    m = PSIConnectivityStrategy(tr=2.0).compute(small_roi_ts)
    assert m.shape == (4, 4)


# ---------------------------------------------------------------------------
# Strategy __repr__
# ---------------------------------------------------------------------------

def test_pearson_repr():
    assert repr(PearsonConnectivityStrategy()) == "PearsonConnectivityStrategy()"


def test_cross_repr():
    assert "CrossCorrelation" in repr(CrossCorrelationConnectivityStrategy())


def test_corrected_cross_repr():
    assert "CorrectedCrossCorrelation" in repr(CorrectedCrossCorrelationConnectivityStrategy())


def test_partial_repr():
    assert "PartialCorrelation" in repr(PartialCorrelationConnectivityStrategy())


def test_coherence_repr():
    s = CoherenceConnectivityStrategy(tr=2.0)
    assert "Coherence" in repr(s)
    assert "2.0" in repr(s)


def test_imag_coherence_repr():
    s = ImaginaryCoherenceConnectivityStrategy(tr=2.0)
    assert "ImaginaryCoherence" in repr(s)


def test_lagged_coherence_repr():
    s = LaggedCoherenceConnectivityStrategy(tr=2.0)
    assert "LaggedCoherence" in repr(s)


def test_aec_repr():
    assert "AEC" in repr(AECConnectivityStrategy())


def test_aec_orth_repr():
    assert "AECOrth" in repr(AECOrthConnectivityStrategy())


def test_mutual_information_repr():
    assert "MutualInformation" in repr(MutualInformationConnectivityStrategy())


def test_sync_likelihood_repr():
    assert "SyncLikelihood" in repr(SyncLikelihoodConnectivityStrategy())


def test_granger_causality_repr():
    assert "GrangerCausality" in repr(GrangerCausalityConnectivityStrategy())


def test_transfer_entropy_repr():
    assert "TransferEntropy" in repr(TransferEntropyConnectivityStrategy())


def test_pdc_repr():
    s = PDCConnectivityStrategy(tr=2.0)
    assert "PDC" in repr(s)


def test_psi_repr():
    s = PSIConnectivityStrategy(tr=2.0)
    assert "PSI" in repr(s)
