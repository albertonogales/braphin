"""
Tests for braphin/tools.py and braphin/strategy.py

Covers:
- Pearson correlation: shape, symmetry, diagonal, range, dtype
- Cross-correlation: shape, no NaN
- Corrected cross-correlation: Bug 4 fix — negative_lag reversal
- Partial correlation: shape, symmetry, diagonal, range
- Coherence / imaginary coherence / lagged coherence: shape, range
- AEC / AEC-c: shape, symmetry, range
- Mutual Information: shape, symmetry, non-negative
- Synchronisation Likelihood: shape, symmetry, range
- Granger Causality: shape, non-negative, zero diagonal (directed)
- Transfer Entropy: shape, non-negative, zero diagonal (directed)
- PDC: shape, non-negative (directed)
- PSI: shape, antisymmetry (directed)
- Threshold application
- Method validation and alias resolution
- Strategy factory

Note: PLV / PLI / wPLI / dWPLI / PPC were removed from the fMRI pipeline
(unreliable on raw BOLD signal due to insufficient temporal resolution).
They remain available in EEGraph for EEG data via modality="eeg".
"""

import numpy as np
import pytest

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
    get_connectivity_strategy,
)
from braphin.tools import (
    _corr_cross_correlation_coef,
    apply_connectivity_threshold,
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
    list_connectivity_measures,
    validate_connectivity_method,
)

N = 6
T = 60


@pytest.fixture
def roi_ts():
    rng = np.random.default_rng(99)
    return rng.random((N, T)).astype(np.float32)


@pytest.fixture
def perfectly_correlated_ts():
    rng = np.random.default_rng(7)
    base = rng.random(T).astype(np.float32)
    ts = rng.random((N, T)).astype(np.float32)
    ts[0] = base
    ts[1] = base
    return ts


# ---------------------------------------------------------------------------
# Pearson
# ---------------------------------------------------------------------------

def test_pearson_shape(roi_ts):
    m = compute_pearson_correlation(roi_ts)
    assert m.shape == (N, N)


def test_pearson_symmetric(roi_ts):
    m = compute_pearson_correlation(roi_ts)
    np.testing.assert_allclose(m, m.T, atol=1e-5)


def test_pearson_diagonal_ones(roi_ts):
    m = compute_pearson_correlation(roi_ts)
    np.testing.assert_allclose(np.diag(m), np.ones(N), atol=1e-5)


def test_pearson_values_in_range(roi_ts):
    m = compute_pearson_correlation(roi_ts)
    assert np.all(m >= -1.0 - 1e-5)
    assert np.all(m <= 1.0 + 1e-5)


def test_pearson_dtype(roi_ts):
    m = compute_pearson_correlation(roi_ts)
    assert m.dtype == np.float32


def test_pearson_perfect_correlation(perfectly_correlated_ts):
    m = compute_pearson_correlation(perfectly_correlated_ts)
    assert m[0, 1] == pytest.approx(1.0, abs=1e-4)


def test_pearson_no_nan(roi_ts):
    m = compute_pearson_correlation(roi_ts)
    assert not np.any(np.isnan(m))


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def test_cross_correlation_shape(roi_ts):
    m = compute_cross_correlation(roi_ts)
    assert m.shape == (N, N)


def test_cross_correlation_no_nan(roi_ts):
    m = compute_cross_correlation(roi_ts)
    assert not np.any(np.isnan(m))


def test_cross_correlation_dtype(roi_ts):
    m = compute_cross_correlation(roi_ts)
    assert m.dtype == np.float32


# ---------------------------------------------------------------------------
# Corrected cross-correlation  — Bug 4 fix verification
# ---------------------------------------------------------------------------

def test_corrected_cross_correlation_shape(roi_ts):
    m = compute_corrected_cross_correlation(roi_ts)
    assert m.shape == (N, N)


def test_corrected_cross_correlation_no_nan(roi_ts):
    m = compute_corrected_cross_correlation(roi_ts)
    assert not np.any(np.isnan(m))


def test_corrected_cross_correlation_antisymmetry():
    """
    Bug 4 fix: after reversing negative_lag, corCC(x,y) == -corCC(y,x).

    Mathematical derivation:
        corCC(x,y)[k] = Rxy(+1+k) - Rxy(-1-k)
        corCC(y,x)[k] = Ryx(+1+k) - Ryx(-1-k)
                      = Rxy(-1-k) - Rxy(+1+k)
                      = -corCC(x,y)[k]
    Therefore mean(corCC(x,y)) == -mean(corCC(y,x)).
    """
    rng = np.random.default_rng(42)
    x = rng.random(80).astype(np.float32)
    y = rng.random(80).astype(np.float32)
    cc_xy = _corr_cross_correlation_coef(x, y)
    cc_yx = _corr_cross_correlation_coef(y, x)
    assert cc_xy == pytest.approx(-cc_yx, abs=1e-5)


def test_corrected_cross_correlation_same_signal_zero_diagonal():
    """corCC(x, x) must be 0 because positive_lag == reversed negative_lag."""
    rng = np.random.default_rng(11)
    x = rng.random(60).astype(np.float32)
    cc = _corr_cross_correlation_coef(x, x)
    assert cc == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

def test_threshold_zeroes_below_value(roi_ts):
    m = compute_pearson_correlation(roi_ts)
    thresh = apply_connectivity_threshold(m, 0.4)
    below = np.abs(m) < 0.4
    np.testing.assert_array_equal(thresh[below], 0.0)


def test_threshold_preserves_above_value(roi_ts):
    m = compute_pearson_correlation(roi_ts)
    thresh = apply_connectivity_threshold(m, 0.4)
    above = np.abs(m) >= 0.4
    np.testing.assert_allclose(thresh[above], m[above], atol=1e-6)


def test_threshold_zero_keeps_all(roi_ts):
    m = compute_pearson_correlation(roi_ts)
    thresh = apply_connectivity_threshold(m, 0.0)
    np.testing.assert_allclose(thresh, m, atol=1e-6)


def test_threshold_one_zeroes_all_offdiag(roi_ts):
    m = compute_pearson_correlation(roi_ts)
    thresh = apply_connectivity_threshold(m, 1.0)
    n = thresh.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                assert thresh[i, j] == 0.0 or abs(thresh[i, j]) >= 1.0


def test_threshold_negative_raises(roi_ts):
    m = compute_pearson_correlation(roi_ts)
    with pytest.raises(ConnectivityError):
        apply_connectivity_threshold(m, -0.1)


# ---------------------------------------------------------------------------
# Method validation
# ---------------------------------------------------------------------------

def test_validate_method_canonical_pearson():
    assert validate_connectivity_method("pearson_correlation") == "pearson_correlation"


def test_validate_method_alias_pearson():
    assert validate_connectivity_method("pearson") == "pearson_correlation"


def test_validate_method_alias_cross():
    assert validate_connectivity_method("cross-correlation") == "cross_correlation"


def test_validate_method_alias_corrected():
    assert validate_connectivity_method("corrected_cross_correlation") == "corr_cross_correlation"


def test_validate_method_case_insensitive():
    assert validate_connectivity_method("Pearson") == "pearson_correlation"


def test_validate_method_unsupported_raises():
    with pytest.raises(ConnectivityError):
        validate_connectivity_method("dtf")


def test_validate_method_none_raises():
    with pytest.raises(ConnectivityError):
        validate_connectivity_method(None)


def test_list_connectivity_measures():
    measures = list_connectivity_measures()
    assert "pearson_correlation" in measures
    assert "cross_correlation" in measures
    assert "corr_cross_correlation" in measures


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def test_get_strategy_pearson():
    s = get_connectivity_strategy("pearson_correlation")
    assert isinstance(s, PearsonConnectivityStrategy)


def test_get_strategy_cross():
    s = get_connectivity_strategy("cross_correlation")
    assert isinstance(s, CrossCorrelationConnectivityStrategy)


def test_get_strategy_corrected_cross():
    s = get_connectivity_strategy("corr_cross_correlation")
    assert isinstance(s, CorrectedCrossCorrelationConnectivityStrategy)


def test_strategy_compute_returns_matrix(roi_ts):
    s = get_connectivity_strategy("pearson_correlation")
    m = s.compute(roi_ts)
    assert m.shape == (N, N)


def test_get_strategy_unsupported_raises():
    with pytest.raises(Exception):
        get_connectivity_strategy("not_a_method")


# ---------------------------------------------------------------------------
# Partial correlation
# ---------------------------------------------------------------------------

def test_partial_correlation_shape(roi_ts):
    m = compute_partial_correlation(roi_ts)
    assert m.shape == (N, N)


def test_partial_correlation_symmetric(roi_ts):
    m = compute_partial_correlation(roi_ts)
    np.testing.assert_allclose(m, m.T, atol=1e-5)


def test_partial_correlation_diagonal_ones(roi_ts):
    m = compute_partial_correlation(roi_ts)
    np.testing.assert_allclose(np.diag(m), np.ones(N), atol=1e-5)


def test_partial_correlation_range(roi_ts):
    m = compute_partial_correlation(roi_ts)
    assert np.all(m >= -1.0 - 1e-5)
    assert np.all(m <= 1.0 + 1e-5)


def test_partial_correlation_no_nan(roi_ts):
    m = compute_partial_correlation(roi_ts)
    assert not np.any(np.isnan(m))


def test_partial_correlation_dtype(roi_ts):
    m = compute_partial_correlation(roi_ts)
    assert m.dtype == np.float32


def test_get_strategy_partial_correlation():
    s = get_connectivity_strategy("partial_correlation")
    assert isinstance(s, PartialCorrelationConnectivityStrategy)


def test_validate_method_alias_partial():
    assert validate_connectivity_method("partial corr") == "partial_correlation"


# ---------------------------------------------------------------------------
# Coherence (magnitude-squared, requires TR)
# ---------------------------------------------------------------------------

TR = 2.0  # seconds — typical fMRI TR


def test_coherence_shape(roi_ts):
    m = compute_coherence(roi_ts, TR)
    assert m.shape == (N, N)


def test_coherence_symmetric(roi_ts):
    m = compute_coherence(roi_ts, TR)
    np.testing.assert_allclose(m, m.T, atol=1e-5)


def test_coherence_diagonal_ones(roi_ts):
    m = compute_coherence(roi_ts, TR)
    np.testing.assert_allclose(np.diag(m), np.ones(N), atol=1e-5)


def test_coherence_range(roi_ts):
    m = compute_coherence(roi_ts, TR)
    assert np.all(m >= 0.0 - 1e-5)
    assert np.all(m <= 1.0 + 1e-5)


def test_coherence_no_nan(roi_ts):
    m = compute_coherence(roi_ts, TR)
    assert not np.any(np.isnan(m))


def test_coherence_dtype(roi_ts):
    m = compute_coherence(roi_ts, TR)
    assert m.dtype == np.float32


def test_coherence_missing_tr_raises(roi_ts):
    with pytest.raises(ConnectivityError):
        compute_coherence(roi_ts, None)


def test_coherence_negative_tr_raises(roi_ts):
    with pytest.raises(ConnectivityError):
        compute_coherence(roi_ts, -1.0)


def test_get_strategy_coherence():
    s = get_connectivity_strategy("coherence", tr=TR)
    assert isinstance(s, CoherenceConnectivityStrategy)


def test_get_strategy_coherence_missing_tr_raises():
    with pytest.raises(ConnectivityError):
        get_connectivity_strategy("coherence")


def test_validate_method_alias_coherence():
    assert validate_connectivity_method("squared coherence") == "coherence"


# ---------------------------------------------------------------------------
# Imaginary coherence (requires TR)
# ---------------------------------------------------------------------------

def test_imag_coherence_shape(roi_ts):
    m = compute_imaginary_coherence(roi_ts, TR)
    assert m.shape == (N, N)


def test_imag_coherence_antisymmetric(roi_ts):
    # Signed imaginary coherence (Nolte 2004): IC(i,j) = -IC(j,i)
    m = compute_imaginary_coherence(roi_ts, TR)
    np.testing.assert_allclose(m, -m.T, atol=1e-5)


def test_imag_coherence_diagonal_zero(roi_ts):
    # Self-coherence imaginary part is always 0
    m = compute_imaginary_coherence(roi_ts, TR)
    np.testing.assert_allclose(np.diag(m), 0.0, atol=1e-5)


def test_imag_coherence_no_nan(roi_ts):
    m = compute_imaginary_coherence(roi_ts, TR)
    assert not np.any(np.isnan(m))


def test_imag_coherence_dtype(roi_ts):
    m = compute_imaginary_coherence(roi_ts, TR)
    assert m.dtype == np.float32


def test_imag_coherence_missing_tr_raises(roi_ts):
    with pytest.raises(ConnectivityError):
        compute_imaginary_coherence(roi_ts, None)


def test_get_strategy_imag_coherence():
    s = get_connectivity_strategy("imag_coherence", tr=TR)
    assert isinstance(s, ImaginaryCoherenceConnectivityStrategy)


def test_get_strategy_imag_coherence_missing_tr_raises():
    with pytest.raises(ConnectivityError):
        get_connectivity_strategy("imag_coherence")


def test_validate_method_alias_imag_coherence():
    assert validate_connectivity_method("imaginary coherence") == "imag_coherence"


# ---------------------------------------------------------------------------
# list_connectivity_measures includes new measures
# ---------------------------------------------------------------------------

def test_list_connectivity_measures_includes_new():
    measures = list_connectivity_measures()
    for method in ("partial_correlation", "coherence", "imag_coherence", "aec", "aec_orth"):
        assert method in measures


# ---------------------------------------------------------------------------
# AEC — Amplitude Envelope Correlation
# ---------------------------------------------------------------------------

def test_aec_shape(roi_ts):
    assert compute_aec(roi_ts).shape == (N, N)

def test_aec_symmetric(roi_ts):
    m = compute_aec(roi_ts)
    np.testing.assert_allclose(m, m.T, atol=1e-5)

def test_aec_diagonal_ones(roi_ts):
    np.testing.assert_allclose(np.diag(compute_aec(roi_ts)), np.ones(N), atol=1e-5)

def test_aec_range(roi_ts):
    m = compute_aec(roi_ts)
    assert np.all(m >= -1.0 - 1e-5) and np.all(m <= 1.0 + 1e-5)

def test_aec_no_nan(roi_ts):
    assert not np.any(np.isnan(compute_aec(roi_ts)))

def test_aec_dtype(roi_ts):
    assert compute_aec(roi_ts).dtype == np.float32

def test_get_strategy_aec():
    assert isinstance(get_connectivity_strategy("aec"), AECConnectivityStrategy)

def test_validate_alias_aec():
    assert validate_connectivity_method("amplitude envelope correlation") == "aec"


# ---------------------------------------------------------------------------
# AEC-c — Orthogonalized AEC
# ---------------------------------------------------------------------------

def test_aec_orth_shape(roi_ts):
    assert compute_aec_orth(roi_ts).shape == (N, N)

def test_aec_orth_symmetric(roi_ts):
    m = compute_aec_orth(roi_ts)
    np.testing.assert_allclose(m, m.T, atol=1e-5)

def test_aec_orth_diagonal_ones(roi_ts):
    np.testing.assert_allclose(np.diag(compute_aec_orth(roi_ts)), np.ones(N), atol=1e-5)

def test_aec_orth_no_nan(roi_ts):
    assert not np.any(np.isnan(compute_aec_orth(roi_ts)))

def test_aec_orth_dtype(roi_ts):
    assert compute_aec_orth(roi_ts).dtype == np.float32

def test_get_strategy_aec_orth():
    assert isinstance(get_connectivity_strategy("aec_orth"), AECOrthConnectivityStrategy)

def test_validate_alias_aec_orth():
    assert validate_connectivity_method("aec_c") == "aec_orth"


# ---------------------------------------------------------------------------
# Mutual Information
# ---------------------------------------------------------------------------

def test_mi_shape(roi_ts):
    assert compute_mutual_information(roi_ts).shape == (N, N)

def test_mi_symmetric(roi_ts):
    m = compute_mutual_information(roi_ts)
    np.testing.assert_allclose(m, m.T, atol=1e-5)

def test_mi_non_negative(roi_ts):
    assert np.all(compute_mutual_information(roi_ts) >= -1e-7)

def test_mi_no_nan(roi_ts):
    assert not np.any(np.isnan(compute_mutual_information(roi_ts)))

def test_mi_dtype(roi_ts):
    assert compute_mutual_information(roi_ts).dtype == np.float32

def test_get_strategy_mi():
    assert isinstance(get_connectivity_strategy("mutual_information"),
                      MutualInformationConnectivityStrategy)

def test_validate_alias_mi():
    assert validate_connectivity_method("mi") == "mutual_information"


# ---------------------------------------------------------------------------
# Synchronisation Likelihood
# ---------------------------------------------------------------------------

def test_sl_shape(roi_ts):
    assert compute_sync_likelihood(roi_ts).shape == (N, N)

def test_sl_symmetric(roi_ts):
    m = compute_sync_likelihood(roi_ts)
    np.testing.assert_allclose(m, m.T, atol=1e-5)

def test_sl_range(roi_ts):
    m = compute_sync_likelihood(roi_ts)
    assert np.all(m >= -1e-5) and np.all(m <= 1.0 + 1e-5)

def test_sl_no_nan(roi_ts):
    assert not np.any(np.isnan(compute_sync_likelihood(roi_ts)))

def test_sl_dtype(roi_ts):
    assert compute_sync_likelihood(roi_ts).dtype == np.float32

def test_get_strategy_sl():
    assert isinstance(get_connectivity_strategy("sync_likelihood"),
                      SyncLikelihoodConnectivityStrategy)

def test_validate_alias_sl():
    assert validate_connectivity_method("synchronisation likelihood") == "sync_likelihood"


# ---------------------------------------------------------------------------
# Lagged Coherence (needs TR)
# ---------------------------------------------------------------------------

def test_lagged_coherence_shape(roi_ts):
    assert compute_lagged_coherence(roi_ts, TR).shape == (N, N)

def test_lagged_coherence_symmetric(roi_ts):
    m = compute_lagged_coherence(roi_ts, TR)
    np.testing.assert_allclose(m, m.T, atol=1e-5)

def test_lagged_coherence_non_negative(roi_ts):
    assert np.all(compute_lagged_coherence(roi_ts, TR) >= -1e-7)

def test_lagged_coherence_no_nan(roi_ts):
    assert not np.any(np.isnan(compute_lagged_coherence(roi_ts, TR)))

def test_lagged_coherence_dtype(roi_ts):
    assert compute_lagged_coherence(roi_ts, TR).dtype == np.float32

def test_lagged_coherence_missing_tr_raises(roi_ts):
    with pytest.raises(ConnectivityError):
        compute_lagged_coherence(roi_ts, None)

def test_get_strategy_lagged_coherence():
    assert isinstance(get_connectivity_strategy("lagged_coherence", tr=TR),
                      LaggedCoherenceConnectivityStrategy)

def test_validate_alias_lagged_coherence():
    assert validate_connectivity_method("lagged coherence") == "lagged_coherence"


# ---------------------------------------------------------------------------
# Granger Causality (directed)
# ---------------------------------------------------------------------------

def test_gc_shape(roi_ts):
    assert compute_granger_causality(roi_ts).shape == (N, N)

def test_gc_non_negative(roi_ts):
    assert np.all(compute_granger_causality(roi_ts) >= -1e-7)

def test_gc_zero_diagonal(roi_ts):
    m = compute_granger_causality(roi_ts)
    np.testing.assert_array_equal(np.diag(m), np.zeros(N))

def test_gc_no_nan(roi_ts):
    assert not np.any(np.isnan(compute_granger_causality(roi_ts)))

def test_gc_dtype(roi_ts):
    assert compute_granger_causality(roi_ts).dtype == np.float32

def test_gc_asymmetric(roi_ts):
    """GC is in general asymmetric (directed)."""
    m = compute_granger_causality(roi_ts)
    # Not all off-diagonal pairs should be equal (probabilistic; passes for random data)
    assert not np.allclose(m, m.T, atol=1e-5)

def test_get_strategy_gc():
    assert isinstance(get_connectivity_strategy("granger_causality"),
                      GrangerCausalityConnectivityStrategy)

def test_validate_alias_gc():
    assert validate_connectivity_method("gc") == "granger_causality"


# ---------------------------------------------------------------------------
# Transfer Entropy (directed)
# ---------------------------------------------------------------------------

def test_te_shape(roi_ts):
    assert compute_transfer_entropy(roi_ts).shape == (N, N)

def test_te_non_negative(roi_ts):
    assert np.all(compute_transfer_entropy(roi_ts) >= -1e-7)

def test_te_zero_diagonal(roi_ts):
    m = compute_transfer_entropy(roi_ts)
    np.testing.assert_array_equal(np.diag(m), np.zeros(N))

def test_te_no_nan(roi_ts):
    assert not np.any(np.isnan(compute_transfer_entropy(roi_ts)))

def test_te_dtype(roi_ts):
    assert compute_transfer_entropy(roi_ts).dtype == np.float32

def test_get_strategy_te():
    assert isinstance(get_connectivity_strategy("transfer_entropy"),
                      TransferEntropyConnectivityStrategy)

def test_validate_alias_te():
    assert validate_connectivity_method("te") == "transfer_entropy"


# ---------------------------------------------------------------------------
# PDC (directed, needs TR)
# ---------------------------------------------------------------------------

def test_pdc_shape(roi_ts):
    assert compute_pdc(roi_ts, TR).shape == (N, N)

def test_pdc_non_negative(roi_ts):
    assert np.all(compute_pdc(roi_ts, TR) >= -1e-7)

def test_pdc_range(roi_ts):
    assert np.all(compute_pdc(roi_ts, TR) <= 1.0 + 1e-5)

def test_pdc_no_nan(roi_ts):
    assert not np.any(np.isnan(compute_pdc(roi_ts, TR)))

def test_pdc_dtype(roi_ts):
    assert compute_pdc(roi_ts, TR).dtype == np.float32

def test_pdc_missing_tr_raises(roi_ts):
    with pytest.raises(ConnectivityError):
        compute_pdc(roi_ts, None)

def test_get_strategy_pdc():
    assert isinstance(get_connectivity_strategy("pdc", tr=TR), PDCConnectivityStrategy)

def test_validate_alias_pdc():
    assert validate_connectivity_method("partial directed coherence") == "pdc"


# ---------------------------------------------------------------------------
# PSI (directed, needs TR)
# ---------------------------------------------------------------------------

def test_psi_shape(roi_ts):
    assert compute_psi(roi_ts, TR).shape == (N, N)

def test_psi_no_nan(roi_ts):
    assert not np.any(np.isnan(compute_psi(roi_ts, TR)))

def test_psi_dtype(roi_ts):
    assert compute_psi(roi_ts, TR).dtype == np.float32

def test_psi_antisymmetric(roi_ts):
    """PSI is antisymmetric: PSI[i,j] = -PSI[j,i]."""
    m = compute_psi(roi_ts, TR)
    np.testing.assert_allclose(m, -m.T, atol=1e-5)

def test_psi_missing_tr_raises(roi_ts):
    with pytest.raises(ConnectivityError):
        compute_psi(roi_ts, None)

def test_get_strategy_psi():
    assert isinstance(get_connectivity_strategy("psi", tr=TR), PSIConnectivityStrategy)

def test_validate_alias_psi():
    assert validate_connectivity_method("phase slope index") == "psi"


# ---------------------------------------------------------------------------
# list_connectivity_measures — all new methods present
# ---------------------------------------------------------------------------

def test_list_all_new_measures():
    measures = list_connectivity_measures()
    for m in (
        "aec", "aec_orth", "mutual_information",
        "sync_likelihood", "lagged_coherence", "granger_causality",
        "transfer_entropy", "pdc", "psi",
    ):
        assert m in measures, f"{m!r} missing from CONNECTIVITY_MEASURES"
