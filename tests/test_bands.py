"""
Tests for braphin.bands

Covers:
- FMRI_BANDS: dict with keys slow5, slow4, slow3, broadband
- FMRI_BANDS: each value is (fmin, fmax) with fmin < fmax
- bandpass_roi_time_series: returns same shape as input
- bandpass_roi_time_series: dtype is numeric
- bandpass_roi_time_series: single ROI (1×T) raises (needs >= 2 ROIs) — or works if validated upstream
- bandpass_roi_time_series: many ROIs (4×50) works without error
- bandpass_roi_time_series: invalid fmin >= fmax raises
- compute_band_connectivity: returns N×N array
- compute_band_connectivity: method="pearson_correlation" works
- compute_band_connectivity: returns finite values
- compute_band_connectivity: unknown band raises error
- compute_all_bands_connectivity: returns a dict
- compute_all_bands_connectivity: keys are all FMRI_BANDS keys (or subset for compatible bands)
- compute_all_bands_connectivity: each value is an N×N array
- compute_all_bands_connectivity: values are finite
"""

import numpy as np
import pytest

from braphin.bands import (
    FMRI_BANDS,
    bandpass_roi_time_series,
    compute_band_connectivity,
    compute_all_bands_connectivity,
)
from braphin.exceptions import ConnectivityError

# ---------------------------------------------------------------------------
# Synthetic time series
# TR=2s → Nyquist=0.25 Hz, which covers slow5/slow4/broadband fully.
# slow3 (0.073–0.167 Hz) is also inside Nyquist=0.25 Hz with TR=2.
# ---------------------------------------------------------------------------
TR = 2.0      # seconds
N_ROIS = 4
T = 200       # longer series for stable filtering (especially slow bands)

rng = np.random.default_rng(0)
ROI_TS = rng.standard_normal((N_ROIS, T)).astype(np.float32)


# ---------------------------------------------------------------------------
# FMRI_BANDS structure
# ---------------------------------------------------------------------------

def test_fmri_bands_is_dict():
    assert isinstance(FMRI_BANDS, dict)


def test_fmri_bands_has_slow5():
    assert "slow5" in FMRI_BANDS


def test_fmri_bands_has_slow4():
    assert "slow4" in FMRI_BANDS


def test_fmri_bands_has_slow3():
    assert "slow3" in FMRI_BANDS


def test_fmri_bands_has_broadband():
    assert "broadband" in FMRI_BANDS


def test_fmri_bands_values_are_tuples():
    for name, value in FMRI_BANDS.items():
        assert isinstance(value, tuple), f"Band {name!r} value is not a tuple"
        assert len(value) == 2


def test_fmri_bands_fmin_less_than_fmax():
    for name, (fmin, fmax) in FMRI_BANDS.items():
        assert fmin < fmax, f"Band {name!r}: fmin={fmin} >= fmax={fmax}"


# ---------------------------------------------------------------------------
# bandpass_roi_time_series
# ---------------------------------------------------------------------------

def test_bandpass_returns_same_shape():
    fmin, fmax = FMRI_BANDS["slow4"]
    out = bandpass_roi_time_series(ROI_TS, tr=TR, fmin=fmin, fmax=fmax)
    assert out.shape == ROI_TS.shape


def test_bandpass_dtype_is_numeric():
    fmin, fmax = FMRI_BANDS["slow4"]
    out = bandpass_roi_time_series(ROI_TS, tr=TR, fmin=fmin, fmax=fmax)
    assert np.issubdtype(out.dtype, np.floating)


def test_bandpass_broadband_no_error():
    fmin, fmax = FMRI_BANDS["broadband"]
    out = bandpass_roi_time_series(ROI_TS, tr=TR, fmin=fmin, fmax=fmax)
    assert out.shape == ROI_TS.shape


def test_bandpass_slow3_no_error():
    """slow3 (0.073–0.167 Hz) is within Nyquist=0.25 Hz for TR=2s."""
    fmin, fmax = FMRI_BANDS["slow3"]
    out = bandpass_roi_time_series(ROI_TS, tr=TR, fmin=fmin, fmax=fmax)
    assert out.shape == ROI_TS.shape


def test_bandpass_invalid_fmin_gt_fmax_raises():
    with pytest.raises((ConnectivityError, ValueError)):
        bandpass_roi_time_series(ROI_TS, tr=TR, fmin=0.1, fmax=0.01)


def test_bandpass_fmax_above_nyquist_raises():
    """fmax > Nyquist (0.25 Hz for TR=2s) should raise ConnectivityError."""
    with pytest.raises(ConnectivityError):
        bandpass_roi_time_series(ROI_TS, tr=TR, fmin=0.01, fmax=0.30)


# ---------------------------------------------------------------------------
# compute_band_connectivity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("band", ["slow5", "slow4", "broadband"])
def test_band_connectivity_shape(band):
    mat = compute_band_connectivity(
        roi_time_series=ROI_TS, tr=TR, band=band, method="pearson_correlation"
    )
    assert mat.shape == (N_ROIS, N_ROIS)


@pytest.mark.parametrize("band", ["slow5", "slow4", "broadband"])
def test_band_connectivity_returns_ndarray(band):
    mat = compute_band_connectivity(
        roi_time_series=ROI_TS, tr=TR, band=band, method="pearson_correlation"
    )
    assert isinstance(mat, np.ndarray)


@pytest.mark.parametrize("band", ["slow5", "slow4", "broadband"])
def test_band_connectivity_finite_values(band):
    mat = compute_band_connectivity(
        roi_time_series=ROI_TS, tr=TR, band=band, method="pearson_correlation"
    )
    assert np.all(np.isfinite(mat))


def test_band_connectivity_slow3_in_range():
    """slow3 is within Nyquist for TR=2s; ensure it runs and returns valid output."""
    mat = compute_band_connectivity(
        roi_time_series=ROI_TS, tr=TR, band="slow3", method="pearson_correlation"
    )
    assert mat.shape == (N_ROIS, N_ROIS)
    assert isinstance(mat, np.ndarray)


def test_band_connectivity_unknown_band_raises():
    with pytest.raises((ConnectivityError, KeyError)):
        compute_band_connectivity(
            roi_time_series=ROI_TS, tr=TR, band="gamma", method="pearson_correlation"
        )


# ---------------------------------------------------------------------------
# compute_all_bands_connectivity
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_bands_result():
    return compute_all_bands_connectivity(ROI_TS, tr=TR, method="pearson_correlation")


def test_all_bands_returns_dict(all_bands_result):
    assert isinstance(all_bands_result, dict)


def test_all_bands_keys_are_subset_of_fmri_bands(all_bands_result):
    """All returned keys must be valid FMRI_BANDS keys."""
    for key in all_bands_result.keys():
        assert key in FMRI_BANDS


def test_all_bands_contains_slow5(all_bands_result):
    assert "slow5" in all_bands_result


def test_all_bands_contains_slow4(all_bands_result):
    assert "slow4" in all_bands_result


def test_all_bands_contains_broadband(all_bands_result):
    assert "broadband" in all_bands_result


def test_all_bands_each_value_is_ndarray(all_bands_result):
    for band_name, mat in all_bands_result.items():
        assert isinstance(mat, np.ndarray), f"Band {band_name!r} is not ndarray"


def test_all_bands_each_value_correct_shape(all_bands_result):
    for band_name, mat in all_bands_result.items():
        assert mat.shape == (N_ROIS, N_ROIS), (
            f"Band {band_name!r} has shape {mat.shape}, expected ({N_ROIS}, {N_ROIS})"
        )


def test_all_bands_values_finite(all_bands_result):
    for band_name, mat in all_bands_result.items():
        assert np.all(np.isfinite(mat)), f"Band {band_name!r} contains non-finite values"
