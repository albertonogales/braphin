"""
Tests for configuration dataclass __post_init__ validation.

Covers the validation branches in PreprocessConfig, DenoiseConfig,
ConnectivityConfig that are not reached by other tests.
"""

import pytest

from braphin.config import (
    AtlasConfig,
    ConnectivityConfig,
    DenoiseConfig,
    PreprocessConfig,
)


# ---------------------------------------------------------------------------
# PreprocessConfig
# ---------------------------------------------------------------------------

def test_preprocess_slice_timing_requires_tr():
    with pytest.raises(ValueError, match="tr"):
        PreprocessConfig(apply_slice_timing=True, tr=None)


def test_preprocess_smoothing_fwhm_positive():
    with pytest.raises(ValueError, match="smoothing_fwhm"):
        PreprocessConfig(apply_smoothing=True, smoothing_fwhm=-1.0)


def test_preprocess_dvars_threshold_positive():
    with pytest.raises(ValueError, match="outlier_threshold_dvars"):
        PreprocessConfig(outlier_threshold_dvars=-0.5)


def test_preprocess_scrubbing_strategy_invalid():
    with pytest.raises(ValueError, match="scrubbing_strategy"):
        PreprocessConfig(scrubbing_strategy="delete")


def test_preprocess_slice_axis_invalid():
    with pytest.raises(ValueError, match="slice_axis"):
        PreprocessConfig(slice_axis=3)


def test_preprocess_valid_defaults():
    cfg = PreprocessConfig()
    assert cfg.scrubbing_strategy == "interpolate"
    assert cfg.slice_axis == 2


# ---------------------------------------------------------------------------
# DenoiseConfig
# ---------------------------------------------------------------------------

def test_denoise_bandpass_requires_tr():
    with pytest.raises(ValueError, match="tr"):
        DenoiseConfig(apply_bandpass=True, tr=None)


def test_denoise_bandpass_low_positive():
    with pytest.raises(ValueError, match="bandpass_low"):
        DenoiseConfig(apply_bandpass=True, tr=2.0, bandpass_low=0.0)


def test_denoise_bandpass_high_gt_low():
    with pytest.raises(ValueError, match="bandpass_high"):
        DenoiseConfig(apply_bandpass=True, tr=2.0, bandpass_low=0.1, bandpass_high=0.05)


def test_denoise_valid_bandpass():
    cfg = DenoiseConfig(apply_bandpass=True, tr=2.0, bandpass_low=0.008, bandpass_high=0.1)
    assert cfg.apply_bandpass is True


# ---------------------------------------------------------------------------
# ConnectivityConfig
# ---------------------------------------------------------------------------

def test_connectivity_negative_threshold_raises():
    with pytest.raises(ValueError, match="threshold"):
        ConnectivityConfig(threshold=-0.1)


def test_connectivity_zero_window_size_raises():
    with pytest.raises(ValueError, match="window_size"):
        ConnectivityConfig(window_size=0.0)


def test_connectivity_zero_tr_raises():
    with pytest.raises(ValueError, match="tr"):
        ConnectivityConfig(tr=0.0)


def test_connectivity_model_order_less_than_one_raises():
    with pytest.raises(ValueError, match="model_order"):
        ConnectivityConfig(model_order=0)


def test_connectivity_zero_step_size_raises():
    with pytest.raises(ValueError, match="step_size"):
        ConnectivityConfig(step_size=0.0)


def test_connectivity_valid_defaults():
    cfg = ConnectivityConfig()
    assert cfg.method == "pearson_correlation"
    assert cfg.threshold is None
