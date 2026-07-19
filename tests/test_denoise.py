"""
Tests for braphin/denoise.py

Covers:
- Output shape and dtype preservation
- Confound regression (1D, 2D, transposed)
- Bug 5 fix: incompatible confound file is skipped, next one is used
- Pending steps when no confounds are found
- Validation errors
"""

import dataclasses

import numpy as np
import pytest

from braphin.config import DenoiseConfig
from braphin.denoise import DenoiseBRAPHINData, BRAPHINDenoiseBundle
from braphin.exceptions import DenoisingError
from braphin.preprocess import BRAPHINPreprocessBundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_denoise(preprocess_bundle, aux_files=None, **cfg_kwargs):
    cfg = DenoiseConfig(
        regress_confounds=cfg_kwargs.pop("regress_confounds", False),
        apply_scrubbing=cfg_kwargs.pop("apply_scrubbing", False),
        apply_bandpass=cfg_kwargs.pop("apply_bandpass", False),
        tr=cfg_kwargs.pop("tr", None),
    )
    bundle = preprocess_bundle
    if aux_files is not None:
        bundle = dataclasses.replace(preprocess_bundle, auxiliary_files=aux_files)
    return DenoiseBRAPHINData(bundle, cfg).run()


# ---------------------------------------------------------------------------
# Basic output
# ---------------------------------------------------------------------------

def test_denoise_returns_bundle(denoise_bundle):
    assert isinstance(denoise_bundle, BRAPHINDenoiseBundle)


def test_denoised_data_shape_matches(denoise_bundle, preprocess_bundle):
    assert denoise_bundle.denoised_data.shape == preprocess_bundle.preprocessed_data.shape


def test_denoised_voxel_ts_shape_matches(denoise_bundle, preprocess_bundle):
    assert denoise_bundle.voxel_time_series.shape == preprocess_bundle.voxel_time_series.shape


def test_denoised_data_dtype(denoise_bundle):
    assert denoise_bundle.denoised_data.dtype == np.float32


def test_no_steps_applied_when_all_off(denoise_bundle):
    assert denoise_bundle.applied_steps == []


# ---------------------------------------------------------------------------
# Confound regression
# ---------------------------------------------------------------------------

def test_confound_regression_2d(preprocess_bundle, num_timepoints):
    rng = np.random.default_rng(1)
    confounds = rng.random((num_timepoints, 6)).astype(np.float32)
    result = _run_denoise(
        preprocess_bundle,
        aux_files={"confounds_timeseries.tsv": confounds},
        regress_confounds=True,
    )
    assert "confound_regression" in result.applied_steps
    assert result.denoise_metadata["confounds_file_used"] == "confounds_timeseries.tsv"
    assert result.denoise_metadata["confounds_shape"] == (num_timepoints, 6)


def test_confound_regression_1d(preprocess_bundle, num_timepoints):
    """1D confound vector must be accepted."""
    confounds = np.random.rand(num_timepoints).astype(np.float32)
    result = _run_denoise(
        preprocess_bundle,
        aux_files={"confounds.tsv": confounds},
        regress_confounds=True,
    )
    assert "confound_regression" in result.applied_steps
    assert result.denoise_metadata["confounds_shape"] == (num_timepoints, 1)


def test_confound_regression_transposed(preprocess_bundle, num_timepoints):
    """Matrix with shape (K, T) must be transposed automatically."""
    confounds = np.random.rand(3, num_timepoints).astype(np.float32)
    result = _run_denoise(
        preprocess_bundle,
        aux_files={"confounds.tsv": confounds},
        regress_confounds=True,
    )
    assert "confound_regression" in result.applied_steps
    assert result.denoise_metadata["confounds_shape"] == (num_timepoints, 3)


def test_confound_regression_removes_variance(preprocess_bundle, num_timepoints):
    """After regressing out a confound, the signal orthogonal to it should change."""
    rng = np.random.default_rng(7)
    confounds = rng.random((num_timepoints, 4)).astype(np.float32)
    result_no_reg = _run_denoise(preprocess_bundle, regress_confounds=False)
    result_reg = _run_denoise(
        preprocess_bundle,
        aux_files={"confounds.tsv": confounds},
        regress_confounds=True,
    )
    assert not np.allclose(
        result_no_reg.voxel_time_series,
        result_reg.voxel_time_series,
    )


def test_confound_regression_output_finite(preprocess_bundle, num_timepoints):
    confounds = np.random.rand(num_timepoints, 6).astype(np.float32)
    result = _run_denoise(
        preprocess_bundle,
        aux_files={"confounds.tsv": confounds},
        regress_confounds=True,
    )
    assert np.isfinite(result.voxel_time_series).all()


# ---------------------------------------------------------------------------
# Bug 5 fix: skips incompatible confound file, finds the next valid one
# ---------------------------------------------------------------------------

def test_skips_incompatible_confound_and_finds_next(preprocess_bundle, num_timepoints):
    """
    Bug 5 fix: if the first confound file has the wrong shape, the search
    must continue to the next file instead of raising immediately.
    """
    rng = np.random.default_rng(99)
    bad = rng.random((num_timepoints + 10, 3)).astype(np.float32)   # wrong T
    good = rng.random((num_timepoints, 3)).astype(np.float32)        # correct

    result = _run_denoise(
        preprocess_bundle,
        aux_files={
            "bad_confounds.tsv": bad,    # iterated first — must be skipped
            "good_confounds.tsv": good,
        },
        regress_confounds=True,
    )
    assert "confound_regression" in result.applied_steps
    assert result.denoise_metadata["confounds_file_used"] == "good_confounds.tsv"


# ---------------------------------------------------------------------------
# No confound file found
# ---------------------------------------------------------------------------

def test_no_confound_found_listed_as_pending(preprocess_bundle):
    result = _run_denoise(preprocess_bundle, regress_confounds=True)
    assert "confound_regression" not in result.applied_steps
    assert any("confound" in s for s in result.pending_steps)


def test_non_confound_aux_file_ignored(preprocess_bundle, num_timepoints):
    """Files without 'confound' in the name are ignored by the confound search."""
    other = np.random.rand(num_timepoints, 2).astype(np.float32)
    result = _run_denoise(
        preprocess_bundle,
        aux_files={"events.tsv": other},
        regress_confounds=True,
    )
    assert "confound_regression" not in result.applied_steps


# ---------------------------------------------------------------------------
# Pending steps
# ---------------------------------------------------------------------------

def test_scrubbing_in_applied_steps(preprocess_bundle):
    """Scrubbing is now implemented — it must appear in applied_steps."""
    result = _run_denoise(preprocess_bundle, apply_scrubbing=True)
    assert "scrubbing" in result.applied_steps


def test_bandpass_in_applied_steps(preprocess_bundle, num_timepoints):
    """Bandpass is now implemented — it must appear in applied_steps."""
    result = _run_denoise(preprocess_bundle, apply_bandpass=True, tr=2.0)
    assert "bandpass_filtering" in result.applied_steps


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_none_bundle_raises():
    with pytest.raises(DenoisingError):
        DenoiseBRAPHINData(BRAPHINPreprocessBundle()).run()


# ---------------------------------------------------------------------------
# Bandpass + confound regression together
# ---------------------------------------------------------------------------

def test_bandpass_and_confound_regression(preprocess_bundle, num_timepoints):
    """Bandpass filtering of confounds runs when both bandpass and regression are on."""
    rng = np.random.default_rng(7)
    confounds = rng.random((num_timepoints, 3)).astype(np.float32)
    aux = {"confounds_timeseries.tsv": confounds}
    result = _run_denoise(
        preprocess_bundle,
        aux_files=aux,
        regress_confounds=True,
        apply_bandpass=True,
        tr=2.0,
    )
    assert "bandpass_filtering" in result.applied_steps
    assert "confound_regression" in result.applied_steps


def test_bandpass_with_no_confounds_still_filters(preprocess_bundle):
    """When regression is requested but no confound file matches, bandpass still runs."""
    result = _run_denoise(
        preprocess_bundle,
        aux_files={"irrelevant.tsv": np.zeros((10, 3), dtype=np.float32)},
        regress_confounds=True,
        apply_bandpass=True,
        tr=2.0,
    )
    assert "bandpass_filtering" in result.applied_steps


# ---------------------------------------------------------------------------
# Fallback confound pattern matching
# ---------------------------------------------------------------------------

def test_fallback_motion_pattern(preprocess_bundle, num_timepoints):
    """Confound is found via fallback 'motion' pattern when 'confound' is absent."""
    rng = np.random.default_rng(11)
    confounds = rng.random((num_timepoints, 2)).astype(np.float32)
    result = _run_denoise(
        preprocess_bundle,
        aux_files={"motion_params.tsv": confounds},
        regress_confounds=True,
    )
    assert "confound_regression" in result.applied_steps


# ---------------------------------------------------------------------------
# display_info
# ---------------------------------------------------------------------------

def test_display_info_no_crash(preprocess_bundle, caplog):
    import logging
    cfg = DenoiseConfig(regress_confounds=False, apply_scrubbing=False, apply_bandpass=False)
    model = DenoiseBRAPHINData(preprocess_bundle, cfg)
    bundle = model.run()
    with caplog.at_level(logging.INFO):
        model.display_info(bundle)


def test_display_info_with_steps(preprocess_bundle, num_timepoints, caplog):
    import logging
    rng = np.random.default_rng(5)
    confounds = rng.random((num_timepoints, 2)).astype(np.float32)
    cfg = DenoiseConfig(regress_confounds=True, apply_bandpass=True, tr=2.0)
    bundle = dataclasses.replace(
        preprocess_bundle,
        auxiliary_files={"confounds_timeseries.tsv": confounds},
    )
    model = DenoiseBRAPHINData(bundle, cfg)
    result = model.run()
    with caplog.at_level(logging.INFO):
        model.display_info(result)
