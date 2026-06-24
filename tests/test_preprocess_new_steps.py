"""
Tests for the four newly-implemented preprocessing steps:
  - Spatial smoothing         (apply_smoothing)
  - Slice-timing correction   (apply_slice_timing)
  - Motion correction         (apply_motion_correction)
  - Outlier detection         (apply_outlier_detection)

And for the two newly-implemented denoising steps:
  - Bandpass filtering        (apply_bandpass)
  - Scrubbing                 (apply_scrubbing)
"""

import dataclasses

import nibabel as nib
import numpy as np
import pytest

from braphin.config import DenoiseConfig, PreprocessConfig
from braphin.denoise import DenoiseBRAPHINData
from braphin.exceptions import DenoisingError, PreprocessingError
from braphin.importBRAPHINData import BRAPHINInputBundle
from braphin.io.nifti import get_nifti_metadata
from braphin.preprocess import BRAPHINPreprocessBundle, PreprocessBRAPHINData


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

SPATIAL = (8, 8, 8)
T = 60           # enough timepoints for bandpass (≥ 40 required)
TR = 2.0         # seconds


def _make_input_bundle(arr=None, affine=None, zooms=None):
    """Build a minimal BRAPHINInputBundle from a synthetic array."""
    if arr is None:
        rng = np.random.default_rng(0)
        arr = (rng.random((*SPATIAL, T)) * 200.0 + 600.0).astype(np.float32)
    if affine is None:
        scale = 2.0 if zooms is None else zooms
        if isinstance(scale, (int, float)):
            affine = np.diag([scale, scale, scale, 1.0])
        else:
            affine = np.diag([*scale[:3], 1.0])
    img = nib.Nifti1Image(arr, affine)
    meta = get_nifti_metadata(img)
    return BRAPHINInputBundle(fmri_path="fake.nii.gz", fmri_image=img, fmri_metadata=meta)


def _run_preprocess(arr=None, affine=None, **cfg_kwargs):
    bundle = _make_input_bundle(arr, affine)
    defaults = dict(
        apply_motion_correction=False,
        apply_slice_timing=False,
        apply_outlier_detection=False,
        apply_normalization=False,
        apply_smoothing=False,
    )
    defaults.update(cfg_kwargs)   # caller's values win
    cfg = PreprocessConfig(**defaults)
    return PreprocessBRAPHINData(bundle, cfg).run()


def _run_denoise_from_preprocess(preprocess_bundle, **cfg_kwargs):
    cfg = DenoiseConfig(
        regress_confounds=False,
        apply_scrubbing=cfg_kwargs.pop("apply_scrubbing", False),
        apply_bandpass=cfg_kwargs.pop("apply_bandpass", False),
        tr=cfg_kwargs.pop("tr", None),
    )
    return DenoiseBRAPHINData(preprocess_bundle, cfg).run()


# ─────────────────────────────────────────────────────────────────────────────
# Spatial smoothing
# ─────────────────────────────────────────────────────────────────────────────

class TestSpatialSmoothing:

    def test_smoothing_step_in_applied_steps(self):
        result = _run_preprocess(apply_smoothing=True, smoothing_fwhm=4.0)
        assert "spatial_smoothing" in result.applied_steps

    def test_smoothing_changes_data(self):
        rng = np.random.default_rng(1)
        arr = rng.random((*SPATIAL, T)).astype(np.float32) * 1000
        raw = _run_preprocess(arr=arr).preprocessed_data
        smoothed = _run_preprocess(arr=arr, apply_smoothing=True, smoothing_fwhm=4.0).preprocessed_data
        assert not np.allclose(raw, smoothed)

    def test_smoothed_data_is_finite(self):
        result = _run_preprocess(apply_smoothing=True, smoothing_fwhm=4.0)
        assert np.isfinite(result.preprocessed_data).all()

    def test_smoothed_data_same_shape(self):
        result = _run_preprocess(apply_smoothing=True, smoothing_fwhm=6.0)
        assert result.preprocessed_data.shape == (*SPATIAL, T)

    def test_smoothing_reduces_spatial_variance(self):
        """Smoothing should lower per-volume spatial variance."""
        rng = np.random.default_rng(2)
        arr = rng.random((*SPATIAL, T)).astype(np.float32)
        raw = _run_preprocess(arr=arr).preprocessed_data
        smoothed = _run_preprocess(arr=arr, apply_smoothing=True, smoothing_fwhm=6.0).preprocessed_data
        assert float(np.var(smoothed)) < float(np.var(raw))

    def test_smoothing_metadata_flag(self):
        result = _run_preprocess(apply_smoothing=True, smoothing_fwhm=4.0)
        assert result.preprocess_metadata["smoothing_applied"] is True

    def test_smoothing_invalid_fwhm_raises(self):
        # Config validation now fires before the pipeline, raising ValueError
        with pytest.raises((ValueError, PreprocessingError)):
            _run_preprocess(apply_smoothing=True, smoothing_fwhm=-1.0)

    def test_smoothing_uses_voxel_sizes(self):
        """Larger voxels → smaller sigma in voxels → less blurring."""
        rng = np.random.default_rng(3)
        arr = rng.random((*SPATIAL, T)).astype(np.float32)
        # Fine voxels: more sigma-in-voxels → more blur
        fine = _run_preprocess(
            arr=arr,
            affine=np.diag([1.0, 1.0, 1.0, 1.0]),
            apply_smoothing=True,
            smoothing_fwhm=4.0,
        ).preprocessed_data
        # Coarse voxels: fewer sigma-in-voxels → less blur
        coarse = _run_preprocess(
            arr=arr,
            affine=np.diag([4.0, 4.0, 4.0, 1.0]),
            apply_smoothing=True,
            smoothing_fwhm=4.0,
        ).preprocessed_data
        # Fine should produce lower variance (more blurred) than coarse
        assert float(np.var(fine)) < float(np.var(coarse))


# ─────────────────────────────────────────────────────────────────────────────
# Slice-timing correction
# ─────────────────────────────────────────────────────────────────────────────

class TestSliceTimingCorrection:

    def test_stc_step_in_applied_steps(self):
        result = _run_preprocess(apply_slice_timing=True, tr=TR)
        assert "slice_timing_correction" in result.applied_steps

    def test_stc_changes_data(self):
        rng = np.random.default_rng(4)
        arr = rng.random((*SPATIAL, T)).astype(np.float32) * 1000
        raw = _run_preprocess(arr=arr).preprocessed_data
        corrected = _run_preprocess(arr=arr, apply_slice_timing=True, tr=TR).preprocessed_data
        assert not np.allclose(raw, corrected)

    def test_stc_output_finite(self):
        result = _run_preprocess(apply_slice_timing=True, tr=TR)
        assert np.isfinite(result.preprocessed_data).all()

    def test_stc_output_same_shape(self):
        result = _run_preprocess(apply_slice_timing=True, tr=TR)
        assert result.preprocessed_data.shape == (*SPATIAL, T)

    def test_stc_raises_without_tr(self):
        # Config validation raises ValueError before pipeline runs
        with pytest.raises((ValueError, PreprocessingError)):
            _run_preprocess(apply_slice_timing=True, tr=None)

    def test_stc_interleaved_differs_from_sequential(self):
        rng = np.random.default_rng(5)
        arr = rng.random((*SPATIAL, T)).astype(np.float32) * 1000
        seq = _run_preprocess(
            arr=arr, apply_slice_timing=True, tr=TR, slice_order="sequential"
        ).preprocessed_data
        iln = _run_preprocess(
            arr=arr, apply_slice_timing=True, tr=TR, slice_order="interleaved"
        ).preprocessed_data
        assert not np.allclose(seq, iln)

    def test_stc_invalid_slice_order_raises(self):
        with pytest.raises(PreprocessingError, match="slice_order"):
            _run_preprocess(apply_slice_timing=True, tr=TR, slice_order="random")

    def test_stc_ref_slice_zero_vs_middle(self):
        rng = np.random.default_rng(6)
        arr = rng.random((*SPATIAL, T)).astype(np.float32) * 1000
        first = _run_preprocess(
            arr=arr, apply_slice_timing=True, tr=TR, slice_timing_ref_slice=0
        ).preprocessed_data
        middle = _run_preprocess(
            arr=arr, apply_slice_timing=True, tr=TR, slice_timing_ref_slice=-1
        ).preprocessed_data
        assert not np.allclose(first, middle)

    def test_stc_metadata_flag(self):
        result = _run_preprocess(apply_slice_timing=True, tr=TR)
        assert result.preprocess_metadata["slice_timing_applied"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Motion correction
# ─────────────────────────────────────────────────────────────────────────────

class TestMotionCorrection:

    def test_mc_step_in_applied_steps(self):
        result = _run_preprocess(apply_motion_correction=True)
        assert "motion_correction" in result.applied_steps

    def test_mc_returns_motion_params(self):
        result = _run_preprocess(apply_motion_correction=True)
        assert result.motion_params is not None
        assert result.motion_params.shape == (T, 6)

    def test_mc_first_volume_params_are_zero(self):
        """Volume 0 is the reference — its parameters must be zero."""
        result = _run_preprocess(apply_motion_correction=True)
        assert np.allclose(result.motion_params[0], 0.0)

    def test_mc_output_finite(self):
        result = _run_preprocess(apply_motion_correction=True)
        assert np.isfinite(result.preprocessed_data).all()

    def test_mc_output_same_shape(self):
        result = _run_preprocess(apply_motion_correction=True)
        assert result.preprocessed_data.shape == (*SPATIAL, T)

    def test_mc_static_volume_stays_close_to_reference(self):
        """A dataset with no motion should have near-zero motion parameters."""
        # All volumes identical → no motion to correct
        base = np.random.default_rng(7).random(SPATIAL).astype(np.float32) * 200
        arr = np.stack([base] * T, axis=-1)
        result = _run_preprocess(arr=arr, apply_motion_correction=True)
        # Parameters should be very small
        assert np.all(np.abs(result.motion_params) < 1.0)

    def test_mc_corrects_translated_volume(self):
        """A shifted volume should be pulled back toward the reference."""
        rng = np.random.default_rng(8)
        arr = rng.random((*SPATIAL, T)).astype(np.float32) * 200
        # Introduce a 1-voxel shift in the last volume
        arr[..., -1] = np.roll(arr[..., 0], shift=1, axis=0)
        result = _run_preprocess(arr=arr, apply_motion_correction=True)
        # The x-translation of the last volume should be non-zero
        assert abs(result.motion_params[-1, 0]) > 0.01

    def test_mc_metadata_flag(self):
        result = _run_preprocess(apply_motion_correction=True)
        assert result.preprocess_metadata["motion_correction_applied"] is True

    def test_motion_params_stored_in_bundle(self):
        result = _run_preprocess(apply_motion_correction=True)
        assert "motion_params_shape" in result.preprocess_metadata
        assert result.preprocess_metadata["motion_params_shape"] == (T, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Outlier detection
# ─────────────────────────────────────────────────────────────────────────────

class TestOutlierDetection:

    def test_od_step_in_applied_steps(self):
        result = _run_preprocess(apply_outlier_detection=True)
        assert "outlier_detection" in result.applied_steps

    def test_od_returns_outlier_mask(self):
        result = _run_preprocess(apply_outlier_detection=True)
        assert result.outlier_mask is not None
        assert result.outlier_mask.shape == (T,)
        assert result.outlier_mask.dtype == bool

    def test_od_output_finite(self):
        result = _run_preprocess(apply_outlier_detection=True)
        assert np.isfinite(result.preprocessed_data).all()

    def test_od_output_same_shape(self):
        result = _run_preprocess(apply_outlier_detection=True)
        assert result.preprocessed_data.shape == (*SPATIAL, T)

    def test_od_detects_spiked_volume(self):
        """A volume with extreme values should be flagged as an outlier."""
        rng = np.random.default_rng(10)
        arr = rng.random((*SPATIAL, T)).astype(np.float32) * 100
        spike_idx = T // 2
        arr[..., spike_idx] = arr[..., spike_idx] * 100.0  # artificial spike
        result = _run_preprocess(
            arr=arr,
            apply_outlier_detection=True,
            outlier_threshold_dvars=0.5,
        )
        assert result.outlier_mask[spike_idx], "Spiked volume should be flagged"

    def test_od_clean_data_no_outliers(self):
        """Perfectly constant data should produce no outliers."""
        arr = np.ones((*SPATIAL, T), dtype=np.float32) * 500.0
        result = _run_preprocess(arr=arr, apply_outlier_detection=True)
        assert not np.any(result.outlier_mask)

    def test_od_interpolation_strategy_replaces_outliers(self):
        """With 'interpolate' strategy, outlier volumes should differ from the original spike."""
        rng = np.random.default_rng(11)
        arr = rng.random((*SPATIAL, T)).astype(np.float32) * 100
        spike_idx = T // 2
        arr[..., spike_idx] *= 100.0
        original_spike = arr[..., spike_idx].copy()
        result = _run_preprocess(
            arr=arr,
            apply_outlier_detection=True,
            outlier_threshold_dvars=0.5,
            scrubbing_strategy="interpolate",
        )
        if result.outlier_mask[spike_idx]:
            assert not np.allclose(result.preprocessed_data[..., spike_idx], original_spike)

    def test_od_mark_strategy_does_not_change_data(self):
        """With 'mark' strategy, the data array is returned unchanged."""
        rng = np.random.default_rng(12)
        arr = rng.random((*SPATIAL, T)).astype(np.float32) * 100
        spike_idx = T // 2
        arr[..., spike_idx] *= 100.0
        result = _run_preprocess(
            arr=arr,
            apply_outlier_detection=True,
            outlier_threshold_dvars=0.5,
            scrubbing_strategy="mark",
        )
        # Data should be bit-for-bit equal to original for the spike volume
        assert np.allclose(result.preprocessed_data[..., spike_idx], arr[..., spike_idx])

    def test_od_n_outliers_in_metadata(self):
        rng = np.random.default_rng(13)
        arr = rng.random((*SPATIAL, T)).astype(np.float32) * 100
        arr[..., T // 2] *= 100.0
        result = _run_preprocess(
            arr=arr, apply_outlier_detection=True, outlier_threshold_dvars=0.5
        )
        assert result.preprocess_metadata["n_outlier_volumes"] >= 1

    def test_od_metadata_flag(self):
        result = _run_preprocess(apply_outlier_detection=True)
        assert result.preprocess_metadata["outlier_detection_applied"] is True

    def test_od_with_motion_params_computes_fd(self):
        """Running motion correction then outlier detection should work end-to-end."""
        result = _run_preprocess(
            apply_motion_correction=True,
            apply_outlier_detection=True,
        )
        assert result.outlier_mask is not None
        assert result.motion_params is not None


# ─────────────────────────────────────────────────────────────────────────────
# Bandpass filtering (in denoise.py)
# ─────────────────────────────────────────────────────────────────────────────

def _make_preprocess_bundle(arr=None):
    """Return a minimal BRAPHINPreprocessBundle with no steps applied."""
    if arr is None:
        rng = np.random.default_rng(20)
        arr = (rng.random((*SPATIAL, T)) * 200.0 + 600.0).astype(np.float32)
    ib = _make_input_bundle(arr)
    cfg = PreprocessConfig(
        apply_motion_correction=False, apply_slice_timing=False,
        apply_outlier_detection=False, apply_normalization=False,
        apply_smoothing=False,
    )
    return PreprocessBRAPHINData(ib, cfg).run()


class TestBandpassFiltering:

    def test_bandpass_step_in_applied_steps(self):
        pp = _make_preprocess_bundle()
        result = _run_denoise_from_preprocess(pp, apply_bandpass=True, tr=TR)
        assert "bandpass_filtering" in result.applied_steps

    def test_bandpass_changes_data(self):
        rng = np.random.default_rng(21)
        arr = rng.random((*SPATIAL, T)).astype(np.float32) * 200
        pp = _make_preprocess_bundle(arr)
        raw = DenoiseBRAPHINData(pp, DenoiseConfig(
            regress_confounds=False, apply_scrubbing=False, apply_bandpass=False
        )).run().voxel_time_series
        filtered = _run_denoise_from_preprocess(pp, apply_bandpass=True, tr=TR).voxel_time_series
        assert not np.allclose(raw, filtered)

    def test_bandpass_output_finite(self):
        pp = _make_preprocess_bundle()
        result = _run_denoise_from_preprocess(pp, apply_bandpass=True, tr=TR)
        assert np.isfinite(result.voxel_time_series).all()

    def test_bandpass_output_same_shape(self):
        pp = _make_preprocess_bundle()
        result = _run_denoise_from_preprocess(pp, apply_bandpass=True, tr=TR)
        assert result.voxel_time_series.shape == pp.voxel_time_series.shape

    def test_bandpass_raises_without_tr(self):
        pp = _make_preprocess_bundle()
        # Config validation raises ValueError before pipeline runs
        with pytest.raises((ValueError, DenoisingError)):
            _run_denoise_from_preprocess(pp, apply_bandpass=True, tr=None)

    def test_bandpass_attenuates_out_of_band_frequency(self):
        """
        A pure sine wave outside the passband should have lower power after
        filtering than one inside the passband.
        """
        times = np.arange(T) * TR
        # In-band: 0.03 Hz (between 0.008 and 0.09)
        inband = np.sin(2 * np.pi * 0.03 * times).astype(np.float32)
        # Out-of-band: 0.2 Hz (above 0.09 Hz high cutoff for TR=2s → nyq=0.25Hz)
        outband = np.sin(2 * np.pi * 0.2 * times).astype(np.float32)

        rng = np.random.default_rng(22)
        spatial_flat = rng.random((np.prod(SPATIAL), 1)).astype(np.float32)

        # Build (V, T) time series with the test signal
        ts_inband = np.outer(np.ones(np.prod(SPATIAL)), inband).astype(np.float32)
        ts_outband = np.outer(np.ones(np.prod(SPATIAL)), outband).astype(np.float32)

        def _filter(ts):
            arr4d = ts.reshape(*SPATIAL, T)
            ib = _make_input_bundle(arr4d)
            pp = PreprocessBRAPHINData(ib, PreprocessConfig(
                apply_motion_correction=False, apply_slice_timing=False,
                apply_outlier_detection=False, apply_normalization=False,
                apply_smoothing=False,
            )).run()
            cfg = DenoiseConfig(
                regress_confounds=False, apply_scrubbing=False,
                apply_bandpass=True, tr=TR,
            )
            return DenoiseBRAPHINData(pp, cfg).run().voxel_time_series

        filt_in = _filter(ts_inband)
        filt_out = _filter(ts_outband)

        power_in = float(np.mean(filt_in ** 2))
        power_out = float(np.mean(filt_out ** 2))
        # In-band signal should retain much more power than out-of-band
        assert power_in > power_out * 10

    def test_bandpass_raises_high_cutoff_above_nyquist(self):
        pp = _make_preprocess_bundle()
        cfg = DenoiseConfig(
            regress_confounds=False, apply_scrubbing=False,
            apply_bandpass=True, tr=TR,
            bandpass_low=0.008, bandpass_high=0.5,  # 0.5 Hz = Nyquist for TR=1s
        )
        with pytest.raises(DenoisingError):
            DenoiseBRAPHINData(pp, cfg).run()

    def test_bandpass_raises_low_cutoff_not_positive(self):
        pp = _make_preprocess_bundle()
        # bandpass_low=0.0 is caught by config __post_init__ (ValueError) before pipeline runs
        with pytest.raises((ValueError, DenoisingError)):
            DenoiseConfig(
                regress_confounds=False, apply_scrubbing=False,
                apply_bandpass=True, tr=TR,
                bandpass_low=0.0, bandpass_high=0.09,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Scrubbing (in denoise.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestDenoiseScrubbing:

    def test_scrubbing_step_in_applied_steps(self):
        pp = _make_preprocess_bundle()
        result = _run_denoise_from_preprocess(pp, apply_scrubbing=True)
        assert "scrubbing" in result.applied_steps

    def test_scrubbing_output_finite(self):
        pp = _make_preprocess_bundle()
        result = _run_denoise_from_preprocess(pp, apply_scrubbing=True)
        assert np.isfinite(result.voxel_time_series).all()

    def test_scrubbing_output_same_shape(self):
        pp = _make_preprocess_bundle()
        result = _run_denoise_from_preprocess(pp, apply_scrubbing=True)
        assert result.voxel_time_series.shape == pp.voxel_time_series.shape

    def test_scrubbing_uses_preprocess_outlier_mask(self):
        """When preprocess has run outlier_detection, denoise uses the stored mask."""
        rng = np.random.default_rng(30)
        arr = rng.random((*SPATIAL, T)).astype(np.float32) * 100
        arr[..., T // 2] *= 100.0  # spike
        ib = _make_input_bundle(arr)
        pp = PreprocessBRAPHINData(ib, PreprocessConfig(
            apply_motion_correction=False, apply_slice_timing=False,
            apply_outlier_detection=True, apply_normalization=False,
            apply_smoothing=False, outlier_threshold_dvars=0.5,
        )).run()
        result = _run_denoise_from_preprocess(pp, apply_scrubbing=True)
        assert "scrubbing" in result.applied_steps
