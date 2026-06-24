# Changelog

All notable changes to BRAPHIN are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2024

First stable release accompanying the publication of the undergraduate thesis
*"BRAPHIN: A Python Library for fMRI Functional Connectivity Graph Analysis"*
(Universidad de Alcalá, David Ortega Lozano).

### Added

#### braphin — fMRI pipeline
- `InputMRIData`: NIfTI loading and validation with support for auxiliary files (TSV, CSV, NPY, JSON).
- `PreprocessMRIData`: NaN/inf cleaning and optional global intensity normalisation.
- `DenoiseMRIData`: least-squares confound regression.
- `TransformMRIData`: atlas parcellation, automatic nearest-neighbour resampling to fMRI space, ROI mean time-series extraction, centroid computation and JSON caching.
- `ModelMRIConnectivityData`: Pearson correlation, cross-correlation, corrected cross-correlation; optional absolute threshold.
- Four bundled atlases: AAL (116 ROIs), Schaefer 100 / 200 / 400.
- Typed bundle dataclasses (`MRIInputBundle`, `MRIPreprocessBundle`, `MRIDenoiseBundle`, `MRITransformBundle`, `MRIConnectivityBundle`) for full pipeline traceability.
- `AtlasDefinition` registry and helpers (`list_supported_atlases`, `get_atlas_definition`, `get_atlas_roi_name_map`).
- Strategy pattern for connectivity (`ConnectivityStrategy` ABC + three concrete strategies).
- Hierarchical exception system (`BRAPHINError` → 7 specific subclasses).
- Config dataclasses (`InputConfig`, `PreprocessConfig`, `DenoiseConfig`, `AtlasConfig`, `ConnectivityConfig`, `BRAPHINConfig`).

#### eegraph — unified API
- `Graph.load_data(modality="fmri")` and `Graph.modelate()` route to the braphin pipeline.
- Original EEG support preserved without breaking changes.

#### Tests
- 167 pytest unit and integration tests covering every pipeline stage.
- All tests use synthetic data; no external files required.

### Fixed

- **Bug 1** (`transform.py`): crash when `atlas_data` was passed directly as an ndarray or NIfTI image with `atlas_name=None`; the centroid builder now falls back to computing centroids from the supplied data.
- **Bug 2** (`io/tabular.py`): BIDS TSV confound files with text header rows (e.g. `trans_x\ttrans_y\t…`) were parsed as a NaN row, producing an array of shape `(T+1, K)` and breaking confound regression; the loader now auto-detects and skips the header.
- **Bug 3** (`atlas.py`): the AAL ROI name map used coded anatomical IDs (2001, 2101, …) instead of the sequential integer labels (1–116) stored in the NIfTI file; anatomical names were never returned.
- **Bug 4** (`tools.py`): `_corr_cross_correlation_coef` computed `corCC = positive_lag − negative_lag` without reversing `negative_lag`, misaligning lags and breaking the antisymmetry property `corCC(x,y) = −corCC(y,x)`.
- **Bug 5** (`denoise.py`): `_find_confounds_matrix` raised `DenoisingError` on the first incompatible confound file instead of continuing the search; if a second, compatible file followed it was never found.
- **Bug 6** (`transform.py`): `centroid_coordinate_space` was unconditionally overwritten to `"world"` regardless of whether the atlas had an affine matrix; voxel-space atlases now correctly report `"voxel"`.
- **Bug 7** (`__init__.py`): two import lines for `strategy` and `modelateData` were duplicated.
- **Bug 8** (`transform.py`): unused `from importlib import metadata` removed.

---

## [Unreleased]

### Added

- **Motion correction** (`preprocess.py`): rigid-body 6-parameter realignment of each volume to the first volume using Powell optimisation (`scipy.optimize.minimize`) and `scipy.ndimage.affine_transform`. Motion parameters (T × 6) are stored in `MRIPreprocessBundle.motion_params`.
- **Slice-timing correction** (`preprocess.py`): linear interpolation of each slice's time series to a common reference acquisition time. Supports sequential and interleaved slice orders. Requires `PreprocessConfig.tr`.
- **Outlier detection / scrubbing** (`preprocess.py`): DVARS-based outlier detection with configurable IQR multiplier threshold. Outlier volumes are interpolated from neighbours (`"interpolate"`) or just flagged (`"mark"`). Outlier mask stored in `MRIPreprocessBundle.outlier_mask`.
- **Spatial smoothing** (`preprocess.py`): isotropic Gaussian kernel applied per 3-D volume. FWHM (mm) is converted to per-axis sigma using voxel sizes from the NIfTI header.
- **Bandpass filtering** (`denoise.py`): zero-phase 5th-order Butterworth bandpass filter (`scipy.signal.butter` + `sosfiltfilt`) applied to each voxel time series. Requires `DenoiseConfig.tr`.
- **Scrubbing** (`denoise.py`): reuses the `outlier_mask` from the preprocess bundle if available; otherwise recomputes DVARS on the denoised signal. Interpolates flagged volumes from clean neighbours.
- **New `PreprocessConfig` fields**: `tr`, `slice_order`, `slice_timing_ref_slice`, `outlier_threshold_dvars`, `scrubbing_strategy`.
- **New `DenoiseConfig` field**: `tr`.
- **New `MRIPreprocessBundle` fields**: `motion_params` (ndarray T×6), `outlier_mask` (bool ndarray T).
- **49 new pytest tests** across `tests/test_preprocess_new_steps.py` covering all six new implementations.

### Planned

- Per-voxel percent signal change (PSC) normalisation.
- Brain masking to exclude non-brain voxels.
- Dynamic (windowed) functional connectivity.
- TR extraction from NIfTI header and automatic propagation through the pipeline.
