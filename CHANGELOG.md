# Changelog

All notable changes to BRAPHIN are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.2.0] — 2025

### Fixed (Critical)

- **Regression/bandpass order** (`denoise.py`, #1): The pipeline now correctly bandpass-filters both the signal and the confound regressors **before** OLS regression (Lindquist et al. 2019). The previous order (regression → bandpass) inflated functional connectivity estimates. New order: scrubbing → bandpass(signal + confounds) → regression.
- **Motion correction now in mm world space** (`preprocess.py`, #2): The rigid-body optimiser now operates in physical mm space by composing the NIfTI affine. `motion_params` stores mm translations and radian rotations in world space. Previously the optimiser operated in voxel index space, which is incorrect for anisotropic voxels.
- **`get_motion_confounds(bundle)` utility added** (`preprocess.py`, #3): Motion parameters must be negated before use as confound regressors. The new function `get_motion_confounds(preprocess_bundle)` returns `-motion_params` (shape T × 6). Exported from `braphin`.

### Fixed (Moderate)

- **Partial correlation raises error when N_ROIs ≥ T** (`tools.py`, #4): Previously silently returned nonsensical results when the covariance matrix was rank-deficient. Now raises `ConnectivityError` with a clear diagnostic message.
- **`PreprocessConfig.slice_axis` field added** (`config.py`, `preprocess.py`, #5): New integer field (default `2`, i.e. Z-axis) that allows coronal or sagittal acquisitions to specify the correct slice dimension for slice-timing correction. Validated as 0, 1, or 2.
- **BIDS JSON sidecar parsing** (`importBRAPHINData.py`, #6): Auxiliary JSON files are now parsed for `"SliceTiming"` (per-slice acquisition offsets) and `"RepetitionTime"`. Stored in `fmri_metadata["slice_timing_offsets"]` and `fmri_metadata["tr"]`. A `"RepetitionTime"` found in the sidecar overrides the value read from the NIfTI header.
- **NaN in confound matrix raises error** (`denoise.py`, #7): Confound regression now raises `DenoisingError` listing the affected column indices if any confound contains NaN (e.g. missing first-volume FD), instead of silently propagating NaN through the residuals.
- **TR auto-extracted from NIfTI header** (`importBRAPHINData.py`, #8): TR is now read from `pixdim[4]` at load time when its value is plausible (0.1–20 s) and stored in `fmri_metadata["tr"]`. The BIDS sidecar `"RepetitionTime"` takes precedence. Users no longer need to set TR manually when the NIfTI header is correctly populated.

### Fixed (Minor)

- **Stale PLV/PLI/wPLI removed from `ConnectivityConfig` docstring** (`config.py`, #9): The `method` field description now points to `list_fmri_connectivity_measures()` instead of listing unsupported EEG-only phase measures.
- **MI and TE histogram bias warnings** (`tools.py`, #10): `UserWarning` is now raised when the sample size is too small for reliable histogram estimation (T < n_bins² × 5 for MI; T < n_bins³ × 5 for TE).
- **Bivariate Granger causality warning** (`tools.py`, #11): `UserWarning` raised when N > 5 ROIs, citing Ding et al. (2006). The docstring explains why pairwise GC produces spurious connections in larger networks.
- **Signed imaginary coherence** (`tools.py`, #12): `imag_coherence` now computes the standard Nolte et al. (2004) **signed** imaginary coherence `Im(C_xy)`, which is antisymmetric: IC(i,j) = −IC(j,i). Previously computed the unsigned absolute value `|Im(C_xy)|`, which discards phase-direction information.
- **Confound file fallback patterns** (`denoise.py`, #13): `_find_confounds_matrix` now also matches filenames containing `"motion"`, `"regressors"`, `"nuisance"`, or `"timeseries"` as fallbacks when no `"confound"` file is found. A logged warning is emitted for fallback matches.
- **Centroid space mismatch warning** (`transform.py`, #14): A `UserWarning` is issued when the atlas was resampled to fMRI space, indicating that the stored centroid coordinates are from the reference atlas space (typically MNI152), not the subject's native space.

---

## [1.1.0] — 2025

### Added

#### braphin — new preprocessing steps
- **Motion correction** (`preprocess.py`): rigid-body 6-parameter realignment of each volume to the first volume using Powell optimisation (`scipy.optimize.minimize`) and `scipy.ndimage.affine_transform`. Motion parameters (T × 6) are stored in `BRAPHINPreprocessBundle.motion_params`.
- **Slice-timing correction** (`preprocess.py`): linear interpolation of each slice's time series to a common reference acquisition time. Supports sequential and interleaved slice orders. Requires `PreprocessConfig.tr`.
- **Outlier detection / scrubbing** (`preprocess.py`): DVARS-based outlier detection with configurable IQR-multiplier threshold. Outlier volumes are interpolated from neighbours (`"interpolate"`) or just flagged (`"mark"`). Outlier mask stored in `BRAPHINPreprocessBundle.outlier_mask`.
- **Spatial smoothing** (`preprocess.py`): isotropic Gaussian kernel applied per 3-D volume. FWHM (mm) is converted to per-axis sigma using voxel sizes from the NIfTI header.

#### braphin — new denoising steps
- **Bandpass filtering** (`denoise.py`): zero-phase 5th-order Butterworth bandpass filter (`scipy.signal.butter` + `sosfiltfilt`) applied to each voxel time series. Requires `DenoiseConfig.tr`.
- **Scrubbing** (`denoise.py`): reuses the `outlier_mask` from the preprocess bundle if available; otherwise recomputes DVARS on the denoised signal. Interpolates flagged volumes from clean neighbours.

#### braphin — frequency bands module (`braphin.bands`)
- `FMRI_BANDS` — predefined BOLD oscillation bands based on Zuo et al. (2010): `slow5 (0.010–0.027 Hz)`, `slow4 (0.027–0.073 Hz)`, `slow3 (0.073–0.167 Hz)`, `broadband (0.010–0.100 Hz)`.
- `bandpass_roi_time_series(roi_time_series, tr, low, high)` — zero-phase Butterworth bandpass applied to ROI time series.
- `compute_band_connectivity(roi_time_series, tr, band_name, method)` — compute connectivity for a single named BOLD band.
- `compute_all_bands_connectivity(roi_time_series, tr, method)` — compute connectivity for all four BOLD bands; returns a dict `{band_name: ndarray}`.

#### braphin — visualisation module (`braphin.visualize`)
- `build_fmri_graph(connectivity_matrix, roi_labels, roi_centroids_3d, centroid_coordinate_space, projection)` — render a functional connectivity graph projected onto a brain slice. Projection can be `"axial"`, `"coronal"`, or `"sagittal"`. Returns a Matplotlib figure.
- `visualize_html` and `visualize_png` — helper renderers.

#### braphin — orchestrator
- `ModelMRIData` — high-level orchestrator that runs the full fMRI pipeline (Input → Preprocess → Denoise → Transform → Connectivity) in a single `connectivity_workflow()` call.

#### braphin — graph entry point
- `BRAPHINGraph` (exported as `Graph`) — subclasses `eegraph.graph.Graph` to add fMRI routing. Import via `from braphin import Graph`.
- `Graph.load_data(path, modality="fmri")` and `Graph.modelate(...)` now supported for both fMRI and EEG.

#### Connectivity methods expanded
Connectivity methods grew from 3 to 15. New methods added in this release:
- `partial_correlation` — via precision matrix inversion
- `coherence` — magnitude-squared coherence (TR required)
- `imag_coherence` — mean absolute imaginary coherence (TR required)
- `lagged_coherence` — Pascual-Marqui lagged coherence (TR required)
- `aec` — Amplitude Envelope Correlation (Hilbert)
- `aec_orth` — Orthogonalized AEC (Hipp et al. 2012)
- `mutual_information` — joint-histogram mutual information
- `sync_likelihood` — Synchronisation Likelihood (Stam & van Dijk 2002)
- `granger_causality` — bivariate linear Granger Causality (directed)
- `transfer_entropy` — histogram-based Transfer Entropy (directed)
- `pdc` — Partial Directed Coherence (Baccalá & Sameshima 2001; TR required)
- `psi` — Phase Slope Index (Nolte et al. 2008; TR required)

Full list: `pearson_correlation`, `partial_correlation`, `cross_correlation`, `corr_cross_correlation`, `coherence`, `imag_coherence`, `lagged_coherence`, `aec`, `aec_orth`, `mutual_information`, `sync_likelihood`, `granger_causality`, `transfer_entropy`, `pdc`, `psi`.

#### New `PreprocessConfig` fields
`tr`, `slice_order`, `slice_timing_ref_slice`, `outlier_threshold_dvars`, `scrubbing_strategy`.

#### New `DenoiseConfig` field
`tr`.

#### New bundle fields
- `BRAPHINPreprocessBundle.motion_params` (ndarray T×6): estimated rigid-body motion parameters per volume. `None` if motion correction was not applied.
- `BRAPHINPreprocessBundle.outlier_mask` (bool ndarray T): DVARS-detected outlier volumes. `None` if outlier detection was not applied.

#### Standardised connectivity registries
- `FMRI_CONNECTIVITY_MEASURES` — dict mapping fMRI method names to descriptions.
- `EEG_CONNECTIVITY_MEASURES` — dict mapping EEG method names (from eegraph).
- `list_fmri_connectivity_measures()` — returns list of fMRI method names.
- `list_eeg_connectivity_measures()` — returns list of EEG method names.

#### Tests
- 49 new pytest tests across `tests/test_preprocess_new_steps.py` covering all six new preprocessing implementations.
- Additional tests for new connectivity methods, bands module, and graph visualisation.
- Total test suite: 400+ tests.

### Refactored

- **All pipeline classes renamed** to use the `BRAPHIN` prefix consistently:
  - `InputMRIData` → `InputfMRIData` (input stage now also reflects the fMRI modality explicitly)
  - `PreprocessMRIData` → `PreprocessBRAPHINData`
  - `DenoiseMRIData` → `DenoiseBRAPHINData`
  - `TransformMRIData` → `TransformBRAPHINData`
  - `ModelMRIConnectivityData` → `ModelBRAPHINConnectivityData`
- **All bundle dataclasses renamed** to use the `BRAPHIN` prefix:
  - `MRIInputBundle` → `BRAPHINInputBundle`
  - `MRIPreprocessBundle` → `BRAPHINPreprocessBundle`
  - `MRIDenoiseBundle` → `BRAPHINDenoiseBundle`
  - `MRITransformBundle` → `BRAPHINTransformBundle`
  - `MRIConnectivityBundle` → `BRAPHINConnectivityBundle`
- **`BRAPHINGraph`** introduced as the unified entry point. Subclasses `eegraph.graph.Graph`. Exported as `Graph` for convenience.
- **`ModelMRIData`** orchestrator added to `braphin/model.py`.
- **Layout helpers** extracted from `ModelMRIData` to `braphin/visualize.py` as `build_fmri_graph`.
- **Strict one-way dependency enforced**: `eegraph` has zero imports from `braphin`. All cross-modality routing lives in `braphin.graph.BRAPHINGraph`.
- **`apply_normalization`** field in `PreprocessConfig` renamed to `apply_voxel_zscore` to better reflect the per-voxel temporal z-score operation it performs.
- **`[BRAPHIN]` prefix** now used in `display_info()` output across all pipeline classes, replacing the old `[EEGraph]` prefix.
- **All Spanish comments and docstrings** translated to English.
- **Phase-based measures removed from fMRI** (PLV, PLI, wPLI, dWPLI, PPC): these measures rely on instantaneous phase estimates, which are unreliable for BOLD signals due to the haemodynamic response function smoothing. They remain available for EEG via the eegraph pipeline.

---

## [1.0.0] — 2024

First stable release accompanying the publication of the undergraduate thesis
*"BRAPHIN: A Python Library for fMRI Functional Connectivity Graph Analysis"*
(Universidad de Alcalá, David Ortega Lozano).

### Added

#### braphin — fMRI pipeline
- `InputfMRIData`: NIfTI loading and validation with support for auxiliary files (TSV, CSV, NPY, JSON).
- `PreprocessBRAPHINData`: NaN/inf cleaning and optional global intensity normalisation.
- `DenoiseBRAPHINData`: least-squares confound regression.
- `TransformBRAPHINData`: atlas parcellation, automatic nearest-neighbour resampling to fMRI space, ROI mean time-series extraction, centroid computation and JSON caching.
- `ModelBRAPHINConnectivityData`: Pearson correlation, cross-correlation, corrected cross-correlation; optional absolute threshold.
- Four bundled atlases: AAL (116 ROIs), Schaefer 100 / 200 / 400.
- Typed bundle dataclasses (`BRAPHINInputBundle`, `BRAPHINPreprocessBundle`, `BRAPHINDenoiseBundle`, `BRAPHINTransformBundle`, `BRAPHINConnectivityBundle`) for full pipeline traceability.
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
