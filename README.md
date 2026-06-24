# BRAPHIN

[![Tests](https://github.com/ufvceiec/braphin/actions/workflows/tests.yml/badge.svg)](https://github.com/ufvceiec/braphin/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/ufvceiec/braphin/branch/main/graph/badge.svg)](https://codecov.io/gh/ufvceiec/braphin)
[![PyPI version](https://badge.fury.io/py/braphin.svg)](https://pypi.org/project/braphin/)
[![Python versions](https://img.shields.io/pypi/pyversions/braphin.svg)](https://pypi.org/project/braphin/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**BRAPHIN** is a Python library for analysing resting-state functional MRI (fMRI) data through a brain-connectivity graph pipeline. It extends [EEGraph](https://github.com/ufvceiec/EEGRAPH) — originally designed for EEG — to support fMRI, providing a unified interface for both modalities and enabling downstream Graph Neural Network (GNN) analyses such as Parkinson's disease classification.

---

## Features

- **Full fMRI pipeline**: NIfTI loading → preprocessing (slice timing, motion correction, outlier detection, normalisation, smoothing) → denoising (confound regression, scrubbing, bandpass) → atlas parcellation → ROI time-series extraction → functional connectivity
- **Three connectivity measures**: Pearson correlation, cross-correlation, corrected cross-correlation
- **Four bundled atlases**: AAL (116 ROIs), Schaefer 100 / 200 / 400
- **Automatic atlas resampling** to the subject's fMRI space via nearest-neighbour interpolation
- **ROI centroid caching** in world coordinates (JSON)
- **GNN-ready output**: NetworkX graphs with node positions, compatible with PyTorch Geometric
- **Unified EEG + fMRI API** through `eegraph.Graph`
- **387 passing unit and integration tests**

---

## Pipeline overview

```
NIfTI file (.nii / .nii.gz)
      │
      ▼
 InputMRIData          ← validates format, loads auxiliaries (confounds, events)
      │
      ▼
 PreprocessMRIData     ← NaN/inf cleaning, slice-timing correction, motion correction,
                          outlier detection, per-voxel temporal z-score normalisation, spatial smoothing
      │
      ▼
 DenoiseMRIData        ← confound regression, scrubbing, bandpass filtering
      │
      ▼
 TransformMRIData      ← atlas parcellation, ROI mean time-series extraction
      │
      ▼
 ModelMRIConnectivity  ← connectivity matrix (Pearson / cross-corr)
      │
      ▼
 NetworkX Graph        ← nodes = ROIs, edges = connectivity weights
```

---

## Installation

### Core library (fMRI only)

```bash
pip install -r requirements.txt
```

### With EEG support

```bash
pip install -r requirements-eeg.txt
```

### With GNN support

```bash
pip install -r requirements-gnn.txt
```

### From source (editable install)

```bash
git clone https://github.com/<your-org>/braphin.git
cd braphin
pip install -e .
```

---

## Quick start — fMRI

```python
from braphin import (
    InputMRIData, PreprocessMRIData, DenoiseMRIData,
    TransformMRIData, ModelMRIConnectivityData,
    PreprocessConfig, DenoiseConfig, AtlasConfig, ConnectivityConfig,
)

# 1. Load
input_bundle = InputMRIData(
    "sub-01_task-rest_bold.nii.gz",
    auxiliary_paths=["sub-01_task-rest_confounds_timeseries.tsv"],
).load()

# 2. Preprocess
pp_bundle = PreprocessMRIData(
    input_bundle,
    PreprocessConfig(apply_normalization=True),
).run()

# 3. Denoise — regress out motion confounds
dn_bundle = DenoiseMRIData(
    pp_bundle,
    DenoiseConfig(regress_confounds=True),
).run()

# 4. Parcellate with AAL atlas (116 ROIs)
tx_bundle = TransformMRIData(
    dn_bundle,
    config=AtlasConfig(atlas_name="aal"),
).run()

# 5. Compute Pearson functional connectivity
conn_bundle = ModelMRIConnectivityData(
    tx_bundle,
    ConnectivityConfig(method="pearson_correlation", threshold=0.3),
).run()

matrix = conn_bundle.connectivity_matrix   # shape (116, 116)
graph  = conn_bundle                       # NetworkX graph built downstream
```

---

## Quick start — unified EEG + fMRI API

```python
from eegraph.graph import Graph

# fMRI
G = Graph()
G.load_data("sub-01_bold.nii.gz", modality="fmri")
graph, matrix = G.modelate(
    window_size=None,
    connectivity="pearson_correlation",
    threshold=0.3,
    atlas_config=AtlasConfig(atlas_name="schaefer_100"),
)

# EEG (unchanged from original EEGraph)
G = Graph()
G.load_data("sub-01_eeg.edf", modality="eeg")
graph, matrix = G.modelate(window_size=2, connectivity="pearson_correlation")
```

---

## Supported atlases

| Name | ROIs | Type | Description |
|---|---|---|---|
| `aal` | 116 | Anatomical | Automated Anatomical Labeling — classic anatomical parcellation |
| `schaefer_100` | 100 | Functional | Schaefer 2018 functional atlas |
| `schaefer_200` | 200 | Functional | Schaefer 2018 — medium granularity |
| `schaefer_400` | 400 | Functional | Schaefer 2018 — high granularity |

Custom atlases are also supported via `AtlasConfig(atlas_path="my_atlas.nii.gz")` or by passing a NIfTI image or NumPy array directly to `TransformMRIData`.

---

## Connectivity methods

| Method key | Description | Symmetric |
|---|---|---|
| `pearson_correlation` | Pearson r between ROI time series | Yes |
| `cross_correlation` | Normalised cross-correlation (lag 0 – 10%) | No |
| `corr_cross_correlation` | Corrected cross-correlation: `Rxy(+lag) − Rxy(−lag)` | No (antisymmetric) |

---

## Configuration reference

All pipeline stages are controlled by dataclasses in `braphin.config`:

```python
PreprocessConfig(
    apply_slice_timing      = False,   # linear interpolation per slice to reference time
    tr                      = None,    # repetition time in seconds (required for slice timing)
    slice_order             = "sequential",   # "sequential" or "interleaved"
    slice_timing_ref_slice  = 0,       # reference slice index
    apply_motion_correction = False,   # rigid-body realignment to volume 0 (Powell optimisation)
    apply_outlier_detection = False,   # DVARS-based outlier detection
    outlier_threshold_dvars = 1.5,     # IQR multiplier for DVARS threshold
    scrubbing_strategy      = "interpolate",  # "interpolate" or "mark"
    apply_normalization     = True,    # per-voxel temporal z-score normalisation
    apply_smoothing         = False,   # isotropic Gaussian spatial smoothing
    smoothing_fwhm          = 6.0,     # FWHM in mm
)

DenoiseConfig(
    regress_confounds = True,    # least-squares confound regression
    apply_scrubbing   = False,   # interpolate outlier volumes (uses preprocess outlier_mask if available)
    apply_bandpass    = False,   # zero-phase 5th-order Butterworth bandpass filter
    bandpass_low      = 0.008,   # Hz — lower cutoff
    bandpass_high     = 0.1,     # Hz — upper cutoff (Biswal 1995: 0.008–0.1 Hz)
    tr                = None,    # repetition time in seconds (required for bandpass)
)

AtlasConfig(
    atlas_name = "aal",          # one of the bundled atlas names
    atlas_path = None,           # path to a custom NIfTI atlas
    roi_labels = None,           # optional list of custom ROI names
)

ConnectivityConfig(
    method      = "pearson_correlation",
    threshold   = None,          # absolute threshold (keeps |r| >= threshold)
    window_size = None,          # None = static; float (seconds) = windowed dynamic (planned)
)
```

---

## Running the tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run the full test suite (387 tests)
pytest

# With coverage report
pytest --cov=braphin --cov-report=term-missing
```

---

## Project structure

```
braphin-main/
├── braphin/                  # Core fMRI library
│   ├── config.py              # Configuration dataclasses
│   ├── importMRIData.py       # Stage 1: NIfTI loading
│   ├── preprocess.py          # Stage 2: preprocessing
│   ├── denoise.py             # Stage 3: denoising
│   ├── transform.py           # Stage 4: atlas parcellation
│   ├── modelateData.py        # Stage 5: connectivity
│   ├── strategy.py            # Connectivity strategy pattern
│   ├── tools.py               # Connectivity measures
│   ├── atlas.py               # Atlas registry and helpers
│   ├── exceptions.py          # Exception hierarchy
│   ├── io/
│   │   ├── nifti.py           # NIfTI I/O
│   │   └── tabular.py         # CSV / TSV / NPY I/O
│   ├── atlases/               # Bundled atlas NIfTI files
│   └── atlas_centroids/       # Pre-computed centroid JSON files
├── eegraph/                   # Unified EEG + fMRI API (extends original EEGraph)
├── tests/                     # Pytest test suite (387 tests)
├── Examples/                  # Jupyter notebook tutorials
├── requirements.txt           # Core dependencies
├── requirements-eeg.txt       # EEG optional dependencies
├── requirements-gnn.txt       # GNN optional dependencies
├── requirements-dev.txt       # Development dependencies
└── pyproject.toml             # Package metadata
```

---

## Exception hierarchy

```
BRAPHINError
├── MRIInputError       — file not found, wrong extension
├── MRIFormatError      — wrong NIfTI dimensions
├── AtlasError          — unsupported atlas, shape mismatch
├── PreprocessingError  — invalid bundle, normalisation failure
├── DenoisingError      — confound shape mismatch
├── TransformationError — atlas/fMRI space mismatch, ROI extraction failure
└── ConnectivityError   — unsupported method, matrix computation failure
```

---

## Citation

If you use BRAPHIN in your research, please cite:

```bibtex
@thesis{ortega2024braphin,
  author  = {Ortega Lozano, David},
  title   = {BRAPHIN: A Python Library for fMRI Functional Connectivity Graph Analysis},
  school  = {Universidad de Alcalá},
  year    = {2024},
  type    = {Trabajo de Fin de Grado},
}
```

---

## Known limitations

The following are acknowledged limitations of the current release:

- **No global signal regression (GSR).** GSR is a common preprocessing step in resting-state fMRI but is not yet implemented. Users comparing results with HCP pipelines should be aware of this difference.
- **No partial correlation / precision matrix.** Only full Pearson correlation, normalised cross-correlation, and corrected cross-correlation are supported. Regularised partial correlation (e.g. graphical lasso) is a planned future addition.
- **Cross-correlation is O(N²).** The cross-correlation and corrected cross-correlation strategies iterate over all ROI pairs in pure Python. For large atlases (Schaefer 400) this is ~80,000 iterations per call; vectorisation with NumPy/SciPy FFT is planned.
- **Motion correction convention.** `scipy.ndimage.affine_transform` maps output→input coordinates. The estimated `motion_params` therefore represent the *inverse* rigid-body transform, not physical head displacement. They should not be used directly as quality-control motion parameters without sign-reversal.
- **Slice-timing uses first-order (linear) interpolation.** This is acceptable for typical TR ≥ 1 s, but higher-order methods may be preferable for sub-second TRs.
- **`BRAPHINConfig` is not consumed by the pipeline.** The convenience aggregator dataclass is exported for user-side configuration management but is not yet accepted as an input to any pipeline stage.
- **No real-data validation.** The test suite uses synthetic NumPy data generated at runtime. Validation against a reference neuroimaging dataset (e.g. OpenNeuro ds000031) is planned.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
