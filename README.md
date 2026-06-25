# BRAPHIN

[![Tests](https://github.com/albertonogales/braphin/actions/workflows/tests.yml/badge.svg)](https://github.com/albertonogales/braphin/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/albertonogales/braphin/main/coverage.json)](https://github.com/albertonogales/braphin/actions)
[![PyPI version](https://badge.fury.io/py/braphin.svg)](https://pypi.org/project/braphin/)
[![Python versions](https://img.shields.io/pypi/pyversions/braphin.svg)](https://pypi.org/project/braphin/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**BRAPHIN** is a Python library that turns fMRI brain scans into functional connectivity graphs — ready for machine learning and graph neural network analysis.

---

## What it does

- **Reads NIfTI fMRI files** — the standard brain scan file format (`.nii` / `.nii.gz`)
- **Cleans and denoises the data** — removes invalid values, corrects for head movement, applies bandpass filtering
- **Parcellates the brain using an atlas** — divides the brain into regions (ROIs) and extracts the average signal from each region over time
- **Computes functional connectivity** — measures how similarly each pair of brain regions behaves over time (15 methods available)
- **Outputs a NetworkX graph** — nodes are brain regions, edges are connectivity weights, ready for GNN classification

---

## Quick start

The simplest way to run the full pipeline is the four-line `Graph` API:

```python
from braphin import Graph

g = Graph()
g.load_data("sub-01_bold.nii.gz", modality="fmri")
graph, matrix = g.modelate(connectivity="pearson_correlation", window_size=None)
# matrix — NumPy array (N_regions × N_regions) of connectivity weights
# graph  — NetworkX graph with brain region nodes
```

---

## Step-by-step pipeline

For more control, you can run each stage individually. Each stage returns a bundle you can inspect before passing to the next step.

```python
from braphin import (
    InputfMRIData, PreprocessBRAPHINData, DenoiseBRAPHINData,
    TransformBRAPHINData, ModelBRAPHINConnectivityData,
    PreprocessConfig, DenoiseConfig, AtlasConfig, ConnectivityConfig,
)

# Stage 1 — Load the NIfTI file (and any confound files)
input_bundle = InputfMRIData(
    "sub-01_task-rest_bold.nii.gz",
    auxiliary_paths=["sub-01_task-rest_confounds_timeseries.tsv"],
).load()

# Stage 2 — Clean and normalise the data
pp_bundle = PreprocessBRAPHINData(
    input_bundle,
    PreprocessConfig(apply_voxel_zscore=True),
).run()

# Stage 3 — Remove the influence of head movement from the signal
dn_bundle = DenoiseBRAPHINData(
    pp_bundle,
    DenoiseConfig(regress_confounds=True),
).run()

# Stage 4 — Divide the brain into 116 regions using the AAL atlas
tx_bundle = TransformBRAPHINData(
    dn_bundle,
    config=AtlasConfig(atlas_name="aal"),
).run()

# Stage 5 — Compute Pearson correlation between every pair of regions
conn_bundle = ModelBRAPHINConnectivityData(
    tx_bundle,
    ConnectivityConfig(method="pearson_correlation", threshold=0.3),
).run()

matrix = conn_bundle.connectivity_matrix   # NumPy array, shape (116, 116)
graph  = conn_bundle.graph                 # NetworkX graph
```

---

## Graph visualisation

```python
from braphin import build_fmri_graph, visualize_html, visualize_png

G = build_fmri_graph(
    connectivity_matrix=conn_bundle.connectivity_matrix,
    roi_labels=conn_bundle.roi_labels,
    roi_centroids_3d=conn_bundle.roi_centroids_3d,
    projection="axial",   # "axial", "coronal", or "sagittal"
)

visualize_html(G, "my_subject")   # saves my_subject_plot.html (interactive)
visualize_png(G, "my_subject")    # saves my_subject.png
```

---

## Supported atlases

An atlas is a brain map that divides the brain into named regions. BRAPHIN includes four bundled atlases and also supports custom atlases.

| Name | ROIs | Type | Description |
|---|---|---|---|
| `aal` | 116 | Anatomical | Automated Anatomical Labeling — classic anatomical parcellation |
| `schaefer_100` | 100 | Functional | Schaefer 2018 functional atlas |
| `schaefer_200` | 200 | Functional | Schaefer 2018 — medium granularity |
| `schaefer_400` | 400 | Functional | Schaefer 2018 — high granularity |

Custom atlases: `AtlasConfig(atlas_path="my_atlas.nii.gz")` or pass a NIfTI image or NumPy array directly to `TransformBRAPHINData`.

---

## Connectivity methods

Functional connectivity measures how similar two brain regions' activity patterns are over time. BRAPHIN supports 15 methods.

| Method key | Description | Symmetric | TR required |
|---|---|---|---|
| `pearson_correlation` | Linear correlation between two region signals | Yes | No |
| `partial_correlation` | Correlation after removing shared influence of all other regions | Yes | No |
| `cross_correlation` | Correlation at the lag that maximises similarity (up to 10% of signal length) | No | No |
| `corr_cross_correlation` | Signed cross-correlation: positive lag minus negative lag | No | No |
| `coherence` | Frequency-domain similarity (averaged over frequencies) | Yes | Yes |
| `imag_coherence` | Imaginary part of coherence — less sensitive to shared noise | No | Yes |
| `lagged_coherence` | Coherence at non-zero phase lag only | Yes | Yes |
| `aec` | Amplitude Envelope Correlation — similarity of signal power envelopes | Yes | No |
| `aec_orth` | Orthogonalized AEC — removes spurious correlation from signal leakage | Yes | No |
| `mutual_information` | Non-linear dependency between two region signals | Yes | No |
| `sync_likelihood` | Generalised synchronisation measure | Yes | No |
| `granger_causality` | Whether signal in region A helps predict region B | No (directed) | No |
| `transfer_entropy` | Information flow from one region to another | No (directed) | No |
| `pdc` | Partial Directed Coherence — directed frequency-domain influence | No (directed) | Yes |
| `psi` | Phase Slope Index — which signal leads in phase | No (directed) | Yes |

> **TR** is the time between brain scans (in seconds). It is required by spectral methods that work in the frequency domain.

---

## Frequency bands

BRAPHIN provides predefined BOLD fMRI frequency bands. These are different from EEG bands and are based on Zuo et al. (2010).

```python
from braphin import FMRI_BANDS, compute_all_bands_connectivity

# FMRI_BANDS = {
#     "slow5":     (0.010, 0.027),   # Hz
#     "slow4":     (0.027, 0.073),
#     "slow3":     (0.073, 0.167),
#     "broadband": (0.010, 0.100),
# }

# Compute connectivity separately for each frequency band
band_results = compute_all_bands_connectivity(
    roi_time_series=tx_bundle.roi_time_series,
    tr=2.0,
    method="pearson_correlation",
)
# Returns a dict: {"slow5": ndarray, "slow4": ndarray, "slow3": ndarray, "broadband": ndarray}
```

---

## Motion confounds

Head movement during scanning introduces noise. After running motion correction, use `get_motion_confounds()` to get the movement parameters in the right format for denoising.

```python
from braphin import get_motion_confounds

confounds = get_motion_confounds(pp_bundle)   # shape (T, 6) — ready for confound regression
```

The six columns are translation and rotation parameters [tx, ty, tz, rx, ry, rz]. BRAPHIN applies the necessary sign convention automatically.

---

## Configuration reference

All pipeline stages are configured with simple dataclasses. The most common options are shown below.

```python
PreprocessConfig(
    apply_voxel_zscore      = True,    # normalise each voxel's signal over time (recommended)
    apply_motion_correction = False,   # rigid-body alignment to the first scan volume
    apply_smoothing         = False,   # spatial smoothing (Gaussian kernel)
    smoothing_fwhm          = 6.0,     # smoothing width in mm
    apply_slice_timing      = False,   # correct for the fact that slices are scanned at different times
    tr                      = None,    # time between scans in seconds (required for slice timing)
)

DenoiseConfig(
    regress_confounds = True,    # remove the influence of confound signals (e.g. head movement)
    apply_bandpass    = False,   # keep only signals in a frequency range
    bandpass_low      = 0.008,   # lower frequency cutoff in Hz
    bandpass_high     = 0.1,     # upper frequency cutoff in Hz (Biswal 1995 resting-state band)
    tr                = None,    # time between scans in seconds (required for bandpass)
    apply_scrubbing   = False,   # interpolate or remove scan volumes with excessive motion
)

AtlasConfig(
    atlas_name = "aal",          # bundled atlas: "aal", "schaefer_100", "schaefer_200", "schaefer_400"
    atlas_path = None,           # path to a custom NIfTI atlas file
    roi_labels = None,           # optional list of custom region names
)

ConnectivityConfig(
    method      = "pearson_correlation",   # any method key from the table above
    threshold   = None,                    # keep only edges where |weight| >= threshold
    window_size = None,                    # None = static connectivity across the whole scan
)
```

<details>
<summary>Less common options</summary>

```python
PreprocessConfig(
    slice_order             = "sequential",  # "sequential" or "interleaved"
    slice_timing_ref_slice  = 0,             # reference slice index for slice-timing correction
    slice_axis              = 2,             # which axis holds the slices (0=X, 1=Y, 2=Z)
    apply_outlier_detection = False,         # DVARS-based outlier detection
    outlier_threshold_dvars = 1.5,           # IQR multiplier for the DVARS threshold
    scrubbing_strategy      = "interpolate", # "interpolate" or "mark"
)
```

</details>

---

## Installation

### Core library

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

### From source

```bash
git clone https://github.com/<your-org>/braphin.git
cd braphin
pip install -e .
```

---

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest

# With coverage report
pytest --cov=braphin --cov-report=term-missing
```

---

## Project structure

```
braphin-main/
├── braphin/
│   ├── config.py              # Configuration dataclasses (PreprocessConfig, etc.)
│   ├── importBRAPHINData.py   # Stage 1: NIfTI loading
│   ├── preprocess.py          # Stage 2: preprocessing
│   ├── denoise.py             # Stage 3: denoising
│   ├── transform.py           # Stage 4: atlas parcellation
│   ├── connectivity.py        # Stage 5: connectivity
│   └── visualize.py           # build_fmri_graph, visualize_html, visualize_png
├── eegraph/                   # Unified EEG + fMRI API (EEGraph dependency)
├── tests/                     # Pytest test suite (459 tests)
├── Examples/                  # Jupyter notebook tutorials
└── pyproject.toml
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

- **No global signal regression (GSR).** A common preprocessing step in resting-state fMRI pipelines; not yet implemented.
- **Cross-correlation is slow for large atlases.** The cross-correlation methods iterate over all region pairs in Python. For Schaefer 400 this is ~80,000 iterations; a faster vectorised version is planned.
- **Motion correction convention.** `motion_params` stores world-space rigid-body parameters. Use `get_motion_confounds(bundle)` to obtain the sign-corrected version for regression.
- **Slice-timing uses linear interpolation.** Acceptable for TR ≥ 1 s; higher-order methods may be preferable for sub-second TRs.
- **No real-data validation.** The test suite uses synthetic data only. Validation against a public neuroimaging dataset is planned.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
