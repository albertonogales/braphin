# Experiments

Reproduces the results reported in the BRAPHIN paper (Sections 4–5).

## Setup

```bash
pip install braphin torch torch-geometric scikit-learn numpy scipy
```

## Data layout

```
data/
  matrices/
    controls/   # .npy files, shape (116,116), one per subject
    patients/   # .npy files, shape (116,116), one per subject
  splits/
    neurocon_controls.txt   # filenames of Neurocon control subjects
    neurocon_patients.txt   # filenames of Neurocon patient subjects
```

Each `.npy` file is a Pearson correlation connectivity matrix produced by
BRAPHIN (AAL atlas, 116 ROIs):

```python
from braphin import Graph

g = Graph()
g.load_data("subject.nii.gz", modality="fmri")
G, matrix = g.modelate(window_size=None, connectivity="pearson_correlation")

import numpy as np
np.save("subject.npy", matrix)
```

## Experiment 1 — GCN classification (Tables 1 & 2)

Trains a 3-layer GCN on BRAPHIN-generated graphs, reports train/validation/test
metrics (Table 1) and 5-fold cross-validation (Table 2).

```bash
python experiments/gnn_parkinson.py \
    --data_dir data/matrices \
    --splits_dir data/splits \
    --seed 64 --epochs 100 --patience 10
```

Key hyperparameters (from grid search during development):

| Parameter | Value |
|---|---|
| K (edges per node) | 115 |
| Learning rate | 0.0005 |
| Weight decay | 0.001 |
| Dropout | 0.5 |
| Batch size | 64 |
| Random seed | 64 |

## Experiment 2 — Graph-theoretical analysis (Table 3)

Computes 12 graph-theoretical metrics on the Neurocon external test set,
compares patients vs. controls with Mann-Whitney U + Benjamini-Hochberg FDR.

```bash
python experiments/graph_metrics_analysis.py \
    --data_dir data/matrices \
    --splits_dir data/splits \
    --threshold 0.5
```

## Expected results

### Table 1 — Classification performance

| Metric | Train | Validation | Test (Neurocon) |
|---|---|---|---|
| Accuracy | 95.28% | 87.50% | 76.00% |
| Sensitivity | 100.00% | 87.50% | 84.00% |
| Specificity | — | — | 68.00% |
| F1-score | 95.52% | 87.50% | 77.78% |
| AUC | 99.80% | 87.50% | 76.48% |

### Table 2 — 5-fold cross-validation

| Metric | Mean ± Std |
|---|---|
| Accuracy | 88.92% ± 6.66% |
| AUC | 95.94% ± 2.33% |

### Table 3 — Graph-theoretical metrics (Neurocon test set, n=50)

| Metric | Controls | Patients | Δ% | p | p_adj |
|---|---|---|---|---|---|
| Number of edges | 2110.00 | 1314.60 | −37.70% | 0.0017 | 0.0039 |
| Network density | 0.316 | 0.197 | — | 0.0017 | 0.0039 |
| Mean degree | 36.38 | 22.67 | — | 0.0017 | 0.0039 |
| Mean weighted strength | 24.33 | 14.89 | −38.79% | 0.0023 | 0.0039 |
| Local efficiency | — | — | −10.91% | 0.0016 | 0.0039 |
| Weighted clustering coeff. | — | — | −15.09% | 0.0022 | 0.0039 |
| Global efficiency | 0.630 | 0.557 | −11.61% | 0.0070 | 0.0105 |
| Mean path length | — | — | +14.13% | 0.0079 | 0.0105 |
| Modularity | 0.231 | 0.316 | +36.71% | 0.0145 | 0.0174 |
| Number of communities | 9.92 | 16.08 | +62.10% | 0.0205 | 0.0224 |
| Rich-club coefficient | 0.692 | 0.603 | −12.85% | 0.0283 | 0.0283 |
| Small-world sigma | 2.116 | 2.767 | +30.80% | 0.0006 | 0.0039 |
