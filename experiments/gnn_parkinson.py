"""
experiments/gnn_parkinson.py
============================

GCN experiment for Parkinson's disease detection using BRAPHIN-generated
functional connectivity graphs.

Reproduces the results reported in Section 5.1 of the paper:
  - Single train/validation/test split (Table 1)
  - 5-fold cross-validation on internal data (Table 2)

Dataset layout expected in data/matrices/:
    data/matrices/controls/   -- .npy files, shape (116, 116), one per subject
    data/matrices/patients/   -- .npy files, shape (116, 116), one per subject
    data/splits/neurocon_controls.txt  -- filenames reserved for external test
    data/splits/neurocon_patients.txt  -- filenames reserved for external test

Each .npy file is a Pearson correlation connectivity matrix produced by
BRAPHIN (AAL atlas, 116 ROIs, threshold=0.5).

Usage
-----
    python experiments/gnn_parkinson.py --data_dir data/matrices \
        --splits_dir data/splits --seed 64 --epochs 100 --patience 10

Requirements
------------
    torch>=2.0, torch-geometric>=2.4, scikit-learn>=1.3, numpy, scipy
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.nn import Linear
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_matrix(path: str) -> np.ndarray:
    """Load a connectivity matrix from a .npy file."""
    m = np.load(path).astype(np.float32)
    assert m.shape == (116, 116), f"Expected 116×116, got {m.shape}: {path}"
    return m


def matrix_to_graph(matrix: np.ndarray, label: int, k: int = 115) -> Data:
    """
    Convert a 116×116 Pearson connectivity matrix to a PyTorch Geometric graph.

    Node features: the full 116-dimensional connectivity row for each ROI.
    Edges: the K strongest positive connections per node (K=115 by default,
    as determined by grid search during model development).
    """
    n = matrix.shape[0]

    # Node features: each ROI's full connectivity profile
    x = torch.tensor(matrix, dtype=torch.float)  # (116, 116)

    # Build edges: for each node retain the K strongest positive connections
    edge_index_list = []
    for i in range(n):
        row = matrix[i].copy()
        row[i] = -np.inf  # exclude self-loop
        row[row < 0] = -np.inf  # exclude negative correlations
        top_k = np.argsort(row)[-k:]
        for j in top_k:
            if row[j] > -np.inf:
                edge_index_list.append([i, j])

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))


def load_dataset(data_dir: str, splits_dir: str, k: int = 115):
    """
    Load all subjects, split into (train_val, test) based on Neurocon files.

    Returns:
        train_val_data : list[Data]
        train_val_labels : list[int]
        test_data : list[Data]
        test_labels : list[int]
    """
    neurocon_files = set()
    for fname in ("neurocon_controls.txt", "neurocon_patients.txt"):
        fpath = os.path.join(splits_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                neurocon_files.update(line.strip() for line in f if line.strip())

    all_data, all_labels = [], []
    test_data, test_labels = [], []

    for label, subdir in ((0, "controls"), (1, "patients")):
        folder = os.path.join(data_dir, subdir)
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".npy"):
                continue
            matrix = load_matrix(os.path.join(folder, fname))
            graph = matrix_to_graph(matrix, label, k=k)
            if fname in neurocon_files:
                test_data.append(graph)
                test_labels.append(label)
            else:
                all_data.append(graph)
                all_labels.append(label)

    return all_data, all_labels, test_data, test_labels


# ---------------------------------------------------------------------------
# GCN architecture
# ---------------------------------------------------------------------------

class ParkinsonGCN(torch.nn.Module):
    """
    3-layer GCN for binary Parkinson's disease classification.

    Architecture:
        GCNConv(116 → 64) → ReLU → Dropout
        GCNConv(64  → 64) → ReLU → Dropout
        GCNConv(64  → 32) → ReLU → Dropout
        Global mean + max pooling → concat (32+32=64)
        Linear(64 → 1)   [BCEWithLogitsLoss; no sigmoid here]
    """

    def __init__(self, in_channels: int = 116, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 32)
        self.fc = Linear(64, 1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level readout: mean + max pooling concatenated
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        return self.fc(x).squeeze(-1)  # (batch_size,)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def compute_class_weight(labels):
    """BCEWithLogitsLoss positive-class weight from training label ratio."""
    n_neg = labels.count(0)
    n_pos = labels.count(1)
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (accuracy, f1, auc, loss, y_true, y_pred, y_prob)."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        loss = criterion(logits, batch.y.float())
        total_loss += loss.item() * batch.num_graphs
        y_true.extend(batch.y.cpu().numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    try:
        auc = roc_auc_score(y_true, y_prob) * 100
    except ValueError:
        auc = float("nan")
    avg_loss = total_loss / len(loader.dataset)

    return acc, f1, auc, avg_loss, y_true, y_pred, y_prob


def print_metrics(split: str, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    try:
        auc = roc_auc_score(y_true, y_prob) * 100
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / max(tn + fp, 1) * 100
    bal_acc = (rec + specificity) / 2

    print(f"\n{'='*50}")
    print(f"  {split} results")
    print(f"{'='*50}")
    print(f"  Accuracy       : {acc:.2f}%")
    print(f"  Precision      : {prec:.2f}%")
    print(f"  Sensitivity    : {rec:.2f}%")
    print(f"  Specificity    : {specificity:.2f}%")
    print(f"  F1-score       : {f1:.2f}%")
    print(f"  AUC            : {auc:.2f}%")
    print(f"  Balanced Acc   : {bal_acc:.2f}%")
    print(f"  Confusion matrix (TN={tn}, FP={fp}, FN={fn}, TP={tp})")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_single_split(
    train_data, train_labels, val_data, val_labels, test_data, test_labels,
    args, device
):
    """Train on the fixed 80/20 split and evaluate on external Neurocon test."""
    pos_weight = compute_class_weight(train_labels).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    model = ParkinsonGCN(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    best_state = None
    patience_count = 0

    print("\nTraining (single split)...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_acc, train_f1, train_auc, _, _, _, _ = evaluate(model, train_loader, device)
        val_acc, val_f1, val_auc, val_loss, _, _, _ = evaluate(model, val_loader, device)

        # Early stopping criterion from paper:
        # train_acc > 80%, gap < 10%, then maximise val_acc, tiebreak by F1 then AUC
        gap = train_acc - val_acc
        is_candidate = train_acc > 80.0 and gap <= 10.0
        improved = is_candidate and (
            val_acc > best_val_acc
            or (val_acc == best_val_acc and val_f1 > getattr(run_single_split, "_best_f1", 0))
        )

        if improved:
            best_val_acc = val_acc
            run_single_split._best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0:
            print(
                f"  Epoch {epoch:3d} | train {train_acc:.1f}% | "
                f"val {val_acc:.1f}% | val_loss {val_loss:.4f}"
            )

        if patience_count >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    _, _, _, _, tr_true, tr_pred, tr_prob = evaluate(model, train_loader, device)
    _, _, _, _, vl_true, vl_pred, vl_prob = evaluate(model, val_loader, device)
    _, _, _, _, te_true, te_pred, te_prob = evaluate(model, test_loader, device)

    print_metrics("TRAINING", tr_true, tr_pred, tr_prob)
    print_metrics("VALIDATION", vl_true, vl_pred, vl_prob)
    print_metrics("TEST (Neurocon)", te_true, te_pred, te_prob)

    return model


def run_cross_validation(train_val_data, train_val_labels, test_data, test_labels, args, device):
    """5-fold CV on internal data; Neurocon test set excluded from all folds."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    labels_array = np.array(train_val_labels)
    data_array = np.array(train_val_data, dtype=object)

    fold_val_accs, fold_val_aucs = [], []

    print("\n5-fold cross-validation (Neurocon excluded from all folds)...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(data_array, labels_array), 1):
        fold_train = data_array[train_idx].tolist()
        fold_val = data_array[val_idx].tolist()
        fold_train_labels = labels_array[train_idx].tolist()

        pos_weight = compute_class_weight(fold_train_labels).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_loader = DataLoader(fold_train, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(fold_val, batch_size=args.batch_size)

        model = ParkinsonGCN(dropout=args.dropout).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        best_val_acc, best_state, patience_count = 0.0, None, 0
        for epoch in range(1, args.epochs + 1):
            train_epoch(model, train_loader, optimizer, criterion, device)
            train_acc, train_f1, _, _, _, _, _ = evaluate(model, train_loader, device)
            val_acc, val_f1, val_auc, _, _, _, _ = evaluate(model, val_loader, device)

            gap = train_acc - val_acc
            if train_acc > 80.0 and gap <= 10.0 and val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= args.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        _, _, final_val_auc, _, _, _, _ = evaluate(model, val_loader, device)
        fold_val_accs.append(best_val_acc)
        fold_val_aucs.append(final_val_auc)
        print(f"  Fold {fold}: val_acc={best_val_acc:.2f}%  val_auc={final_val_auc:.2f}%")

    mean_acc = np.mean(fold_val_accs)
    std_acc = np.std(fold_val_accs)
    mean_auc = np.mean(fold_val_aucs)
    std_auc = np.std(fold_val_aucs)

    print(f"\n  CV summary: val_acc = {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"              val_auc = {mean_auc:.2f}% ± {std_auc:.2f}%")

    return mean_acc, std_acc, mean_auc, std_auc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GCN Parkinson's disease detection via BRAPHIN")
    p.add_argument("--data_dir", default="data/matrices",
                   help="Directory with controls/ and patients/ subdirs of .npy matrices")
    p.add_argument("--splits_dir", default="data/splits",
                   help="Directory with neurocon_controls.txt and neurocon_patients.txt")
    p.add_argument("--seed", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--weight_decay", type=float, default=0.001)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--k", type=int, default=115,
                   help="Top-K edges per node (K=115 from grid search)")
    p.add_argument("--val_ratio", type=float, default=0.2,
                   help="Fraction of non-Neurocon data held out for validation")
    p.add_argument("--no_cv", action="store_true", help="Skip cross-validation")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading graphs...")
    train_val_data, train_val_labels, test_data, test_labels = load_dataset(
        args.data_dir, args.splits_dir, k=args.k
    )
    print(f"  Train+val: {len(train_val_data)} graphs | Test (Neurocon): {len(test_data)}")

    # Fixed 80/20 split (reproducible with seed)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(train_val_data))
    n_val = int(len(train_val_data) * args.val_ratio)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_data = [train_val_data[i] for i in train_idx]
    train_labels = [train_val_labels[i] for i in train_idx]
    val_data = [train_val_data[i] for i in val_idx]
    val_labels = [train_val_labels[i] for i in val_idx]

    print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # Single split experiment (Table 1)
    run_single_split(
        train_data, train_labels,
        val_data, val_labels,
        test_data, test_labels,
        args, device,
    )

    # 5-fold CV (Table 2)
    if not args.no_cv:
        run_cross_validation(train_val_data, train_val_labels, test_data, test_labels, args, device)


if __name__ == "__main__":
    main()
