"""Dataset loaders and graph construction.

Splits are redrawn for every seed; the feature scaler is fit on training
accounts only.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _scale(x_raw, train_idx):
    scaler = StandardScaler().fit(x_raw[train_idx])
    return torch.from_numpy(np.clip(scaler.transform(x_raw), -3, 3).astype(np.float32))


def _stratified_70_15_15(labels, seed):
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for c in (1, 0):
        idx = np.where(labels == c)[0].copy()
        rng.shuffle(idx)
        n_tr, n_va = int(0.70 * len(idx)), int(0.15 * len(idx))
        parts["train"].append(idx[:n_tr])
        parts["val"].append(idx[n_tr:n_tr + n_va])
        parts["test"].append(idx[n_tr + n_va:])
    return {k: np.concatenate(v) for k, v in parts.items()}


def load_tabular(data_dir, seed=0):
    """Fox8-23 / BotSim-24 after preprocessing (features.csv, labels.csv)."""
    feats = pd.read_csv(os.path.join(data_dir, "features.csv"))
    feats = feats.drop(columns=[c for c in ("user_id",) if c in feats])
    x_raw = feats.fillna(0.0).values.astype(np.float32)
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))["label"].values.astype(np.int64)
    split = _stratified_70_15_15(labels, seed)
    return _scale(x_raw, split["train"]), labels, split


def load_mgtab(data_dir, seed=0, degree_features=False):
    """MGTAB (edge_index.pt, features.pt, labels_bot.pt[, edge_type.pt])."""
    load = lambda f: torch.load(os.path.join(data_dir, f), map_location="cpu", weights_only=True)
    edge_index = load("edge_index.pt").long().contiguous()
    x_raw = load("features.pt").float()
    labels = load("labels_bot.pt").long().numpy()
    n = x_raw.shape[0]
    train, rest = train_test_split(np.arange(n), test_size=0.30, stratify=labels, random_state=seed)
    val, test = train_test_split(rest, test_size=1 / 3, stratify=labels[rest], random_state=seed)
    if degree_features:
        # Label-free popularity coordinates: log in/out-degree per relation type.
        etype = load("edge_type.pt").long()
        R = int(etype.max()) + 1
        deg = torch.zeros(n, 2 * R)
        for r in range(R):
            e = edge_index[:, etype == r]
            deg[:, 2 * r] += torch.bincount(e[1], minlength=n).float()
            deg[:, 2 * r + 1] += torch.bincount(e[0], minlength=n).float()
        x_raw = torch.cat([x_raw, torch.log1p(deg)], dim=1)
    split = dict(train=train, val=val, test=test)
    return _scale(x_raw.numpy(), train), edge_index, labels, split


@torch.no_grad()
def knn_graph(x, k=10):
    """Directed cosine k-NN edges (row: account, column: its neighbors)."""
    x = F.normalize(x.float(), dim=-1)
    n = x.shape[0]
    chunk = max(1, min(512, n // 4))
    rows, cols = [], []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sim = x[s:e] @ x.T
        sim[:, s:e].fill_diagonal_(-2.0)
        nbr = sim.topk(k, dim=-1).indices
        rows.append(torch.arange(s, e, device=x.device).unsqueeze(1).expand_as(nbr).reshape(-1))
        cols.append(nbr.reshape(-1))
    return torch.stack([torch.cat(rows), torch.cat(cols)])


def mean_adjacency(edge_index, n):
    """Sparse CSR mean-aggregation matrix (rows: targets), keeping repeated edges."""
    flipped = edge_index.flip(0)
    counts = torch.sparse_coo_tensor(flipped, torch.ones(flipped.shape[1]), (n, n)).coalesce()
    row = counts.indices()[0]
    deg = torch.zeros(n).index_add_(0, row, counts.values())
    vals = counts.values() / deg[row].clamp_min(1)
    return torch.sparse_coo_tensor(counts.indices(), vals, counts.shape).to_sparse_csr()
