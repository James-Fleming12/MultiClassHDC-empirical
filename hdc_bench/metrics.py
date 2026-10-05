"""Evaluation metrics: top-1 accuracy, Hungarian cluster accuracy, AUROC.

The clustering metric follows the NCD convention used in the old repo: cluster
the *unlabelled* novel-class features, then Hungarian-match the discovered
clusters to the true classes and report the matched top-1 accuracy.  Clustering
is spherical k-means on L2-normalized vectors, implemented in torch so it runs
on the GPU (the sklearn fallback is used only if torch is unavailable).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None


def top1(logits: np.ndarray, y: np.ndarray) -> float:
    """Plain top-1 accuracy."""
    return float((logits.argmax(axis=1) == y).mean())


def hungarian_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                       n_clusters: int | None = None) -> float:
    """Cluster accuracy: one-to-one matching of clusters to classes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if n_clusters is None:
        n_clusters = int(max(y_pred.max(), y_true.max())) + 1
    n_classes = int(y_true.max()) + 1
    cost = np.zeros((n_clusters, n_classes), dtype=np.int64)
    np.add.at(cost, (y_pred, y_true), 1)
    rows, cols = linear_sum_assignment(-cost)
    return float(cost[rows, cols].sum() / len(y_true))


def auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """AUROC with ``scores_pos`` = positive (novel) class, ties averaged."""
    scores = np.concatenate([scores_pos, scores_neg])
    labels = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    u = ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def spherical_kmeans(X, n_clusters: int, seed: int, device=None) -> np.ndarray:
    """Spherical k-means on L2-normalized rows; returns hard cluster ids.

    ``device`` selects the torch implementation (GPU preferred); when torch is
    missing the sklearn KMeans fallback runs on normalized vectors.
    """
    if torch is not None and device is not None:
        return _torch_spherical_kmeans(X, n_clusters, seed, device)
    from sklearn.cluster import KMeans

    X = np.asarray(X, dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    km = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=seed)
    return km.fit_predict(X)


def _kmeanspp(X, n_clusters: int, gen):
    n = X.shape[0]
    first = torch.randint(0, n, (1,), generator=gen, device=X.device)
    centers = [X[first]]
    d2 = 1.0 - (X @ centers[0].t()).view(-1)
    for _ in range(1, n_clusters):
        probs = d2.clamp_min(0.0)
        if float(probs.sum()) <= 0:
            probs = torch.ones_like(probs)
        idx = torch.multinomial(probs, 1, generator=gen)
        c = X[idx]
        centers.append(c)
        d2 = torch.minimum(d2, 1.0 - (X @ c.t()).view(-1))
    return torch.cat(centers, dim=0)


def _torch_spherical_kmeans(X, n_clusters: int, seed: int, device,
                            n_init: int = 3, max_iter: int = 100) -> np.ndarray:
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    if X.ndim == 1:
        X = X.view(-1, 1)
    X = F.normalize(X, p=2, dim=1)
    n = X.shape[0]
    if n_clusters >= n:
        return np.arange(n)
    torch.backends.cuda.matmul.allow_tf32 = True
    gen = torch.Generator(device=X.device if X.is_cuda else "cpu").manual_seed(int(seed))
    best_labels, best_score = None, -float("inf")

    for _ in range(n_init):
        centers = _kmeanspp(X, n_clusters, gen)
        labels = None
        for _ in range(max_iter):
            new_labels = torch.empty(n, dtype=torch.long, device=X.device)
            for i in range(0, n, 4096):
                new_labels[i:i + 4096] = (X[i:i + 4096] @ centers.t()).argmax(1)
            if labels is not None and torch.equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(n_clusters, device=X.device)
            new_centers.index_add_(0, labels, X)
            counts.index_add_(0, labels, torch.ones(n, device=X.device))
            empty = counts == 0
            if empty.any():
                ridx = torch.randint(0, n, (int(empty.sum()),), generator=gen,
                                     device=X.device)
                new_centers[empty] = X[ridx]
                counts[empty] = 1
            centers = F.normalize(new_centers / counts.unsqueeze(1), p=2, dim=1)
        score = float((X * centers[labels]).sum())
        if score > best_score:
            best_score, best_labels = score, labels
    return best_labels.cpu().numpy()
