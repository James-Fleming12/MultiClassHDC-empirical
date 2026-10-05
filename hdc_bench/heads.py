"""Classifier heads compared in the suite.

* ``HDCHead``: the old repo's HDC prototype classifier -- fixed random Gaussian
  projection, sign (bipolar code), L2-normalized class-mean prototype, score =
  cosine similarity (equivalently normalized Hamming distance ``(1-cos)/2``).
  The projection is random and never trained; only class means are built.
  For the synthetic ``bits`` dataset the input *is* the code, so the projection
  is the identity (``identity=True``).
* ``FeatureProtoHead``: class-mean prototype in the raw feature space, same
  cosine scoring.  This is the "prototype but no HDC" control.
* ``LinearHead`` / ``MLPHead``: a standard network classification layer trained
  with softmax cross-entropy on exactly the same frozen features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tensor(X, device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(X, torch.Tensor):
        return X.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(X), dtype=dtype, device=device)


def _inference_batches(n: int, batch: int = 8192):
    for i in range(0, n, batch):
        yield i, min(i + batch, n)


class HDCHead:
    """Random-projection + sign prototype classifier (the old-repo convention)."""

    def __init__(self, fdim: int, hd_dim: int, seed: int, device, identity: bool = False):
        self.fdim = fdim
        self.hd_dim = hd_dim
        self.device = device
        self.identity = bool(identity or fdim == hd_dim)
        if not self.identity:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            self.proj = torch.randn(fdim, hd_dim, generator=gen).to(device)
        self.prototypes: torch.Tensor | None = None

    def encode(self, X) -> torch.Tensor:
        """Features -> L2-normalized bipolar codes, shape (n, hd_dim)."""
        X = _as_tensor(X, self.device)
        out = torch.empty(X.shape[0], self.hd_dim, device=self.device)
        with torch.no_grad():
            for i, j in _inference_batches(X.shape[0]):
                xb = X[i:j]
                code = xb if self.identity else xb @ self.proj
                out[i:j] = F.normalize(code.sign().float(), p=2, dim=1)
        return out

    def fit(self, X, y, n_classes: int) -> "HDCHead":
        codes = self.encode(X)
        y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device)
        protos = torch.zeros(n_classes, self.hd_dim, device=self.device)
        protos.index_add_(0, y, codes)
        counts = torch.zeros(n_classes, device=self.device)
        counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float32))
        self.prototypes = F.normalize(protos / counts.clamp_min(1).unsqueeze(1), p=2, dim=1)
        return self

    @torch.no_grad()
    def similarities(self, X) -> torch.Tensor:
        assert self.prototypes is not None, "fit() first"
        codes = self.encode(X)
        return codes @ self.prototypes.t()

    def predict(self, X) -> np.ndarray:
        return self.similarities(X).argmax(dim=1).cpu().numpy()

    def novelty(self, X) -> np.ndarray:
        """1 - max cosine similarity to the known prototypes (higher = novel)."""
        return (1.0 - self.similarities(X).max(dim=1).values).cpu().numpy()


class FeatureProtoHead:
    """Class-mean prototype in the raw (L2-normalized) feature space."""

    def __init__(self, device):
        self.device = device
        self.prototypes: torch.Tensor | None = None

    def fit(self, X, y, n_classes: int) -> "FeatureProtoHead":
        X = F.normalize(_as_tensor(X, self.device), p=2, dim=1)
        y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device)
        protos = torch.zeros(n_classes, X.shape[1], device=self.device)
        protos.index_add_(0, y, X)
        counts = torch.zeros(n_classes, device=self.device)
        counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float32))
        self.prototypes = F.normalize(protos / counts.clamp_min(1).unsqueeze(1), p=2, dim=1)
        return self

    @torch.no_grad()
    def similarities(self, X) -> torch.Tensor:
        assert self.prototypes is not None, "fit() first"
        X = F.normalize(_as_tensor(X, self.device), p=2, dim=1)
        return X @ self.prototypes.t()

    def predict(self, X) -> np.ndarray:
        return self.similarities(X).argmax(dim=1).cpu().numpy()

    def novelty(self, X) -> np.ndarray:
        return (1.0 - self.similarities(X).max(dim=1).values).cpu().numpy()


class _MLP(nn.Module):
    def __init__(self, fdim: int, n_classes: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fdim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class CEConfig:
    epochs: int = 50
    batch: int = 512
    lr: float = 1e-3
    wd: float = 1e-4
    warmup: float = 0.05
    hidden: int = 1024


def train_ce(features, labels, n_classes: int, device, seed: int,
             cfg: CEConfig | None = None, mlp: bool = False) -> nn.Module:
    """Train a linear (or 2-hidden-layer MLP) softmax classification layer."""
    cfg = cfg or CEConfig()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    X = F.normalize(_as_tensor(features, device), p=2, dim=1)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
    model = (_MLP(X.shape[1], n_classes, cfg.hidden) if mlp else
             nn.Linear(X.shape[1], n_classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    n = X.shape[0]
    steps_per_epoch = max(1, int(np.ceil(n / cfg.batch)))
    total = cfg.epochs * steps_per_epoch
    warm = max(1, int(cfg.warmup * total))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=total, pct_start=warm / total,
        anneal_strategy="cos")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model


@torch.no_grad()
def model_logits(model: nn.Module, features, device) -> np.ndarray:
    X = F.normalize(_as_tensor(features, device), p=2, dim=1)
    out = []
    for i, j in _inference_batches(X.shape[0]):
        out.append(model(X[i:j]).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def model_softmax_novelty(model: nn.Module, features, device) -> np.ndarray:
    logits = model_logits(model, features, device)
    p = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return (1.0 - p.max(axis=1)).astype(np.float64)
