"""The shared head evaluation used by every suite.

``evaluate_heads`` is given one representation (features or HDC codes) for
base-train / base-test / novel-train / novel-test and compares:

  * ``hdc{4096,10000}`` : random-projection + sign class prototypes (old-repo HDC)
  * ``linear``          : trained linear softmax layer
  * ``mlp``             : trained 2-hidden-layer softmax layer
  * ``proto``           : class-mean prototype in the raw feature space
  * optional ``net``    : a network classifier already trained on the same
                          features (used by the pretraining-scaling suite)

Metrics per run:
  * ID accuracy (+ accuracy on robustness variants supplied by the caller)
  * novel-class clustering accuracy (k-means with the oracle class count, then
    Hungarian matching) in feature space and in each HDC code space
  * OOD-detection AUROC (known test vs novel test) for every head
  * 5-shot novel accuracy for prototypes/HDC and a linear layer trained on the
    same 5 examples per class
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .heads import (
    CEConfig,
    FeatureProtoHead,
    HDCHead,
    model_logits,
    model_softmax_novelty,
    train_ce,
)
from .metrics import auroc, hungarian_accuracy, spherical_kmeans
from .metrics import top1 as _top1


def _acc_ids(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred == y).mean()) if len(y) else float("nan")


def _sample_fewshot(X, y, m: int, seed: int):
    rng = np.random.default_rng([seed, 555])
    picks = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        picks.append(idx if len(idx) <= m else rng.choice(idx, size=m, replace=False))
    idx = np.concatenate(picks)
    return X[idx], y[idx]


def _logits_on(model, X, device) -> np.ndarray:
    return model_logits(model, X, device)


def evaluate_heads(*, Xtr, ytr, Xte, yte, n_classes, fdim, seed, device,
                   Xntr=None, yntr=None, Xnte=None, ynte=None, n_novel=0,
                   robust: dict | None = None, hdc_dims=(4096, 10000),
                   identity_hdc: bool = False, extra_heads: dict | None = None,
                   fewshot_m: int = 5) -> dict:
    """Fit all heads on (Xtr, ytr) and evaluate.  Returns a nested metric dict."""
    robust = robust or {}
    extra_heads = extra_heads or {}
    t0 = time.time()
    out: dict = {"train_size": int(len(ytr)), "test_size": int(len(yte))}

    # ---- fit heads -------------------------------------------------------
    hdc_heads: dict[int, HDCHead] = {}
    for dim in hdc_dims:
        t = time.time()
        head = HDCHead(fdim, dim, seed=seed * 10007 + dim, device=device,
                       identity=identity_hdc)
        head.fit(Xtr, ytr, n_classes)
        hdc_heads[dim] = head
        out[f"fit_time_hdc{dim}"] = round(time.time() - t, 3)

    t = time.time()
    proto = FeatureProtoHead(device).fit(Xtr, ytr, n_classes)
    out["fit_time_proto"] = round(time.time() - t, 3)
    t = time.time()
    linear = train_ce(Xtr, ytr, n_classes, device, seed, CEConfig(epochs=50))
    out["fit_time_linear"] = round(time.time() - t, 3)
    t = time.time()
    mlp = train_ce(Xtr, ytr, n_classes, device, seed, CEConfig(epochs=50, lr=8e-4), mlp=True)
    out["fit_time_mlp"] = round(time.time() - t, 3)

    # ---- ID accuracy -----------------------------------------------------
    id_metrics = {
        "linear": _top1(_logits_on(linear, Xte, device), yte),
        "mlp": _top1(_logits_on(mlp, Xte, device), yte),
        "proto": _acc_ids(proto.predict(Xte), yte),
    }
    for dim, head in hdc_heads.items():
        id_metrics[f"hdc{dim}"] = _acc_ids(head.predict(Xte), yte)
    for name, logits_fn in extra_heads.items():
        id_metrics[name] = _top1(logits_fn(Xte), yte)
    out["id"] = {k: round(v, 4) for k, v in id_metrics.items()}

    # ---- robustness variants (feature-space perturbations) ---------------
    if robust:
        rob = {}
        for label, (Xr, yr) in robust.items():
            m = {
                "linear": _top1(_logits_on(linear, Xr, device), yr),
                "mlp": _top1(_logits_on(mlp, Xr, device), yr),
                "proto": _acc_ids(proto.predict(Xr), yr),
            }
            for dim, head in hdc_heads.items():
                m[f"hdc{dim}"] = _acc_ids(head.predict(Xr), yr)
            for name, logits_fn in extra_heads.items():
                m[name] = _top1(logits_fn(Xr), yr)
            rob[label] = {k: round(v, 4) for k, v in m.items()}
        out["robust"] = rob

    # ---- novel-class metrics ---------------------------------------------
    if n_novel and nte_ok(ynte):
        novel: dict = {}
        spaces = {"feat": None}
        for dim in hdc_dims:
            spaces[f"hdc{dim}"] = hdc_heads[dim]
        for sname, head in spaces.items():
            t = time.time()
            codes = (head.encode(Xnte) if head is not None
                     else torch.as_tensor(np.asarray(Xnte), dtype=torch.float32, device=device))
            pred = spherical_kmeans(codes.cpu().numpy(), n_novel, seed, device=device)
            novel[f"cluster_{sname}"] = round(hungarian_accuracy(ynte, pred, n_novel), 4)
            novel[f"cluster_time_{sname}"] = round(time.time() - t, 2)
        # OOD detection: known (base test) vs novel (novel test)
        for dim, head in hdc_heads.items():
            novel[f"auroc_hdc{dim}"] = round(auroc(head.novelty(Xnte), head.novelty(Xte)), 4)
        novel["auroc_proto"] = round(auroc(proto.novelty(Xnte), proto.novelty(Xte)), 4)
        novel["auroc_linear"] = round(
            auroc(model_softmax_novelty(linear, Xnte, device),
                  model_softmax_novelty(linear, Xte, device)), 4)
        novel["auroc_mlp"] = round(
            auroc(model_softmax_novelty(mlp, Xnte, device),
                  model_softmax_novelty(mlp, Xte, device)), 4)
        for name, logits_fn in extra_heads.items():
            p_nov = torch.softmax(torch.from_numpy(logits_fn(Xnte)), 1).numpy()
            p_kn = torch.softmax(torch.from_numpy(logits_fn(Xte)), 1).numpy()
            novel[f"auroc_{name}"] = round(auroc(1 - p_nov.max(1), 1 - p_kn.max(1)), 4)

        # few-shot novel prototypes / linear probe
        if Xntr is not None and len(yntr) >= n_novel:
            Xs, ys = _sample_fewshot(Xntr, yntr, fewshot_m, seed)
            if len(ys):
                m = {}
                for dim in hdc_dims:
                    h = HDCHead(fdim, dim, seed=seed * 10007 + dim, device=device,
                                identity=identity_hdc)
                    h.fit(Xs, ys, n_novel)
                    m[f"hdc{dim}"] = _acc_ids(h.predict(Xnte), ynte)
                m["proto"] = _acc_ids(
                    FeatureProtoHead(device).fit(Xs, ys, n_novel).predict(Xnte), ynte)
                try:
                    lin_fs = train_ce(Xs, ys, n_novel, device, seed,
                                      CEConfig(epochs=120, lr=1e-3,
                                               batch=min(256, max(len(ys), 8))))
                    m["linear"] = _top1(_logits_on(lin_fs, Xnte, device), ynte)
                except Exception as e:  # pragma: no cover
                    m["linear"] = float("nan")
                    m["linear_error"] = str(e)[:80]
                novel[f"fewshot{fewshot_m}"] = {k: round(v, 4) for k, v in m.items()}
        out["novel"] = novel

    out["eval_time"] = round(time.time() - t0, 2)
    return out


def nte_ok(ynte) -> bool:
    return ynte is not None and len(ynte) > 0
