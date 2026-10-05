#!/usr/bin/env python
"""Run the HDC configuration ablation (projections / quantization / prototypes).

For each (dataset, backbone, seed, K) with N novel classes it evaluates:
  * the standard heads (linear, MLP, feature prototype) for reference,
  * every VariantSpec at code length 4096 and 10000 (matched bit budget),
  * the projection-draw variance of the baseline Gaussian head.

Records go to ``results/raw/variants/``; the tables are built by
``scripts/make_report.py`` (suite ``variants``).

Usage:
  python scripts/run_variants.py --cases all
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hdc_bench.features import has_perturb, load_features
from hdc_bench.hdc_variants import VARIANTS, HDCVariantHead, VariantSpec
from hdc_bench.heads import CEConfig, FeatureProtoHead, model_logits, model_softmax_novelty, train_ce
from hdc_bench.metrics import auroc, hungarian_accuracy, spherical_kmeans
from hdc_bench.paths import IMAGE_DATASETS
from hdc_bench.records import has_record, save_record
from hdc_bench.synthetic import build_synthetic

CODE_LENS = (4096, 10000)
N_NOVEL = 20

CASES = {
    "cifar_dino": ("cifar100", "dinov2_vitb14_reg", [5, 50, 100], [0, 1, 2]),
    "cifar_resnet": ("cifar100", "resnet18", [50, 100], [0, 1, 2]),
    "tiny_dino": ("tinyimagenet", "dinov2_vitb14_reg", [10, 100, 200], [0, 1, 2]),
    "gmm": ("gmm", "", [5, 50, 200], [0, 1]),
}


def _slice(X, y, classes, remap):
    mask = np.isin(y, classes)
    return X[mask], np.array([remap[c] for c in y[mask]], dtype=np.int64)


def _cap(dataset: str) -> int:
    return {"cifar100": 100, "tinyimagenet": 200}.get(dataset, 10 ** 6)


def load_case(dataset, backbone, seed, K):
    """Return (Xtr,ytr,Xte,yte,Xntr,yntr,Xnte,ynte,fdim,robust)."""
    N = min(N_NOVEL, _cap(dataset) - K)
    if dataset == "gmm":
        d = build_synthetic("gmm", seed, K, N)
        robust = {label: (Xr, np.arange(K).repeat(len(Xr) // max(K, 1)))
                  for label, Xr in d["robust"].items()}
        return (d["Xtr"], d["ytr"], d["Xte"], d["yte"], d["Xntr"], d["yntr"],
                d["Xnte"], d["ynte"], d["fdim"], robust)
    n_total = IMAGE_DATASETS[dataset]["n_classes"]
    clean = load_features(dataset, backbone, "clean")
    Xtr_all, ytr_all = clean["X_train"].numpy(), clean["y_train"].numpy()
    Xte_all, yte_all = clean["X_test"].numpy(), clean["y_test"].numpy()
    perm = np.random.default_rng(seed).permutation(n_total)
    base, novel = perm[:K], perm[K:K + N]
    rb = {c: i for i, c in enumerate(base)}
    rn = {c: i for i, c in enumerate(novel)}
    Xtr, ytr = _slice(Xtr_all, ytr_all, base, rb)
    Xte, yte = _slice(Xte_all, yte_all, base, rb)
    if N:
        Xntr, yntr = _slice(Xtr_all, ytr_all, novel, rn)
        Xnte, ynte = _slice(Xte_all, yte_all, novel, rn)
    else:
        Xntr = yntr = Xnte = ynte = None
    robust = {}
    for perturb in ("noise0.05", "noise0.10"):
        if has_perturb(dataset, backbone, perturb):
            p = load_features(dataset, backbone, perturb)
            Xr, yr = p["X_test"].numpy(), p["y_test"].numpy()
            robust[perturb] = _slice(Xr, yr, base, rb)
    return Xtr, ytr, Xte, yte, Xntr, yntr, Xnte, ynte, int(clean["fdim"]), robust


def evaluate_case(dataset, backbone, seed, K, device):
    t0 = time.time()
    (Xtr, ytr, Xte, yte, Xntr, yntr, Xnte, ynte, fdim, robust) = load_case(dataset, backbone, seed, K)
    N = 0 if ynte is None else len(np.unique(ynte))
    rec = {"dataset": dataset, "backbone": backbone, "seed": seed, "K": K,
           "N": N, "fdim": fdim, "code_lens": list(CODE_LENS),
           "n_train": int(len(ytr)), "n_test": int(len(yte))}

    # standard heads for reference
    t = time.time()
    lin = train_ce(Xtr, ytr, K, device, seed, CEConfig(epochs=50))
    mlp = train_ce(Xtr, ytr, K, device, seed, CEConfig(epochs=50, lr=8e-4), mlp=True)
    proto = FeatureProtoHead(device).fit(Xtr, ytr, K)
    ref = {"linear": float((model_logits(lin, Xte, device).argmax(1) == yte).mean()),
           "mlp": float((model_logits(mlp, Xte, device).argmax(1) == yte).mean()),
           "proto": float((proto.predict(Xte) == yte).mean()),
           "robust": {}, "auroc": {}}
    for label, (Xr, yr) in robust.items():
        ref["robust"][label] = {
            "linear": float((model_logits(lin, Xr, device).argmax(1) == yr).mean()),
            "mlp": float((model_logits(mlp, Xr, device).argmax(1) == yr).mean()),
            "proto": float((proto.predict(Xr) == yr).mean())}
    if N:
        ref["auroc"] = {
            "linear": auroc(model_softmax_novelty(lin, Xnte, device),
                            model_softmax_novelty(lin, Xte, device)),
            "mlp": auroc(model_softmax_novelty(mlp, Xnte, device),
                         model_softmax_novelty(mlp, Xte, device)),
            "proto": auroc(proto.novelty(Xnte), proto.novelty(Xte)),
            "cluster_feat": hungarian_accuracy(
                ynte, spherical_kmeans(Xnte, N, seed, device=device), N)}
    rec["heads"] = ref
    rec["fit_time_heads"] = round(time.time() - t, 1)

    variants = {}
    for spec in VARIANTS:
        vrec = {}
        for L in CODE_LENS:
            t = time.time()
            head = HDCVariantHead(spec, fdim, L, seed=seed, device=device)
            head.fit(Xtr, ytr, K)
            out = {"id": float((head.predict(Xte) == yte).mean()),
                   "novelty": head.novelty(Xte) if N else None,
                   "fit_time": round(time.time() - t, 1)}
            if robust:
                out["robust"] = {}
                for label, (Xr, yr) in robust.items():
                    out["robust"][label] = float((head.predict(Xr) == yr).mean())
            if N:
                out["auroc"] = float(auroc(head.novelty(Xnte), out["novelty"]))
                out["cluster"] = float(hungarian_accuracy(
                    ynte, spherical_kmeans(head.cluster_codes(Xnte), N, seed, device=device), N))
                out.pop("novelty")
            else:
                out.pop("novelty")
            vrec[str(L)] = out
            del head
            torch.cuda.empty_cache()
        variants[spec.name] = vrec
    rec["variants"] = variants

    # projection-draw variance of the baseline
    draws = {}
    for L in CODE_LENS:
        accs = []
        for off in (0, 7919, 104729):
            head = HDCVariantHead(VariantSpec("gauss_draw"), fdim, L,
                                  seed=seed * 10007 + off, device=device)
            head.fit(Xtr, ytr, K)
            accs.append(float((head.predict(Xte) == yte).mean()))
            del head
        draws[str(L)] = {"min": float(np.min(accs)), "mean": float(np.mean(accs)),
                         "max": float(np.max(accs))}
    rec["gauss_draws"] = draws
    rec["wall_time"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=list(CASES),
                    choices=list(CASES) + ["all"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cases = list(CASES) if "all" in args.cases else args.cases
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for case in cases:
        dataset, backbone, ks, seeds = CASES[case]
        for seed in seeds:
            for K in ks:
                N = min(N_NOVEL, _cap(dataset) - K)
                name = f"{dataset}__{backbone or 'raw'}_K{K}_N{N}_seed{seed}"
                if has_record("variants", name) and not args.force:
                    continue
                rec = evaluate_case(dataset, backbone, seed, K, device)
                save_record("variants", name, rec)
                v = rec["variants"]
                print(f"  {name}: linear={rec['heads']['linear']:.3f} "
                      f"gauss4k={v['gauss']['4096']['id']:.3f} "
                      f"lincodes4k={v['lincodes']['4096']['id']:.3f} "
                      f"th2_4k={v['th2']['4096']['id']:.3f} "
                      f"median4k={v['median']['4096']['id']:.3f} ({rec['wall_time']}s)",
                      flush=True)


if __name__ == "__main__":
    main()
