#!/usr/bin/env python
"""Run the frozen-feature class-scaling suite on CIFAR-100 / TinyImageNet.

For each (seed, K, N): pick K base classes and N disjoint novel classes, fit all
heads on the base train split, and evaluate ID accuracy, robustness (Gaussian
pixel noise) and no-label/5-shot novel metrics.

  K-series: K grows, N fixed at 20 (N=0 when K is the last class).
  N-series: K fixed, N grows.

Usage:
  python scripts/run_images.py --datasets cifar100 --backbones dinov2_vitb14_reg \
      --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hdc_bench.features import BACKBONES, load_features
from hdc_bench.paths import IMAGE_DATASETS
from hdc_bench.records import has_record, save_record
from hdc_bench.runner import evaluate_heads

GRIDS = {
    "cifar100": dict(Ks=[5, 10, 20, 50, 100], N_for_K=20,
                     Ns=[5, 10, 20, 40], K_for_N=50),
    "tinyimagenet": dict(Ks=[10, 20, 50, 100, 200], N_for_K=20,
                         Ns=[5, 10, 25, 50, 100], K_for_N=100),
}


def configs_for(dataset: str):
    g = GRIDS[dataset]
    cfgs = [(k, min(g["N_for_K"], IMAGE_DATASETS[dataset]["n_classes"] - k))
            for k in g["Ks"]]
    cfgs += [(g["K_for_N"], n) for n in g["Ns"]]
    # dedupe, keep order
    seen, out = set(), []
    for c in cfgs:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def slice_by_classes(X, y, classes, remap: dict | None = None):
    mask = np.isin(y, classes)
    Xs, ys = X[mask], y[mask]
    if remap is not None:
        ys = np.array([remap[c] for c in ys], dtype=np.int64)
    return Xs, ys


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar100", "tinyimagenet"],
                    choices=list(GRIDS))
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=list(BACKBONES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--perturbations", nargs="+", default=["noise0.05", "noise0.10"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="limit configs per (dataset, backbone)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for dataset in args.datasets:
        n_total = IMAGE_DATASETS[dataset]["n_classes"]
        for backbone in args.backbones:
            print(f"\n=== {dataset} / {backbone} ===", flush=True)
            clean = load_features(dataset, backbone, "clean")
            Xtr_all, ytr_all = clean["X_train"].numpy(), clean["y_train"].numpy()
            Xte_all, yte_all = clean["X_test"].numpy(), clean["y_test"].numpy()
            class_names = clean["class_names"]
            robust_all = {}
            for perturb in args.perturbations:
                try:
                    d = load_features(dataset, backbone, perturb)
                    robust_all[perturb] = (d["X_test"].numpy(), d["y_test"].numpy())
                except FileNotFoundError:
                    print(f"  [warn] no cached {perturb} features; skipping robustness")

            for seed in args.seeds:
                perm = np.random.default_rng(seed).permutation(n_total)
                cfgs = configs_for(dataset)
                if args.limit:
                    cfgs = cfgs[:args.limit]
                for K, N in cfgs:
                    name = f"{backbone}_K{K}_N{N}_seed{seed}"
                    suite = f"images/{dataset}"
                    if has_record(suite, name) and not args.force:
                        continue
                    t0 = time.time()
                    base, novel = perm[:K], perm[K:K + N]
                    remap_base = {c: i for i, c in enumerate(base)}
                    remap_novel = {c: i for i, c in enumerate(novel)}
                    Xtr, ytr = slice_by_classes(Xtr_all, ytr_all, base, remap_base)
                    Xte, yte = slice_by_classes(Xte_all, yte_all, base, remap_base)
                    if N:
                        Xntr, yntr = slice_by_classes(Xtr_all, ytr_all, novel, remap_novel)
                        Xnte, ynte = slice_by_classes(Xte_all, yte_all, novel, remap_novel)
                    else:
                        Xntr = yntr = Xnte = ynte = None
                    robust = {}
                    for perturb, (Xr, yr) in robust_all.items():
                        # only known classes have perturbed features cached for now
                        mask = np.isin(yr, base)
                        robust[perturb] = (Xr[mask],
                                           np.array([remap_base[c] for c in yr[mask]]))
                    rec = {
                        "dataset": dataset, "backbone": backbone, "seed": seed,
                        "K": K, "N": N,
                        "base_classes": [class_names[c] for c in base],
                        "novel_classes": [class_names[c] for c in novel],
                        "n_test": int(len(yte)),
                    }
                    rec.update(evaluate_heads(
                        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, n_classes=K,
                        fdim=int(clean["fdim"]), seed=seed, device=device,
                        Xntr=Xntr, yntr=yntr, Xnte=Xnte, ynte=ynte, n_novel=N,
                        robust=robust, hdc_dims=(4096, 10000)))
                    rec["wall_time"] = round(time.time() - t0, 1)
                    save_record(suite, name, rec)
                    idm = rec["id"]
                    msg = (f"K={K:3d} N={N:3d} s{seed} "
                           f"id[hdc4k {idm.get('hdc4096', float('nan')):.3f} "
                           f"hdc10k {idm.get('hdc10000', float('nan')):.3f} "
                           f"lin {idm['linear']:.3f} mlp {idm['mlp']:.3f} "
                           f"proto {idm['proto']:.3f}]")
                    if N:
                        nv = rec["novel"]
                        msg += (f" | novel[clu4k {nv.get('cluster_hdc4096', float('nan')):.3f} "
                                f"clu10k {nv.get('cluster_hdc10000', float('nan')):.3f} "
                                f"clufeat {nv['cluster_feat']:.3f}]")
                    print(f"  {msg}  ({rec['wall_time']}s)", flush=True)


if __name__ == "__main__":
    main()
