#!/usr/bin/env python
"""Pretraining-class-count scaling: train the feature extractor on K classes.

Three variants:
  * ``mlp_synthetic`` : MLP encoder on the synthetic ``subspace`` generator.
  * ``cifar``         : SmallResNet on CIFAR-100 (official train/test pickles).
  * ``tiny``          : SmallResNet on TinyImageNet (packed uint8 cache).

For each K the network is trained from scratch on K base classes, then all the
usual heads are compared on its frozen 256-d penultimate features, including
the network's own classification head (which is exactly a linear layer on those
features).  This isolates "the network degrades with K" from "the HDC prototype
head degrades with K": the HDC/linear probes and the network head share the same
penultimate representation, while the network head also shares the network's
training.

Usage:
  python scripts/run_pretrain.py --suite mlp_synthetic cifar --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hdc_bench.pretrain import (
    CIFAR_MEAN,
    CIFAR_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_cifar100,
    load_tinyimagenet,
    mlp_features,
    penultimate_features,
    train_mlp,
    train_network,
)
from hdc_bench.records import has_record, save_record
from hdc_bench.runner import evaluate_heads
from hdc_bench.synthetic import build_synthetic

KS = [5, 10, 20, 50, 100]
KS_TINY = [10, 20, 50, 100, 200]
N_NOVEL = 20
NOISE_LEVELS = {"noise0.05": 0.05, "noise0.10": 0.10}


def net_head_logits_from_features(model, device):
    """Network head applied directly to *penultimate* features."""
    def fn(F):
        x = torch.as_tensor(np.asarray(F), dtype=torch.float32, device=device)
        with torch.no_grad():
            return model.forward_from_features(x).cpu().numpy()
    return fn

def run_synthetic_mlp(seed: int, K: int, N: int, device, epochs: int):
    data = build_synthetic("subspace", seed, K, N)
    ytr = data["ytr"]
    model = train_mlp(data["Xtr"], ytr, K, device, seed, epochs=epochs)
    Ftr = mlp_features(model, data["Xtr"], device)
    Fte = mlp_features(model, data["Xte"], device)
    Fntr = mlp_features(model, data["Xntr"], device) if N else None
    Fnte = mlp_features(model, data["Xnte"], device) if N else None
    robust = {label: (mlp_features(model, Xr, device), np.arange(K).repeat(len(Xr) // max(K, 1)))
              for label, Xr in data["robust"].items()}
    rec = {"suite": "mlp_synthetic", "seed": seed, "K": K, "N": N,
           "fdim": model.feature_dim}
    rec.update(evaluate_heads(
        Xtr=Ftr, ytr=ytr, Xte=Fte, yte=data["yte"], n_classes=K,
        fdim=model.feature_dim, seed=seed, device=device,
        Xntr=Fntr, yntr=data["yntr"], Xnte=Fnte, ynte=data["ynte"], n_novel=N,
        robust=robust, hdc_dims=(4096, 10000),
        extra_heads={"net": net_head_logits_from_features(model, device)}))
    return rec


def run_image(seed: int, K: int, N: int, device, dataset: str, epochs: int):
    n_classes = 100 if dataset == "cifar100" else 200
    if dataset == "cifar100":
        Xtr, ytr, Xte, yte = load_cifar100()
        mean, std, pad = CIFAR_MEAN, CIFAR_STD, 4
    else:
        Xtr, ytr, Xte, yte = load_tinyimagenet()
        mean, std, pad = IMAGENET_MEAN, IMAGENET_STD, 8
    perm = np.random.default_rng(seed).permutation(n_classes)
    base, novel = perm[:K], perm[K:K + N]
    remap = {c: i for i, c in enumerate(base)}
    remap_n = {c: i for i, c in enumerate(novel)}
    trmask = np.isin(ytr, base)
    temask = np.isin(yte, base)
    Xb, yb = Xtr[trmask], np.array([remap[c] for c in ytr[trmask]])
    Xbt, ybt = Xte[temask], np.array([remap[c] for c in yte[temask]])
    model = train_network(torch.as_tensor(Xb), yb, K, device, seed,
                          epochs=epochs, pad=pad, mean=mean, std=std)
    Ftr = penultimate_features(model, torch.as_tensor(Xb), device, mean=mean, std=std)
    Fte = penultimate_features(model, torch.as_tensor(Xbt), device, mean=mean, std=std)
    if N:
        trnmask = np.isin(ytr, novel)
        tenmask = np.isin(yte, novel)
        Xntr = Xtr[trnmask]
        yntr = np.array([remap_n[c] for c in ytr[trnmask]])
        Xnte = Xte[tenmask]
        ynte = np.array([remap_n[c] for c in yte[tenmask]])
        Fntr = penultimate_features(model, torch.as_tensor(Xntr), device, mean=mean, std=std)
        Fnte = penultimate_features(model, torch.as_tensor(Xnte), device, mean=mean, std=std)
    else:
        Fntr = Fnte = None
        yntr = ynte = None
    robust = {}
    for label, sigma in NOISE_LEVELS.items():
        Fr = penultimate_features(model, torch.as_tensor(Xbt), device, mean=mean,
                                  std=std, sigma=sigma, seed=seed + 1)
        robust[label] = (Fr, ybt)
    rec = {"suite": f"pretrain_{dataset}", "dataset": dataset, "seed": seed,
           "K": K, "N": N, "fdim": model.feature_dim,
           "base_classes": [int(c) for c in base],
           "novel_classes": [int(c) for c in novel]}
    rec.update(evaluate_heads(
        Xtr=Ftr, ytr=yb, Xte=Fte, yte=ybt, n_classes=K, fdim=model.feature_dim,
        seed=seed, device=device, Xntr=Fntr, yntr=yntr, Xnte=Fnte, ynte=ynte,
        n_novel=N, robust=robust, hdc_dims=(4096, 10000),
        extra_heads={"net": net_head_logits_from_features(model, device)}))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+", default=["mlp_synthetic"],
                    choices=["mlp_synthetic", "cifar", "tiny"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ks", nargs="+", type=int, default=None,
                    help="override the K grid (default: per-suite)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for suite in args.suite:
        ks = args.ks if args.ks else (KS if suite in ("mlp_synthetic", "cifar") else KS_TINY)
        for seed in args.seeds:
            for K in ks:
                cap = {"mlp_synthetic": 10**6, "cifar": 100, "tiny": 200}[suite]
                N = min(N_NOVEL, cap - K)
                name = f"K{K}_N{N}_seed{seed}"
                if has_record(f"pretrain/{suite}", name) and not args.force:
                    continue
                t0 = time.time()
                if suite == "mlp_synthetic":
                    rec = run_synthetic_mlp(seed, K, N, device, args.epochs)
                else:
                    rec = run_image(seed, K, N, device, "cifar100" if suite == "cifar" else "tinyimagenet",
                                    args.epochs)
                rec["wall_time"] = round(time.time() - t0, 1)
                save_record(f"pretrain/{suite}", name, rec)
                idm = rec["id"]
                print(f"  {suite:14s} K={K:3d} s{seed} "
                      f"net={idm.get('net', float('nan')):.3f} "
                      f"netprobe_lin={idm['linear']:.3f} "
                      f"hdc4k={idm['hdc4096']:.3f} hdc10k={idm['hdc10000']:.3f} "
                      f"({rec['wall_time']}s)", flush=True)


if __name__ == "__main__":
    main()
