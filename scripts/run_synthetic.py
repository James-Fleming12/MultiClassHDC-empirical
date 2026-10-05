#!/usr/bin/env python
"""Run the synthetic class-scaling suite.

Generators: ``bits`` (HDC-native random codes), ``gmm`` (Gaussian mixture),
``subspace`` (class subspace + center).  For each (generator, K, N, seed):
fit the heads and evaluate ID / robustness / novel metrics, exactly as the
image suites do.

``bits`` is HDC-native: its data dimension *is* the HD dimension, so it is run
once per HD dim (4096, 10000).  The other generators use 256-d features and
project to both HD dims.

Usage:
  python scripts/run_synthetic.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hdc_bench.records import has_record, save_record
from hdc_bench.runner import evaluate_heads
from hdc_bench.synthetic import SYNTH_CONFIGS, build_synthetic

GENERATORS = ["bits", "gmm", "subspace"]
KS = [5, 10, 20, 50, 100, 200]
N_FOR_K = 20
NS = [5, 10, 20, 50, 100]
K_FOR_N = 50
HD_DIMS = [4096, 10000]


def configs():
    cfgs = [(k, min(N_FOR_K, 250 - k)) for k in KS]
    cfgs += [(K_FOR_N, n) for n in NS]
    seen, out = set(), []
    for c in cfgs:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def run_config(gen: str, seed: int, K: int, N: int, hd_dim: int, device):
    data = build_synthetic(gen, seed, K, N, hd_dim=hd_dim)
    robust = {label: (Xr, np.arange(K).repeat(len(Xr) // max(K, 1)))
              for label, Xr in data["robust"].items()}
    rec = {
        "generator": gen, "seed": seed, "K": K, "N": N, "hd_dim": hd_dim,
        "fdim": data["fdim"],
    }
    rec.update(evaluate_heads(
        Xtr=data["Xtr"], ytr=data["ytr"], Xte=data["Xte"], yte=data["yte"],
        n_classes=K, fdim=data["fdim"], seed=seed, device=device,
        Xntr=data["Xntr"], yntr=data["yntr"], Xnte=data["Xnte"], ynte=data["ynte"],
        n_novel=N, robust=robust, hdc_dims=(hd_dim,),
        identity_hdc=bool(SYNTH_CONFIGS[gen].get("hd_native"))))
    return rec


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--generators", nargs="+", default=GENERATORS, choices=GENERATORS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="limit configs per generator (smoke test)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for gen in args.generators:
        native = bool(SYNTH_CONFIGS[gen].get("hd_native"))
        dims = HD_DIMS if native else [4096]  # non-native: one record, both head dims
        cfgs = configs()
        if args.limit:
            cfgs = cfgs[:args.limit]
        for seed in args.seeds:
            for K, N in cfgs:
                for dim in dims:
                    suffix = f"_D{dim}" if native else ""
                    name = f"{gen}_K{K}_N{N}_seed{seed}{suffix}"
                    suite = "synthetic"
                    if has_record(suite, name) and not args.force:
                        continue
                    t0 = time.time()
                    if native:
                        rec = run_config(gen, seed, K, N, hd_dim=dim, device=device)
                    else:
                        data = build_synthetic(gen, seed, K, N)
                        robust = {label: (Xr, np.arange(K).repeat(len(Xr) // max(K, 1)))
                                  for label, Xr in data["robust"].items()}
                        rec = {"generator": gen, "seed": seed, "K": K, "N": N,
                               "fdim": data["fdim"], "hd_dims": [4096, 10000],
                               "identity_hdc": False}
                        rec.update(evaluate_heads(
                            Xtr=data["Xtr"], ytr=data["ytr"], Xte=data["Xte"],
                            yte=data["yte"], n_classes=K, fdim=data["fdim"],
                            seed=seed, device=device, Xntr=data["Xntr"],
                            yntr=data["yntr"], Xnte=data["Xnte"], ynte=data["ynte"],
                            n_novel=N, robust=robust, hdc_dims=(4096, 10000)))
                    rec["wall_time"] = round(time.time() - t0, 1)
                    save_record(suite, name, rec)
                    idm = rec["id"]
                    msg = (f"{gen:9s} D={dim if native else '4k+10k':>7} K={K:3d} N={N:3d} s{seed} "
                           f"hdc4k={idm.get('hdc4096', float('nan')):.3f} "
                           f"hdc10k={idm.get('hdc10000', float('nan')):.3f} "
                           f"lin={idm['linear']:.3f} mlp={idm['mlp']:.3f} "
                           f"proto={idm['proto']:.3f}")
                    print(f"  {msg}  ({rec['wall_time']}s)", flush=True)


if __name__ == "__main__":
    main()
