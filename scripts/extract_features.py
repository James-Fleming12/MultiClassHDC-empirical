#!/usr/bin/env python
"""Extract and cache image features for CIFAR-100 / TinyImageNet.

Usage:
  python scripts/extract_features.py --datasets cifar100 tinyimagenet \
      --backbones dinov2_vitb14_reg resnet18 --perturbations clean noise0.05 noise0.10
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hdc_bench.features import BACKBONES, PERTURBATIONS, extract_features
from hdc_bench.paths import IMAGE_DATASETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(IMAGE_DATASETS),
                    choices=list(IMAGE_DATASETS))
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=list(BACKBONES))
    ap.add_argument("--perturbations", nargs="+", default=["clean"],
                    choices=list(PERTURBATIONS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for dataset in args.datasets:
        for backbone in args.backbones:
            for perturb in args.perturbations:
                path = extract_features(dataset, backbone, perturb, device=device,
                                        force=args.force)
                print(f"  -> {path}")


if __name__ == "__main__":
    main()
