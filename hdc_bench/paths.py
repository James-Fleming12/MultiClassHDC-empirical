"""Filesystem layout for the MultiClassHDC empirical suite.

The image datasets live in the older NCD_HDC repository; this repo never copies
the raw images, it only caches extracted features under ``results/features``.
The old repository is also the reference implementation for the HDC encoding
conventions (random Gaussian projection + sign, normalized class-mean
prototypes, normalized-Hamming/cosine scoring).
"""

from __future__ import annotations

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FEATURE_DIR = os.path.join(RESULTS_DIR, "features")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURE_DIR = os.path.join(RESULTS_DIR, "figures")

# The dataset archive/repo described in the task.
OLD_REPO = os.environ.get("NCD_HDC_ROOT", "/home/james/Research/SEE/Old/NCD_HDC")
DATA_DIR = os.path.join(OLD_REPO, "data")

# Folder-layout image datasets (built by the old repo's dataset_setup scripts).
IMAGE_DATASETS = {
    "cifar100": {
        "root": os.path.join(DATA_DIR, "cifar-100", "organized"),
        "images_subdir": "",
        "n_classes": 100,
        "note": "CIFAR-100 fine labels, all 60k images, 70/30 per-class split",
    },
    "tinyimagenet": {
        # the organized/ tree in the old repo is a symlink farm whose targets
        # still point at the pre-move path, so read the raw train layout.
        "root": os.path.join(DATA_DIR, "tiny-imagenet-200", "train"),
        "images_subdir": "images",
        "n_classes": 200,
        "note": "TinyImageNet train split (500/class), 70/30 per-class split",
    },
}

for _d in (FEATURE_DIR, RAW_DIR, TABLE_DIR, FIGURE_DIR):
    os.makedirs(_d, exist_ok=True)
