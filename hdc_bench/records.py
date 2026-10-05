"""Small JSON record store: one record per run under ``results/raw/<suite>/``."""

from __future__ import annotations

import json
import os

import numpy as np

from .paths import RAW_DIR


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def record_path(suite: str, name: str) -> str:
    d = os.path.join(RAW_DIR, suite)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}.json")


def has_record(suite: str, name: str) -> bool:
    return os.path.exists(record_path(suite, name))


def save_record(suite: str, name: str, record: dict) -> str:
    path = record_path(suite, name)
    with open(path, "w") as f:
        json.dump(_jsonable(record), f, indent=1, sort_keys=True)
    return path


def load_records(suite: str) -> list[dict]:
    d = os.path.join(RAW_DIR, suite)
    if not os.path.isdir(d):
        return []
    out = []
    for fname in sorted(os.listdir(d)):
        if fname.endswith(".json"):
            with open(os.path.join(d, fname)) as f:
                out.append(json.load(f))
    return out
