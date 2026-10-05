"""Synthetic datasets with controllable class count and geometry.

Three generators:

* ``bits``     : "HDC-native".  Each class is a random Rademacher hypervector in
  ``{-1,+1}^D``; samples are that vector with a fraction ``flip`` of bits
  flipped.  The features are already the codes, so the HDC projection is the
  identity and the HD dimension *is* the data dimension (D in {4096, 10000}).
  This isolates the prototype-vs-network-head question with no feature-learning
  confound at all, and directly probes HDC capacity as K grows.
* ``gmm``      : isotropic Gaussian mixture in R^256, class means on a sphere of
  radius ``sep``, noise std ``sigma``.  Well-behaved prototype geometry.
* ``subspace`` : class center + a random low-rank class subspace (rank 16) +
  isotropic noise.  A more realistic "feature manifold" geometry where
  class-conditional distributions are anisotropic.

Classes are generated in a nested way (classes 0..K+N-1 are the same vectors
for every ``K``/``N``), so accuracy-vs-K curves are comparable across K.

All synthetic features are L2-normalized before the heads, matching the image
features.  Robustness variants add extra test-time noise (Gaussian for
gmm/subspace, extra bit flips for bits).
"""

from __future__ import annotations

import numpy as np

SYNTH_CONFIGS = {
    "bits": dict(
        n_train=60, n_test=30, flip=0.44,
        robust={"flip0.46": 0.46, "flip0.48": 0.48},
        hd_native=True,
    ),
    "gmm": dict(
        fdim=256, sep=4.0, sigma=1.0, n_train=200, n_test=100,
        robust={"noise1.5x": 1.5, "noise2.5x": 2.5},
        hd_native=False,
    ),
    "subspace": dict(
        fdim=256, rank=16, sep=3.5, sigma=0.8, n_train=200, n_test=100,
        robust={"noise1.5x": 1.5, "noise2.5x": 2.5},
        hd_native=False,
    ),
}


def _normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _bits_vectors(seed: int, n_classes: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng([seed, 7])
    return (rng.integers(0, 2, size=(n_classes, dim)).astype(np.float32) * 2.0 - 1.0)


def _bit_samples(protos: np.ndarray, n: int, flip: float, seed: int) -> np.ndarray:
    D = protos.shape[1]
    out = np.empty((len(protos) * n, D), dtype=np.float32)
    for c, proto in enumerate(protos):
        g = np.random.default_rng([seed, 1234 + c])
        x = np.tile(proto, (n, 1))
        x[g.random((n, D)) < flip] *= -1.0
        out[c * n:(c + 1) * n] = x
    return out


def _gmm_centers(seed: int, n_classes: int, dim: int, sep: float) -> np.ndarray:
    rng = np.random.default_rng([seed, 11])
    v = rng.standard_normal((n_classes, dim)).astype(np.float32)
    return _normalize(v) * sep


def _subspace_bases(seed: int, n_classes: int, dim: int, rank: int) -> np.ndarray:
    rng = np.random.default_rng([seed, 13])
    basis = rng.standard_normal((n_classes, dim, rank)).astype(np.float32)
    # QR per class for an orthonormal basis
    for c in range(n_classes):
        q, _ = np.linalg.qr(basis[c])
        basis[c] = q
    return basis


def _gmm_samples(centers: np.ndarray, n: int, sigma: float, seed: int) -> np.ndarray:
    out = np.empty((len(centers) * n, centers.shape[1]), dtype=np.float32)
    for c, mu in enumerate(centers):
        g = np.random.default_rng([seed, 4321 + c])
        out[c * n:(c + 1) * n] = mu + sigma * g.standard_normal((n, len(mu))).astype(np.float32)
    return out


def _subspace_samples(centers: np.ndarray, bases: np.ndarray, n: int, sigma: float,
                      seed: int) -> np.ndarray:
    k = bases.shape[2]
    out = np.empty((len(centers) * n, centers.shape[1]), dtype=np.float32)
    for c, (mu, U) in enumerate(zip(centers, bases)):
        g = np.random.default_rng([seed, 999 + c])
        z = g.standard_normal((n, k)).astype(np.float32)
        eps = g.standard_normal((n, len(mu))).astype(np.float32)
        out[c * n:(c + 1) * n] = mu + z @ U.T + sigma * eps
    return out


def build_synthetic(name: str, seed: int, K: int, N: int, hd_dim: int | None = None,
                    overrides: dict | None = None) -> dict:
    """Build a synthetic run: K base classes and N novel classes.

    Returns a dict with Xtr/ytr (base train), Xte/yte (base test), Xntr/yntr
    (novel train, for few-shot), Xnte/ynte (novel test), the feature dim, and
    ``robust`` = {label: X_perturbed_test_of_base} variants.
    """
    cfg = dict(SYNTH_CONFIGS[name])
    if overrides:
        cfg.update(overrides)
    if name == "bits":
        dim = int(hd_dim or 10000)
    else:
        dim = int(cfg["fdim"])
    C = K + N

    if name == "bits":
        protos = _bits_vectors(seed, C, dim)
        flip = float(cfg["flip"])
        Xtr = _normalize(_bit_samples(protos[:K], cfg["n_train"], flip, seed))
        Xte = _normalize(_bit_samples(protos[:K], cfg["n_test"], flip, seed + 1))
        Xntr = _normalize(_bit_samples(protos[K:], max(cfg["n_train"] // 2, 5), flip, seed + 1))
        Xnte = _normalize(_bit_samples(protos[K:], cfg["n_test"], flip, seed + 2))
        robust = {}
        for label, extra_flip in cfg["robust"].items():
            robust[label] = _normalize(_bit_samples(protos[:K], cfg["n_test"],
                                                    float(extra_flip), seed + 3))
    elif name == "gmm":
        sep, sigma = float(cfg["sep"]), float(cfg["sigma"])
        centers = _gmm_centers(seed, C, dim, sep)
        Xtr = _normalize(_gmm_samples(centers[:K], cfg["n_train"], sigma, seed))
        Xte = _normalize(_gmm_samples(centers[:K], cfg["n_test"], sigma, seed + 1))
        Xntr = _normalize(_gmm_samples(centers[K:], max(cfg["n_train"] // 2, 5), sigma, seed + 1))
        Xnte = _normalize(_gmm_samples(centers[K:], cfg["n_test"], sigma, seed + 2))
        robust = {}
        for label, mult in cfg["robust"].items():
            robust[label] = _normalize(_gmm_samples(centers[:K], cfg["n_test"],
                                                    sigma * float(mult), seed + 3))
    elif name == "subspace":
        rank, sep, sigma = int(cfg["rank"]), float(cfg["sep"]), float(cfg["sigma"])
        centers = _gmm_centers(seed, C, dim, sep)
        bases = _subspace_bases(seed, C, dim, rank)
        Xtr = _normalize(_subspace_samples(centers[:K], bases[:K], cfg["n_train"], sigma, seed))
        Xte = _normalize(_subspace_samples(centers[:K], bases[:K], cfg["n_test"], sigma, seed + 1))
        Xntr = _normalize(_subspace_samples(centers[K:], bases[K:], max(cfg["n_train"] // 2, 5),
                                            sigma, seed + 1))
        Xnte = _normalize(_subspace_samples(centers[K:], bases[K:], cfg["n_test"], sigma, seed + 2))
        robust = {}
        for label, mult in cfg["robust"].items():
            robust[label] = _normalize(_subspace_samples(centers[:K], bases[:K], cfg["n_test"],
                                                         sigma * float(mult), seed + 3))
    else:
        raise ValueError(f"unknown synthetic generator {name}")

    ytr = np.repeat(np.arange(K), len(Xtr) // max(K, 1)).astype(np.int64) if K else np.zeros(0, np.int64)
    yntr = np.repeat(np.arange(N), len(Xntr) // max(N, 1)).astype(np.int64) if N else np.zeros(0, np.int64)
    y_te = np.arange(K).repeat(len(Xte) // max(K, 1)).astype(np.int64) if K else np.zeros(0, np.int64)
    y_nte = np.arange(N).repeat(len(Xnte) // max(N, 1)).astype(np.int64) if N else np.zeros(0, np.int64)
    return {
        "name": name, "seed": seed, "K": K, "N": N, "fdim": dim,
        "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": y_te,
        "Xntr": Xntr, "yntr": yntr, "Xnte": Xnte, "ynte": y_nte,
        "robust": robust,
    }
