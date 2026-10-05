"""HDC configuration ablation: projections, quantization and prototype rules.

The baseline HDC head in the rest of the repo is: Gaussian random projection ->
sign (1 bit) -> real-valued class-mean prototype -> cosine (normalized Hamming).
This module lets the same evaluation be run with alternative choices:

projections   gauss | rade (Rademacher) | sparse (Achlioptas 1/3) | ens2
quantization  sign (1 bit) | th2 (2-bit thermometer) | th3 (3-bit thermometer)
prototype     mean_cos (baseline) | majority (binary majority + Hamming) |
              median | projmean (class-mean feature encoded once) | lin_codes
              (trained linear layer on the codes = capacity ceiling)

``ens2`` averages the cosine scores of two independent half-length Gaussian
projections.  All variants use the *same total code length* (4k / 10k), so the
comparison is at matched bit budget; the thermometer thresholds are the
Gaussian quantiles that split the projected values into equiprobable bins.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .heads import CEConfig, train_ce

# Gaussian quantiles for equiprobable thermometer bins
THERMO_THRESHOLDS = {
    2: (-0.4307273, 0.4307273),
    3: (-0.6744898, 0.0, 0.6744898),
}


@dataclass
class VariantSpec:
    name: str
    proj: str = "gauss"
    quant: str = "sign"
    proto: str = "mean_cos"
    opts: dict = field(default_factory=dict)


VARIANTS = [
    VariantSpec("gauss", proj="gauss", quant="sign", proto="mean_cos"),
    VariantSpec("rade", proj="rade", quant="sign", proto="mean_cos"),
    VariantSpec("sparse", proj="sparse", quant="sign", proto="mean_cos"),
    VariantSpec("ens2", proj="ens2", quant="sign", proto="mean_cos"),
    VariantSpec("th2", proj="gauss", quant="th2", proto="mean_cos"),
    VariantSpec("th3", proj="gauss", quant="th3", proto="mean_cos"),
    VariantSpec("majority", proj="gauss", quant="sign", proto="majority"),
    VariantSpec("median", proj="gauss", quant="sign", proto="median"),
    VariantSpec("projmean", proj="gauss", quant="sign", proto="projmean"),
    VariantSpec("lincodes", proj="gauss", quant="sign", proto="lin_codes"),
]


def _gauss(g, *shape):
    return torch.randn(*shape, generator=g)


def _make_projection(kind: str, fdim: int, dim: int, seed: int, device):
    g = torch.Generator().manual_seed(int(seed))
    if kind == "gauss":
        proj = _gauss(g, fdim, dim)
    elif kind == "rade":
        proj = torch.randint(0, 2, (fdim, dim), generator=g).float() * 2 - 1
    elif kind == "sparse":
        u = torch.rand(fdim, dim, generator=g)
        proj = torch.where(u < 1.0 / 6, -1.0, torch.where(u < 1.0 / 3, 1.0, 0.0))
        proj = proj * (3.0 ** 0.5)
    else:
        raise ValueError(kind)
    return proj.to(device)


def _to_tensor(X, device):
    if isinstance(X, torch.Tensor):
        return X.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)


class HDCVariantHead:
    """One HDC configuration: fit prototypes / train the code readout."""

    def __init__(self, spec: VariantSpec, fdim: int, code_len: int, seed: int, device):
        self.spec = spec
        self.fdim = fdim
        self.code_len = int(code_len)
        self.seed = int(seed)
        self.device = device
        self.bits = {"sign": 1, "th2": 2, "th3": 3}[spec.quant]
        self.base_dim = self.code_len // self.bits
        self.out_dim = self.base_dim * self.bits
        self.prototypes = None
        self.model = None
        self.encoders: list[torch.Tensor] = []
        if spec.proj == "ens2":
            half = max(1, self.base_dim // 2)
            for k in range(2):
                self.encoders.append(_make_projection(
                    "gauss", fdim, half, seed * 10007 + 17 * k, device))
            self.base_dim = half  # per-space length (2*half ~= code_len)
            self.out_dim = half
        else:
            self.encoders.append(_make_projection(
                spec.proj, fdim, self.base_dim, seed * 10007, device))
        if self.bits > 1:
            self.thresholds = torch.tensor(
                THERMO_THRESHOLDS[self.bits], device=device).view(1, 1, -1)
        else:
            self.thresholds = None

    # ------------------------------------------------------------------
    def _project_space(self, X, k: int):
        return X @ self.encoders[k]

    def _quantize(self, z):
        if self.spec.quant == "sign":
            return z.sign().float()
        # thermometer: bits are 1 above each threshold; (n, dim, bits)
        bits = (z.unsqueeze(-1) > self.thresholds).float()
        n, d, b = bits.shape
        return bits.reshape(n, d * b)

    def encode(self, X, batch: int = 4096):
        """Return a list of code tensors (one per projection space)."""
        X = _to_tensor(X, self.device)
        spaces = [[] for _ in self.encoders]
        with torch.no_grad():
            for i in range(0, X.shape[0], batch):
                xb = X[i:i + batch]
                for k in range(len(self.encoders)):
                    spaces[k].append(self._quantize(self._project_space(xb, k)))
        return [torch.cat(s) for s in spaces]

    def _encode_once(self, X):
        codes = self.encode(X)
        return codes

    # ------------------------------------------------------------------
    def fit(self, X, y, n_classes: int):
        spec = self.spec
        y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device)
        if spec.proto == "median":
            protos = []
            for c in range(n_classes):
                Xc = X[torch.as_tensor(np.asarray(y.cpu())) == c] if not isinstance(X, torch.Tensor) \
                    else X[y == c]
                codes = self.encode(Xc)[0]
                protos.append(torch.median(codes, dim=0).values)
            self.prototypes = [F.normalize(torch.stack(protos), p=2, dim=1)]
            return self
        if spec.proto == "projmean":
            Xt = _to_tensor(X, "cpu")
            sums = torch.zeros(n_classes, Xt.shape[1])
            sums.index_add_(0, y.cpu(), Xt)
            counts = torch.zeros(n_classes)
            counts.index_add_(0, y.cpu(), torch.ones(len(y)))
            means = (sums / counts.clamp_min(1).unsqueeze(1)).to(self.device)
            protos = self.encode(means)[0]
            self.prototypes = [F.normalize(protos, p=2, dim=1)]
            return self
        # streaming class sums for mean_cos / majority / lin_codes
        sums = [torch.zeros(n_classes, self.out_dim, device=self.device)
                for _ in self.encoders]
        codes_keep = [] if spec.proto == "lin_codes" else None
        Xt = _to_tensor(X, self.device)
        with torch.no_grad():
            for i in range(0, Xt.shape[0], 4096):
                xb = Xt[i:i + 4096]
                yb = y[i:i + 4096]
                for k in range(len(self.encoders)):
                    code = self._quantize(self._project_space(xb, k))
                    sums[k].index_add_(0, yb, code)
                    if codes_keep is not None:
                        codes_keep.append(code)
        if spec.proto == "lin_codes":
            codes = torch.cat(codes_keep) if len(self.encoders) == 1 else None
            codes_cpu = codes.cpu()
            del codes, codes_keep
            self.model = train_ce(codes_cpu, np.asarray(y.cpu()), n_classes,
                                  self.device, self.seed, CEConfig(epochs=50))
            del codes_cpu
            return self
        protos = []
        for s in sums:
            if spec.proto == "majority":
                s = s.sign()
            protos.append(F.normalize(s, p=2, dim=1))
        self.prototypes = protos
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def similarities(self, X) -> torch.Tensor:
        assert self.prototypes is not None, "prototypes not fit"
        codes = self.encode(X)
        sims = None
        for k, code in enumerate(codes):
            c = F.normalize(code, p=2, dim=1)
            s = c @ self.prototypes[k].t()
            sims = s if sims is None else sims + s
        return sims / len(codes)

    def predict(self, X) -> np.ndarray:
        if self.spec.proto == "lin_codes":
            from .heads import model_logits
            return model_logits(self.model, self._encode_feature(X), self.device).argmax(1)
        return self.similarities(X).argmax(dim=1).cpu().numpy()

    def _encode_feature(self, X):
        return self.encode(X)[0]

    def novelty(self, X) -> np.ndarray:
        if self.spec.proto == "lin_codes":
            from .heads import model_softmax_novelty
            return model_softmax_novelty(self.model, self._encode_feature(X), self.device)
        return (1.0 - self.similarities(X).max(dim=1).values).cpu().numpy()

    def cluster_codes(self, X) -> np.ndarray:
        codes = self.encode(X)
        if len(codes) == 1:
            return codes[0].cpu().numpy()
        return torch.cat([F.normalize(c, p=2, dim=1) for c in codes], dim=1).cpu().numpy()
