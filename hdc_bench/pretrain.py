"""Small supervised networks trained from scratch on K classes.

This is the only place where a *network* is trained end-to-end.  It exists to
answer the part of Q1 that the frozen-feature suites cannot: when the number of
pretraining classes K grows, does the network itself degrade, and does the HDC
prototype head on top of the network's penultimate layer degrade more or less
than a trained classification layer on the same frozen penultimate features?

* ``SmallResNet``: a compact ResNet (3 stages, 6 conv layers, 256-d pooled
  penultimate) trained with SGD/AdamW + standard crop/flip augmentation.
* ``train_network``: trains on K classes of an in-memory uint8 image array.
* ``penultimate_features``: pooled 256-d features for any (possibly perturbed)
  image array.
* CIFAR-100 is read from the old repo's pickles; TinyImageNet images are packed
  once into a uint8 array cache.

The same code also trains an MLP encoder on synthetic data (``train_mlp``).
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .paths import DATA_DIR, FEATURE_DIR

CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2764)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        s = x if self.short is None else self.short(x)
        return F.relu(y + s, inplace=True)


class SmallResNet(nn.Module):
    """3-stage ResNet; penultimate = 256-d pooled features."""

    feature_dim = 256

    def __init__(self, n_classes: int, width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False), nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(BasicBlock(width, width), BasicBlock(width, width))
        self.layer2 = nn.Sequential(BasicBlock(width, 2 * width, 2), BasicBlock(2 * width, 2 * width))
        self.layer3 = nn.Sequential(BasicBlock(2 * width, 4 * width, 2), BasicBlock(4 * width, 4 * width))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4 * width, n_classes)
        assert 4 * width == self.feature_dim

    def forward_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.pool(x).flatten(1)

    def forward_from_features(self, f):
        return self.head(f)

    def forward(self, x):
        return self.head(self.forward_features(x))


def load_cifar100():
    """Return uint8 NHWC arrays: (Xtr, ytr, Xte, yte), 100 fine classes."""
    root = os.path.join(DATA_DIR, "cifar-100-python")
    out = []
    for name in ("train", "test"):
        with open(os.path.join(root, name), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        X = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y = np.array(d[b"fine_labels"], dtype=np.int64)
        out.append((X, y))
    return out[0][0], out[0][1], out[1][0], out[1][1]


def load_tinyimagenet(cache_name: str = "tinyimagenet_uint8.pt"):
    """Pack the TinyImageNet folder layout into uint8 NHWC arrays (cached).

    The train split (500/class, 100k images) is packed; the per-class last 30%
    (sorted filename order, same as the pipeline protocol) is the test split.
    """
    from .features import FolderItems

    cache = os.path.join(FEATURE_DIR, cache_name)
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=False)
        X = d.get("X", d.get("Xtr"))
        y = d.get("y", d.get("ytr"))
        return X[d["train"]], y[d["train"]].numpy(), X[d["test"]], y[d["test"]].numpy()
    from PIL import Image

    from .paths import IMAGE_DATASETS
    cfg = IMAGE_DATASETS["tinyimagenet"]
    items = FolderItems(cfg["root"], cfg["n_classes"], cfg.get("images_subdir", ""))
    Xtr = np.zeros((len(items.items), 64, 64, 3), dtype=np.uint8)
    y = np.zeros(len(items.items), dtype=np.int64)
    train = np.array([i for i, s in enumerate(items.splits) if s == "train"], dtype=np.int64)
    test = np.array([i for i, s in enumerate(items.splits) if s == "test"], dtype=np.int64)
    for i in range(0, len(items.items), 2000):
        for j in range(i, min(i + 2000, len(items.items))):
            path, label = items.items[j]
            Xtr[j] = np.asarray(Image.open(path).convert("RGB").resize((64, 64)))
            y[j] = label
    d = {"X": torch.from_numpy(Xtr), "y": torch.from_numpy(y),
         "train": torch.from_numpy(train), "test": torch.from_numpy(test)}
    torch.save(d, cache)
    return Xtr[train], y[train], Xtr[test], y[test]


def _augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random translate + horizontal flip (on a [0,1] float GPU batch)."""
    if pad:
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    n, _, _, W = x.shape
    size = W - 2 * pad
    if size <= 0:  # no crop room
        return x
    dev = x.device
    i = torch.randint(0, 2 * pad + 1, (n,), device=dev)
    j = torch.randint(0, 2 * pad + 1, (n,), device=dev)
    rows = i[:, None] + torch.arange(size, device=dev)[None, :]
    cols = j[:, None] + torch.arange(size, device=dev)[None, :]
    x = x[torch.arange(n, device=dev)[:, None, None], :, rows[:, :, None], cols[:, None, :]]
    x = x.permute(0, 3, 1, 2).contiguous()  # NHWC gather -> NCHW
    flip = torch.rand(n, device=dev) < 0.5
    if flip.any():
        x[flip] = x[flip].flip(-1)
    return x


def train_network(X: torch.Tensor, y: torch.Tensor, n_classes: int, device,
                  seed: int, epochs: int = 60, batch: int = 256, lr: float = 3e-3,
                  wd: float = 5e-4, pad: int = 4, mean=CIFAR_MEAN, std=CIFAR_STD,
                  width: int = 64) -> SmallResNet:
    """Train SmallResNet on (X uint8 NHWC, y) with GPU augmentation."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X)
    if X.shape[-1] == 3:  # NHWC -> NCHW
        X = X.permute(0, 3, 1, 2).contiguous()
    Xg = X.to(device)
    yg = torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
    model = SmallResNet(n_classes, width=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    n = Xg.shape[0]
    steps = max(1, int(np.ceil(n / batch))) * epochs
    pct_start = min(0.3, max(0.05, 2.0 / steps))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=pct_start, anneal_strategy="cos")
    gen = torch.Generator().manual_seed(seed)
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            idx = perm[i:i + batch].to(device)
            xb = Xg[idx].float().div_(255.0)
            xb = _augment(xb, pad)
            xb = (xb - mean_t) / std_t
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=str(device).startswith("cuda")):
                loss = F.cross_entropy(model(xb), yg[idx])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    model.eval()
    return model


@torch.no_grad()
def penultimate_features(model: nn.Module, X: torch.Tensor, device,
                         mean=CIFAR_MEAN, std=CIFAR_STD, batch: int = 512,
                         sigma: float = 0.0, seed: int = 0) -> np.ndarray:
    """Pooled penultimate features for a uint8 NHWC array (optionally Gaussian noised)."""
    model.eval()
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X)
    if X.shape[-1] == 3:  # NHWC -> NCHW
        X = X.permute(0, 3, 1, 2).contiguous()
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    out = []
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch].to(device).float().div_(255.0)
        if sigma > 0:
            noise = torch.randn(xb.shape, generator=gen).to(device)
            xb = torch.clamp(xb + sigma * noise, 0.0, 1.0)
        xb = (xb - mean_t) / std_t
        f = model.forward_features(xb)
        out.append(F.normalize(f.float(), p=2, dim=1).cpu().numpy())
    return np.concatenate(out)


class MLPEncoder(nn.Module):
    """2-layer MLP encoder + linear head used for the synthetic pretrain suite."""

    def __init__(self, fdim: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(fdim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.head = nn.Linear(hidden, n_classes)
        self.feature_dim = hidden

    def forward_features(self, x):
        return self.enc(x)

    def forward_from_features(self, f):
        return self.head(f)

    def forward(self, x):
        return self.head(self.enc(x))


def train_mlp(X: np.ndarray, y: np.ndarray, n_classes: int, device, seed: int,
              epochs: int = 120, batch: int = 256, lr: float = 1e-3,
              hidden: int = 256) -> MLPEncoder:
    torch.manual_seed(seed)
    Xg = torch.as_tensor(X, dtype=torch.float32)
    yg = torch.as_tensor(y, dtype=torch.long)
    model = MLPEncoder(Xg.shape[1], n_classes, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(Xg)
    steps = max(1, int(np.ceil(n / batch))) * epochs
    pct_start = min(0.3, max(0.05, 2.0 / steps))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=pct_start)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(Xg[idx].to(device)), yg[idx].to(device))
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model


@torch.no_grad()
def mlp_features(model: MLPEncoder, X: np.ndarray, device) -> np.ndarray:
    X = torch.as_tensor(X, dtype=torch.float32)
    out = []
    for i in range(0, len(X), 8192):
        out.append(F.normalize(model.forward_features(X[i:i + 8192].to(device)), p=2, dim=1).cpu())
    return torch.cat(out).numpy()
