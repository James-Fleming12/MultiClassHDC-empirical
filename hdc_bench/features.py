"""Image feature extraction and caching for CIFAR-100 / TinyImageNet.

The raw datasets are the folder layouts built by the old NCD_HDC repo.  We
follow that repo's test-time protocol exactly: per class, the first 70% of the
sorted filenames are the train split and the last 30% the test split; images
are resized to 256 and center-cropped to 224 before the backbone; the pooled
features are L2-normalized.

Backbones:
  * ``dinov2_vitb14_reg``: frozen DINOv2 ViT-B/14-reg (LVD-142M, 768-d).  This
    is the strong self-supervised feature source used as the main configuration.
  * ``resnet18``: torchvision ResNet-18 ImageNet-1k (512-d).  A cheap supervised
    control backbone so conclusions are not DINOv2-specific.

Perturbations for the robustness test are applied to the *test* images only
(Gaussian pixel noise, std 0.05 / 0.10 in [0,1] pixel space) and their features
are cached separately; heads are always fit on clean train features.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .paths import FEATURE_DIR, IMAGE_DATASETS

warnings.filterwarnings("ignore", message=".*xFormers.*")

TRAIN_FRACTION = 0.7
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONES = ("dinov2_vitb14_reg", "resnet18", "dinov2_vits14_reg",
             "dino_vits16", "resnet50", "mobilenet_v2", "tinyvit_11m")
PERTURBATIONS = ("clean", "noise0.05", "noise0.10")


def _resize_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


class FolderItems:
    """(path, class_idx, split) items for a folder-layout dataset.

    ``images_subdir`` is appended under each class directory (empty for the
    CIFAR layout, "images" for the raw tiny-imagenet-200 train layout).
    """

    def __init__(self, root: str, n_classes: int, images_subdir: str = ""):
        assert os.path.isdir(root), f"missing dataset root {root}"
        self.root = root
        self.class_names = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        if len(self.class_names) != n_classes:
            raise RuntimeError(
                f"{root}: found {len(self.class_names)} class folders, expected {n_classes}")
        self.items: list[tuple[str, int]] = []
        self.splits: list[str] = []
        for ci, cname in enumerate(self.class_names):
            cdir = os.path.join(root, cname, images_subdir)
            files = sorted(
                f for f in os.listdir(cdir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".jpeg")))
            cut = int(len(files) * TRAIN_FRACTION)
            for f in files[:cut]:
                self.items.append((os.path.join(cdir, f), ci))
                self.splits.append("train")
            for f in files[cut:]:
                self.items.append((os.path.join(cdir, f), ci))
                self.splits.append("test")

    def split_indices(self, split: str) -> np.ndarray:
        return np.array([i for i, s in enumerate(self.splits) if s == split], dtype=np.int64)


class _ImageDataset(Dataset):
    def __init__(self, items, indices, sigma: float = 0.0, seed: int = 0):
        self.items = items
        self.indices = indices
        self.tf = _resize_transform()
        self.sigma = sigma
        self.seed = seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        path, label = self.items[idx]
        img = self.tf(Image.open(path).convert("RGB"))
        if self.sigma > 0:
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
            img = torch.clamp(img + self.sigma * torch.randn(img.shape, generator=g), 0.0, 1.0)
        return img, label


def load_backbone(name: str, device):
    """Return (frozen backbone, feature_dim)."""
    if name == "dinov2_vitb14_reg":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg",
                               verbose=False)
        dim = 768
    elif name == "dinov2_vits14_reg":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg",
                               verbose=False)
        dim = 384
    elif name == "dino_vits16":
        model = torch.hub.load("facebookresearch/dino:main", "dino_vits16",
                               verbose=False)
        dim = model.embed_dim
    elif name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        dim = 512
    elif name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        dim = 2048
    elif name == "mobilenet_v2":
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Identity()
        dim = 1280
    elif name == "tinyvit_11m":
        import timm
        model = timm.create_model("tiny_vit_11m_224.dist_in22k", pretrained=True,
                                  num_classes=0)
        dim = model.num_features
    else:
        raise ValueError(f"unknown backbone {name}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, dim


def _cache_path(dataset: str, backbone: str, perturb: str) -> str:
    return os.path.join(FEATURE_DIR, f"{dataset}__{backbone}__{perturb}.pt")


def extract_features(dataset: str, backbone: str, perturb: str = "clean",
                     device=None, batch: int = 256, workers: int = 12,
                     force: bool = False) -> str:
    """Extract and cache features; returns the cache path."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = _cache_path(dataset, backbone, perturb)
    if os.path.exists(path) and not force:
        return path
    cfg = IMAGE_DATASETS[dataset]
    items = FolderItems(cfg["root"], cfg["n_classes"], cfg.get("images_subdir", ""))
    model, dim = load_backbone(backbone, device)
    tf = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    sigma = float(perturb.split("noise")[1]) if perturb.startswith("noise") else 0.0
    splits = ["train", "test"] if perturb == "clean" else ["test"]
    out = {"dataset": dataset, "backbone": backbone, "perturb": perturb,
           "class_names": items.class_names, "fdim": dim}
    for split in splits:
        idx = items.split_indices(split)
        ds = _ImageDataset(items.items, idx, sigma=sigma, seed=0)
        loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                            pin_memory=True, persistent_workers=workers > 0)
        feats, labels = [], []
        with torch.no_grad():
            for imgs, labs in loader:
                imgs = tf(imgs.to(device, non_blocking=True))
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=device == "cuda" or str(device).startswith("cuda")):
                    f = model(imgs)
                f = F.normalize(f.float(), p=2, dim=1)
                feats.append(f.cpu())
                labels.append(labs)
        out[f"X_{split}"] = torch.cat(feats)
        out[f"y_{split}"] = torch.cat(labels)
        print(f"[features] {dataset}/{backbone}/{perturb} {split}: "
              f"{tuple(out[f'X_{split}'].shape)}")
    torch.save(out, path)
    return path


def load_features(dataset: str, backbone: str, perturb: str = "clean", device="cpu"):
    path = _cache_path(dataset, backbone, perturb)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device, weights_only=False)


def has_perturb(dataset: str, backbone: str, perturb: str) -> bool:
    return os.path.exists(_cache_path(dataset, backbone, perturb))
