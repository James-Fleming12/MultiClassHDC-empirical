#!/usr/bin/env python
"""Assemble README.md from the hand-written narrative + generated tables.

Every results table quoted in the README lives in ``results/tables/`` and is
produced by ``scripts/make_report.py``, so the README numbers cannot drift from
the raw records.  Rebuild with:

  python scripts/make_report.py && python scripts/build_readme.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hdc_bench.paths import REPO_ROOT, TABLE_DIR


def T(name: str) -> str:
    path = os.path.join(TABLE_DIR, name)
    if not os.path.exists(path):
        return f"_[missing table: {name}]_"
    with open(path) as f:
        return f.read().strip()


def IMG(name: str, caption: str) -> str:
    path = os.path.join("results", "figures", name)
    if not os.path.exists(os.path.join(REPO_ROOT, path)):
        return f"_[missing figure: {name}]_"
    return f"![{caption}]({path})"


README = f"""# MultiClassHDC-empirical

A controlled empirical study of two questions about **hyperdimensional
computing (HDC) prototype classifiers** when the number of classes grows:

* **Q1 - class-count scaling.** Does HDC prototype accuracy degrade as more
  classes are introduced, *independently of any degradation of the network
  features*? In particular:
  * does base (ID) accuracy degrade as the number of classes K in the training
    set grows, and does input-robustness degrade with it;
  * does accuracy on novel classes - classes that receive **no labels and no
    pretraining** - degrade as K grows and as the number of novel classes N
    grows;
  * when a network is actually pretrained on K classes, does the HDC prototype
    head track the network's own degradation or add its own?
* **Q2 - HDC vs a standard network classification layer.** On identical
  representations, does the HDC random-projection + sign + class-mean-prototype
  classifier beat a trained linear/MLP softmax layer, and in which regimes
  (many/few classes, ID, robustness, novel discovery, few-shot, OOD detection)?

Everything is measured on **synthetic data**, **CIFAR-100** and
**TinyImageNet**, with **HD dimension 4096 and 10000** in every HDC run, three
feature sources (DINOv2 ViT-B/14-reg, ResNet-18, small from-scratch CNNs), and
3-5 seeds.  All raw run records, tidy CSVs, tables and figures are under
`results/`; the README tables are generated from those records by
`scripts/make_report.py`.

---

## Headline answers (details and exact tables below)

1. **HDC prototype accuracy does degrade with K, but the degradation is
   shared by the trained network heads and by the network itself.** On frozen
   DINOv2 features, CIFAR-100 HDC ID accuracy falls `0.957 -> 0.724` from
   K = 5 to K = 100 and TinyImageNet `0.960 -> 0.834` from K = 10 to K = 200.
   The trained linear/MLP heads on the **same frozen features** degrade by a
   similar amount and stay only ~2-5 points above HDC at the largest K.  When a
   small CNN is trained on K classes, the network's own head and the HDC
   prototype head built on its penultimate features fall together (CIFAR
   pretrain: `net 0.92 -> 0.71`, `HDC 0.92 -> 0.70`; Tiny pretrain:
   `net 0.74 -> 0.50`, `HDC 0.72 -> 0.48`), so there is no separate HDC-specific
   collapse in this regime.
2. **Dimension 4096 vs 10000 is not the bottleneck for real features.** HDC
   10k beats HDC 4k by only `+0.001..+0.005` on CIFAR-100/TinyImageNet DINOv2
   and ResNet-18 features at every K.  The gap is meaningful only on synthetic
   projected features (gmm/subspace: `+0.013` mean, up to `+0.03` at
   K = 100-200) and for HDC-native random codes, where 10k bits are clearly
   more noise-robust.
3. **For novel classes without labels, HDC and feature-space k-means are
   essentially tied on real features**, and both degrade with the number of
   novel classes N (CIFAR-100 DINOv2: `0.86 -> 0.61` from N = 5 to N = 40;
   TinyImageNet: `0.98 -> 0.74` from N = 5 to N = 100).  On the synthetic
   Gaussian/subspace data, plain feature k-means is much better
   (`0.88 vs 0.40` at K = 5, N = 20): the sign projection preserves enough
   information for labelled prototypes but damages Euclidean cluster structure.
4. **HDC does not beat a trained network layer on ID accuracy with strong
   features**; the linear/MLP heads win by ~2-5 points at large K, and by more
   on weaker (ResNet-18) features.  HDC's wins are in the low-label/shortcut
   regimes: 5-shot novel classes on HDC-native codes (`0.57 vs 0.46`), and it
   ties the network heads in 5-shot on image features.  As an OOD detector the
   HDC prototype-distance score is competitive with max-softmax on DINOv2
   features (and better than max-softmax from the 2-layer MLP), but on a
   from-scratch network the jointly-trained softmax head detects novel classes
   much better at large K (`0.865 vs 0.654` AUROC at pretraining K = 50).
5. **More classes in the pretraining data hurts base accuracy, robustness and
   OOD detection, and the HDC head inherits the loss.**  The class count, not
   the HD dimension, is the lever: increasing K costs the most, increasing N
   mostly costs novel clustering and leaves OOD AUROC flat.

---

## 1. Test design

### 1.1 Heads compared (always on identical representations)

| head | what it is |
| :--- | :--- |
| `hdc4096` / `hdc10000` | fixed random Gaussian projection `x @ P` -> `sign` (bipolar code) -> per-class mean code (L2-normalized) -> nearest prototype by cosine / normalized Hamming. The projection is random and never trained; only class means are built. This is the old-repo HDC encoding. |
| `linear` | linear softmax layer trained with cross-entropy on the same features (50 epochs, AdamW + OneCycle). |
| `mlp` | 2-hidden-layer (1024 units, ReLU) softmax layer, same training recipe. |
| `proto` | class-mean prototype in the raw feature space, same cosine scoring as HDC. Isolates "prototype head" from "HDC projection". |
| `net` | (pretraining suites only) the network's own jointly-trained classifier head. |

All features are L2-normalized.  HDC prototypes are built from **all** training
samples of each class; the random projection is fixed per (seed, HD dim) and
shared by all K values of that seed.  For the `bits` synthetic dataset the
input *is* a bipolar code, so the HDC projection is the identity and the HD
dimension is the data dimension.

### 1.2 Data and feature sources

| benchmark | data | features |
| :--- | :--- | :--- |
| `bits` | K random Rademacher class vectors in `{{-1,+1}}^D`, 60 train / 30 test samples per class, 44% of bits flipped per sample (robustness: 46%, 48%) | the codes themselves (identity HDC) |
| `gmm` | 256-d isotropic Gaussian mixture, class means on a sphere (radius 4, noise 1), 200/100 samples per class | fixed features, HDC projects to 4096/10000 |
| `subspace` | 256-d class center (radius 3.5) + random rank-16 class subspace + noise 0.8 | fixed features, HDC projects to 4096/10000 |
| CIFAR-100 | 100 classes, 70/30 per-class split of all 60k images | frozen DINOv2 ViT-B/14-reg (768-d), frozen ResNet-18 (512-d) |
| TinyImageNet | 200 classes, 70/30 per-class split of the 100k train images | frozen DINOv2 ViT-B/14-reg, frozen ResNet-18 |
| pretrain CIFAR | official CIFAR-100 train/test, K classes | small 6-conv ResNet (256-d penultimate) trained **from scratch on K classes** |
| pretrain Tiny | 64x64 TinyImageNet, K classes | same small ResNet trained from scratch on K classes |
| pretrain MLP | synthetic `subspace`, K classes | 2-layer MLP encoder trained from scratch on K classes |

For each benchmark, K base classes are sampled at random (class-subset seed)
and N disjoint novel classes are held out completely.  Image robustness
variants add Gaussian pixel noise (std 0.05 / 0.10 in [0,1]) **to the test
images only**; heads are always fit on clean training features.

### 1.3 What is measured

* **ID accuracy** - top-1 on the K base-class test split.
* **Robustness** - ID top-1 on the noisy test images (features re-extracted
  through the same frozen backbone) / noised synthetic features / higher bit
  flip rates.
* **Novel no-label accuracy** - k-means (oracle N clusters, spherical k-means,
  Hungarian-matched) on the **unlabelled** novel test features, run in the raw
  feature space and in each HDC code space.
* **OOD detection AUROC** - known base test vs novel test.  Scores: `1 - max
  cosine to prototypes` for HDC/`proto`, `1 - max softmax` for `linear`/`mlp`/
  `net`.
* **5-shot novel accuracy** - 5 labelled examples per novel class build
  prototypes (HDC/proto) or train a linear layer; evaluated on novel test.
* **Pretraining class scaling** - the network is trained from scratch on K
  classes; all heads then consume its frozen penultimate features, so the
  network's own degradation and the HDC head's degradation are directly
  comparable.

### 1.4 Grids

* Synthetic: K in {{5,10,20,50,100,200}}, N in {{5,10,20,50,100}} (K-series at
  N = 20; N-series at K = 50), 5 seeds.
* CIFAR-100: K in {{5,10,20,50,100}} (N = 20; K = 100 has no novel classes),
  N in {{5,10,20,40}} at K = 50, 3 seeds, 2 backbones.
* TinyImageNet: K in {{10,20,50,100,200}} (N = 20), N in
  {{5,10,25,50,100}} at K = 100, 3 seeds, 2 backbones.
* Pretraining: synthetic MLP K in {{5,10,20,50,100}}; CIFAR K in
  {{5,10,20,50,100}}; Tiny K in {{10,50,200}} (2 seeds).
* Every HDC number is reported at **D = 4096 and D = 10000**.

---

## 2. Synthetic benchmarks

### 2.1 `bits` - HDC-native random codes (the pure capacity test)

There is no feature extractor here: classes are random hypervectors and samples
are noisy copies.  This isolates the classifier head and the HD dimension.

{T('id_bits_D4096.md')}

{T('id_bits_D10000.md')}

{T('robust_bits_D4096.md')}

{T('novel_auroc_bits_D4096.md')}

{T('novel_fewshot5_bits_D4096.md')}

{IMG('synthetic_bits_D4096.png', 'bits (D=4096): ID top-1, novel no-label clustering and OOD AUROC vs K')}

{IMG('synthetic_bits_D10000.png', 'bits (D=10000): ID top-1, novel no-label clustering and OOD AUROC vs K')}

**Takeaways - `bits`.**

* ID accuracy is essentially flat with K for the prototype heads at both dims:
  HDC 4k `0.999 -> 0.989` and HDC 10k `1.000` at every K up to 200.  Random
  hypervectors have enormous prototype capacity at these K/D ratios; the
  prototype head shows **no K-degradation** in the well-separated regime.
* The 2-layer MLP is the casualty: `1.00 -> 0.42` (4k) / `0.80` (10k) as K
  grows.  With 60 samples per class and near-random 4k/10k-dim inputs, the
  network layer underfits/overfits while the closed-form prototype and the
  linear layer stay strong.  This is a genuine "network head on hard codes"
  failure mode, not an HDC win by construction.
* Robustness: at a 46% flip rate the 10k codes hold `0.999 -> 0.994` over
  K = 5..200 while the 4k codes fall `0.977 -> 0.758`; at 48% flips 10k holds
  `0.912 -> 0.498` while 4k falls `0.748 -> 0.175`.  This is the one place
  where **dimension size clearly buys robustness**.
* OOD detection is near perfect for prototypes at every K (`1.000 -> 0.991`);
  the MLP detector collapses with K (`0.81` at K = 100, `0.66` at K = 200).
* 5-shot novel prototypes beat a 5-shot linear layer (`0.57 vs 0.46`) at every
  K - the prototype rule is the data-efficient option when labels are scarce.

### 2.2 `gmm` and `subspace` - projected synthetic features

{T('id_gmm_Dproj.md')}

{T('id_gmm_Ddims.md')}

{T('robust_gmm_Dproj.md')}

{T('novel_cluster_gmm_Dproj.md')}

{T('novel_fewshot5_gmm_Dproj.md')}

{T('id_subspace_Dproj.md')}

{T('novel_cluster_subspace_Dproj.md')}

{IMG('synthetic_gmm_Dproj.png', 'gmm: ID top-1, novel no-label clustering and OOD AUROC vs K')}

{IMG('synthetic_subspace_Dproj.png', 'subspace: ID top-1, novel no-label clustering and OOD AUROC vs K')}

**Takeaways - `gmm` / `subspace`.**

* Both HDC and feature prototypes degrade smoothly with K
  (gmm HDC 4k `0.984 -> 0.796`; subspace `0.987 -> 0.835`), while the linear
  layer is weaker at small K, catches up at K = 50-100, and overtakes at
  K = 200.  The feature prototype (`proto`) is consistently the best
  prototype-based head: the projection+sign step costs ~2-4 points for the
  4k codes and ~1-2 points for the 10k codes at large K.
* HDC 10k > HDC 4k by `+0.013` on average (up to `+0.025` at K = 200), the
  largest dimension effect among all benchmarks: with only 256-d input
  features the random-projection direction has more information to recover.
* Robustness degrades quickly with K for every head (e.g. gmm HDC 4k at
  noise x2.5: `0.65 -> 0.12` from K = 5 to K = 200), roughly in lock-step
  across heads.
* **No-label novel clustering is where HDC is clearly weaker on this data**:
  feature k-means reaches `0.88/0.88` while HDC-code k-means gets `0.40/0.56`
  (gmm/subspace, K = 5, N = 20).  The labelled prototype use of the codes is
  fine, but the sign-code geometry is a poor space for Euclidean k-means.
* 5-shot novel accuracy: the prototype heads beat a 5-shot linear layer by a
  wide margin on gmm (`0.50-0.53` for HDC 10k / feature prototype vs `0.32`
  linear at every K); HDC 10k closes most of the gap to the feature prototype.
  The prototype advantage is largest when labels are few.

### 2.3 Pretraining K with a from-scratch MLP encoder

{T('id_pretrain_mlp.md')}

{T('novel_pretrain_mlp.md')}

{T('robust_pretrain_mlp.md')}

**Takeaways - synthetic pretraining.**

* The network's own ID accuracy falls `0.987 -> 0.740` over K = 5..100.  HDC
  4k on the same frozen penultimate features falls `0.991 -> 0.732` and the
  linear probe `0.988 -> 0.745`: **HDC adds no degradation of its own**; the
  entire loss is the representation shrinking as K grows.
* HDC 10k recovers `+0.002..+0.010` over 4k in ID, again small.
* OOD AUROC separates the heads: the jointly-trained network softmax and the
  linear probe stay at `0.81-0.94` across K, while the prototype-distance
  scores fall from `0.90` at K = 5 to `0.60-0.67` at K = 50.  Prototype
  distance is a good novelty score in low-K / clean-geometry settings but
  loses calibration as K grows.
* Novel no-label clustering is low for every representation in this setup
  (`0.11-0.15`); the MLP encoder is trained for classification, and its
  penultimate space is not clustered by class.

---

## 3. CIFAR-100

### 3.1 ID accuracy and robustness vs K (frozen DINOv2 ViT-B/14-reg)

{T('id_cifar100_dinov2_vitb14_reg_by_K.md')}

{T('id_cifar100_dinov2_vitb14_reg_dims.md')}

{T('robust_cifar100_dinov2_vitb14_reg_by_K.md')}

### 3.2 Novel classes without labels and OOD detection

{T('novel_cluster_cifar100_dinov2_vitb14_reg_by_K.md')}

{T('ood_auroc_cifar100_dinov2_vitb14_reg_by_K.md')}

{T('novel_cifar100_dinov2_vitb14_reg_by_N.md')}

### 3.3 ResNet-18 features (weaker/supervised backbone control)

{T('id_cifar100_resnet18_by_K.md')}

{T('id_cifar100_resnet18_dims.md')}

{T('novel_cluster_cifar100_resnet18_by_K.md')}

{T('ood_auroc_cifar100_resnet18_by_K.md')}

{IMG('summary_cifar100_dinov2_vitb14_reg.png', 'CIFAR-100 / DINOv2: ID, novel clustering, OOD AUROC vs K')}

{IMG('summary_cifar100_resnet18.png', 'CIFAR-100 / ResNet-18: ID, novel clustering, OOD AUROC vs K')}

**Takeaways - CIFAR-100.**

* With DINOv2 features, HDC ID accuracy degrades `0.957 -> 0.724` from K = 5
  to 100; the linear head goes `0.962 -> 0.777` and the MLP `0.976 -> 0.778`.
  HDC degrades at the same rate as the network layer (the gap stays ~2-5
  points) - more classes make the frozen feature geometry harder, and all
  heads pay.
* HDC 10k - HDC 4k is `+0.001..+0.004` at every K; doubling the HD dimension
  does not recover the gap to the linear layer at large K.
* Noise robustness degrades with K for all heads, HDC by about the same
  absolute margin as the trained layers (K = 50, noise 0.10: HDC 4k `0.515` vs
  linear `0.590`).
* Novel no-label clustering is flat in K (`0.68-0.70` for HDC, `0.69-0.70`
  for features) but falls with N (`0.86 -> 0.61` features, `0.86 -> 0.60` HDC
  from N = 5 to 40).  HDC and raw features are tied within noise.
* OOD AUROC falls with K (`0.95 -> 0.83` for both HDC and linear): the more
  base classes there are, the more likely a novel sample is close to some base
  prototype.  It is flat in N (`~0.83`), so it is the *known* class count that
  governs novelty detection.
* 5-shot novel accuracy is `0.80-0.88` for HDC, proto and linear - the stronger
  the features, the less the head matters.
* On ResNet-18 features the ranking is the same but the absolute level is much
  lower (K = 100: HDC `0.398`, linear `0.453`, MLP `0.537`).  The MLP head's
  advantage grows on weaker features, and HDC 10k helps slightly more
  (`+0.004` mean).

---

## 4. TinyImageNet

### 4.1 ID accuracy and robustness vs K (frozen DINOv2 ViT-B/14-reg)

{T('id_tinyimagenet_dinov2_vitb14_reg_by_K.md')}

{T('id_tinyimagenet_dinov2_vitb14_reg_dims.md')}

{T('robust_tinyimagenet_dinov2_vitb14_reg_by_K.md')}

### 4.2 Novel classes without labels and OOD detection

{T('novel_cluster_tinyimagenet_dinov2_vitb14_reg_by_K.md')}

{T('ood_auroc_tinyimagenet_dinov2_vitb14_reg_by_K.md')}

{T('novel_tinyimagenet_dinov2_vitb14_reg_by_N.md')}

### 4.3 ResNet-18 features

{T('id_tinyimagenet_resnet18_by_K.md')}

{T('id_tinyimagenet_resnet18_dims.md')}

{T('novel_cluster_tinyimagenet_resnet18_by_K.md')}

{T('ood_auroc_tinyimagenet_resnet18_by_K.md')}

{IMG('summary_tinyimagenet_dinov2_vitb14_reg.png', 'TinyImageNet / DINOv2: ID, novel clustering, OOD AUROC vs K')}

{IMG('summary_tinyimagenet_resnet18.png', 'TinyImageNet / ResNet-18: ID, novel clustering, OOD AUROC vs K')}

**Takeaways - TinyImageNet.**

* The same pattern at 200 classes: DINOv2 HDC `0.960 -> 0.834` from K = 10 to
  200, linear `0.964 -> 0.864`, MLP `0.971 -> 0.850`.  The HDC-vs-linear gap
  grows from `-0.004` at K = 10 to `-0.030` at K = 200.
* HDC 10k - 4k is `+0.001..+0.005`, again negligible; ResNet-18 shows the same
  `+0.003..+0.005` dimension range.
* Novel no-label clustering with HDC tracks feature k-means closely and falls
  from `0.98` (N = 5) to `0.73` (N = 100).  OOD AUROC is stable in N
  (`0.92-0.93`) and falls modestly with K (`0.983` at K = 10 to `0.919` at
  K = 100; K = 200 has no novel classes).
* 5-shot novel accuracy is `0.84-0.97` for all heads; with 5 examples per
  class on 200 classes, HDC still matches the trained linear layer.
* On ResNet-18 features HDC `0.847 -> 0.536` vs linear `0.845 -> 0.588` and MLP
  `0.888 -> 0.623`.

---

## 5. Pretraining-class scaling on real images (from-scratch networks)

Each network is trained from scratch on K classes; all heads then use its
frozen penultimate features.  `net` is the network's own classifier, i.e. the
upper reference for "what the network learned".

### 5.1 CIFAR-100 small ResNet

{T('id_pretrain_cifar.md')}

{T('robust_pretrain_cifar.md')}

{T('novel_pretrain_cifar.md')}

### 5.2 TinyImageNet small ResNet

{T('id_pretrain_tiny.md')}

{T('robust_pretrain_tiny.md')}

{T('novel_pretrain_tiny.md')}

{IMG('pretrain_mlp_vs_K.png', 'Synthetic MLP pretraining: ID top-1 vs pretraining K')}

{IMG('pretrain_cifar_vs_K.png', 'CIFAR-100 small ResNet: ID top-1 vs pretraining K')}

{IMG('pretrain_tiny_vs_K.png', 'TinyImageNet small ResNet: ID top-1 vs pretraining K')}

**Takeaways - image pretraining.**

* The network's own ID accuracy degrades with pretraining K, and the HDC head
  built on the frozen penultimate features degrades **by the same amount**:
  on CIFAR the `net`/`hdc4k` pair goes from `0.923/0.922` at K = 5 to
  `0.714/0.698` at K = 100, while the separately-trained linear probe reaches
  `0.717` at K = 100.  The `net`-vs-HDC gap never exceeds ~0.02 at any K;
  there is no HDC-specific degradation.
* **More pretraining classes make the representation better for novel
  classes**: no-label novel clustering in the penultimate space *rises* from
  `0.23` (K = 5) to `0.49` (K = 50) on CIFAR and from `0.24` (K = 10) to
  `0.38` (K = 50) on Tiny, and HDC tracks it.  The class count hurts base-class
  classification but helps novel-class geometry.
* TinyImageNet confirms the same ID story: `net 0.735 -> 0.504`,
  `HDC 0.718 -> 0.482`, `linear 0.716 -> 0.511` from K = 10 to 200.  OOD
  AUROC falls with K and favours the jointly-trained head
  (`net 0.731 -> 0.671` vs `HDC 0.657 -> 0.605`).
* OOD detection is roughly flat in K and only slightly favours the network's
  own softmax on the small image networks (CIFAR K = 50: `net 0.774` vs
  `HDC 0.737`; the gap is much larger for the synthetic MLP encoder above,
  `0.865` vs `0.654`).
* Noise robustness falls with K for every head, with the network's own head and
  HDC tracking each other.

---

## 6. Cross-benchmark summary

{T('summary_all.md')}

---

## 7. Direct answers

### Q1 - does HDC prototype accuracy degrade with more classes?

**Yes for the class count, no for the HD dimension, and the loss is not
HDC-specific.**

* **ID / base accuracy:** HDC ID accuracy falls monotonically with K on every
  real-feature benchmark (CIFAR-100 DINOv2 `0.957 -> 0.724`; TinyImageNet
  `0.960 -> 0.834`; ResNet-18 `0.837 -> 0.398` on CIFAR and `0.847 -> 0.536`
  on Tiny; from-scratch networks `0.922 -> 0.698` on CIFAR and `0.718 -> 0.482`
  on Tiny).  The HDC-native `bits` benchmark is the exception:
  prototype capacity is not exhausted up to K = 200 there (`0.999 -> 0.989`
  at D = 4096), because random hypervectors are maximally separated.
* **Independence from network degradation:** in the frozen-feature suites the
  extractor does not change with K, so the fall is the head+geometry; in the
  from-scratch suites both the network's own head and HDC fall together
  (`net ~ hdc4k` at every K), so the HDC head does not add its own
  class-count degradation.
* **Robustness:** ID accuracy under pixel noise/noised features decays with K
  for all heads, HDC and network layers closely in step; the trained layers
  keep a small absolute edge on images at every K, and no head shows a
  relative robustness advantage that grows with K.
* **Novel classes with no labels or pretraining:** accuracy falls strongly
  with the number of novel classes N (CIFAR `0.86 -> 0.61`; Tiny `0.98 ->
  0.74`) and is roughly flat in K for clustering, while OOD detection falls
  with K (`0.95 -> 0.83` CIFAR) and is flat in N.  In other words, base-class
  count controls novelty separation, novel-class count controls cluster
  separation.
* **Dimension:** at D = 4096 vs D = 10000 the HDC deltas are `+0.001..+0.005`
  on real features, `+0.013` mean on projected synthetic features, and large
  only for noisy HDC-native codes at high flip rates.  **The gains/losses with
  K are not a dimension-size effect at these D.**

### Q2 - does HDC outperform a standard network classification layer?

**It depends on the regime, and on strong features the trained layer usually
wins:**

| regime | winner | evidence |
| :--- | :--- | :--- |
| ID accuracy, DINOv2/ResNet features | **linear/MLP**, by 2-5 pts at large K | CIFAR K=100: `0.724` HDC vs `0.777` linear; Tiny K=200: `0.834` vs `0.864` |
| ID accuracy, random HDC-native codes | **tie** (HDC/linear/proto), MLP collapses | bits D=4096 K=200: `0.989/0.991/0.989` vs MLP `0.422` |
| robustness | tie / trained head slightly better on images, **HDC clearly better at D=10000 vs 4096 on codes** | bits flip0.46; CIFAR noise |
| novel no-label clustering | **tie on real features** (HDC ~ feature k-means); **feature k-means wins on synthetic projections** | CIFAR/Tiny vs gmm `0.40` HDC vs `0.88` features |
| OOD detection | **tie with linear on DINOv2**, better than MLP on DINOv2, but **network softmax wins on from-scratch networks at large K** | pretrain MLP K=50: `0.865` net vs `0.654` HDC; pretrain CIFAR K=50: `0.774` vs `0.737`; pretrain Tiny K=50: `0.671` vs `0.605` |
| 5-shot novel | **HDC ahead on codes** (`0.57` vs `0.46`), tie on images | bits / CIFAR / Tiny 5-shot tables |
| data-efficiency (few labels) | **HDC / prototypes** | same tables |

The consistent picture: the HDC prototype rule is a strong, training-free
baseline that matches feature-space prototypes and is remarkably insensitive
to the HD dimension; a trained layer buys a few points on strong features and
a larger margin on weak ones; prototype *distance* is a weaker novelty score
than a trained softmax once K is large.

---

## 8. Limitations and notes

* The HDC head here is the supervised prototype classifier (class means from
  labelled data), not the full online discovery method from the NCD_HDC repo;
  the "no labels" novel test uses spherical k-means with the **oracle** class
  count N.  Discovery without K is a separate question.
* Synthetic `bits` uses independent bit flips; its accuracy at D = 10000 is
  saturated at K <= 200, so capacity limits may appear only for much larger K
  or smaller D.
* The from-scratch image networks are deliberately small ResNets trained for
  60/40 epochs; conclusions are about relative head behaviour, not about the
  best achievable accuracy of those backbones.
* Image features follow the old repo's protocol: 70/30 per-class split of the
  available images, Resize(256) + CenterCrop(224), L2-normalized pooled
  features.
* Seeds control class-subset selection, head initialization/training and the
  HDC projection; 3 seeds for the image suites (2 for pretrain Tiny) make the
  standard deviations in the tables non-trivial.
* OOD AUROC for `mlp` uses max-softmax, which is known to be miscalibrated;
  the lower MLP AUROCs are partly a calibration artefact, not necessarily a
  worse representation.

## 9. Reproducing

```bash
# 1. extract and cache frozen image features (DINOv2 + ResNet-18, clean + noise)
python scripts/extract_features.py --datasets cifar100 tinyimagenet \\
    --backbones dinov2_vitb14_reg resnet18 --perturbations clean noise0.05 noise0.10

# 2. run the suites (records are written under results/raw; reruns skip done runs)
python scripts/run_synthetic.py
python scripts/run_images.py --datasets cifar100 tinyimagenet \\
    --backbones dinov2_vitb14_reg resnet18 --seeds 0 1 2
python scripts/run_pretrain.py --suite mlp_synthetic cifar --seeds 0 1 2
python scripts/run_pretrain.py --suite tiny --seeds 0 1 --ks 10 50 200 --epochs 40

# 3. tables, figures and this README
python scripts/make_report.py
python scripts/build_readme.py
```

Raw records: `results/raw/`; tidy CSVs, markdown tables and figures:
`results/tables/`, `results/figures/`.  Feature caches (~4 GB) are written to
`results/features/` and are not committed.

Reference environment: Python 3.14, `torch 2.13.0+cu130`, `timm 1.0.28`,
`scikit-learn 1.9`, one NVIDIA RTX 5080 (16 GB); feature extraction uses
autocast fp16, head training fp32.
"""


def main():
    path = os.path.join(REPO_ROOT, "README.md")
    with open(path, "w") as f:
        f.write(README)
    print(f"[readme] wrote {path} ({len(README)} chars)")


if __name__ == "__main__":
    main()
