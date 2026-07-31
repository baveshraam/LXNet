# LXNet — Lightweight CNN for Lung Disease Classification

A replication of Humayan et al., *A Lightweight CNN for Lung Disease Classification from Chest X-ray with XAI-based Interpretability* (PLOS One, 2026).

**Aim.** Reproduce the paper's central claim: a ~0.35M-parameter CNN that classifies 9 lung-disease categories from chest X-rays as well as ImageNet backbones 50–70× its size, with CLAHE preprocessing, 5-fold cross-validation, and Grad-CAM / Score-CAM / LIME interpretability.

## The finding

The dataset contains the same X-ray many times over, re-saved under different
filenames. After removing byte-identical files there are still **6,657 images but
only 4,635 visually distinct X-rays** — 30% redundancy.

Split those at random, as the paper does, and **40.4% of the test set has a
near-duplicate sitting in the training set**. The model is then scored largely on
pictures it has already memorised: every architecture lands at 98–99% and the
ranking between them stops carrying information.

This repository runs both protocols over identical data:

| Arm | Split | What it measures |
|---|---|---|
| **Group-aware** | every copy of an X-ray confined to one split | generalisation to unseen images |
| **Random** | images dealt out independently (paper protocol) | reproduces the published-style number |

Measured results, figures and the per-model gap: **[docs/RESULTS.md](docs/RESULTS.md)**.

## Status

| Component | State |
|---|---|
| Data indexing, dedup, splitting | done, 20 tests |
| Near-duplicate grouping + leak-free splits | done, 9 tests |
| CLAHE preprocessing | done, 14 tests |
| LXNet + 3 baselines | done, 356,585 params, 16 tests |
| Cached tf.data pipeline | done, 10 tests |
| Metrics + Wilcoxon | done, 14 tests |
| Grad-CAM / Score-CAM / LIME | done, 13 tests |
| Figures + results table | done, 8 tests |
| Interpretability panel | done, 4 tests |
| Full training run | see `runs/`, results in `docs/` |

## Dataset

[X-Ray Lung Diseases Images (9 classes)](https://www.kaggle.com/datasets/fernando2rad/x-ray-lung-diseases-images-9-classes) — 6,743 images, extracted to `dataset/` (gitignored).

| Label | Class | Images |
|---|---|---|
| 0 | Normal | 1340 |
| 1 | Pneumonia | 1060 |
| 2 | Higher Density | 678 |
| 3 | Lower Density | 629 |
| 4 | Obstructive Pulmonary | 644 |
| 5 | Degenerative Infectious | 594 |
| 6 | Encapsulated Lesions | 658 |
| 7 | Mediastinal Changes | 596 |
| 8 | Chest Changes | 544 |

Audited on load: 0 corrupt files, 40 distinct resolutions (6263 are 450×450), mixed L/RGB/RGBA, **74 groups of byte-identical duplicates**, and 1 image filed under two different classes.

## Deliberate deviations from the paper

These are choices where following the paper exactly would produce a number that is not real. Each is implemented, tested, and reported alongside the paper's approach rather than silently substituted.

**1. Deduplicate before splitting.** 85 files are byte-identical copies of another image. Splitting naively puts the same X-ray in train and test, so the model is scored on images it memorised. `lxnet.data.dedupe` collapses each group to one copy. The single cross-class duplicate — the same pixels labelled both *Pneumonia* and *Lower Density* — is dropped entirely, since keeping either copy trains on a coin flip.

**2. Balance after splitting, not before.** The paper oversamples minority classes to equalise the dataset, then splits. That places duplicated minority images on both sides of the split, which inflates recall on exactly the rare classes the balancing was meant to help. Here, the split comes first and oversampling touches the training fold only (`_oversample_indices`), inside each CV fold.

**3. Global average pooling instead of Flatten.** Required to hit the stated parameter budget: flattening a 28×28×128 feature map into a 256-unit dense layer costs 25.7M weights on its own — 70× the entire published model. With GAP the architecture lands at **356,585 parameters**, matching the paper's "approximately 0.35 million".

**4. Conservative augmentation.** Horizontal flip, mild brightness/contrast. No vertical flip or large rotation: an inverted chest X-ray is not a plausible clinical image, and mirroring is itself diagnostically meaningful (dextrocardia, situs inversus).

**5. Split whole near-duplicate groups, not images.** Exact hashing catches only byte-identical files. Re-saving an X-ray at a different JPEG quality changes every byte while leaving the picture the same, and this dataset is full of those: 6,657 unique files reduce to 4,635 visually distinct images. Splitting per-image therefore leaks 40.4% of the test set. `group_aware_split` and `group_aware_folds` deal out whole perceptual groups instead, so no image in evaluation has a twin in training. `--split-mode random` reproduces the leaky protocol for comparison.

Why exact-match grouping: chest X-rays are near-identical by construction — same anatomy, same framing, low contrast — so any Hamming tolerance chains distinct images together transitively. Largest resulting group by threshold:

| Hamming threshold | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| largest group | 21 | 143 | 374 | **1,680** |

At threshold 3 that group spans 8 of the 9 classes with a median internal RMSE of 47 grey levels — a transitive-closure artefact, not duplicates. So `GROUP_MAX_DISTANCE = 0`: it under-counts mild recompression but never merges different images.

Expect these to *lower* the headline accuracy relative to the paper's 96.1%. That is the point — the gap is the measurement.

## Setup

```bash
conda activate lxnet          # py3.10, TF 2.10.1, cudatoolkit 11.2, cudnn 8.1
pip install -r requirements.txt
```

GPU note: TF 2.10 finds CUDA only when the env's `Library/bin` is on `PATH`, which `conda activate` handles. Invoking `envs/lxnet/python.exe` by absolute path silently falls back to CPU.

## Usage

```bash
# honest protocol: all models hold-out, LXNet cross-validated
python -m lxnet.train --out-dir runs/grouped --split-mode grouped --cv-models LXNet

# paper protocol, for comparison
python -m lxnet.train --out-dir runs/random --split-mode random --models LXNet --cv-models LXNet

# figures + docs/RESULTS.md
python -m lxnet.report

# interpretability panel: one row per class, Grad-CAM / Score-CAM / LIME
python -m lxnet.explain --run-dir runs/grouped --out docs/fig_xai.png

# quick sanity check
python -m lxnet.train --models LXNet --epochs 2 --limit 300 --skip-cv --out-dir runs/smoke
```

| Flag | Default | Meaning |
|---|---|---|
| `--data-root` | `dataset` | dataset directory |
| `--out-dir` | `runs/latest` | results, checkpoints, cache |
| `--models` | all four | subset to train |
| `--epochs` | 40 | max epochs (early stopping, patience 8) |
| `--folds` | 5 | CV folds |
| `--limit` | — | subsample for smoke tests |
| `--split-mode` | `grouped` | `grouped` = leak-free; `random` = paper protocol |
| `--cv-models` | all `--models` | subset to cross-validate |

Preprocessing is cached to `runs/<name>/preprocessed.npz` as CLAHE'd grayscale uint8 — 338 MB for the full dataset, versus 4 GB if kept as float32 RGB. Delete it to force a re-preprocess.

## Tests

```bash
pytest -q               # 108 tests
pytest -m "not slow"    # skips baseline construction (downloads ImageNet weights)
```

The suite targets the failure modes that stay silent: train/test leakage, image–label misalignment, augmentation reaching evaluation data, macro-F1 hiding an ignored minority class, and CAM maps that are constant or class-independent.

## Layout

```
lxnet/
  data.py        indexing, dedup, perceptual grouping, splits, folds
  preprocess.py  CLAHE, image loading
  pipeline.py    cached tf.data input pipeline
  models.py      LXNet + DenseNet201 / ResNet50V2 / InceptionV3
  train.py       hold-out and k-fold orchestration, CLI
  evaluate.py    metrics, fold summaries, Wilcoxon signed-rank
  xai.py         Grad-CAM, Score-CAM, LIME, overlays
  explain.py     per-class interpretability panel from a checkpoint
  report.py      figures and the results table
tests/           the leakage guarantees are pinned here
```
