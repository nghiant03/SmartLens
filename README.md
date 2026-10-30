# SmartLens — Visual Defect Detection: Baselines & Evaluation Suite

Implementation of the **baselines** and the **leakage-free evaluation protocol** described in
[`TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) (§4.2, §5, §6) for the NUS-ISS PRS
project *Visual Defect Detection* (Group 30).

**Proposed main model: HiRes-ProtoAD, not a baseline ensemble.** Its INP conditioning,
uncertainty, guided upsampling and local high-frequency branch are specified but **not yet
implemented**. Dinomaly is the A0 reference baseline, not the proposed model.
See [`IMPLEMENTATION_AUDIT.md`](IMPLEMENTATION_AUDIT.md) for implementation gaps,
benchmark saturation findings and the measured training-environment bottleneck.

| ID | Baseline | Paradigm | Setting | Code |
|---|---|---|---|---|
| B1 | **SSIM-CAE** | Conv. autoencoder, L1 + multi-scale SSIM | per category | [`ssim_cae.py`](smartlens/models/baselines/ssim_cae.py) |
| B2a | **PatchCore** (CVPR'22) | WRN50-2 layer2+3, 10 % greedy coreset, k-NN | per category | [`patchcore.py`](smartlens/models/baselines/patchcore.py) |
| B3 | **Dinomaly-Repro** (CVPR'25) = ablation A0 | Frozen DINOv2-Reg + noisy bottleneck + linear-attention decoder | **unified** (1 model, 15 classes) | [`dinomaly.py`](smartlens/models/baselines/dinomaly.py) |

## Setup

For another GPU machine, follow [REMOTE_TRAINING.md](REMOTE_TRAINING.md), including
project/data transfer, environment verification, full-suite commands and compute estimates.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e ".[dev]"
```

All artefacts stay inside the repository (all git-ignored):

| Folder | Content |
|---|---|
| `datasets/mvtec_ad/` | MVTec AD (`<category>/{train,test,ground_truth}`) |
| `datasets/visa/` | VisA (`<category>/Data/...`, `split_csv/1cls.csv`) |
| `weights/` | pretrained backbone weights (torchvision / timm), fetched on first use |
| `runs/` | checkpoints, logs, evaluation outputs |

The datasets are **not** downloaded by any code in this repository: place them in `datasets/`
yourself (or point `SMARTLENS_DATA_ROOT` at a folder containing `mvtec_ad/` and `visa/`).

## Leakage-free protocol (spec §5.2)

`scripts/prepare_splits.py` writes deterministic manifests (`seed=42`) to `configs/splits/`
(already committed for MVTec AD and VisA):

| Split | Source | Used for |
|---|---|---|
| `train` | 85 % of official `train/good` | fitting |
| `val_normal` | 15 % of official `train/good` | checkpoint selection, pixel quantiles, unsupervised threshold τ_unsup (99th pct) |
| `test_val` | stratified 20 % of official `test` | F1-tuned threshold τ_f1 only |
| `test` | stratified 80 % of official `test` | **all reported numbers** |

```bash
python scripts/prepare_splits.py --data configs/mvtec_unified.yaml
python scripts/prepare_splits.py --data configs/visa_pcb.yaml --all-categories
```

## Train / evaluate

The experiment plan uses **one fixed seed (42)** for each baseline and A0–A4 variant.
The baseline runner executes SSIM-CAE, PatchCore and Dinomaly once per dataset.

```bash
# one baseline
python scripts/train.py    --data configs/mvtec_unified.yaml --model configs/models/patchcore.yaml
python scripts/evaluate.py --run runs/mvtec_ad/patchcore --optshift

# quick experiments: subset of categories + config overrides
python scripts/train.py --data configs/mvtec_unified.yaml --model configs/models/ssim_cae.yaml \
    --categories bottle screw --set model.train.iters=1000

# full baseline benchmark including the planned robustness tests
python scripts/run_baselines.py --data configs/mvtec_unified.yaml --optshift
python scripts/run_baselines.py --data configs/visa_pcb.yaml --optshift
python scripts/summarize.py --runs-root runs/mvtec_ad --tag test_optshift
```

Existing `runs/smoke_mvtec` results cover **bottle only**, with 300 training iterations;
`runs/smoke_visa` covers **PCB1 only**. These are smoke tests, not full benchmarks or SOTA
evidence. Evaluation now fails on missing requested checkpoints; use `--categories`
explicitly for a subset. Reports show category and image counts and reject comparisons
with differing category sets or evaluation protocols. Historical ensemble and EfficientAD artifacts are
excluded from regenerated comparisons.

### Training hardware

The audited RTX 5050 Laptop (8GB) runs Dinomaly at 448px/batch 16 with BF16 and gradient
checkpointing; the same batch without checkpointing runs out of memory. Full experiments
will use the **user-selected 24GB+ GPU**. Its model config preserves the architecture and
optimization budget:

```bash
# On the target GPU: first check memory using a disposable, short training run.
python scripts/train.py --data configs/mvtec_unified.yaml --model configs/models/dinomaly_24gb.yaml \
    --run-dir runs/hardware_probe/dinomaly --set model.train.iters=20

# Full unified A0; the target machine and dataset paths must be provisioned first.
python scripts/train.py --data configs/mvtec_unified.yaml --model configs/models/dinomaly_24gb.yaml
python scripts/evaluate.py --run runs/mvtec_ad/dinomaly --optshift
```

Training summaries record actual categories, iteration counts, GPU, peak training VRAM
and model config. `--skip-existing` cannot overwrite run metadata with a changed config.

## Evaluation outputs (`<run>/eval/`)

* `metrics_<tag>.csv` / `summary_<tag>.md` — per category + mean:
  **P-AP, AU-PRO (≤5 % and ≤30 % FPR)**, **I-FPR@95%TPR, I-TPR@1%FPR**, **I-AUROC, I-AP, I-F1max**,
  **Precision / Recall / F1 / FPR** at τ_unsup and τ_f1, **P-AUROC, P-F1max**;
  recall per defect sub-type in `results_<tag>.json`.
* `results_<tag>.json` — also contains calibration (pixel quantiles, thresholds) and the system profile:
  **latency (CUDA events, bs=1, 50 warm-up / 200 timed), FPS, peak VRAM, parameters**.
* `metrics_<tag>_optshift_*.csv` — **MVTec-OptShift** robustness (illumination gradient ±30 %,
  gamma 0.65 / 1.45, specular glare, ±3° / ±4 px affine), thresholds from *clean* validation.
* `galleries/<category>/{false_alarms,missed_defects,true_positives}.png` + `errors.csv` — error review.

New experiments evaluate all models on a common **448×448 nearest-neighbour mask grid**
(`data.mask_size`, `data.mask_resize`); saved legacy runs retain **256×256 area/0.5**.
The protocols must not be mixed. Neither is native-resolution evaluation: resampling
can lose or merge tiny defects, so native-region analysis is still required before
claiming 1–3px micro-defect gains. Reports expose anomalous images with empty resized masks.

Curve metrics are exact (no binning) and tested against scikit-learn (`pytest`).
F1max and TPR@1%FPR are test-curve diagnostics, not deployment thresholds. Small normal
sample counts limit low-FPR precision. `--split test_val` omits F1-tuned operating metrics
to avoid fitting and reporting the threshold on the same split. Current calibration is
per category, including for the unified baseline.

## Layout

```text
configs/                 data/eval configs, model configs, split manifests
smartlens/data/          splits.py, datasets.py, perturbations.py (MVTec-OptShift)
smartlens/models/        base.py (interface, blur, StableAdamW), backbones/, baselines/
smartlens/losses/        ssim.py
smartlens/evaluation/    metrics.py, calibration.py, latency.py, error_analysis.py, evaluator.py
smartlens/pipeline.py    train / evaluate orchestration
scripts/                 prepare_splits, train, evaluate, run_baselines, summarize
tests/                   metric / split / perturbation / model unit tests
```
