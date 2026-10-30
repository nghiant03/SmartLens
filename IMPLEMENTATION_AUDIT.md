# Baseline, evaluation and training audit

Audit date: 30 October 2026.

**Subsequent scope decision:** use one fixed seed (42) for all baseline and ablation
experiments. EfficientAD has been removed from the active implementation and comparisons;
SSIM-CAE is retained. EfficientAD measurements below are historical audit evidence.
Current runnable baselines are SSIM-CAE, PatchCore and Dinomaly; the evaluation-only budget is in
[REMOTE_TRAINING.md](REMOTE_TRAINING.md).

## 1. Main model and research claim

The current proposal is **HiRes-ProtoAD**, a single jointly trained architecture. Its
internal semantic/local branch fusion is distinct from combining independently trained
SSIM-CAE and PatchCore models. The latter ensemble has been removed from the model
registry, implementation, configs, baseline runner, tests and active comparisons.

**HiRes-ProtoAD is not implemented.** Missing components include intrinsic prototypes,
conditioned decoder blocks, uncertainty heads, the high-resolution teacher/student
branch, learned JBU, their training losses and A1–A4 ablations. Only Dinomaly/A0 exists.
Consequently there is no trained proposed model, no measured proposed-model improvement,
and no measured proposed-model memory or latency yet. The specification's claimed gains,
38M trainable parameters (ViT-B) and 20–25ms latency must be treated as targets.

## 2. What the saved results actually show

Sources: `runs/smoke_mvtec/*/{config.yaml,train_summary.json,eval/results_test_optshift.json}`
and `runs/smoke_visa/patchcore/eval/results_test.json`.

| Existing experiment | Coverage | Training budget | Evidence |
|---|---|---|---|
| MVTec SSIM-CAE | bottle only | 300 / configured default 4,000 steps | I-AUROC 94.1%; P-AP 62.1% |
| MVTec PatchCore | bottle only | memory bank fit, 18,228 vectors | I-AUROC 100%; P-AP 79.5% |
| MVTec EfficientAD | bottle only | 300 / 70,000 steps | I-AUROC 97.2%; AU-PRO 56.4% |
| MVTec Dinomaly | bottle only, despite `unified.pt` | 300 / 10,000 steps | I-AUROC 100%; P-AP 85.1% |
| VisA PatchCore | PCB1 only | bank capped at 25,000 vectors | I-AUROC 95.2%; P-AP 82.7% |

All four MVTec fits used 178 training images and 31 normal validation images. Their test
split contains **67 images: 16 normal and 51 anomalous**. Full MVTec training would contain
3,084 normal training images across 15 categories. A unified checkpoint filename does
not establish multi-category training.

The apparent saturation is **clean bottle image ranking**, not the entire evaluation:

- Dinomaly illumination-gradient I-AUROC falls from 100% to **69.0%**, P-AP to **38.6%**,
  AU-PRO to **76.9%**, and the clean-calibrated unsupervised FPR becomes **100%**.
- Dinomaly glare I-AUROC is **93.0%**, also with **100%** clean-calibrated FPR.
- VisA PCB1 PatchCore I-FPR@95%TPR is **20.0%**; unsupervised recall is **72.5%**.
- P-AUROC near 99% coexists with much lower pixel AP because normal background dominates.

Thus the plan already contains useful unsaturated tasks, but only a small smoke-test
slice was executed. Reporting the mean without coverage hid this. More difficult metrics
alone do not turn a one-category smoke run into a full benchmark.

A fresh end-to-end evaluation of the existing PCB1 PatchCore checkpoint with the new
448px protocol (`runs/smoke_visa/patchcore/eval/summary_audit_448.md`) gives **P-AP 82.7%,
AU-PRO@5%FPR 74.3%, and I-TPR@1%FPR 41.2%** on 80 normal / 80 anomalous images.
Its I-AUROC remains 95.2%. This confirms useful headroom without training a new model;
it is still a one-category smoke evaluation, and the mask protocol differs from legacy results.

Verification: 30 tests passed, Python compilation succeeded, the historical MVTec
comparison was regenerated with explicit 1/15 coverage and without the retired ensemble,
and the real-checkpoint PCB1 evaluation completed successfully.

## 3. Evaluation findings and corrections

### Coverage and leakage

The saved split manifests separate train, normal validation, threshold-validation and
test paths. The locked 80% test subset is a **project-specific protocol**, not directly
comparable to paper scores on the full official test set. Normal-only training is enforced
by these generated manifests. This audit did not perform a content-level duplicate scan.

Previously missing checkpoints were silently skipped and remaining categories averaged.
Evaluation now fails instead; subsets require explicit category selection. New results
record dataset, category coverage, mask protocol, split-manifest SHA256, perturbations,
training summaries and model config. Comparisons reject mismatched protocols and display
sample counts and partial coverage. Legacy results are identified as lacking provenance.

Previously `--split test_val` could tune and report its F1 threshold on the same data.
It now reports only normal-validation-calibrated operating points and diagnostic curves.
Test F1max remains an oracle curve summary, explicitly labeled, not a deployment score.

All calibration is currently **per category**, even for Dinomaly. True category-blind
unified deployment still requires a separately evaluated pooled-calibration protocol.

### Localization resolution

The former 256px area resize followed by a 0.5 cutoff does not preserve thin defects.
Measured connected-component counts on locked test masks:

| VisA category | Native regions | 256px area/0.5 | 448px area/0.5 |
|---|---:|---:|---:|
| PCB1 | 213 | 200 | 197 |
| PCB2 | 275 | 238 | 250 |
| PCB3 | 228 | 201 | 239 |
| PCB4 | 422 | 333 | 379 |
| capsules | 134 | 108 | 132 |
| candle | 525 | 182 | 303 |

Counts measure topology changes, **not a one-to-one count of erased regions**: resizing
can remove, merge or fragment components. No entire anomalous mask became empty in this
audit, so checking only empty masks would miss the region-level problem. Detailed counts
are in `runs/audit/mask_audit.json`.

New shared configs use a 448px nearest-neighbour mask grid, record the interpolation rule,
and expose empty-anomaly-mask counts. Legacy saved configs retain 256px area/0.5. Neither
protocol is native-resolution evaluation; nearest-neighbour can also lose off-grid defects.
Native-resolution region/size analysis remains necessary for the proposal's 1–3px claim.
Do not interpret a score change caused by changing mask resolution as a model improvement.

### Metrics and experiment requirements

- Reports now lead with pixel AP and AU-PRO, including **AU-PRO up to 5% FPR** alongside
  the existing 30% measure; both use the same exact region-weighted curve.
- **TPR@1%FPR** is an attainable test-curve diagnostic (ties are not interpolated).
  Validation-calibrated recall/FPR remain the operational measures.
- Shift reports now include pixel AP and clean-threshold recall/FPR, not only AUROC/PRO.
- With 16 test normals, FPR increments are 6.25%; a 1% operating claim is unsupported.
  Likewise a 99th-percentile threshold from 31 validation normals does not guarantee a
  true 1% false alarm rate, especially after distribution shift.
- Required evidence: full MVTec-15, VisA's planned six-category slice (and full VisA-12
  for paper comparisons), all five OptShift conditions, matched A0–A4 ablations using
  the agreed single seed (42), and image-level paired uncertainty estimates. Avoid treating correlated pixels
  as independent confidence-interval samples.
- For native micro-defects, stratify by original connected-component size/width and
  evaluate boundaries/region recall. This analysis is not yet implemented.
- Actual MVTec AD 2 is not supported by the current loaders. OptShift is synthetic and
  must not be presented as MVTec AD 2 results.

## 4. Baseline fidelity and training

| Baseline | Implementation review | Consequence |
|---|---|---|
| SSIM-CAE | L1 + averaged multi-scale SSIM, per-category AE, best normal-val checkpoint | Useful historical baseline, not current SOTA; runs the full budget rather than actual early termination |
| PatchCore | Frozen WRN layer2/3 concatenation, greedy random-projection coreset, k-NN | Adapted implementation; 25k cap yields only ~3.18% of PCB1's 786,432 patches, not advertised 10%; summaries now record the effective ratio |
| EfficientAD | PDN student/teacher + AE; ImageNet penalty disabled by default | Label as no-penalty variant unless external natural images are supplied; 300 steps cannot establish full-model performance |
| Dinomaly | Frozen DINOv2-Reg, dropout, linear decoder, hard mining, BF16/checkpointing | Sound A0 structure but adapted protocol; 300 steps ends before its 1,000-step hard-mining ramp completes |

The local Dinomaly uses direct 448px resizing; upstream defaults to resize 448 then crop
392. Upstream initialization clips normal weights to ±0.03, whereas local initialization
uses ±2.0 with std 0.01. Target layers `[2..9]` are **zero-indexed** in the implementation
and upstream; the proposed-model spec describes them as one-indexed. Local raw-token
fusion also differs from the proposed spec's layer-normalized grouping. These differences
must be resolved or explicitly retained as A0 choices before building A1–A4.

Checkpoint files contain model weights, not optimizer/scheduler/RNG state. `save_every`
is an inference snapshot, not fault-tolerant training resume. This matters for preemptible
cloud instances. Training summaries now capture actual categories, iterations, device,
model config and training VRAM; empty drop-last loaders fail instead of hanging forever.

## 5. Measured environment constraint and decision

Host: i7-13620H, ~31GiB RAM, RTX 5050 Laptop 8GB; NVIDIA power limit reported as 50W.
PyTorch 2.11.0+cu128, CUDA runtime 12.8. The runtime reports 7,799MiB usable GPU memory.

Probe: actual unified MVTec training loader, ViT-B/14, 448px, batch 16, BF16, real
hard-mining loss, backward, clipping and StableAdamW/AMSGrad updates. Three warm-up steps
then five synchronized timed steps; four loader workers. No model-quality conclusions
are drawn from this short performance probe.

| Setting | Result | Mean step | Data wait + copy | Peak allocated / reserved |
|---|---|---:|---:|---:|
| Gradient checkpointing on | completed | 1.757s | 0.006s | 3,016 / 3,890 MiB |
| Gradient checkpointing off | CUDA OOM | unavailable | unavailable | 7,334 / 7,632 MiB before failure |

Measured Dinomaly parameters: **147.74M total, 61.43M trainable**. The frozen encoder
does not eliminate decoder activation or optimizer-state memory. The hardware constrains
throughput and requires recomputation at this batch size; it does **not** establish an
accuracy ceiling or explain clean-bottle saturation. The new model's memory is unknown.
Raw probe results: `runs/audit/training_profile.json`.

**User decision: move full training/ablations to a 24GB+ GPU.** RTX 3090/4090, A10 or L4
are candidate capacity classes, not equal-speed devices. Use at least 32GB host RAM;
64GB provides more room for full-resolution exact pixel-metric sorts and cached features.
Provision the target first, then profile `configs/models/dinomaly_24gb.yaml` (checkpointing
off, same 448px/batch-16 recipe) before full training. Profile HiRes-ProtoAD after it exists.
No remote machine was provisioned and no full training was launched during this audit.

## 6. Current-SOTA positioning

The plan's 2025 reference set is no longer sufficient by itself to support “current SOTA”
in October 2026. The official Dinomaly repository links **Dinomaly2**, whose preview code
was released in May 2026. Its authors explicitly note possible config/paper discrepancies.
INP-Former is cited in the proposal but also absent as a runnable comparator.

Before claiming an improvement: pin the official comparator revisions, verify their
settings/checkpoints, run them on matched project splits and retain a separate official
benchmark protocol for comparisons to published numbers. Compare trained A4 against both
the matched A0 and contemporary strong methods. The aim is an improvement hypothesis,
not an already demonstrated SOTA result.

Primary sources checked during this audit:
- [Official Dinomaly README](https://github.com/guojiajeremy/Dinomaly)
- [Official unified MVTec training recipe](https://github.com/guojiajeremy/Dinomaly/blob/master/dinomaly_mvtec_uni.py)
- [Official Dinomaly2 README / preview status](https://github.com/guojiajeremy/Dinomaly2)
