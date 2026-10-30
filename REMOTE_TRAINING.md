# Set up training and evaluation on another machine

These steps cover the implemented baselines. HiRes-ProtoAD and A1–A4 are not yet
implemented. Commands assume Ubuntu 22.04/24.04, SSH access and one NVIDIA GPU.

## 1. Select the machine

- **GPU:** one dedicated GPU with at least 24GB VRAM. RTX 4090/3090, A10 and L4 have
  similar capacity but different throughput; do not select by VRAM alone.
- **CPU:** at least 8 vCPUs; 16 preferable for loading and exact pixel metrics.
- **Host RAM:** 32GB minimum; 64GB recommended, particularly for larger evaluation grids.
- **Storage:** 100–150GB persistent SSD is a practical starting allocation for code,
  environments, datasets, cached backbones, multiple checkpoints and results.
- Use a GPU image with a working, sufficiently recent NVIDIA driver for CUDA 12.8
  PyTorch wheels. `nvidia-smi` must work before creating the Python environment.

Current local footprint: datasets ~6.8GB, active weights ~547MB (excluding retired
EfficientAD weights), Python environment ~7.1GB; full experiment outputs will exceed the
existing smoke-run footprint.

## 2. Copy the project and data from the local checkout

Source code, configs and locked split manifests (`configs/splits/*.json`) are tracked in
Git, while `datasets/` and `weights/` are git-ignored. Syncing from the local repository
root transfers the working tree, split manifests, datasets and active cached weights in
one step without requiring GitHub credentials on the remote host.

From the **local repository root**, replace the SSH destination and target path:

```bash
export REMOTE=user@GPU_HOST
export REMOTE_DIR=/home/user/SmartLens
ssh "$REMOTE" "mkdir -p '$REMOTE_DIR'"
rsync -avh --progress \
    --exclude='.git/' --exclude='.jj/' --exclude='.crush/' \
    --exclude='.venv/' --exclude='__pycache__/' --exclude='.pytest_cache/' \
    --exclude='*.egg-info/' --exclude='runs/' --exclude='weights/efficientad/' \
    ./ "$REMOTE:$REMOTE_DIR/"
```

The trailing slash on `./` is intentional. This transfers `datasets/` and active
`weights/` (`torch/` and `huggingface/`), while omitting VCS/editor metadata, the local
virtual environment, retired EfficientAD weights and smoke-run outputs. Do not regenerate
the split manifests on the new machine: every model must use the same held-out images.

## 3. Create the Python environment on the GPU machine

On the **GPU machine** (set `REMOTE_DIR` to match step 2):

```bash
sudo apt-get update && sudo apt-get install -y curl rsync tmux
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$PATH"

export REMOTE_DIR=/home/user/SmartLens
cd "$REMOTE_DIR"
nvidia-smi

uv sync --locked
source .venv/bin/activate
```

`uv sync --locked` reads `.python-version` (`3.12`), `pyproject.toml` (including the
explicit `pytorch-cu128` wheel index for `torch` and `torchvision`) and `uv.lock` to
provision Python 3.12, `.venv`, hash-verified dependencies (`torch==2.11.0+cu128`,
`torchvision==0.26.0+cu128`, `timm==1.0.30`, `pytest`) and editable `smartlens` in one
step. The PyTorch wheels supply their CUDA user-space dependencies, so no separate CUDA
toolkit installation is needed; a compatible NVIDIA host driver is still required.

## 4. Verify GPU, data and software

```bash
python - <<'PY'
import torch
from smartlens.config import load_yaml
from smartlens.data.datasets import DataModule

assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'
p = torch.cuda.get_device_properties(0)
print('GPU:', p.name)
print('VRAM GiB:', round(p.total_memory / 2**30, 2))
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('BF16:', torch.cuda.is_bf16_supported())
assert torch.cuda.is_bf16_supported(), 'The configured AMP path uses BF16'
for config in ['configs/mvtec_unified.yaml', 'configs/visa_pcb.yaml', 'configs/visa_all.yaml']:
    dm = DataModule(load_yaml(config)['data'])
    for category in dm.categories:
        for split in ['train', 'val_normal', 'test_val', 'test']:
            for record in dm.records(category, split):
                assert (dm.root / record['image']).is_file(), record['image']
                if record['mask']:
                    assert (dm.root / record['mask']).is_file(), record['mask']
    print(config, 'verified categories:', len(dm.categories))
PY

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q
mkdir -p runs/environment
uv pip freeze --python .venv/bin/python > runs/environment/requirements.freeze.txt
nvidia-smi > runs/environment/nvidia-smi.txt
sha256sum configs/splits/*.json > runs/environment/split_hashes.txt
```

The expected test count at this revision is 29. Default dataset locations are
`datasets/mvtec_ad/` and `datasets/visa/`. For another storage mount, set
`SMARTLENS_DATA_ROOT` to the parent directory containing those two folders.

## 5. Profile the target before the full run

Use a disposable run directory. This tests actual forward, backward and optimizer
allocations at 448px / batch 16 without gradient checkpointing:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/train.py \
    --data configs/mvtec_unified.yaml \
    --model configs/models/dinomaly_24gb.yaml \
    --run-dir runs/hardware_probe/dinomaly \
    --num-workers 6 --set model.train.iters=20

cat runs/hardware_probe/dinomaly/train_summary.json
rm -rf runs/hardware_probe
```

Check `peak_train_vram_MB`, `peak_train_reserved_MB` and `train_time_s`, then remove
`runs/hardware_probe` so disposable probe checkpoints are not synced back in step 7.
The short probe includes image-preload startup and is not a reliable steady-state
throughput measurement. For a scheduling estimate, repeat with
`--set model.train.iters=1000`, inspect the `time_s` difference between `iter: 500` and
`iter: 1000` in `runs/hardware_probe/dinomaly/train_log_unified.jsonl`, and delete
`runs/hardware_probe` afterward. Do not treat shortened-training checkpoints as benchmark
models.

The local memory-scaling probe measured 3.13GiB allocated at batch 4 and 4.64GiB at
batch 8, with BF16 and no checkpointing. Extrapolated batch-16 allocation is ~7.65GiB;
runtime/workspace/allocator overhead is additional. **24GB should comfortably fit the
current Dinomaly recipe**, subject to this target-machine check. HiRes-ProtoAD needs
a new profile after its implementation, especially for the guided upsampler.

## 6. Train and evaluate the complete baseline suite

Start an SSH-independent `tmux` session on the **GPU machine**:

```bash
tmux new -s smartlens
```

Inside the `tmux` session, activate the environment and prepare logging:

```bash
export REMOTE_DIR=/home/user/SmartLens
cd "$REMOTE_DIR"
source .venv/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
set -o pipefail
mkdir -p runs/logs
```

Run these **sequentially** on the single GPU:

```bash
python scripts/run_baselines.py \
    --data configs/mvtec_unified.yaml \
    --runs-root runs/mvtec_ad --optshift --num-workers 6 \
    --set model.grad_checkpoint=false \
    2>&1 | tee runs/logs/mvtec_baselines.log

python scripts/run_baselines.py \
    --data configs/visa_pcb.yaml \
    --runs-root runs/visa_pcb --optshift --num-workers 6 \
    --set model.grad_checkpoint=false \
    2>&1 | tee runs/logs/visa_pcb_baselines.log
```

The override disables checkpointing for Dinomaly; the other models do not use it.
All three baselines (SSIM-CAE, PatchCore and Dinomaly) run once with **seed 42**
at their full configured budgets. Do not pass `--categories`
for these full experiments. The VisA command covers the planned six-category slice.
For a separate full 12-category VisA experiment, use `configs/visa_all.yaml` and a
distinct run root such as `runs/visa_all`; do not overwrite the six-category run.

Detach tmux with Ctrl-B then D; reconnect using `tmux attach -t smartlens`.

Outputs include each model's `train_summary.json`, checkpoints, and
`eval/results_test_optshift.json`. Comparison tables are written under the run roots.
Evaluation includes clean and all five shift conditions with clean-validation thresholds.

To evaluate an existing run again without training:

```bash
python scripts/evaluate.py --run runs/mvtec_ad/dinomaly --optshift
python scripts/summarize.py --runs-root runs/mvtec_ad --tag test_optshift
```

Current model snapshots
do not include optimizer/scheduler/RNG state, so interrupted gradient training cannot
resume exactly. `--skip-existing` skips completed checkpoints; it is not step-level resume.

## 7. Retrieve the results

From the **local repository root** (using the variables from step 2):

```bash
mkdir -p runs/remote_gpu
rsync -avh --progress --exclude='hardware_probe/' "$REMOTE:$REMOTE_DIR/runs/" runs/remote_gpu/
```

Keep the environment snapshot, split hashes and logs with the results. Report the
actual GPU model, training budgets, category coverage and evaluation mask protocol.
These baseline runs do not establish a HiRes-ProtoAD improvement: that model and
the contemporary SOTA comparators still need implementation/validation.

## 8. Evaluation-only runtime budget

The retained baselines are **SSIM-CAE, PatchCore and Dinomaly**, with one seed (42).
EfficientAD is removed. This estimate excludes all training, setup and transfers.
No new evaluation was launched to produce it; it uses existing timings and manifest counts.

Assumed suite: common 448px mask grid, clean test + five OptShift conditions, normal-val
calibration and test-val threshold tuning, exact image/pixel metrics including AU-PRO at
5% and 30% FPR, galleries, and the 50-warm-up/200-timed latency profile. Models run
sequentially on a GPU with a CPU/SSD at least comparable to the audited laptop.

| Scope | Estimated wall-clock time for all retained baselines |
|---|---:|
| Available checkpoints only: bottle × 3 models + PCB1 PatchCore | ~10–20 minutes |
| Full MVTec-15 | ~1–2 hours |
| Full planned VisA-6 | ~45–90 minutes |
| **MVTec-15 + VisA-6, full suite** | **~2–4 hours** |
| Both datasets, clean evaluation only (no OptShift) | ~30–60 minutes |

Reserve **4–6 hours** for a first full-suite pass if caches are cold or CPU/storage
performance is uncertain. These are planning ranges, not measurements of a complete
benchmark or of the unprovisioned remote GPU. Full coverage requires checkpoints that
do not yet exist; the estimate assumes they have already been trained.

Evidence and scaling:
- Saved bottle evaluations at 256px, including all shifts: SSIM-CAE 26.8s, PatchCore
  42.9s, Dinomaly 38.2s. These are category evaluation times, not full process runtimes.
- The saved PCB1 PatchCore clean evaluation took 33.9s at 256px and 77.1s under the
  448px suite. This ~2.27× ratio is only a rough scaling anchor: it includes startup
  and calibration differences, not an isolated resolution benchmark.
- Current manifests contain 1,382 MVTec test images and 932 VisA-6 test images.
  Including both validation splits and six test conditions yields **47,283 image
  forwards across the three models**, and **378 category/condition metric evaluations**.
- Scaling the historical bottle timings by image counts and the observed resolution
  ratio gives roughly 1.4 hours for MVTec and 1 hour for VisA before allowing for
  category-specific cost, loading, galleries and profiling; the ranges above account
  for that uncertainty.
- Exact pixel metrics repeatedly sort large arrays on the CPU. A faster 24GB GPU
  accelerates inference but does not scale this entire workload by the same factor.
  CPU performance, RAM and local SSD storage remain important.

HiRes-ProtoAD is excluded because it has no implementation/checkpoint or measured
inference throughput. Full VisA-12 is also outside this six-category budget.

24GB is well supported for the current default baselines run sequentially. It is not
yet a verified capacity bound for HiRes-ProtoAD: a naive BF16 JBU neighborhood tensor
of shape `[16, 32, 25, 448, 448]` alone takes **4.79GiB**, before other tensors/backward.
Implement the upsampler with bounded memory and profile the full training step. A larger
backbone, higher resolution or simultaneous runs require a separate memory budget.
