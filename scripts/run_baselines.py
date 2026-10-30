#!/usr/bin/env python
"""Train + evaluate every baseline sequentially with seed 42, then write the comparison table.

    python scripts/run_baselines.py --data configs/mvtec_unified.yaml
    python scripts/run_baselines.py --data configs/visa_pcb.yaml --models patchcore dinomaly --optshift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartlens.utils.runtime import setup_env  # noqa: E402

setup_env()

from smartlens.config import PROJECT_ROOT, resolve_path  # noqa: E402
from smartlens.pipeline import default_run_dir, evaluate, load_experiment, train  # noqa: E402
from smartlens.utils.runtime import get_device, seed_everything, setup_logging  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from summarize import summarize  # noqa: E402

ALL = ["ssim_cae", "patchcore", "dinomaly"]
CHOICES = ["ssim_cae", "patchcore", "patchcore_448", "dinomaly"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="configs/mvtec_unified.yaml")
    ap.add_argument("--models", nargs="*", default=ALL, choices=CHOICES)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--runs-root", default=None, help="default: runs/<dataset>")
    ap.add_argument("--calibration-scope", default=None, choices=["per_category", "pooled"],
                    help="threshold calibration scope (default: per_category)")
    ap.add_argument("--optshift", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="skip categories that already have checkpoints")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--set", nargs="*", default=[], help="overrides applied to every model config")
    args = ap.parse_args()

    setup_logging()
    device = get_device(args.device)
    tag = ("test_optshift" if args.optshift else "test") + ("_pooled" if args.calibration_scope == "pooled" else "")
    runs_root = None
    for name in args.models:
        seed_everything(42)
        cfg = load_experiment(args.data, f"configs/models/{name}.yaml", args.set)
        if args.runs_root:
            runs_root = resolve_path(args.runs_root)
            run_dir = runs_root / name
        else:
            run_dir = default_run_dir(cfg)
            runs_root = run_dir.parent
        if not args.eval_only:
            train(cfg, run_dir, device, args.categories, args.num_workers, args.skip_existing)
        evaluate(run_dir, device, categories=args.categories, optshift=args.optshift,
                 tag=tag, calibration_scope=args.calibration_scope)
    if runs_root is not None:
        summarize(runs_root, tag)


if __name__ == "__main__":
    main()
