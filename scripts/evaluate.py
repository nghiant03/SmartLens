#!/usr/bin/env python
"""Evaluate a trained run on the locked test split (+ optional OptShift / latency / galleries).

    python scripts/evaluate.py --run runs/mvtec_ad/patchcore
    python scripts/evaluate.py --run runs/mvtec_ad/patchcore --optshift
    python scripts/evaluate.py --run runs/mvtec_ad/dinomaly --optshift --calibration-scope pooled
    python scripts/evaluate.py --run runs/mvtec_ad/dinomaly --split test --no-gallery

Outputs land in ``<run>/eval/``: ``metrics_<tag>.csv`` (per category + mean),
``results_<tag>.json`` (everything incl. calibration and latency), ``summary_<tag>.md``
and ``galleries/<category>/{false_alarms,missed_defects,true_positives}.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartlens.utils.runtime import setup_env  # noqa: E402

setup_env()

from smartlens.config import apply_overrides, load_yaml, resolve_path  # noqa: E402
from smartlens.pipeline import evaluate  # noqa: E402
from smartlens.utils.runtime import get_device, seed_everything, setup_logging  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory produced by scripts/train.py")
    ap.add_argument("--data", default=None, help="optional data config to override the run's data/eval section")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--split", default="test", choices=["test", "test_val"])
    ap.add_argument("--calibration-scope", default=None, choices=["per_category", "pooled"],
                    help="threshold calibration scope (default: per_category)")
    ap.add_argument("--optshift", action="store_true", help="also evaluate MVTec-OptShift perturbations")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--no-gallery", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="eval overrides, e.g. compute_pro=false")
    args = ap.parse_args()

    setup_logging()
    seed_everything(0)
    data_cfg = load_yaml(args.data) if args.data else None
    eval_over = apply_overrides({}, args.set)
    evaluate(resolve_path(args.run), get_device(args.device), data_cfg, args.categories, args.split,
             optshift=args.optshift, latency=not args.no_latency, gallery=not args.no_gallery,
             eval_overrides=eval_over, tag=args.tag, calibration_scope=args.calibration_scope)


if __name__ == "__main__":
    main()
