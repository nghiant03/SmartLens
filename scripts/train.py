#!/usr/bin/env python
"""Train a baseline (per category or unified) on normal images only.

    python scripts/train.py --data configs/mvtec_unified.yaml --model configs/models/patchcore.yaml
    python scripts/train.py --data configs/mvtec_unified.yaml --model configs/models/ssim_cae.yaml \
        --categories bottle screw --set model.train.iters=1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartlens.utils.runtime import setup_env  # noqa: E402

setup_env()

from smartlens.config import resolve_path  # noqa: E402
from smartlens.pipeline import default_run_dir, load_experiment, train  # noqa: E402
from smartlens.utils.runtime import get_device, seed_everything, setup_logging  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--run-dir", default=None, help="default: runs/<dataset>/<model>")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. model.train.iters=500")
    args = ap.parse_args()

    setup_logging()
    seed_everything(args.seed)
    cfg = load_experiment(args.data, args.model, args.set)
    run_dir = resolve_path(args.run_dir) if args.run_dir else default_run_dir(cfg)
    train(cfg, run_dir, get_device(args.device), args.categories, args.num_workers, args.skip_existing)


if __name__ == "__main__":
    main()
