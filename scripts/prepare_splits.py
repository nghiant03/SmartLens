#!/usr/bin/env python
"""Generate deterministic, leakage-free train / val_normal / test_val / test manifests.

The dataset must already be present locally (default: <repo>/datasets/<name>).

    python scripts/prepare_splits.py --data configs/mvtec_unified.yaml
    python scripts/prepare_splits.py --data configs/visa_pcb.yaml --all-categories
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from smartlens.config import apply_overrides, load_yaml, resolve_path  # noqa: E402
from smartlens.data.splits import (MVTEC_CATEGORIES, VISA_CATEGORIES, build_manifest, save_manifest,  # noqa: E402
                                   summarize_manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data config yaml")
    ap.add_argument("--all-categories", action="store_true", help="index every category of the dataset")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = ap.parse_args()

    cfg = apply_overrides(load_yaml(args.data), args.set)["data"]
    name = cfg["name"]
    cats = ({"mvtec_ad": MVTEC_CATEGORIES, "visa": VISA_CATEGORIES}[name] if args.all_categories
            else cfg["categories"])
    root = resolve_path(cfg["root"])
    manifest = build_manifest(name, root, cats, seed=cfg.get("seed", 42),
                              val_normal_ratio=cfg.get("val_normal_ratio", 0.15),
                              test_val_ratio=cfg.get("test_val_ratio", 0.20))
    out = resolve_path(cfg["split_manifest"])
    save_manifest(manifest, out)
    df = pd.DataFrame(summarize_manifest(manifest)).set_index("category")
    df.loc["total"] = df.sum()
    print(df.to_string())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
