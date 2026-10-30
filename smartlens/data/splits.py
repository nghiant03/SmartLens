"""Deterministic, leakage-free split manifests for MVTec AD and VisA.

Protocol (TECHNICAL_SPECIFICATION.md §5.2), applied independently per category:

* ``train``       – 85 % of the official normal training images (model fitting only).
* ``val_normal``  – remaining 15 % of the official normal training images (quantile
                    calibration, early stopping, unsupervised threshold).
* ``test_val``    – stratified 20 % of the official test split (preserving defect sub-type
                    proportions); used only for F1-tuned threshold selection.
* ``test``        – stratified remaining 80 % of the official test split (locked, unseen;
                    every reported number is computed here).

All paths in a manifest are relative to the dataset root so manifests are portable.
"""

from __future__ import annotations

import csv
import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
MVTEC_TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
    "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def _record(root: Path, image: Path, mask: Path | None, label: int, defect_type: str) -> dict:
    return {
        "image": image.relative_to(root).as_posix(),
        "mask": mask.relative_to(root).as_posix() if mask is not None else None,
        "label": int(label),
        "defect_type": defect_type,
    }


# --------------------------------------------------------------------------------------
# Raw dataset indexing (official train / test)
# --------------------------------------------------------------------------------------
def index_mvtec(root: Path, category: str) -> tuple[list[dict], list[dict]]:
    cat_dir = root / category
    if not cat_dir.is_dir():
        raise FileNotFoundError(f"MVTec AD category folder not found: {cat_dir}")
    train = [_record(root, p, None, 0, "good") for p in _images(cat_dir / "train" / "good")]
    test = []
    for defect_dir in sorted(d for d in (cat_dir / "test").iterdir() if d.is_dir()):
        defect = defect_dir.name
        for img in _images(defect_dir):
            if defect == "good":
                test.append(_record(root, img, None, 0, "good"))
            else:
                mask = cat_dir / "ground_truth" / defect / f"{img.stem}_mask.png"
                if not mask.exists():
                    raise FileNotFoundError(f"Missing ground-truth mask: {mask}")
                test.append(_record(root, img, mask, 1, defect))
    return train, test


def _visa_defect_types(root: Path, category: str) -> dict[str, str]:
    """Map image rel-path -> defect description from ``<cat>/image_anno.csv`` (if present)."""
    anno = root / category / "image_anno.csv"
    mapping: dict[str, str] = {}
    if not anno.exists():
        return mapping
    with open(anno, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip()
            if label and label.lower() != "normal":
                # multi-defect images are labelled "a,b" -> keep first sub-type for stratification
                mapping[row["image"].strip()] = label.split(",")[0].strip().replace(" ", "_")
    return mapping


def index_visa(root: Path, category: str) -> tuple[list[dict], list[dict]]:
    split_csv = root / "split_csv" / "1cls.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"VisA split file not found: {split_csv}")
    defect_types = _visa_defect_types(root, category)
    train, test = [], []
    with open(split_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["object"] != category:
                continue
            img = root / row["image"]
            is_anom = row["label"].strip().lower() == "anomaly"
            mask = root / row["mask"] if is_anom and row.get("mask") else None
            defect = defect_types.get(row["image"], "anomaly") if is_anom else "good"
            rec = _record(root, img, mask, int(is_anom), defect)
            (train if row["split"] == "train" else test).append(rec)
    train = [r for r in train if r["label"] == 0]  # unsupervised: never train on anomalies
    return train, test


INDEXERS = {"mvtec_ad": index_mvtec, "visa": index_visa}


# --------------------------------------------------------------------------------------
# Split generation
# --------------------------------------------------------------------------------------
def _rng(seed: int, category: str) -> np.random.Generator:
    return np.random.default_rng(seed + zlib.crc32(category.encode()))


def split_category(
    train: list[dict], test: list[dict], category: str, seed: int = 42,
    val_normal_ratio: float = 0.15, test_val_ratio: float = 0.20,
) -> dict[str, list[dict]]:
    rng = _rng(seed, category)

    train = sorted(train, key=lambda r: r["image"])
    perm = rng.permutation(len(train))
    n_val = max(1, int(round(val_normal_ratio * len(train))))
    val_idx = set(perm[:n_val].tolist())
    tr = [r for i, r in enumerate(train) if i not in val_idx]
    vn = [r for i, r in enumerate(train) if i in val_idx]

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(test, key=lambda r: r["image"]):
        groups[r["defect_type"]].append(r)
    tv, te = [], []
    for defect in sorted(groups):
        items = groups[defect]
        perm = rng.permutation(len(items))
        n = int(round(test_val_ratio * len(items)))
        if len(items) >= 2:
            n = min(max(n, 1), len(items) - 1)  # keep >=1 in each side when possible
        else:
            n = 0
        sel = set(perm[:n].tolist())
        tv += [r for i, r in enumerate(items) if i in sel]
        te += [r for i, r in enumerate(items) if i not in sel]
    return {"train": tr, "val_normal": vn, "test_val": tv, "test": te}


def build_manifest(
    dataset: str, root: str | Path, categories: list[str], seed: int = 42,
    val_normal_ratio: float = 0.15, test_val_ratio: float = 0.20,
) -> dict:
    root = Path(root)
    indexer = INDEXERS[dataset]
    out = {
        "meta": {
            "dataset": dataset, "seed": seed, "val_normal_ratio": val_normal_ratio,
            "test_val_ratio": test_val_ratio,
        },
        "categories": {},
    }
    for cat in categories:
        train, test = indexer(root, cat)
        out["categories"][cat] = split_category(train, test, cat, seed, val_normal_ratio, test_val_ratio)
    _check_disjoint(out)
    return out


def _check_disjoint(manifest: dict) -> None:
    for cat, splits in manifest["categories"].items():
        seen: dict[str, str] = {}
        for split, records in splits.items():
            for r in records:
                if r["image"] in seen:
                    raise RuntimeError(f"Leakage: {r['image']} in both {seen[r['image']]} and {split} ({cat})")
                seen[r["image"]] = split
        if any(r["label"] for r in splits["train"] + splits["val_normal"]):
            raise RuntimeError(f"Anomalous image found in a training split for {cat}")


def save_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)


def load_manifest(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_manifest(manifest: dict) -> list[dict]:
    rows = []
    for cat, s in manifest["categories"].items():
        rows.append({
            "category": cat,
            "train": len(s["train"]),
            "val_normal": len(s["val_normal"]),
            "test_val": len(s["test_val"]),
            "test_val_anom": sum(r["label"] for r in s["test_val"]),
            "test": len(s["test"]),
            "test_anom": sum(r["label"] for r in s["test"]),
        })
    return rows
