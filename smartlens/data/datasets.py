"""PyTorch datasets for MVTec AD / VisA built from leakage-free split manifests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from smartlens.config import resolve_path
from smartlens.data.perturbations import apply_perturbation
from smartlens.data.splits import load_manifest

cv2.setNumThreads(0)


def _size2(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    return int(size[0]), int(size[1])


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=np.uint8)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return ((m > 0).astype(np.uint8)) * 255


def resize_image(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    if img.shape[:2] == (h, w):
        return img
    interp = cv2.INTER_AREA if (img.shape[0] > h or img.shape[1] > w) else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def resize_mask(mask: np.ndarray, size: tuple[int, int], mode: str = "area") -> np.ndarray:
    """Resize a binary mask; legacy area/0.5 can erase thin defects.

    Nearest-neighbour preserves label values, but any downsampling can miss small regions.
    The evaluation protocol must record resolution and interpolation.
    """
    if mode not in ("area", "nearest"):
        raise ValueError(f"Unknown mask resize mode: {mode}")
    h, w = size
    if mask.shape[:2] == (h, w):
        return (mask > 127).astype(np.float32)
    interp = cv2.INTER_NEAREST_EXACT if mode == "nearest" else cv2.INTER_AREA
    m = cv2.resize(mask.astype(np.float32) / 255.0, (w, h), interpolation=interp)
    return (m >= 0.5).astype(np.float32)


class AnomalyDataset(Dataset):
    """One category / one split.

    Returns dicts with
      ``image``  float32 tensor (3, H, W) in [0, 1] (models normalize internally),
      ``mask``   float32 tensor (1, h, w) in {0, 1} at ``mask_size``,
      ``label``  int (0 normal / 1 anomalous), plus ``category``, ``defect_type``, ``path``.
    """

    def __init__(
        self,
        root: str | Path,
        records: list[dict],
        category: str,
        image_size: int | Sequence[int] = 256,
        mask_size: int | Sequence[int] = 256,
        perturbation: str | None = None,
        perturbation_seed: int = 0,
        augment: Callable[[torch.Tensor], torch.Tensor] | None = None,
        preload: bool = False,
        preload_workers: int = 8,
        mask_resize: str = "area",
    ) -> None:
        self.root = Path(root)
        self.records = list(records)
        self.category = category
        self.image_size = _size2(image_size)
        self.mask_size = _size2(mask_size)
        self.mask_resize = mask_resize
        self.perturbation = perturbation
        self.perturbation_seed = perturbation_seed
        self.augment = augment
        self._cache_imgs: torch.Tensor | None = None
        self._cache_masks: torch.Tensor | None = None
        if preload and self.records:
            with ThreadPoolExecutor(max_workers=preload_workers) as ex:
                loaded = list(ex.map(self._load, range(len(self.records))))
            imgs = np.stack([np.transpose(im, (2, 0, 1)) for im, _ in loaded], axis=0)
            self._cache_imgs = torch.from_numpy(np.ascontiguousarray(imgs)).share_memory_()
            if any(r["mask"] is not None for r in self.records):
                masks = np.stack([mk for _, mk in loaded], axis=0)
                self._cache_masks = torch.from_numpy(np.ascontiguousarray(masks)).share_memory_()

    def __len__(self) -> int:
        return len(self.records)

    @property
    def labels(self) -> np.ndarray:
        return np.array([r["label"] for r in self.records], dtype=np.int64)

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        rec = self.records[idx]
        img = read_rgb(self.root / rec["image"])
        mask = read_mask(self.root / rec["mask"] if rec["mask"] else None, img.shape[:2])
        if self.perturbation:
            img, mask = apply_perturbation(self.perturbation, img, mask, rec["image"], self.perturbation_seed)
        return resize_image(img, self.image_size), resize_mask(mask, self.mask_size, self.mask_resize)

    def __getitem__(self, idx: int) -> dict:
        if self._cache_imgs is not None:
            image = self._cache_imgs[idx].float().div_(255.0)
            mask_t = (self._cache_masks[idx][None].float() if self._cache_masks is not None
                      else torch.zeros(1, *self.mask_size, dtype=torch.float32))
        else:
            img, mask = self._load(idx)
            image = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
            mask_t = torch.from_numpy(mask)[None]
        if self.augment is not None:
            image = self.augment(image)
        rec = self.records[idx]
        return {
            "image": image,
            "mask": mask_t,
            "label": rec["label"],
            "category": self.category,
            "defect_type": rec["defect_type"],
            "path": rec["image"],
        }


class DataModule:
    """Resolves dataset root + manifest and builds per-category / unified datasets."""

    def __init__(self, data_cfg: dict) -> None:
        self.cfg = data_cfg
        self.name = data_cfg["name"]
        self.root = resolve_path(data_cfg["root"])
        manifest_path = resolve_path(data_cfg["split_manifest"])
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Split manifest {manifest_path} not found. Run `python scripts/prepare_splits.py "
                f"--data <config>` first.")
        self.manifest = load_manifest(manifest_path)
        self.categories: list[str] = list(data_cfg.get("categories") or self.manifest["categories"].keys())
        self.mask_size = data_cfg.get("mask_size", 256)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

    def records(self, category: str, split: str) -> list[dict]:
        return self.manifest["categories"][category][split]

    def dataset(self, category: str, split: str, image_size, **kw) -> AnomalyDataset:
        return AnomalyDataset(self.root, self.records(category, split), category, image_size,
                              mask_size=kw.pop("mask_size", self.mask_size),
                              mask_resize=kw.pop("mask_resize", self.cfg.get("mask_resize", "area")), **kw)

    def unified(self, split: str, image_size, categories: list[str] | None = None, **kw) -> ConcatDataset:
        cats = categories or self.categories
        return ConcatDataset([self.dataset(c, split, image_size, **kw) for c in cats])


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int = 4, drop_last: bool = False) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                      persistent_workers=num_workers > 0 and shuffle)


def infinite(loader: DataLoader) -> Iterator[dict]:
    if len(loader) == 0:
        raise ValueError("Empty training loader: reduce batch_size, disable drop_last, or provide more images.")
    while True:
        for batch in loader:
            yield batch
