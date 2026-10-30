"""MVTec-OptShift: controlled, unseen test-time optical / pose perturbations (spec §5.1.3).

Every perturbation operates on the *original-resolution* uint8 RGB image (H, W, 3) and its
binary mask (H, W) and is deterministic given ``(image path, seed)`` so all models see the
exact same perturbed test set.
"""

from __future__ import annotations

import zlib
from typing import Callable

import cv2
import numpy as np

Perturbation = Callable[[np.ndarray, np.ndarray | None, np.random.Generator], tuple[np.ndarray, np.ndarray | None]]


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def illumination_gradient(img, mask, rng, strength: float = 0.30):
    """Directional linear brightness ramp from (1 - strength) to (1 + strength) across the image."""
    h, w = img.shape[:2]
    theta = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = (xx - w / 2) * np.cos(theta) + (yy - h / 2) * np.sin(theta)
    proj /= np.abs(proj).max() + 1e-6  # in [-1, 1]
    gain = 1.0 + strength * proj
    return _to_uint8(img.astype(np.float32) * gain[..., None]), mask


def gamma_shift(img, mask, rng, gamma: float = 0.65):
    """Exposure change via gamma curve (gamma < 1 brightens, gamma > 1 darkens)."""
    lut = _to_uint8(255.0 * (np.arange(256) / 255.0) ** gamma)
    return lut[img], mask


def specular_glare(img, mask, rng, n_spots: tuple[int, int] = (1, 2), intensity: tuple[float, float] = (0.55, 0.85),
                   radius: tuple[float, float] = (0.04, 0.10)):
    """Localized Gaussian intensity bloom(s) simulating specular highlights."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(int(rng.integers(n_spots[0], n_spots[1] + 1))):
        cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
        sigma = rng.uniform(*radius) * min(h, w)
        a = rng.uniform(*intensity)
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)) * a
        out = out + (255.0 - out) * g[..., None]
    return _to_uint8(out), mask


def affine_misalignment(img, mask, rng, max_rot: float = 3.0, max_shift_px: float = 4.0, ref_size: int = 256):
    """Mild rotation (±max_rot deg) + translation (±max_shift_px at ``ref_size`` resolution)."""
    h, w = img.shape[:2]
    angle = rng.uniform(-max_rot, max_rot)
    tx = rng.uniform(-max_shift_px, max_shift_px) * w / ref_size
    ty = rng.uniform(-max_shift_px, max_shift_px) * h / ref_size
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[:, 2] += (tx, ty)
    img_o = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask_o = None
    if mask is not None:
        mask_o = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_o, mask_o


PERTURBATIONS: dict[str, Perturbation] = {
    "illum_gradient": illumination_gradient,
    "gamma_0.65": lambda i, m, r: gamma_shift(i, m, r, gamma=0.65),
    "gamma_1.45": lambda i, m, r: gamma_shift(i, m, r, gamma=1.45),
    "glare": specular_glare,
    "affine": affine_misalignment,
}


def apply_perturbation(name: str, img: np.ndarray, mask: np.ndarray | None, key: str, seed: int = 0):
    """Apply perturbation ``name`` deterministically (keyed by image path + seed)."""
    if name not in PERTURBATIONS:
        raise KeyError(f"Unknown perturbation {name!r}; available: {sorted(PERTURBATIONS)}")
    rng = np.random.default_rng(zlib.crc32(f"{name}|{key}".encode()) + seed)
    return PERTURBATIONS[name](img, mask, rng)
