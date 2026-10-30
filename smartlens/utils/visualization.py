"""Plotting helpers: heatmap overlays, GT contours and diagnostic panels."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """(3, H, W) or (H, W, 3) float in [0, 1] / uint8 -> (H, W, 3) uint8."""
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def heatmap_overlay(image: np.ndarray, amap: np.ndarray, vmin: float, vmax: float, alpha: float = 0.5) -> np.ndarray:
    img = to_uint8_rgb(image)
    if amap.shape != img.shape[:2]:
        amap = cv2.resize(amap.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    norm = np.clip((amap - vmin) / (vmax - vmin + 1e-12), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)


def draw_contours(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), thickness: int = 1) -> np.ndarray:
    img = to_uint8_rgb(image).copy()
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)
    return img


def put_label(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(img, text[:48], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return img


def diagnostic_panel(image: np.ndarray, mask: np.ndarray, amap: np.ndarray, vmin: float, vmax: float,
                     threshold_map: float | None = None, title: str = "") -> np.ndarray:
    """[input + GT contour | heatmap overlay | thresholded prediction (red) vs GT (green)]."""
    rgb = to_uint8_rgb(image)
    gt = draw_contours(rgb, mask)
    over = heatmap_overlay(rgb, amap, vmin, vmax)
    pred = rgb.copy()
    if threshold_map is not None:
        a = amap if amap.shape == rgb.shape[:2] else cv2.resize(amap, (rgb.shape[1], rgb.shape[0]))
        pred = draw_contours(pred, (a >= threshold_map).astype(np.uint8), color=(255, 0, 0), thickness=2)
    pred = draw_contours(pred, mask)
    return np.concatenate([put_label(gt, title), put_label(over, "anomaly map"),
                           put_label(pred, "pred (red) / GT (green)")], axis=1)


def save_rgb(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
