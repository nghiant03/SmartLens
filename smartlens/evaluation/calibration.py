"""Validation-only calibration: quantile normalization and decision thresholds (spec §4.2).

Nothing in this module ever looks at the locked ``test`` split:
* pixel quantiles and the unsupervised threshold use ``val_normal`` (normal images only);
* the F1-tuned threshold uses the stratified ``test_val`` split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from smartlens.evaluation.metrics import f1_max


@dataclass
class Calibration:
    pix_q50: float
    pix_q90: float
    pix_q995: float
    img_mean: float
    img_std: float
    tau_unsup_p99: float      # 99th percentile of normal-val image scores  (~1 % expected FPR)
    tau_unsup_3sigma: float   # mu + 3 sigma of normal-val image scores
    tau_f1: float | None = None  # F1-optimal threshold on test_val (validation-tuned setting)

    def to_dict(self) -> dict:
        return asdict(self)

    def normalize_map(self, amap: np.ndarray) -> np.ndarray:
        """Map onto a common false-alarm scale: 0 at q90, 0.1 at q99.5 of normal-val pixels."""
        return 0.1 * (amap - self.pix_q90) / (self.pix_q995 - self.pix_q90 + 1e-12)


def pixel_quantiles(maps: np.ndarray, qs=(0.5, 0.90, 0.995), max_samples: int = 20_000_000, seed: int = 0) -> list[float]:
    flat = maps.ravel()
    if flat.size > max_samples:
        flat = np.random.default_rng(seed).choice(flat, max_samples, replace=False)
    return [float(v) for v in np.quantile(flat, qs)]


def calibrate_from_val_normal(val_maps: np.ndarray, val_scores: np.ndarray, percentile: float = 99.0) -> Calibration:
    q50, q90, q995 = pixel_quantiles(val_maps)
    s = np.asarray(val_scores, dtype=np.float64)
    return Calibration(
        pix_q50=q50, pix_q90=q90, pix_q995=q995,
        img_mean=float(s.mean()), img_std=float(s.std()),
        tau_unsup_p99=float(np.percentile(s, percentile)),
        tau_unsup_3sigma=float(s.mean() + 3 * s.std()),
    )


def f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """F1-maximizing threshold on a labelled validation split (None if single-class)."""
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return None
    _, thr = f1_max(labels, np.asarray(scores, dtype=np.float64))
    return thr
