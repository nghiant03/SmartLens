"""Automated false-alarm / missed-defect gallery export (proposal: "review missed defects
and false alarms to understand the strengths and limitations of each method")."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from smartlens.utils.visualization import diagnostic_panel, save_rgb


def export_error_gallery(dataset, preds, tau: float, pixel_tau: float | None, vmin: float, vmax: float,
                         out_dir: str | Path, max_items: int = 8) -> dict:
    """Write ranked galleries for one category.

    * ``false_alarms``   – normal images with score >= tau, highest score first
    * ``missed_defects`` – anomalous images with score < tau, lowest score first
    * ``true_positives`` – detected anomalies (highest score first) for reference

    ``dataset`` must yield images at the same resolution as ``preds.maps``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, scores = preds.labels, preds.scores
    groups = {
        "false_alarms": [i for i in np.argsort(-scores) if labels[i] == 0 and scores[i] >= tau],
        "missed_defects": [i for i in np.argsort(scores) if labels[i] == 1 and scores[i] < tau],
        "true_positives": [i for i in np.argsort(-scores) if labels[i] == 1 and scores[i] >= tau],
    }
    rows, counts = [], {}
    for name, idxs in groups.items():
        counts[name] = len(idxs)
        panels = []
        for rank, i in enumerate(idxs[:max_items]):
            item = dataset[int(i)]
            title = f"{item['defect_type']} s={scores[i]:.3g} t={tau:.3g}"
            panels.append(diagnostic_panel(item["image"].numpy(), preds.masks[i], preds.maps[i], vmin, vmax,
                                           pixel_tau, title))
            rows.append({"group": name, "rank": rank, "path": preds.paths[i], "defect_type": preds.defect_types[i],
                         "score": float(scores[i]), "threshold": tau})
        if panels:
            save_rgb(out_dir / f"{name}.png", np.concatenate(panels, axis=0))
    with open(out_dir / "errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "rank", "path", "defect_type", "score", "threshold"])
        w.writeheader()
        w.writerows(rows)
    return counts
