"""Leakage-free evaluation runner: inference, calibration, metrics, OptShift, galleries."""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from smartlens.data.datasets import DataModule, make_loader
from smartlens.evaluation.calibration import Calibration, calibrate_from_val_normal, f1_threshold
from smartlens.evaluation.error_analysis import export_error_gallery
from smartlens.evaluation.metrics import image_metrics, per_image_pixel_ap, pixel_metrics, threshold_metrics
from smartlens.models.base import AnomalyModel, log


@dataclass
class Predictions:
    scores: np.ndarray                 # (N,)
    labels: np.ndarray                 # (N,)
    maps: np.ndarray                   # (N, h, w) float32 at evaluation resolution
    masks: np.ndarray                  # (N, h, w) uint8
    paths: list[str] = field(default_factory=list)
    defect_types: list[str] = field(default_factory=list)


@torch.no_grad()
def run_inference(model: AnomalyModel, dataset, device: torch.device, batch_size: int = 16,
                  num_workers: int = 4, desc: str = "infer") -> Predictions:
    model.eval()
    loader = make_loader(dataset, batch_size, False, num_workers)
    scores, labels, maps, masks, paths, dtypes = [], [], [], [], [], []
    for b in tqdm(loader, desc=desc, leave=False):
        out = model.predict(b["image"].to(device, non_blocking=True))
        amap = out["anomaly_map"].float()[:, None]
        size = tuple(b["mask"].shape[-2:])
        if tuple(amap.shape[-2:]) != size:
            down = amap.shape[-1] > size[-1]
            amap = F.interpolate(amap, size=size, mode="bilinear", align_corners=False, antialias=down)
        maps.append(amap[:, 0].cpu().numpy().astype(np.float32))
        scores.append(out["image_score"].float().cpu().numpy())
        labels.append(np.asarray(b["label"]))
        masks.append(b["mask"][:, 0].numpy().astype(np.uint8))
        paths += list(b["path"])
        dtypes += list(b["defect_type"])
    return Predictions(np.concatenate(scores).astype(np.float64), np.concatenate(labels), np.concatenate(maps),
                       np.concatenate(masks), paths, dtypes)


def score_predictions(preds: Predictions, calib: Calibration, compute_pro: bool = True, pro_limit: float = 0.3) -> dict:
    res: dict = {"n_images": int(len(preds.labels)), "n_anomalous": int(preds.labels.sum()),
                 "n_normal": int((preds.labels == 0).sum()),
                 "n_anomalous_empty_masks": int(((preds.labels == 1) &
                                                 ~preds.masks.reshape(len(preds.labels), -1).any(axis=1)).sum())}
    res.update(image_metrics(preds.labels, preds.scores))
    res.update(pixel_metrics(preds.maps, preds.masks, pro_limit=pro_limit, compute_pro=compute_pro))
    res["at_tau_unsup"] = threshold_metrics(preds.labels, preds.scores, calib.tau_unsup_p99)
    res["at_tau_unsup_3sigma"] = threshold_metrics(preds.labels, preds.scores, calib.tau_unsup_3sigma)
    if calib.tau_f1 is not None:
        res["at_tau_f1"] = threshold_metrics(preds.labels, preds.scores, calib.tau_f1)
    # recall per defect sub-type at the validation-tuned (else unsupervised) threshold
    tau = calib.tau_f1 if calib.tau_f1 is not None else calib.tau_unsup_p99
    dt = np.asarray(preds.defect_types)
    res["recall_per_defect"] = {d: float(np.mean(preds.scores[dt == d] >= tau))
                                for d in sorted(set(preds.defect_types)) if d != "good"}
    return res


class Evaluator:
    def __init__(self, dm: DataModule, eval_cfg: dict, device: torch.device, out_dir: Path) -> None:
        self.dm, self.cfg, self.device, self.out_dir = dm, eval_cfg, device, Path(out_dir)
        self.bs = eval_cfg.get("batch_size", 16)
        self.nw = eval_cfg.get("num_workers", 4)

    def _ds(self, model, category, split, **kw):
        return self.dm.dataset(category, split, model.input_size, **kw)

    def calibrate(self, model: AnomalyModel, category: str, tune_f1: bool = True) -> Calibration:
        val = run_inference(model, self._ds(model, category, "val_normal"), self.device, self.bs, self.nw,
                            f"{category}:val_normal")
        calib = calibrate_from_val_normal(val.maps, val.scores, self.cfg.get("unsup_percentile", 99.0))
        if tune_f1:
            tv = run_inference(model, self._ds(model, category, "test_val"), self.device, self.bs, self.nw,
                               f"{category}:test_val")
            calib.tau_f1 = f1_threshold(tv.labels, tv.scores)
        return calib

    def calibrate_pooled(self, model_for_category: Callable[[str], AnomalyModel], categories: list[str],
                         tune_f1: bool = True, max_pixels_per_category: int = 2_000_000) -> Calibration:
        """Fit one category-blind calibration across pooled validation splits (true label-free deployment)."""
        map_samples, val_scores = [], []
        tv_labels, tv_scores = [], []
        for category in categories:
            model = model_for_category(category)
            val = run_inference(model, self._ds(model, category, "val_normal"), self.device, self.bs, self.nw,
                                f"{category}:val_normal[pooled]")
            flat = val.maps.ravel()
            if flat.size > max_pixels_per_category:
                rng = np.random.default_rng(zlib.crc32(category.encode()))
                flat = rng.choice(flat, max_pixels_per_category, replace=False)
            map_samples.append(flat)
            val_scores.append(val.scores)
            if tune_f1:
                tv = run_inference(model, self._ds(model, category, "test_val"), self.device, self.bs, self.nw,
                                   f"{category}:test_val[pooled]")
                tv_labels.append(tv.labels)
                tv_scores.append(tv.scores)
        calib = calibrate_from_val_normal(np.concatenate(map_samples), np.concatenate(val_scores),
                                          self.cfg.get("unsup_percentile", 99.0))
        if tune_f1 and tv_labels:
            calib.tau_f1 = f1_threshold(np.concatenate(tv_labels), np.concatenate(tv_scores))
        return calib

    def evaluate_category(self, model: AnomalyModel, category: str, split: str = "test",
                          perturbations: list[str] | None = None, gallery: bool = False,
                          calib: Calibration | None = None) -> dict:
        if calib is None:
            calib = self.calibrate(model, category, tune_f1=split != "test_val")
        preds = run_inference(model, self._ds(model, category, split), self.device, self.bs, self.nw,
                              f"{category}:{split}")
        compute_pro = self.cfg.get("compute_pro", True)
        pro_limit = self.cfg.get("pro_integration_limit", 0.3)
        clean = score_predictions(preds, calib, compute_pro, pro_limit)
        img_pap = per_image_pixel_ap(preds.maps, preds.masks, preds.labels)
        clean["Img-P-AP"] = float(np.nanmean(img_pap)) if np.isfinite(img_pap).any() else float("nan")
        opt_scores: dict[str, list[float]] = {}
        result = {"category": category, "split": split, "calibration": calib.to_dict(),
                  "n_val_normal": len(self.dm.records(category, "val_normal")),
                  "clean": clean}
        if gallery:
            tau = calib.tau_f1 if calib.tau_f1 is not None else calib.tau_unsup_p99
            vmax = float(np.quantile(preds.maps, 0.9995))
            vis_ds = self.dm.dataset(category, split, self.dm.mask_size)
            result["gallery_counts"] = export_error_gallery(
                vis_ds, preds, tau, calib.pix_q995, calib.pix_q90, max(vmax, calib.pix_q995 * 1.0001),
                self.out_dir / "galleries" / category, self.cfg.get("gallery_items", 8))
        for pert in perturbations or []:
            p_preds = run_inference(model, self._ds(model, category, split, perturbation=pert), self.device,
                                    self.bs, self.nw, f"{category}:{split}:{pert}")
            # thresholds stay those calibrated on *clean* validation data (deployment scenario)
            result.setdefault("optshift", {})[pert] = score_predictions(p_preds, calib, compute_pro, pro_limit)
            opt_scores[pert] = [float(v) for v in p_preds.scores.tolist()]
        result["predictions"] = {
            "paths": list(preds.paths),
            "labels": [int(v) for v in preds.labels.tolist()],
            "scores": [float(v) for v in preds.scores.tolist()],
            "anom_p_ap": [None if np.isnan(v) else float(v) for v in img_pap],
            "optshift_scores": opt_scores,
        }
        return result


def pooled_image_summary(results: list[dict], calib: Calibration) -> dict:
    """Global image-level ranking and operating metrics across all categories simultaneously."""
    labels = np.concatenate([np.asarray(r["predictions"]["labels"], dtype=np.int64) for r in results])
    scores = np.concatenate([np.asarray(r["predictions"]["scores"], dtype=np.float64) for r in results])
    out: dict = {"n_images": int(len(labels)), "n_anomalous": int(labels.sum()),
                 "n_normal": int((labels == 0).sum()), **image_metrics(labels, scores)}
    out["at_tau_unsup"] = threshold_metrics(labels, scores, calib.tau_unsup_p99)
    out["at_tau_unsup_3sigma"] = threshold_metrics(labels, scores, calib.tau_unsup_3sigma)
    if calib.tau_f1 is not None:
        out["at_tau_f1"] = threshold_metrics(labels, scores, calib.tau_f1)
    return out


# --------------------------------------------------------------------------------------
# Aggregation / reporting
# --------------------------------------------------------------------------------------
HEADLINE = ["P-AP", "Img-P-AP", "AU-PRO@5%FPR", "AU-PRO", "I-FPR@95TPR", "I-TPR@1%FPR",
            "I-AUROC", "I-AP", "I-F1max", "P-AUROC", "P-F1max"]


def flatten_result(res: dict, prefix: str = "") -> dict:
    row = {}
    for k, v in res.items():
        if isinstance(v, dict):
            if k == "recall_per_defect":
                continue
            row.update(flatten_result(v, f"{prefix}{k}/"))
        elif isinstance(v, (int, float, np.floating, np.integer)) or v is None:
            row[f"{prefix}{k}"] = float(v) if v is not None else math.nan
    return row


def results_table(results: list[dict], key: str = "clean", pert: str | None = None) -> pd.DataFrame:
    rows = []
    for r in results:
        src = r["optshift"][pert] if pert else r[key]
        rows.append({"category": r["category"], **flatten_result(src)})
    df = pd.DataFrame(rows).set_index("category")
    df.loc["mean"] = df.mean(numeric_only=True)
    return df


def save_results(results: list[dict], out_dir: Path, tag: str, extra: dict | None = None) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"results_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, **(extra or {})}, f, indent=1, default=float)
    df = results_table(results)
    df.to_csv(out_dir / f"metrics_{tag}.csv", float_format="%.5f")
    lines = [f"# Results ({tag})", ""]
    protocol = (extra or {}).get("protocol", {})
    if protocol:
        prec = ("BF16 AMP" if protocol.get("amp") else
                ("FP32 (TF32)" if protocol.get("allow_tf32", True) else "FP32 (IEEE)"))
        lines += [f"Dataset: **{protocol['dataset']}**; split: **{protocol['split']}**; "
                  f"category coverage: **{len(results)}/{len(protocol['configured_categories'])}**; "
                  f"mask resolution: **{protocol['mask_size']}** ({protocol['mask_resize']}); "
                  f"calibration: **{protocol.get('calibration_scope', 'per_category')}**; "
                  f"precision: **{prec}**.",
                  "", "Evaluated: " + ", ".join(protocol["evaluated_categories"]) + ".", ""]
        if not protocol["complete_category_coverage"]:
            lines += ["**Partial-category evaluation; not a full benchmark result.**", ""]
    lines += ["F1max and TPR@1%FPR are test-curve diagnostics, not deployable tuned thresholds. "
              "Operating-point metrics use held-out validation thresholds.", "", _markdown(df), ""]
    if extra and "pooled_clean" in extra:
        pooled_df = pd.DataFrame([flatten_result(extra["pooled_clean"])], index=["pooled_all"])
        lines += ["## Category-blind pooled image ranking", "", _markdown(pooled_df, index_name="scope"), ""]
    lines += ["## Sample support", "", "| category | test normal | test anomalous | normal validation | empty anomaly masks |",
              "|---|---:|---:|---:|---:|"]
    for r in results:
        clean = r["clean"]
        lines.append(f"| {r['category']} | {clean.get('n_normal', 'unknown')} | {clean['n_anomalous']} | "
                     f"{r.get('n_val_normal', 'unknown')} | {clean.get('n_anomalous_empty_masks', 'unknown')} |")
    lines.append("")
    perts = sorted({p for r in results for p in r.get("optshift", {})})
    if perts:
        opt_rows = {"clean": df.loc["mean"]}
        for p in perts:
            dfp = results_table(results, pert=p)
            dfp.to_csv(out_dir / f"metrics_{tag}_optshift_{p}.csv", float_format="%.5f")
            opt_rows[p] = dfp.loc["mean"]
        opt = pd.DataFrame(opt_rows).T
        opt.to_csv(out_dir / f"metrics_{tag}_optshift_summary.csv", float_format="%.5f")
        lines += ["## MVTec-OptShift (category mean, thresholds from clean validation)", "",
                  _markdown(opt, index_name="condition"), ""]
    if extra and "latency" in extra:
        lines += ["## System", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in extra["latency"].items()]
    (out_dir / f"summary_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Saved results to %s", out_dir)
    return df


def _markdown(df: pd.DataFrame, index_name: str = "category") -> str:
    cols = [c for c in HEADLINE if c in df.columns] + \
        [c for c in ("at_tau_unsup/precision", "at_tau_unsup/recall", "at_tau_unsup/fpr",
                     "at_tau_f1/precision", "at_tau_f1/recall", "at_tau_f1/f1", "at_tau_f1/fpr") if c in df.columns]
    short = {c: c.replace("at_tau_unsup/", "unsup-").replace("at_tau_f1/", "f1tuned-") for c in cols}
    head = "| " + index_name + " | " + " | ".join(short[c] for c in cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    body = []
    for idx, row in df[cols].iterrows():
        cells = ["–" if pd.isna(row[c]) else f"{100 * row[c]:.1f}" for c in cols]
        name = f"**{idx}**" if idx == "mean" else str(idx)
        body.append("| " + name + " | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])
