"""Image-level and pixel-level anomaly detection metrics (spec §6.1).

Image level : I-AUROC, I-AP, I-F1-max, FPR@95%TPR, and Precision / Recall / F1 / FPR at a
              given decision threshold.
Pixel level : P-AUROC, P-AP, P-F1-max and AU-PRO (integrated up to 30 % FPR, normalized),
              following the official MVTec AD evaluation code.

All curve metrics are computed exactly from a single descending sort (no binning).
"""

from __future__ import annotations

from bisect import bisect

import numpy as np
from scipy.ndimage import label as cc_label

NAN = float("nan")


# --------------------------------------------------------------------------------------
# Exact binary curves
# --------------------------------------------------------------------------------------
def binary_clf_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Cumulative (fps, tps, thresholds) at every distinct score (descending), like sklearn."""
    y_true = np.asarray(y_true).ravel().astype(bool)
    y_score = np.asarray(y_score).ravel()
    order = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[order]
    y_true = y_true[order]
    distinct = np.where(np.diff(y_score))[0]
    idx = np.r_[distinct, y_true.size - 1]
    tps = np.cumsum(y_true, dtype=np.int64)[idx]
    fps = (idx + 1) - tps
    return fps, tps, y_score[idx]


def _curves(y_true, y_score):
    fps, tps, thr = binary_clf_curve(y_true, y_score)
    p, n = tps[-1], fps[-1]
    return fps, tps, thr, p, n


def auroc(y_true, y_score) -> float:
    fps, tps, _, p, n = _curves(y_true, y_score)
    if p == 0 or n == 0:
        return NAN
    tpr = np.r_[0.0, tps / p]
    fpr = np.r_[0.0, fps / n]
    return float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))


def average_precision(y_true, y_score) -> float:
    """Step-wise AP = sum_n (R_n - R_{n-1}) P_n (identical to sklearn)."""
    fps, tps, _, p, n = _curves(y_true, y_score)
    if p == 0:
        return NAN
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / p
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def f1_max(y_true, y_score) -> tuple[float, float]:
    """Maximum F1 over all thresholds and the threshold achieving it."""
    fps, tps, thr, p, n = _curves(y_true, y_score)
    if p == 0:
        return NAN, NAN
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / p
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    i = int(np.argmax(f1))
    return float(f1[i]), float(thr[i])


def fpr_at_tpr(y_true, y_score, tpr_level: float = 0.95) -> float:
    fps, tps, _, p, n = _curves(y_true, y_score)
    if p == 0 or n == 0:
        return NAN
    tpr = tps / p
    fpr = fps / n
    ok = np.where(tpr >= tpr_level)[0]
    return float(fpr[ok[0]]) if ok.size else 1.0


def tpr_at_fpr(y_true, y_score, fpr_limit: float = 0.01) -> float:
    """Best attainable TPR within an empirical FPR budget; no interpolation through ties.

    This is a test-curve diagnostic, not a validation-calibrated operating threshold.
    """
    fps, tps, _, p, n = _curves(y_true, y_score)
    if p == 0 or n == 0:
        return NAN
    allowed = tps[fps / n <= fpr_limit]
    return float(allowed.max() / p) if allowed.size else 0.0


# --------------------------------------------------------------------------------------
# Thresholded (decision-level) metrics
# --------------------------------------------------------------------------------------
def threshold_metrics(y_true, y_score, tau: float) -> dict:
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_score) >= tau
    tp = int(np.sum(pred & y_true)); fp = int(np.sum(pred & ~y_true))
    fn = int(np.sum(~pred & y_true)); tn = int(np.sum(~pred & ~y_true))
    precision = tp / (tp + fp) if tp + fp else NAN
    recall = tp / (tp + fn) if tp + fn else NAN
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {
        "threshold": float(tau), "precision": precision, "recall": recall, "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else NAN,
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def image_metrics(labels, scores) -> dict:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    f1, thr = f1_max(labels, scores)
    return {
        "I-AUROC": auroc(labels, scores),
        "I-AP": average_precision(labels, scores),
        "I-F1max": f1,
        "I-F1max_threshold": thr,
        "I-FPR@95TPR": fpr_at_tpr(labels, scores, 0.95),
        "I-TPR@1%FPR": tpr_at_fpr(labels, scores, 0.01),
    }


# --------------------------------------------------------------------------------------
# Pixel-level metrics
# --------------------------------------------------------------------------------------
def trapezoid(x, y, x_max=None) -> float:
    """Trapezoidal area under (x, y), optionally integrated only up to ``x_max`` with
    linear interpolation at the boundary (official MVTec AD implementation)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            ins = bisect(x, x_max)
            assert 0 < ins < len(x), "x_max must lie inside the curve support"
            y_interp = y[ins - 1] + (y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1])
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])
        mask = x <= x_max
        x, y = x[mask], y[mask]
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction)


def compute_pro_curve(anomaly_maps: np.ndarray, gt_masks: np.ndarray):
    """Exact per-region-overlap vs. FPR curve (Bergmann et al., IJCV 2021).

    ``anomaly_maps`` (N, H, W) float, ``gt_masks`` (N, H, W) {0, 1}. Each connected GT
    component (8-connectivity) contributes equally to PRO irrespective of its size.
    """
    structure = np.ones((3, 3), dtype=int)
    n = anomaly_maps.shape[0]
    fp_changes = np.zeros(anomaly_maps.shape, dtype=np.uint32)
    pro_changes = np.zeros(anomaly_maps.shape, dtype=np.float64)
    num_ok, num_regions = 0, 0
    for i in range(n):
        labeled, n_comp = cc_label(gt_masks[i] > 0, structure)
        num_regions += n_comp
        ok = labeled == 0
        num_ok += int(ok.sum())
        fp_changes[i, ok] = 1
        if n_comp:
            sizes = np.bincount(labeled.ravel())
            region = ~ok
            pro_changes[i, region] = 1.0 / sizes[labeled[region]]
    if num_regions == 0 or num_ok == 0:
        return None, None
    flat = anomaly_maps.ravel()
    order = np.argsort(flat, kind="mergesort")[::-1]
    scores_sorted = flat[order]
    fprs = np.cumsum(fp_changes.ravel()[order], dtype=np.float64) / num_ok
    pros = np.cumsum(pro_changes.ravel()[order]) / num_regions
    keep = np.append(np.diff(scores_sorted) != 0, True)
    fprs, pros = np.clip(fprs[keep], 0, 1), np.clip(pros[keep], 0, 1)
    return np.r_[0.0, fprs, 1.0], np.r_[0.0, pros, 1.0]


def au_pro(anomaly_maps: np.ndarray, gt_masks: np.ndarray, integration_limit: float = 0.3) -> float:
    fprs, pros = compute_pro_curve(anomaly_maps, gt_masks)
    if fprs is None:
        return NAN
    return trapezoid(fprs, pros, x_max=integration_limit) / integration_limit


def pixel_metrics(anomaly_maps: np.ndarray, gt_masks: np.ndarray, pro_limit: float = 0.3,
                  compute_pro: bool = True) -> dict:
    y = gt_masks.ravel() > 0
    s = anomaly_maps.ravel().astype(np.float32)
    f1, thr = f1_max(y, s)
    out = {
        "P-AUROC": auroc(y, s),
        "P-AP": average_precision(y, s),
        "P-F1max": f1,
        "P-F1max_threshold": thr,
    }
    out["AU-PRO"], out["AU-PRO@5%FPR"] = NAN, NAN
    if compute_pro:
        fprs, pros = compute_pro_curve(anomaly_maps.astype(np.float32), gt_masks)
        if fprs is not None:
            out["AU-PRO"] = trapezoid(fprs, pros, x_max=pro_limit) / pro_limit
            out["AU-PRO@5%FPR"] = trapezoid(fprs, pros, x_max=0.05) / 0.05
    return out


def per_image_pixel_ap(anomaly_maps: np.ndarray, gt_masks: np.ndarray, labels: np.ndarray) -> list[float]:
    """Per-image pixel AP on anomalous images with non-empty masks (NaN on normal/empty-mask images)."""
    out: list[float] = []
    for i, lbl in enumerate(labels):
        mask = gt_masks[i] > 0
        if int(lbl) == 1 and mask.any():
            out.append(average_precision(mask.ravel(), anomaly_maps[i].ravel().astype(np.float32)))
        else:
            out.append(NAN)
    return out


# --------------------------------------------------------------------------------------
# Paired image-level bootstrap (treats images, never correlated pixels, as independent)
# --------------------------------------------------------------------------------------
def _pack_for_bootstrap(results: list[dict]) -> dict[str, dict]:
    packed = {}
    for r in results:
        pred = r.get("predictions")
        if not pred:
            raise ValueError(f"Category {r.get('category')!r} lacks per-image predictions for paired bootstrap.")
        cal = r.get("calibration", {})
        labels = np.asarray(pred["labels"], dtype=np.int64)
        packed[r["category"]] = {
            "paths": list(pred["paths"]),
            "labels": labels,
            "norm_idx": np.flatnonzero(labels == 0),
            "anom_idx": np.flatnonzero(labels == 1),
            "scores": np.asarray(pred["scores"], dtype=np.float64),
            "anom_p_ap": (np.array([NAN if v is None else float(v) for v in pred["anom_p_ap"]], dtype=np.float64)
                          if "anom_p_ap" in pred and pred["anom_p_ap"] is not None else None),
            "optshift_scores": {k: np.asarray(v, dtype=np.float64)
                                for k, v in (pred.get("optshift_scores") or {}).items()},
            "tau_unsup": float(cal["tau_unsup_p99"]),
            "tau_f1": float(cal["tau_f1"]) if cal.get("tau_f1") is not None else None,
        }
    return packed


def _eval_bootstrap_sample(packed: dict[str, dict], sample_idx: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                           perts: list[str], has_f1: bool, has_img_pap: bool) -> dict[str, float]:
    cats = list(packed.keys())
    vals: dict[str, list[float]] = {
        "I-AUROC": [], "I-AP": [], "I-FPR@95TPR": [],
        "at_tau_unsup/recall": [], "at_tau_unsup/fpr": [],
    }
    if has_f1:
        vals["at_tau_f1/f1"] = []
    if has_img_pap:
        vals["Img-P-AP"] = []
    if perts:
        vals["optshift_mean/I-AUROC"] = []
        vals["optshift_mean/at_tau_unsup/fpr"] = []

    for c in cats:
        d = packed[c]
        b_norm, b_anom, b_all = sample_idx[c]
        y, s = d["labels"][b_all], d["scores"][b_all]
        vals["I-AUROC"].append(auroc(y, s))
        vals["I-AP"].append(average_precision(y, s))
        vals["I-FPR@95TPR"].append(fpr_at_tpr(y, s, 0.95))
        vals["at_tau_unsup/recall"].append(float(np.mean(d["scores"][b_anom] >= d["tau_unsup"]))
                                           if b_anom.size else NAN)
        vals["at_tau_unsup/fpr"].append(float(np.mean(d["scores"][b_norm] >= d["tau_unsup"]))
                                        if b_norm.size else NAN)
        if has_f1 and d["tau_f1"] is not None:
            vals["at_tau_f1/f1"].append(threshold_metrics(y, s, d["tau_f1"])["f1"])
        if has_img_pap and d["anom_p_ap"] is not None:
            ap = d["anom_p_ap"][b_anom]
            vals["Img-P-AP"].append(float(np.nanmean(ap)) if np.isfinite(ap).any() else NAN)
        if perts:
            opt_auroc, opt_fpr = [], []
            for p in perts:
                sp = d["optshift_scores"][p]
                opt_auroc.append(auroc(y, sp[b_all]))
                opt_fpr.append(float(np.mean(sp[b_norm] >= d["tau_unsup"])) if b_norm.size else NAN)
            vals["optshift_mean/I-AUROC"].append(float(np.nanmean(opt_auroc)))
            vals["optshift_mean/at_tau_unsup/fpr"].append(float(np.nanmean(opt_fpr)))
    return {k: float(np.nanmean(v)) if v else NAN for k, v in vals.items()}


def paired_bootstrap_ci(ref_results: list[dict], cand_results: list[dict], n_boot: int = 1000,
                        seed: int = 42, ci: float = 0.95) -> list[dict]:
    """Stratified image-level paired bootstrap CI for ``candidate - reference`` macro-means."""
    ref = _pack_for_bootstrap(ref_results)
    cand = _pack_for_bootstrap(cand_results)
    if set(ref) != set(cand):
        raise ValueError(f"Mismatched category sets in paired bootstrap: {sorted(ref)} vs {sorted(cand)}")
    cats = sorted(ref)
    for c in cats:
        if ref[c]["paths"] != cand[c]["paths"] or not np.array_equal(ref[c]["labels"], cand[c]["labels"]):
            raise ValueError(f"Unaligned test image order/labels in category {c!r}; cannot pair bootstrap.")

    has_f1 = all(ref[c]["tau_f1"] is not None and cand[c]["tau_f1"] is not None for c in cats)
    has_img_pap = all(ref[c]["anom_p_ap"] is not None and cand[c]["anom_p_ap"] is not None for c in cats)
    perts = sorted(set.intersection(*(set(ref[c]["optshift_scores"]) & set(cand[c]["optshift_scores"]) for c in cats))) \
        if cats else []

    obs_idx = {c: (ref[c]["norm_idx"], ref[c]["anom_idx"], np.arange(len(ref[c]["labels"]))) for c in cats}
    ref_obs = _eval_bootstrap_sample(ref, obs_idx, perts, has_f1, has_img_pap)
    cand_obs = _eval_bootstrap_sample(cand, obs_idx, perts, has_f1, has_img_pap)
    metrics = list(ref_obs.keys())
    deltas = {m: np.empty(n_boot, dtype=np.float64) for m in metrics}

    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        b_idx = {}
        for c in cats:
            ni, ai = ref[c]["norm_idx"], ref[c]["anom_idx"]
            bn = rng.choice(ni, size=ni.size, replace=True) if ni.size else ni
            ba = rng.choice(ai, size=ai.size, replace=True) if ai.size else ai
            b_idx[c] = (bn, ba, np.concatenate([bn, ba]))
        r_b = _eval_bootstrap_sample(ref, b_idx, perts, has_f1, has_img_pap)
        c_b = _eval_bootstrap_sample(cand, b_idx, perts, has_f1, has_img_pap)
        for m in metrics:
            deltas[m][b] = c_b[m] - r_b[m]

    alpha = (1.0 - ci) / 2.0
    rows = []
    for m in metrics:
        d = deltas[m]
        finite = d[np.isfinite(d)]
        lo, hi = (np.percentile(finite, [100 * alpha, 100 * (1 - alpha)]) if finite.size else (NAN, NAN))
        rows.append({
            "metric": m,
            "ref": ref_obs[m],
            "cand": cand_obs[m],
            "diff": cand_obs[m] - ref_obs[m],
            "ci_low": float(lo),
            "ci_high": float(hi),
            "std_err": float(np.std(finite, ddof=1)) if finite.size > 1 else NAN,
            "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
        })
    return rows
