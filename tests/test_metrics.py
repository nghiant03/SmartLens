import numpy as np
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from smartlens.evaluation.calibration import calibrate_from_val_normal, f1_threshold
from smartlens.evaluation.metrics import (au_pro, auroc, average_precision, compute_pro_curve, f1_max, fpr_at_tpr,
                                          pixel_metrics, threshold_metrics, tpr_at_fpr, trapezoid)


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    s = rng.normal(size=2000) + 1.2 * y
    s[:50] = np.round(s[:50], 1)  # ties
    return y, s


def test_auroc_ap_match_sklearn(scores):
    y, s = scores
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-10)
    assert average_precision(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-10)


def test_f1max_matches_sklearn(scores):
    y, s = scores
    p, r, _ = precision_recall_curve(y, s)
    ref = np.max(2 * p * r / np.maximum(p + r, 1e-12))
    f1, thr = f1_max(y, s)
    assert f1 == pytest.approx(ref, abs=1e-10)
    assert threshold_metrics(y, s, thr)["f1"] == pytest.approx(f1, abs=1e-10)


def test_fpr_at_95(scores):
    y, s = scores
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    assert fpr_at_tpr(y, s, 0.95) == pytest.approx(fpr[np.argmax(tpr >= 0.95)], abs=1e-10)


def test_tpr_at_low_fpr_matches_attainable_sklearn_curve(scores):
    y, s = scores
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    assert tpr_at_fpr(y, s, .01) == pytest.approx(tpr[fpr <= .01].max())
    # A tied normal/anomalous pair cannot be fractionally accepted to satisfy the budget.
    assert tpr_at_fpr([0, 1], [.5, .5], .01) == 0.0


def test_low_fpr_pro_uses_equal_region_weighting():
    gt = np.zeros((2, 10, 10), np.uint8)
    gt[0, :1, :1] = 1
    gt[1, :5, :5] = 1
    pred = np.zeros_like(gt, dtype=np.float32)
    pred[1] = gt[1]
    metrics = pixel_metrics(pred, gt)
    assert metrics["AU-PRO@5%FPR"] == pytest.approx(au_pro(pred, gt, .05))
    assert metrics["AU-PRO@5%FPR"] < metrics["AU-PRO"]


def test_degenerate_returns_nan():
    assert np.isnan(auroc(np.zeros(5), np.arange(5)))
    assert np.isnan(average_precision(np.zeros(5), np.arange(5)))


def test_trapezoid_partial():
    x = np.array([0.0, 0.5, 1.0]); y = np.array([0.0, 0.5, 1.0])
    assert trapezoid(x, y, 0.3) == pytest.approx(0.045)
    assert trapezoid(x, y) == pytest.approx(0.5)


def test_aupro_perfect_and_region_weighting():
    gt = np.zeros((2, 32, 32), np.uint8)
    gt[0, 2:4, 2:4] = 1        # small region
    gt[1, 10:30, 10:30] = 1    # large region
    assert au_pro(gt.astype(np.float32), gt) == pytest.approx(1.0)
    # detecting only the large region gives PRO = 0.5 (regions weighted equally, not pixels)
    pred = np.zeros_like(gt, dtype=np.float32); pred[1] = gt[1]
    fprs, pros = compute_pro_curve(pred, gt)
    assert pros[np.searchsorted(fprs, 0.0, side="right") - 1] == pytest.approx(0.5)
    # random scores: PRO ~= FPR, so normalized area up to 0.3 FPR ~= 0.15
    rng = np.random.default_rng(0)
    big = np.zeros((40, 32, 32), np.uint8)
    for i in range(40):
        y0, x0 = rng.integers(0, 24, 2)
        big[i, y0:y0 + 8, x0:x0 + 8] = 1
    assert au_pro(rng.random(big.shape).astype(np.float32), big) == pytest.approx(0.15, abs=0.03)


def test_calibration_thresholds():
    rng = np.random.default_rng(1)
    maps = rng.random((10, 16, 16)); s = rng.normal(size=200)
    c = calibrate_from_val_normal(maps, s)
    assert c.pix_q90 < c.pix_q995
    assert np.mean(s >= c.tau_unsup_p99) == pytest.approx(0.01, abs=0.006)
    assert f1_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(0.8)
    assert f1_threshold(np.zeros(3), np.arange(3)) is None
