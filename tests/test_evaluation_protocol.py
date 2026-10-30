import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.summarize import summarize
from smartlens.config import save_yaml
from smartlens.evaluation.evaluator import Evaluator, Predictions, score_predictions
from smartlens.evaluation.calibration import calibrate_from_val_normal
from smartlens.pipeline import evaluate


def test_test_val_is_not_used_to_tune_its_own_operating_threshold(monkeypatch, tmp_path):
    calls = []

    def infer(model, dataset, *args, **kwargs):
        calls.append(dataset)
        return Predictions(np.array([.1, .2]), np.zeros(2), np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))

    monkeypatch.setattr("smartlens.evaluation.evaluator.run_inference", infer)
    evaluator = Evaluator(None, {}, torch.device("cpu"), tmp_path)
    monkeypatch.setattr(evaluator, "_ds", lambda model, category, split: split)
    calib = evaluator.calibrate(None, "bottle", tune_f1=False)
    assert calls == ["val_normal"]
    assert calib.tau_f1 is None


def test_missing_categories_are_not_silently_averaged(monkeypatch, tmp_path):
    save_yaml({"data": {}, "model": {}}, tmp_path / "config.yaml")
    monkeypatch.setattr("smartlens.pipeline.DataModule", lambda cfg: SimpleNamespace(categories=["bottle", "screw"]))
    with pytest.raises(FileNotFoundError, match="partial benchmarks"):
        evaluate(tmp_path, torch.device("cpu"))


def test_empty_anomaly_masks_are_visible():
    preds = Predictions(np.array([.1, .8]), np.array([0, 1]),
                        np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), defect_types=["good", "crack"])
    calib = calibrate_from_val_normal(np.zeros((2, 2, 2)), np.array([.1, .2]))
    scores = score_predictions(preds, calib)
    assert scores["n_anomalous_empty_masks"] == 1
    assert scores["n_normal"] == 1
    assert np.isnan(scores["AU-PRO"])


def _write_result(root, model, category="bottle", mask_size=448):
    run = root / model
    (run / "eval").mkdir(parents=True)
    save_yaml({"model": {"name": model}, "data": {"name": "mvtec_ad", "categories": ["bottle", "screw"],
                                                 "mask_size": mask_size}}, run / "config.yaml")
    clean = {"n_images": 10, "n_anomalous": 5, "I-AUROC": 1.0}
    result = {"results": [{"category": category, "split": "test", "clean": clean}]}
    (run / "eval" / "results_test.json").write_text(json.dumps(result))


def test_comparison_excludes_retired_models_and_labels_partial_coverage(tmp_path):
    _write_result(tmp_path, "dinomaly")
    _write_result(tmp_path, "ensemble", category="screw")
    _write_result(tmp_path, "efficientad", category="screw")
    table = summarize(tmp_path, "test")
    assert list(table.index) == ["dinomaly"]
    text = (tmp_path / "comparison_test.md").read_text()
    assert "1/2 (PARTIAL/UNVERIFIED)" in text
    assert "test images" in text


@pytest.mark.parametrize("change", [{"category": "screw"}, {"mask_size": 256}])
def test_comparison_rejects_mismatched_protocols(tmp_path, change):
    _write_result(tmp_path, "dinomaly")
    _write_result(tmp_path, "patchcore", **change)
    with pytest.raises(ValueError, match="Incomparable evaluation protocols"):
        summarize(tmp_path, "test")
