import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.summarize import summarize
from smartlens.config import save_yaml
from smartlens.evaluation.calibration import calibrate_from_val_normal
from smartlens.evaluation.evaluator import Evaluator, Predictions, pooled_image_summary, score_predictions
from smartlens.evaluation.latency import _predict_at_resolution, aggregate_latency_profiles, profile_latency
from smartlens.models.base import AnomalyModel
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


def test_latency_resolution_matching_and_multi_checkpoint_aggregation():
    class Dummy224(AnomalyModel):
        def __init__(self, buf_elems: int):
            super().__init__({"name": "dummy", "input_size": 224})
            self.w = torch.nn.Parameter(torch.zeros(100))
            self.register_buffer("bank", torch.zeros(buf_elems))

        def fit(self, ctx):
            return {}

        def predict(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
            b = images.shape[0]
            return {"anomaly_map": torch.zeros(b, 224, 224, device=images.device),
                    "image_score": torch.zeros(b, device=images.device)}

    m1, m2 = Dummy224(1_000_000), Dummy224(3_000_000)
    x = torch.zeros(1, 3, 224, 224)
    assert _predict_at_resolution(m1, x, (448, 448))["anomaly_map"].shape == (1, 448, 448)
    p1 = profile_latency(m1, 224, torch.device("cpu"), warmup=1, iters=3, output_size=448)
    p2 = profile_latency(m2, 224, torch.device("cpu"), warmup=1, iters=3, output_size=448)
    p1["peak_vram_MB"], p2["peak_vram_MB"] = 410.0, 920.0
    p1["latency_ms_mean"], p2["latency_ms_mean"] = 4.0, 8.0
    agg = aggregate_latency_profiles([p1, p2])
    assert agg["input_size"] == [224, 224] and agg["output_size"] == [448, 448]
    assert agg["deployed_checkpoints"] == 2
    assert agg["peak_vram_MB"] == pytest.approx(920.0)
    assert agg["latency_ms_mean"] == pytest.approx(6.0)
    assert agg["latency_ms_max_category"] == pytest.approx(8.0)
    assert agg["deployed_footprint_M"] == pytest.approx(p1["deployed_footprint_M"] + p2["deployed_footprint_M"])


def test_pooled_calibration_shares_single_threshold_and_writes_pooled_summary(monkeypatch, tmp_path):
    # Category A has low normal scores (~0.1), Category B has shifted normal scores (~2.0)
    splits = {
        ("bottle", "val_normal"): Predictions(np.array([0.1, 0.2]), np.zeros(2), np.zeros((2, 4, 4)), np.zeros((2, 4, 4))),
        ("screw", "val_normal"): Predictions(np.array([1.8, 2.0]), np.zeros(2), np.ones((2, 4, 4)), np.zeros((2, 4, 4))),
        ("bottle", "test_val"): Predictions(np.array([0.1, 0.9]), np.array([0, 1]), np.zeros((2, 4, 4)), np.zeros((2, 4, 4))),
        ("screw", "test_val"): Predictions(np.array([1.9, 3.0]), np.array([0, 1]), np.zeros((2, 4, 4)), np.zeros((2, 4, 4))),
        ("bottle", "test"): Predictions(np.array([0.15, 0.95]), np.array([0, 1]), np.zeros((2, 4, 4), np.float32),
                                        np.zeros((2, 4, 4), np.uint8), paths=["b0.png", "b1.png"], defect_types=["good", "broken"]),
        ("screw", "test"): Predictions(np.array([1.95, 3.2]), np.array([0, 1]), np.zeros((2, 4, 4), np.float32),
                                       np.zeros((2, 4, 4), np.uint8), paths=["s0.png", "s1.png"], defect_types=["good", "scratch"]),
    }
    monkeypatch.setattr("smartlens.evaluation.evaluator.run_inference", lambda model, ds, *a, **kw: splits[ds])
    dm = SimpleNamespace(records=lambda c, s: [0, 1], mask_size=(4, 4))
    ev = Evaluator(dm, {"compute_pro": False}, torch.device("cpu"), tmp_path)
    monkeypatch.setattr(ev, "_ds", lambda model, category, split, **kw: (category, split))
    dummy = SimpleNamespace(input_size=(4, 4))

    calib = ev.calibrate_pooled(lambda c: dummy, ["bottle", "screw"], tune_f1=True)
    r_b = ev.evaluate_category(dummy, "bottle", "test", gallery=False, calib=calib)
    r_s = ev.evaluate_category(dummy, "screw", "test", gallery=False, calib=calib)
    assert r_b["calibration"]["tau_unsup_p99"] == pytest.approx(r_s["calibration"]["tau_unsup_p99"])
    assert r_b["calibration"]["tau_f1"] == pytest.approx(r_s["calibration"]["tau_f1"])
    pooled = pooled_image_summary([r_b, r_s], calib)
    assert pooled["n_images"] == 4 and pooled["n_anomalous"] == 2
    # Global ranking sees bottle anomalous (0.95) < screw normal (1.95), exposing cross-category miscalibration (< 1.0)
    assert pooled["I-AUROC"] == pytest.approx(0.75)


def test_comparison_enforces_amp_protocol_and_writes_paired_bootstrap(tmp_path):
    def write_modern(model: str, amp: bool, scores: list[float]):
        run = tmp_path / model
        (run / "eval").mkdir(parents=True, exist_ok=True)
        save_yaml({"model": {"name": model, "amp": amp},
                   "data": {"name": "mvtec_ad", "categories": ["bottle"], "mask_size": 448}}, run / "config.yaml")
        payload = {
            "protocol": {
                "dataset": "mvtec_ad", "model": model, "model_arch": model, "split": "test",
                "configured_categories": ["bottle"], "evaluated_categories": ["bottle"],
                "complete_category_coverage": True, "mask_size": 448, "mask_resize": "nearest",
                "manifest_sha256": "abc123", "pro_integration_limit": 0.3, "unsup_percentile": 99.0,
                "calibration_scope": "pooled", "amp": amp, "allow_tf32": True, "perturbations": [],
            },
            "pooled_clean": {"n_images": 4, "n_anomalous": 2, "n_normal": 2, "I-AUROC": 1.0 if model == "dinomaly" else 0.75},
            "latency": {"latency_ms_mean": 5.0, "latency_ms_max_category": 5.0, "fps": 200.0, "peak_vram_MB": 512.0,
                        "params_total_M": 10.0, "params_trainable_M": 2.0, "buffers_M": 1.0,
                        "deployed_checkpoints": 1, "deployed_footprint_M": 11.0},
            "results": [{
                "category": "bottle", "split": "test",
                "calibration": {"tau_unsup_p99": 0.5, "tau_f1": 0.5},
                "clean": {"n_images": 4, "n_anomalous": 2, "n_normal": 2, "I-AUROC": 1.0 if model == "dinomaly" else 0.5,
                          "Img-P-AP": 0.9 if model == "dinomaly" else 0.6},
                "predictions": {
                    "paths": ["0.png", "1.png", "2.png", "3.png"],
                    "labels": [0, 0, 1, 1],
                    "scores": scores,
                    "anom_p_ap": [None, None, 0.9 if model == "dinomaly" else 0.6, 0.9 if model == "dinomaly" else 0.6],
                    "optshift_scores": {},
                },
            }],
        }
        (run / "eval" / "results_test_pooled.json").write_text(json.dumps(payload))

    write_modern("dinomaly", amp=False, scores=[0.1, 0.2, 0.8, 0.9])
    write_modern("patchcore", amp=True, scores=[0.2, 0.6, 0.55, 0.7])
    with pytest.raises(ValueError, match="amp"):
        summarize(tmp_path, "test_pooled", n_boot=50)

    write_modern("patchcore", amp=False, scores=[0.2, 0.6, 0.55, 0.7])
    summarize(tmp_path, "test_pooled", n_boot=50)
    assert (tmp_path / "comparison_test_pooled_paired_bootstrap.csv").is_file()
    assert (tmp_path / "comparison_test_pooled_pooled.csv").is_file()
    md = (tmp_path / "comparison_test_pooled.md").read_text()
    assert "Paired image-level bootstrap" in md
    assert "Category-blind pooled image ranking" in md
    assert "FP32 (TF32)" in md


def test_pipeline_evaluate_aggregates_multi_checkpoint_latency_and_pooled_scope(monkeypatch, tmp_path):
    from smartlens.models.baselines.ssim_cae import SSIMCAE

    manifest = tmp_path / "splits.json"
    manifest.write_text("{}")
    mcfg = {"name": "ssim_cae", "setting": "per_class", "input_size": 32,
            "latent_dim": 4, "base_channels": 4, "amp": False}
    save_yaml({"data": {"name": "mvtec_ad", "split_manifest": str(manifest), "mask_resize": "nearest"},
               "eval": {"compute_pro": False, "num_workers": 0, "latency": {"warmup": 2, "iters": 4}},
               "model": mcfg}, tmp_path / "config.yaml")
    for cat in ("bottle", "screw"):
        SSIMCAE(mcfg).eval().save(tmp_path / "checkpoints" / f"{cat}.pt")

    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, cat: str, split: str):
            self.cat, self.split = cat, split

        def __len__(self):
            return 2

        def __getitem__(self, i):
            lbl = 0 if self.split == "val_normal" else i
            mask = torch.zeros(1, 32, 32)
            if lbl == 1:
                mask[:, 8:16, 8:16] = 1.0
            return {"image": torch.full((3, 32, 32), 0.2 + 0.5 * lbl), "mask": mask, "label": lbl,
                    "path": f"{self.cat}_{self.split}_{i}.png", "defect_type": "good" if lbl == 0 else "defect"}

    fake_dm = SimpleNamespace(
        name="mvtec_ad", categories=["bottle", "screw"], mask_size=(32, 32),
        dataset=lambda cat, split, size, **kw: TinyDataset(cat, split),
        records=lambda cat, split: [0, 1],
    )
    monkeypatch.setattr("smartlens.pipeline.DataModule", lambda cfg: fake_dm)

    evaluate(tmp_path, torch.device("cpu"), gallery=False, calibration_scope="pooled")
    out = json.loads((tmp_path / "eval" / "results_test_pooled.json").read_text())
    assert out["protocol"]["calibration_scope"] == "pooled"
    assert out["protocol"]["amp"] is False
    assert out["latency"]["deployed_checkpoints"] == 2
    assert out["latency"]["output_size"] == [32, 32]
    assert "pooled_clean" in out and out["pooled_clean"]["n_images"] == 4
    cals = [r["calibration"]["tau_unsup_p99"] for r in out["results"]]
    assert cals[0] == pytest.approx(cals[1])
