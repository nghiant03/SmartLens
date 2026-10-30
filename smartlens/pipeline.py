"""High-level train / evaluate pipeline shared by the CLI scripts."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from smartlens.config import apply_overrides, deep_merge, load_yaml, resolve_path, save_yaml
from smartlens.data.datasets import DataModule
from smartlens.evaluation.evaluator import Evaluator, pooled_image_summary, save_results
from smartlens.evaluation.latency import aggregate_latency_profiles, profile_latency
from smartlens.models.base import AnomalyModel, FitContext, log
from smartlens.models.registry import build_model


def load_experiment(data_cfg: str, model_cfg: str, overrides: list[str] | None = None) -> dict:
    """Merge a data/eval config with a model config into ``{data, eval, model}``."""
    cfg = load_yaml(data_cfg)
    cfg["model"] = load_yaml(model_cfg)
    cfg = apply_overrides(cfg, overrides)
    return cfg


def default_run_dir(cfg: dict) -> Path:
    run_name = cfg["model"].get("run_name", cfg["model"]["name"])
    return resolve_path(f"runs/{cfg['data']['name']}/{run_name}")


def checkpoint_path(run_dir: Path, category: str) -> Path:
    ck = run_dir / "checkpoints"
    if (ck / f"{category}.pt").exists():
        return ck / f"{category}.pt"
    return ck / "unified.pt"


# --------------------------------------------------------------------------------------
def train(cfg: dict, run_dir: Path, device: torch.device, categories: list[str] | None = None,
          num_workers: int = 6, skip_existing: bool = False) -> dict:
    dm = DataModule(cfg["data"])
    mcfg = cfg["model"]
    cats = categories or dm.categories
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    if skip_existing and config_path.exists() and load_yaml(config_path) != cfg:
        raise ValueError("--skip-existing cannot relabel checkpoints with a different config; use a new run directory.")
    save_yaml(cfg, run_dir / "config.yaml")
    summary_path = run_dir / "train_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    preload = mcfg.get("preload", True)

    def fit_one(category: str, train_ds, val_ds):
        ckpt = run_dir / "checkpoints" / f"{category}.pt"
        if skip_existing and ckpt.exists():
            log.info("Skipping %s (checkpoint exists)", category)
            return
        model = build_model(mcfg)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        info = model.fit(FitContext(train_ds, val_ds, device, run_dir, num_workers, category))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        info = {**(info or {}), "train_time_s": round(time.time() - t0, 1), "n_train": len(train_ds),
                "n_val_normal": len(val_ds) if val_ds is not None else 0,
                "categories": cats if category == "unified" else [category],
                "model_config": mcfg,
                "amp": bool(mcfg.get("amp", False)),
                "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "device": str(device), "torch_version": torch.__version__,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "peak_train_vram_MB": torch.cuda.max_memory_allocated(device) / 2 ** 20
                if device.type == "cuda" else None,
                "peak_train_reserved_MB": torch.cuda.max_memory_reserved(device) / 2 ** 20
                if device.type == "cuda" else None}
        model.save(ckpt)
        summary[category] = info
        summary_path.write_text(json.dumps(summary, indent=1, default=float))
        log.info("[%s] %s done in %.1fs", mcfg.get("run_name", mcfg["name"]), category, info["train_time_s"])
        del model
        torch.cuda.empty_cache()

    if mcfg.get("setting", "per_class") == "unified":
        train_ds = dm.unified("train", mcfg["input_size"], categories=cats, preload=preload)
        val_ds = dm.unified("val_normal", mcfg["input_size"], categories=cats)
        fit_one("unified", train_ds, val_ds)
    else:
        for c in cats:
            train_ds = dm.dataset(c, "train", mcfg["input_size"], preload=preload)
            val_ds = dm.dataset(c, "val_normal", mcfg["input_size"])
            fit_one(c, train_ds, val_ds)
    return summary


# --------------------------------------------------------------------------------------
def evaluate(run_dir: Path, device: torch.device, data_cfg: dict | None = None, categories: list[str] | None = None,
             split: str = "test", optshift: bool = False, latency: bool = True, gallery: bool = True,
             eval_overrides: dict | None = None, tag: str | None = None,
             calibration_scope: str | None = None) -> None:
    cfg = load_yaml(run_dir / "config.yaml")
    if data_cfg is not None:  # evaluate on a different data config (e.g. VisA) with same model
        cfg = deep_merge(cfg, {k: v for k, v in data_cfg.items() if k in ("data", "eval")})
    eval_cfg = deep_merge(cfg.get("eval", {}), eval_overrides or {})
    scope = calibration_scope or eval_cfg.get("calibration_scope", "per_category")
    if scope not in ("per_category", "pooled"):
        raise ValueError(f"Unsupported calibration_scope {scope!r}; expected 'per_category' or 'pooled'.")
    dm = DataModule(cfg["data"])
    cats = categories or dm.categories
    missing = [c for c in cats if not checkpoint_path(run_dir, c).exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints for {missing}. Train them or explicitly select --categories; "
                                "partial benchmarks are not silently averaged.")
    out_dir = run_dir / "eval"
    evaluator = Evaluator(dm, eval_cfg, device, out_dir)
    perts = eval_cfg.get("optshift", []) if optshift else []

    unique_ckpts = list(dict.fromkeys(checkpoint_path(run_dir, c) for c in cats))
    lat_cfg = eval_cfg.get("latency", {})
    n_ckpts = max(1, len(unique_ckpts))
    warmup = int(lat_cfg.get("warmup", 50))
    iters = int(lat_cfg.get("iters", 200))
    warmup_k = warmup if n_ckpts == 1 else max(10, math.ceil(warmup / n_ckpts))
    iters_k = iters if n_ckpts == 1 else max(25, math.ceil(iters / n_ckpts))

    model: AnomalyModel | None = None
    loaded: Path | None = None

    def load_for_category(category: str) -> AnomalyModel:
        nonlocal model, loaded
        ckpt = checkpoint_path(run_dir, category)
        if loaded != ckpt:
            model = build_model(torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"])
            model.load(ckpt, map_location=device)
            model.to(device).eval()
            loaded = ckpt
        assert model is not None
        return model

    pooled_calib = (evaluator.calibrate_pooled(load_for_category, cats, tune_f1=split != "test_val")
                    if scope == "pooled" else None)

    results, lat_profiles, profiled_ckpts = [], [], set()
    for c in cats:
        ckpt = checkpoint_path(run_dir, c)
        model = load_for_category(c)
        t0 = time.time()
        res = evaluator.evaluate_category(model, c, split, perts, gallery, calib=pooled_calib)
        res["eval_time_s"] = round(time.time() - t0, 1)
        clean = res["clean"]
        log.info("%-12s I-AUROC %.4f | P-AUROC %.4f | AU-PRO %.4f | P-AP %.4f", c, clean["I-AUROC"],
                 clean["P-AUROC"], clean["AU-PRO"], clean["P-AP"])
        results.append(res)
        if latency and ckpt not in profiled_ckpts:
            lat_profiles.append(profile_latency(model, model.input_size, device, warmup=warmup_k, iters=iters_k,
                                                output_size=dm.mask_size))
            profiled_ckpts.add(ckpt)

    summary_path = run_dir / "train_summary.json"
    mcfg_used = model.cfg if model is not None else cfg["model"]
    extra: dict = {"protocol": {
        "dataset": dm.name,
        "model": cfg["model"].get("run_name", cfg["model"]["name"]),
        "model_arch": cfg["model"]["name"],
        "split": split,
        "configured_categories": dm.categories,
        "evaluated_categories": cats,
        "complete_category_coverage": set(cats) == set(dm.categories),
        "mask_size": dm.mask_size,
        "mask_resize": cfg["data"].get("mask_resize", "area"),
        "manifest_sha256": hashlib.sha256(resolve_path(cfg["data"]["split_manifest"]).read_bytes()).hexdigest(),
        "model_config": cfg["model"],
        "input_size": list(model.input_size) if model is not None else cfg["model"].get("input_size"),
        "amp": bool(mcfg_used.get("amp", False)),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "perturbations": perts,
        "pro_integration_limit": eval_cfg.get("pro_integration_limit", 0.3),
        "unsup_percentile": eval_cfg.get("unsup_percentile", 99.0),
        "calibration_scope": scope,
    }, "training": json.loads(summary_path.read_text()) if summary_path.exists() else {}}
    if scope == "pooled" and pooled_calib is not None and results:
        extra["pooled_clean"] = pooled_image_summary(results, pooled_calib)
    if lat_profiles:
        extra["latency"] = aggregate_latency_profiles(lat_profiles)
        lat = extra["latency"]
        log.info("Latency: %.2f ms/img (max %.2f ms, %.1f FPS), peak VRAM %.0f MB, %d ckpt(s) (%.1fM total)",
                 lat["latency_ms_mean"], lat["latency_ms_max_category"], lat["fps"], lat["peak_vram_MB"],
                 lat["deployed_checkpoints"], lat["deployed_footprint_M"])
    tag = tag or (split + ("_optshift" if optshift else "") + ("_pooled" if scope == "pooled" else ""))
    df = save_results(results, out_dir, tag, extra)
    log.info("Mean over %d categories:\n%s", len(results),
             df.loc[["mean"], [c for c in ("I-AUROC", "I-AP", "I-F1max", "P-AUROC", "P-AP", "Img-P-AP", "AU-PRO")
                               if c in df]].to_string())
    del model
    torch.cuda.empty_cache()
