"""High-level train / evaluate pipeline shared by the CLI scripts."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from smartlens.config import apply_overrides, deep_merge, load_yaml, resolve_path, save_yaml
from smartlens.data.datasets import DataModule
from smartlens.evaluation.evaluator import Evaluator, save_results
from smartlens.evaluation.latency import profile_latency
from smartlens.models.base import FitContext, log
from smartlens.models.registry import build_model


def load_experiment(data_cfg: str, model_cfg: str, overrides: list[str] | None = None) -> dict:
    """Merge a data/eval config with a model config into ``{data, eval, model}``."""
    cfg = load_yaml(data_cfg)
    cfg["model"] = load_yaml(model_cfg)
    cfg = apply_overrides(cfg, overrides)
    return cfg


def default_run_dir(cfg: dict) -> Path:
    return resolve_path(f"runs/{cfg['data']['name']}/{cfg['model']['name']}")


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
                "device": str(device), "torch_version": torch.__version__,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "peak_train_vram_MB": torch.cuda.max_memory_allocated(device) / 2 ** 20
                if device.type == "cuda" else None,
                "peak_train_reserved_MB": torch.cuda.max_memory_reserved(device) / 2 ** 20
                if device.type == "cuda" else None}
        model.save(ckpt)
        summary[category] = info
        summary_path.write_text(json.dumps(summary, indent=1, default=float))
        log.info("[%s] %s done in %.1fs", mcfg["name"], category, info["train_time_s"])
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
             eval_overrides: dict | None = None, tag: str | None = None) -> None:
    cfg = load_yaml(run_dir / "config.yaml")
    if data_cfg is not None:  # evaluate on a different data config (e.g. VisA) with same model
        cfg = deep_merge(cfg, {k: v for k, v in data_cfg.items() if k in ("data", "eval")})
    eval_cfg = deep_merge(cfg.get("eval", {}), eval_overrides or {})
    dm = DataModule(cfg["data"])
    cats = categories or dm.categories
    missing = [c for c in cats if not checkpoint_path(run_dir, c).exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints for {missing}. Train them or explicitly select --categories; "
                                "partial benchmarks are not silently averaged.")
    out_dir = run_dir / "eval"
    evaluator = Evaluator(dm, eval_cfg, device, out_dir)
    perts = eval_cfg.get("optshift", []) if optshift else []

    results, model, loaded = [], None, None
    for c in cats:
        ckpt = checkpoint_path(run_dir, c)
        if loaded != ckpt:
            model = build_model(torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"])
            model.load(ckpt, map_location=device)
            model.to(device).eval()
            loaded = ckpt
        t0 = time.time()
        res = evaluator.evaluate_category(model, c, split, perts, gallery)
        res["eval_time_s"] = round(time.time() - t0, 1)
        clean = res["clean"]
        log.info("%-12s I-AUROC %.4f | P-AUROC %.4f | AU-PRO %.4f | P-AP %.4f", c, clean["I-AUROC"],
                 clean["P-AUROC"], clean["AU-PRO"], clean["P-AP"])
        results.append(res)

    summary_path = run_dir / "train_summary.json"
    extra = {"protocol": {
        "dataset": dm.name, "model": cfg["model"]["name"], "split": split,
        "configured_categories": dm.categories, "evaluated_categories": cats,
        "complete_category_coverage": set(cats) == set(dm.categories),
        "mask_size": dm.mask_size, "mask_resize": cfg["data"].get("mask_resize", "area"),
        "manifest_sha256": hashlib.sha256(resolve_path(cfg["data"]["split_manifest"]).read_bytes()).hexdigest(),
        "model_config": cfg["model"], "perturbations": perts,
        "pro_integration_limit": eval_cfg.get("pro_integration_limit", 0.3),
        "unsup_percentile": eval_cfg.get("unsup_percentile", 99.0),
        "calibration_scope": "per_category",
    }, "training": json.loads(summary_path.read_text()) if summary_path.exists() else {}}
    if latency and model is not None:
        lat_cfg = eval_cfg.get("latency", {})
        extra["latency"] = profile_latency(model, model.input_size, device, lat_cfg.get("warmup", 50),
                                           lat_cfg.get("iters", 200))
        log.info("Latency: %.2f ms/img (%.1f FPS), peak VRAM %.0f MB", extra["latency"]["latency_ms_mean"],
                 extra["latency"]["fps"], extra["latency"]["peak_vram_MB"])
    tag = tag or (split + ("_optshift" if optshift else ""))
    df = save_results(results, out_dir, tag, extra)
    log.info("Mean over %d categories:\n%s", len(results),
             df.loc[["mean"], [c for c in ("I-AUROC", "I-AP", "I-F1max", "P-AUROC", "P-AP", "AU-PRO") if c in df]]
             .to_string())
    del model
    torch.cuda.empty_cache()
