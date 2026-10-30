"""System metrics: latency (CUDA events), throughput, peak VRAM and parameter counts (spec §6.1)."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _hw(size: int | Sequence[int] | None) -> tuple[int, int] | None:
    if size is None:
        return None
    if isinstance(size, int):
        return size, size
    return int(size[0]), int(size[1])


def _predict_at_resolution(model, x: torch.Tensor, out_hw: tuple[int, int] | None) -> dict[str, torch.Tensor]:
    out = model.predict(x)
    if out_hw is not None and tuple(out["anomaly_map"].shape[-2:]) != out_hw:
        amap = out["anomaly_map"].float()[:, None]
        down = amap.shape[-1] > out_hw[-1]
        amap = F.interpolate(amap, size=out_hw, mode="bilinear", align_corners=False, antialias=down)[:, 0]
        out = {**out, "anomaly_map": amap}
    return out


@torch.no_grad()
def profile_latency(model, input_size: tuple[int, int] | int, device: torch.device, warmup: int = 50, iters: int = 200,
                    batch_size: int = 1, output_size: tuple[int, int] | int | None = None) -> dict:
    """End-to-end ``model.predict`` latency (incl. upsampling to ``output_size`` / smoothing / scoring)."""
    model.eval()
    in_hw = _hw(input_size)
    assert in_hw is not None
    out_hw = _hw(output_size)
    x = torch.rand(batch_size, 3, *in_hw, device=device)
    for _ in range(warmup):
        _predict_at_resolution(model, x, out_hw)
    times = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(iters):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            _predict_at_resolution(model, x, out_hw)
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end))
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 20
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            _predict_at_resolution(model, x, out_hw)
            times.append((time.perf_counter() - t0) * 1e3)
        peak = float("nan")
    t = np.asarray(times, dtype=np.float64)
    params = model.num_params()
    mean_ms = float(t.mean())
    out = {
        "batch_size": batch_size,
        "input_size": list(in_hw),
        "output_size": list(out_hw) if out_hw is not None else list(in_hw),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "latency_ms_mean": mean_ms,
        "latency_ms_max_category": mean_ms,
        "latency_ms_std": float(t.std()),
        "latency_ms_p50": float(np.percentile(t, 50)),
        "latency_ms_p95": float(np.percentile(t, 95)),
        "fps": float(1000.0 * batch_size / mean_ms) if mean_ms > 0 else float("inf"),
        "peak_vram_MB": float(peak),
        **params,
        "deployed_checkpoints": 1,
        "deployed_footprint_M": float(params["params_total_M"] + params["buffers_M"]),
    }
    return out


def aggregate_latency_profiles(profiles: list[dict]) -> dict:
    """Aggregate latency / VRAM / parameter profiles across checkpoints (1 unified vs N per-class)."""
    if not profiles:
        return {}
    if len(profiles) == 1:
        return dict(profiles[0])
    first = profiles[0]
    bs = int(first["batch_size"])
    means = np.asarray([p["latency_ms_mean"] for p in profiles], dtype=np.float64)
    mean_ms = float(means.mean())
    peaks = [p["peak_vram_MB"] for p in profiles if not np.isnan(p["peak_vram_MB"])]
    return {
        "batch_size": bs,
        "input_size": first["input_size"],
        "output_size": first.get("output_size", first["input_size"]),
        "device": first["device"],
        "gpu": first["gpu"],
        "latency_ms_mean": mean_ms,
        "latency_ms_max_category": float(means.max()),
        "latency_ms_std": float(np.mean([p["latency_ms_std"] for p in profiles])),
        "latency_ms_p50": float(np.mean([p["latency_ms_p50"] for p in profiles])),
        "latency_ms_p95": float(np.mean([p["latency_ms_p95"] for p in profiles])),
        "fps": float(1000.0 * bs / mean_ms) if mean_ms > 0 else float("inf"),
        "peak_vram_MB": float(max(peaks)) if peaks else float("nan"),
        "params_total_M": float(np.mean([p["params_total_M"] for p in profiles])),
        "params_trainable_M": float(np.mean([p["params_trainable_M"] for p in profiles])),
        "buffers_M": float(np.mean([p["buffers_M"] for p in profiles])),
        "deployed_checkpoints": len(profiles),
        "deployed_footprint_M": float(sum(p["params_total_M"] + p["buffers_M"] for p in profiles)),
    }
