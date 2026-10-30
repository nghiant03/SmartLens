"""System metrics: latency (CUDA events), throughput, peak VRAM and parameter counts (spec §6.1)."""

from __future__ import annotations

import time

import numpy as np
import torch


@torch.no_grad()
def profile_latency(model, input_size: tuple[int, int], device: torch.device, warmup: int = 50, iters: int = 200,
                    batch_size: int = 1) -> dict:
    """End-to-end ``model.predict`` latency (incl. upsampling / smoothing / scoring)."""
    model.eval()
    x = torch.rand(batch_size, 3, *input_size, device=device)
    for _ in range(warmup):
        model.predict(x)
    times = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(iters):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model.predict(x)
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end))
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 20
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            model.predict(x)
            times.append((time.perf_counter() - t0) * 1e3)
        peak = float("nan")
    t = np.asarray(times)
    out = {
        "batch_size": batch_size, "input_size": list(input_size), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "latency_ms_mean": float(t.mean()), "latency_ms_std": float(t.std()),
        "latency_ms_p50": float(np.percentile(t, 50)), "latency_ms_p95": float(np.percentile(t, 95)),
        "fps": float(1000.0 * batch_size / t.mean()), "peak_vram_MB": float(peak),
    }
    out.update(model.num_params())
    return out
