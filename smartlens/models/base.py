"""Shared model interface and utilities."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

log = logging.getLogger("smartlens")


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def gaussian_kernel1d(sigma: float, kernel_size: int | None = None, device=None) -> torch.Tensor:
    if kernel_size is None:
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
    half = (kernel_size - 1) / 2
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - half
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float, kernel_size: int | None = None) -> torch.Tensor:
    """Separable Gaussian blur of a (B, 1, H, W) or (B, H, W) map with reflect padding."""
    if sigma <= 0:
        return x
    squeeze = x.dim() == 3
    if squeeze:
        x = x[:, None]
    k = gaussian_kernel1d(sigma, kernel_size, x.device).to(x.dtype)
    ks = k.numel()
    pad = ks // 2
    c = x.shape[1]
    kx = k.view(1, 1, 1, ks).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, ks, 1).repeat(c, 1, 1, 1)
    mode = "reflect" if min(x.shape[-2:]) > pad else "replicate"
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode=mode), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode=mode), ky, groups=c)
    return x[:, 0] if squeeze else x


def topk_mean(maps: torch.Tensor, ratio: float) -> torch.Tensor:
    """Mean of the top ``ratio`` fraction of pixels per map (B, H, W) -> (B,)."""
    flat = maps.flatten(1)
    k = max(1, int(round(flat.shape[1] * ratio)))
    return flat.topk(k, dim=1).values.mean(dim=1)


@dataclass
class FitContext:
    """Everything a model needs to train on one category (or the unified set)."""

    train_ds: Dataset
    val_ds: Dataset | None
    device: torch.device
    out_dir: Path
    num_workers: int = 4
    category: str = "unified"


class AnomalyModel(nn.Module):
    """Common interface implemented by all baselines.

    * ``fit(ctx)``             – train / build the model using normal images only.
    * ``predict(images)``      – images in [0, 1], (B, 3, H, W) -> dict with
                                 ``anomaly_map`` (B, H', W') and ``image_score`` (B,).
    * ``save(path)`` / ``load(path)`` – persist only non-frozen state (frozen pretrained
                                 backbones are rebuilt from their public weights).
    """

    name: str = "base"
    #: parameter / buffer name prefixes that are frozen pretrained weights (not checkpointed)
    frozen_prefixes: tuple[str, ...] = ()

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_size: tuple[int, int] = _size2(cfg.get("input_size", 256))

    # ---- training ---------------------------------------------------------------------
    def fit(self, ctx: FitContext) -> dict:
        raise NotImplementedError

    # ---- inference --------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def prepare(self, images: torch.Tensor) -> torch.Tensor:
        """Resize to the model input size if needed."""
        if tuple(images.shape[-2:]) != self.input_size:
            images = F.interpolate(images, size=self.input_size, mode="bilinear", align_corners=False,
                                   antialias=True)
        return images

    # ---- persistence ------------------------------------------------------------------
    def extra_state(self) -> dict:
        """Non-tensor-module state to checkpoint (e.g. calibration quantiles)."""
        return {}

    def load_extra_state(self, state: dict) -> None:
        pass

    def _is_frozen(self, key: str) -> bool:
        return any(key.startswith(p) for p in self.frozen_prefixes)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in self.state_dict().items() if not self._is_frozen(k)}
        torch.save({"model": self.name, "cfg": self.cfg, "state_dict": state,
                    "extra": self.extra_state()}, path)

    def load(self, path: str | Path, map_location="cpu") -> "AnomalyModel":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        state = ckpt["state_dict"]
        self._prepare_load(state)
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if not self._is_frozen(k)]
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch for {self.name}: missing={missing} unexpected={unexpected}")
        self.load_extra_state(ckpt.get("extra", {}))
        return self

    def _prepare_load(self, state: dict) -> None:
        """Hook to resize variable-size buffers (e.g. memory banks) before loading."""

    # ---- bookkeeping ------------------------------------------------------------------
    def num_params(self) -> dict[str, float]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffers = sum(b.numel() for b in self.buffers())
        return {"params_total_M": total / 1e6, "params_trainable_M": trainable / 1e6,
                "buffers_M": buffers / 1e6}


def _size2(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    return int(size[0]), int(size[1])


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


class WarmCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear warm-up followed by cosine decay from ``base_lr`` to ``final_lr``."""

    def __init__(self, optimizer, total_iters: int, warmup_iters: int = 0, final_ratio: float = 0.0):
        def fn(it: int) -> float:
            if warmup_iters > 0 and it < warmup_iters:
                return (it + 1) / warmup_iters
            t = (it - warmup_iters) / max(1, total_iters - warmup_iters)
            t = min(max(t, 0.0), 1.0)
            return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * t))

        super().__init__(optimizer, fn)


class StableAdamW(torch.optim.Optimizer):
    """AdamW with per-tensor update clipping (Wortsman et al., 2023), as used by Dinomaly."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    if group["amsgrad"]:
                        st["max_exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                st["step"] += 1
                t = st["step"]
                st["exp_avg"].lerp_(g, 1 - b1)
                st["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1 - b2)
                v = st["exp_avg_sq"]
                if group["amsgrad"]:
                    torch.maximum(st["max_exp_avg_sq"], v, out=st["max_exp_avg_sq"])
                    v = st["max_exp_avg_sq"]
                v_hat = v / (1 - b2 ** t)
                m_hat = st["exp_avg"] / (1 - b1 ** t)
                # update clipping without host sync: lr_t = lr / max(1, RMS(g / sqrt(v_hat)))
                rms = (g.pow(2) / v_hat.clamp_min(group["eps"] ** 2)).mean().sqrt()
                lr_t = group["lr"] / rms.clamp_min(1.0)
                if group["weight_decay"]:
                    p.mul_((1 - lr_t * group["weight_decay"]).to(p.dtype))
                p.sub_((lr_t * m_hat / (v_hat.sqrt() + group["eps"])).to(p.dtype))
        return loss
