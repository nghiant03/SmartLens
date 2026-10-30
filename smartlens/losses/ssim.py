"""Differentiable (multi-scale) SSIM maps and losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from smartlens.models.base import gaussian_kernel1d


def ssim_map(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5,
             data_range: float = 1.0) -> torch.Tensor:
    """Per-pixel, per-channel SSIM map (B, C, H, W) between images in [0, data_range].

    Uses fused separable 1D Gaussian filtering across [x, y, x^2, y^2, x*y] for speed.
    """
    x, y = x.float(), y.float()
    c = x.shape[1]
    c5 = 5 * c
    k1d = gaussian_kernel1d(sigma, window, x.device).to(x.dtype)
    kx = k1d.view(1, 1, 1, window).expand(c5, 1, 1, window).contiguous()
    ky = k1d.view(1, 1, window, 1).expand(c5, 1, window, 1).contiguous()
    pad = window // 2

    stacked = torch.cat([x, y, x * x, y * y, x * y], dim=1)
    stacked = F.conv2d(F.pad(stacked, (pad, pad, 0, 0), mode="reflect"), kx, groups=c5)
    stacked = F.conv2d(F.pad(stacked, (0, 0, pad, pad), mode="reflect"), ky, groups=c5)
    mu_x, mu_y, exx, eyy, exy = stacked.chunk(5, dim=1)

    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    sxx = (exx - mu_x * mu_x).clamp_min(0.0)
    syy = (eyy - mu_y * mu_y).clamp_min(0.0)
    sxy = exy - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sxx + syy + c2)
    return num / den


def multiscale_ssim_maps(x: torch.Tensor, y: torch.Tensor, scales: int = 3, window: int = 11) -> list[torch.Tensor]:
    """SSIM maps on an image pyramid (full, 1/2, 1/4, ...)."""
    x, y = x.float(), y.float()
    maps = []
    for s in range(scales):
        if s > 0:
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
        maps.append(ssim_map(x, y, window=window))
    return maps


def ms_ssim_loss(x: torch.Tensor, y: torch.Tensor, scales: int = 3, window: int = 11) -> torch.Tensor:
    maps = multiscale_ssim_maps(x, y, scales, window)
    return torch.stack([1 - m.mean() for m in maps]).mean()


def ms_ssim_anomaly_map(x: torch.Tensor, y: torch.Tensor, scales: int = 3, window: int = 11) -> torch.Tensor:
    """Average structural dissimilarity (1 - SSIM)/2 over channels and pyramid scales -> (B, H, W)."""
    h, w = x.shape[-2:]
    out = 0
    for m in multiscale_ssim_maps(x, y, scales, window):
        d = ((1 - m) / 2).mean(dim=1, keepdim=True)
        out = out + F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
    return (out / scales)[:, 0]
