"""Baseline 3 — Dinomaly-Repro (Guo et al., CVPR 2025).

Frozen DINOv2-Reg encoder (blocks 2..9) -> noisy MLP bottleneck (dropout) -> 8 ELU
linear-attention decoder blocks; loose two-group reconstruction of encoder features with
hard-mining global cosine loss; bilinear upsampling of uncalibrated cosine-distance maps.
This model is also ablation ``A0`` of the HiRes-ProtoAD study.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from smartlens.data.datasets import infinite, make_loader
from smartlens.models.backbones.dinov2_extractor import DINOv2Extractor
from smartlens.models.base import (AnomalyModel, FitContext, StableAdamW, WarmCosineScheduler, gaussian_blur, log,
                                   topk_mean)
from smartlens.models.baselines.common import JsonlLogger, autocast


class BottleneckMLP(nn.Module):
    """Noisy bottleneck: dropout on input, hidden and output (Dinomaly ``bMlp``)."""

    def __init__(self, dim: int, hidden: int, drop: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(x)
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class LinearAttention(nn.Module):
    """ELU(x)+1 kernelized linear attention (O(N D^2))."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.eps = eps

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = F.elu(qkv[0]) + 1.0, F.elu(qkv[1]) + 1.0, qkv[2]
        kv = k.transpose(-2, -1) @ v
        num = q @ kv
        den = (q.float() * k.float().sum(dim=-2, keepdim=True)).sum(dim=-1, keepdim=True) + self.eps
        out = (num.float() / den).to(x.dtype)
        return self.proj(out.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-8)
        self.norm1, self.attn = norm(dim), LinearAttention(dim, num_heads)
        self.norm2, self.mlp = norm(dim), Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def _modify_grad(grad, inds, factor):
    grad = grad.clone()
    grad[inds.expand_as(grad)] *= factor
    return grad


def global_cosine_hm_percent(en: list[torch.Tensor], de: list[torch.Tensor], p: float = 0.9, factor: float = 0.1):
    """Dinomaly hard-mining global cosine loss: gradients of the easiest ``p`` fraction of
    points (lowest point-wise cosine distance) are down-weighted by ``factor``."""
    loss = 0.0
    for a, b in zip(en, de):
        a = a.detach()
        with torch.no_grad():
            point_dist = 1 - F.cosine_similarity(a, b, dim=1).unsqueeze(1)
            k = max(1, int(point_dist.numel() * (1 - p)))
            thresh = torch.topk(point_dist.reshape(-1), k=k)[0][-1]
        loss = loss + (1 - F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)).mean()
        if b.requires_grad and p > 0:
            b.register_hook(partial(_modify_grad, inds=point_dist < thresh, factor=factor))
    return loss / len(en)


class Dinomaly(AnomalyModel):
    name = "dinomaly"
    frozen_prefixes = ("encoder.",)

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.encoder = DINOv2Extractor(cfg.get("encoder", "dinov2_vitb14_reg"), self.input_size,
                                       cfg.get("target_layers", [2, 3, 4, 5, 6, 7, 8, 9]))
        d, h = self.encoder.embed_dim, self.encoder.num_heads
        self.bottleneck = BottleneckMLP(d, 4 * d, cfg.get("bottleneck_dropout", 0.2))
        self.decoder = nn.ModuleList([DecoderBlock(d, h) for _ in range(cfg.get("decoder_depth", 8))])
        n_t = len(self.encoder.target_layers)
        self.fuse_enc = cfg.get("fuse_layer_encoder", [list(range(n_t // 2)), list(range(n_t // 2, n_t))])
        self.fuse_dec = cfg.get("fuse_layer_decoder", [list(range(len(self.decoder) // 2)),
                                                       list(range(len(self.decoder) // 2, len(self.decoder)))])
        s = cfg.get("score", {})
        self.blur_sigma, self.blur_ks = s.get("blur_sigma", 4.0), s.get("blur_kernel_size", 5)
        self.max_ratio = s.get("image_topk_ratio", 0.01)
        self.amp = cfg.get("amp", True)
        self.grad_checkpoint = cfg.get("grad_checkpoint", True)
        self._init_weights()

    def _init_weights(self):
        for m in list(self.bottleneck.modules()) + list(self.decoder.modules()):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01, a=-2.0, b=2.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _fuse(feats: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(feats, dim=1).mean(dim=1)

    def forward(self, images: torch.Tensor):
        """Returns encoder / decoder feature groups as (B, D, h, w) maps."""
        x = self.prepare(images)
        en_list = self.encoder(x)
        z = self.bottleneck(self._fuse(en_list))
        de_list = []
        use_ckpt = self.grad_checkpoint and torch.is_grad_enabled() and z.requires_grad
        for blk in self.decoder:
            z = checkpoint(blk, z, use_reentrant=False) if use_ckpt else blk(z)
            de_list.append(z)
        de_list = de_list[::-1]  # decoder block k mirrors encoder layer (L - k)
        en = [self._fuse([en_list[i] for i in g]) for g in self.fuse_enc]
        de = [self._fuse([de_list[i] for i in g]) for g in self.fuse_dec]
        npt = self.encoder.num_prefix_tokens
        gh, gw = self.encoder.grid
        to_map = lambda t: t[:, npt:].transpose(1, 2).reshape(t.shape[0], -1, gh, gw)  # noqa: E731
        return [to_map(e) for e in en], [to_map(d) for d in de]

    def trainable_parameters(self):
        return list(self.bottleneck.parameters()) + list(self.decoder.parameters())

    def fit(self, ctx: FitContext) -> dict:
        tcfg = self.cfg.get("train", {})
        dev = ctx.device
        self.to(dev)
        loader = make_loader(ctx.train_ds, tcfg.get("batch_size", 16), True, ctx.num_workers, drop_last=True)
        iters = int(tcfg.get("iters", 10000))
        lr = tcfg.get("lr", 2e-3)
        opt = StableAdamW(self.trainable_parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=tcfg.get("weight_decay", 1e-4),
                          amsgrad=True, eps=1e-10)
        sched = WarmCosineScheduler(opt, iters, tcfg.get("warmup_iters", 100), final_ratio=tcfg.get("final_lr", 2e-4) / lr)
        p_final, p_ramp = tcfg.get("hm_p", 0.9), tcfg.get("hm_ramp_iters", 1000)
        factor, clip = tcfg.get("hm_factor", 0.1), tcfg.get("grad_clip", 0.1)
        logger = JsonlLogger(ctx.out_dir / f"train_log_{ctx.category}.jsonl")
        self.bottleneck.train(); self.decoder.train()
        it_data = infinite(loader)
        pbar = tqdm(range(iters), desc=f"dinomaly[{ctx.category}]", mininterval=5.0)
        for it in pbar:
            x = next(it_data)["image"].to(dev, non_blocking=True)
            with autocast(dev, self.amp):
                en, de = self(x)
            p = min(p_final * it / max(1, p_ramp), p_final)
            loss = global_cosine_hm_percent([e.float() for e in en], [d.float() for d in de], p=p, factor=factor)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(self.trainable_parameters(), max_norm=clip)
            opt.step()
            sched.step()
            if (it + 1) % 50 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
            if (it + 1) % 500 == 0 or it + 1 == iters:
                logger(iter=it + 1, loss=loss.item(), p=p, lr=sched.get_last_lr()[0])
            if tcfg.get("save_every") and (it + 1) % int(tcfg["save_every"]) == 0:
                self.save(ctx.out_dir / "checkpoints" / f"{ctx.category}_iter{it + 1}.pt")
        self.eval()
        log.info("dinomaly[%s] finished %d iters", ctx.category, iters)
        return {"iters": iters}

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        with autocast(images.device, self.amp):
            en, de = self(images)
        maps = []
        for e, d in zip(en, de):
            m = 1 - F.cosine_similarity(e.float(), d.float(), dim=1).unsqueeze(1)
            maps.append(F.interpolate(m, size=self.input_size, mode="bilinear", align_corners=True))
        amap = torch.cat(maps, dim=1).mean(dim=1, keepdim=True)
        amap = gaussian_blur(amap, self.blur_sigma, self.blur_ks)[:, 0]
        return {"anomaly_map": amap, "image_score": topk_mean(amap, self.max_ratio)}
