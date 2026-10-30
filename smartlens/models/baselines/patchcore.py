"""Baseline 2a — PatchCore (Roth et al., CVPR 2022).

Frozen ImageNet WideResNet-50-2, mid-level ``layer2`` + ``layer3`` features with 3x3 local
neighbourhood aggregation, greedy k-center coreset memory bank, nearest-neighbour patch
scoring and the paper's re-weighted image score.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from smartlens.data.datasets import make_loader
from smartlens.models.base import AnomalyModel, FitContext, freeze, gaussian_blur, imagenet_normalize, log
from smartlens.models.baselines.common import autocast


class WRNFeatures(nn.Module):
    def __init__(self, layers=("layer2", "layer3"), backbone: str = "wide_resnet50_2") -> None:
        super().__init__()
        weights = {"wide_resnet50_2": torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1,
                   "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1}[backbone]
        net = getattr(torchvision.models, backbone)(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.layers = layers

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = {}
        x = self.stem(x)
        x = self.layer1(x); feats["layer1"] = x
        x = self.layer2(x); feats["layer2"] = x
        if "layer3" in self.layers:
            x = self.layer3(x); feats["layer3"] = x
        return [feats[l] for l in self.layers]


def greedy_coreset(features: torch.Tensor, ratio: float, proj_dim: int = 128, seed: int = 0,
                   max_points: int | None = None, device: torch.device | None = None) -> torch.Tensor:
    """Approximate greedy k-center selection on a Johnson-Lindenstrauss projection.

    ``features`` may reside on CPU (fp16/fp32) to avoid multi-GB GPU allocations on large
    categories (e.g. VisA); only the low-dimensional projection ``z`` lives on ``device``.
    Returns indices on ``features.device``.
    """
    n, d = features.shape
    dev = device or features.device
    k = max(1, int(math.ceil(n * ratio)))
    if max_points is not None:
        k = min(k, int(max_points))
    if k >= n:
        return torch.arange(n, device=features.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    if proj_dim and proj_dim < d:
        proj = (torch.randn(d, proj_dim, generator=g, dtype=torch.float32) / math.sqrt(proj_dim)).to(dev)
        z = torch.empty((n, proj_dim), dtype=torch.float32, device=dev)
        chunk = 65536
        for s in range(0, n, chunk):
            z[s:s + chunk] = features[s:s + chunk].to(dev, dtype=torch.float32) @ proj
        del proj
    else:
        z = features.to(dev, dtype=torch.float32)
    idx = torch.empty(k, dtype=torch.long, device=dev)
    start = int(torch.randint(n, (1,), generator=g))
    idx[0] = start
    min_d = torch.linalg.vector_norm(z - z[start], dim=1)
    for i in tqdm(range(1, k), desc="coreset", leave=False, mininterval=2.0):
        nxt = torch.argmax(min_d)
        idx[i] = nxt
        min_d = torch.minimum(min_d, torch.linalg.vector_norm(z - z[nxt], dim=1))
    return idx.to(features.device)


def chunked_nn(query: torch.Tensor, bank: torch.Tensor, k: int = 1, chunk: int = 4096):
    """k nearest neighbours (euclidean) of ``query`` rows in ``bank``. Returns (dist, idx)."""
    dists, idxs = [], []
    bank_sq = bank.pow(2).sum(1)
    for s in range(0, query.shape[0], chunk):
        q = query[s:s + chunk]
        d2 = q.pow(2).sum(1, keepdim=True) - 2 * q @ bank.T + bank_sq[None]
        d, i = d2.clamp_min_(0).topk(k, dim=1, largest=False)
        dists.append(d.sqrt())
        idxs.append(i)
    return torch.cat(dists), torch.cat(idxs)


class PatchCore(AnomalyModel):
    name = "patchcore"
    frozen_prefixes = ("backbone.",)

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.backbone = freeze(WRNFeatures(tuple(cfg.get("layers", ["layer2", "layer3"])),
                                           cfg.get("backbone", "wide_resnet50_2")))
        self.coreset_ratio = cfg.get("coreset_ratio", 0.1)
        self.max_coreset_size = cfg.get("max_coreset_size", 25000)
        self.proj_dim = cfg.get("coreset_proj_dim", 128)
        self.num_neighbors = cfg.get("num_neighbors", 9)
        self.blur_sigma = cfg.get("blur_sigma", 4.0)
        self.patch_size = cfg.get("patch_size", 3)
        self.amp = cfg.get("amp", False)
        self.register_buffer("memory_bank", torch.empty(0, 0))
        self.feat_hw: tuple[int, int] | None = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # frozen BN statistics, always
        return self

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = imagenet_normalize(self.prepare(images))
        with autocast(x.device, self.amp):
            feats = self.backbone(x)
        pool = nn.AvgPool2d(self.patch_size, 1, self.patch_size // 2)
        feats = [pool(f.float()) for f in feats]
        hw = feats[0].shape[-2:]
        feats = [feats[0]] + [F.interpolate(f, size=hw, mode="bilinear", align_corners=False) for f in feats[1:]]
        emb = torch.cat(feats, dim=1)  # (B, C, h, w)
        b, c, h, w = emb.shape
        return emb.permute(0, 2, 3, 1).reshape(b, h * w, c), (h, w)

    def fit(self, ctx: FitContext) -> dict:
        dev = ctx.device
        self.to(dev)
        loader = make_loader(ctx.train_ds, self.cfg.get("batch_size", 16), False, ctx.num_workers)
        embs = []
        for b in tqdm(loader, desc=f"patchcore[{ctx.category}] embed", leave=False):
            e, hw = self.embed(b["image"].to(dev))
            embs.append(e.reshape(-1, e.shape[-1]).half().cpu())
        feats = torch.cat(embs)
        del embs
        idx = greedy_coreset(feats, self.coreset_ratio, self.proj_dim,
                             max_points=self.max_coreset_size, device=dev)
        self.memory_bank = feats[idx].float().to(dev).contiguous()
        del feats
        self.feat_hw = hw
        log.info("patchcore[%s] memory bank: %s", ctx.category, tuple(self.memory_bank.shape))
        self.eval()
        return {"memory_bank_size": int(self.memory_bank.shape[0]),
                "n_patch_features": len(ctx.train_ds) * hw[0] * hw[1],
                "effective_coreset_ratio": len(idx) / (len(ctx.train_ds) * hw[0] * hw[1])}

    def _prepare_load(self, state: dict) -> None:
        self.memory_bank = torch.empty_like(state["memory_bank"])

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        emb, (h, w) = self.embed(images)
        b, n, c = emb.shape
        flat = emb.reshape(-1, c)
        dist, nn_idx = chunked_nn(flat, self.memory_bank, k=1)
        patch_scores = dist[:, 0].reshape(b, n)
        locations = nn_idx[:, 0].reshape(b, n)
        image_score = self._image_score(patch_scores, locations, emb)
        amap = patch_scores.reshape(b, 1, h, w)
        amap = F.interpolate(amap, size=self.input_size, mode="bilinear", align_corners=False)
        amap = gaussian_blur(amap, self.blur_sigma)[:, 0]
        return {"anomaly_map": amap, "image_score": image_score}

    def _image_score(self, patch_scores, locations, emb) -> torch.Tensor:
        """Re-weighted max patch score (PatchCore Eq. 7, as in anomalib)."""
        if self.num_neighbors <= 1:
            return patch_scores.amax(1)
        ar = torch.arange(patch_scores.shape[0], device=patch_scores.device)
        max_patch = patch_scores.argmax(1)
        score = patch_scores[ar, max_patch]
        test_feat = emb[ar, max_patch]                        # (B, C)
        nn_sample = self.memory_bank[locations[ar, max_patch]]  # (B, C)
        k = min(self.num_neighbors, self.memory_bank.shape[0])
        _, support = chunked_nn(nn_sample, self.memory_bank, k=k)  # (B, k)
        d = torch.linalg.vector_norm(test_feat[:, None] - self.memory_bank[support], dim=-1)  # (B, k)
        weights = 1 - F.softmax(d, dim=1)[:, 0]
        return weights * score
