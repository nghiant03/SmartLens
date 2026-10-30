"""Frozen DINOv2-with-registers multi-layer token extractor (timm weights)."""

from __future__ import annotations

from typing import Sequence

import timm
import torch
import torch.nn as nn

from smartlens.models.base import freeze, imagenet_normalize

TIMM_NAMES = {
    "dinov2_vits14_reg": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitb14_reg": "vit_base_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitl14_reg": "vit_large_patch14_reg4_dinov2.lvd142m",
}


class DINOv2Extractor(nn.Module):
    """Returns raw token sequences (prefix [CLS]+[REG] tokens included) from ``target_layers``.

    ``target_layers`` are 0-indexed transformer block indices. Blocks after the last target
    layer are never executed.
    """

    def __init__(self, arch: str = "dinov2_vitb14_reg", img_size: Sequence[int] | int = 448,
                 target_layers: Sequence[int] = (2, 3, 4, 5, 6, 7, 8, 9), pretrained: bool = True) -> None:
        super().__init__()
        name = TIMM_NAMES.get(arch, arch)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.vit = timm.create_model(name, pretrained=pretrained, img_size=tuple(img_size), num_classes=0)
        freeze(self.vit)
        self.target_layers = sorted(target_layers)
        self.embed_dim = self.vit.embed_dim
        self.num_heads = self.vit.blocks[0].attn.num_heads
        self.num_prefix_tokens = self.vit.num_prefix_tokens
        self.patch_size = self.vit.patch_embed.patch_size[0]
        self.grid = (img_size[0] // self.patch_size, img_size[1] // self.patch_size)

    def train(self, mode: bool = True):
        super().train(False)  # always frozen / eval
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        v = self.vit
        x = v.patch_embed(imagenet_normalize(images))
        x = v._pos_embed(x)
        x = v.patch_drop(x)
        x = v.norm_pre(x)
        out = []
        for i, blk in enumerate(v.blocks):
            if i > self.target_layers[-1]:
                break
            x = blk(x)
            if i in self.target_layers:
                out.append(x)
        return out
