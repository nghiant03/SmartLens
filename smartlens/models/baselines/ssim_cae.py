"""Baseline 1 — SSIM-CAE: multi-scale convolutional autoencoder trained with L1 + MS-SSIM.

Represents the classic pixel-reconstruction approach of the original proposal
(Bergmann et al., "Improving Unsupervised Defect Segmentation by Applying Structural
Similarity to Autoencoders", VISAPP 2019), adapted to full 256x256 RGB images with a
dense 1x1 latent bottleneck and a multi-scale SSIM objective / anomaly map.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from smartlens.data.datasets import infinite, make_loader
from smartlens.data.splits import MVTEC_TEXTURES
from smartlens.losses.ssim import ms_ssim_anomaly_map, ms_ssim_loss
from smartlens.models.base import AnomalyModel, FitContext, WarmCosineScheduler, gaussian_blur, log, topk_mean
from smartlens.models.baselines.common import BestKeeper, JsonlLogger, autocast


def _block(cin, cout, k, s, p, transpose=False):
    conv = (nn.ConvTranspose2d if transpose else nn.Conv2d)(cin, cout, k, s, p, bias=False)
    return nn.Sequential(conv, nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True))


class ConvAutoencoder(nn.Module):
    """256 -> 8x8 spatial -> dense latent (1x1xD) -> 256 reconstruction."""

    def __init__(self, input_size: int = 256, latent_dim: int = 256, c: int = 32) -> None:
        super().__init__()
        n_down = 5
        assert input_size % (2 ** n_down) == 0, "input_size must be divisible by 32"
        bottom = input_size // (2 ** n_down)
        self.encoder = nn.Sequential(
            _block(3, c, 4, 2, 1), _block(c, c, 3, 1, 1),              # 128
            _block(c, 2 * c, 4, 2, 1), _block(2 * c, 2 * c, 3, 1, 1),  # 64
            _block(2 * c, 4 * c, 4, 2, 1), _block(4 * c, 2 * c, 3, 1, 1),  # 32
            _block(2 * c, 4 * c, 4, 2, 1), _block(4 * c, 4 * c, 3, 1, 1),  # 16
            _block(4 * c, 8 * c, 4, 2, 1),                              # 8
            nn.Conv2d(8 * c, latent_dim, bottom, 1, 0),                 # 1x1 dense latent
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 8 * c, bottom, 1, 0), nn.BatchNorm2d(8 * c), nn.LeakyReLU(0.2, True),
            _block(8 * c, 4 * c, 4, 2, 1, True), _block(4 * c, 4 * c, 3, 1, 1),  # 16
            _block(4 * c, 2 * c, 4, 2, 1, True), _block(2 * c, 4 * c, 3, 1, 1),  # 32
            _block(4 * c, 2 * c, 4, 2, 1, True), _block(2 * c, 2 * c, 3, 1, 1),  # 64
            _block(2 * c, c, 4, 2, 1, True), _block(c, c, 3, 1, 1),              # 128
            nn.ConvTranspose2d(c, 3, 4, 2, 1),                                   # 256
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Orientation-agnostic augmentation (textures only): random flips + 90° rotations."""
    if torch.rand(()) < 0.5:
        x = x.flip(-1)
    if torch.rand(()) < 0.5:
        x = x.flip(-2)
    k = int(torch.randint(0, 4, ()))
    return torch.rot90(x, k, dims=(-2, -1)) if k else x


class SSIMCAE(AnomalyModel):
    name = "ssim_cae"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        assert self.input_size[0] == self.input_size[1], "SSIM-CAE expects square inputs"
        self.net = ConvAutoencoder(self.input_size[0], cfg.get("latent_dim", 256), cfg.get("base_channels", 32))
        s = cfg.get("score", {})
        self.map_ssim_w = s.get("ssim_weight", 0.5)
        self.map_l1_w = s.get("l1_weight", 0.5)
        self.blur_sigma = s.get("blur_sigma", 4.0)
        self.topk_ratio = s.get("image_topk_ratio", 0.0)
        self.ssim_scales = cfg.get("train", {}).get("ssim_scales", 3)
        self.amp = cfg.get("amp", True)

    def _loss(self, x, x_hat, tcfg):
        return tcfg.get("l1_weight", 1.0) * F.l1_loss(x_hat, x) + \
            tcfg.get("ssim_weight", 1.0) * ms_ssim_loss(x_hat, x, scales=self.ssim_scales)

    def fit(self, ctx: FitContext) -> dict:
        tcfg = self.cfg.get("train", {})
        dev = ctx.device
        self.to(dev)
        aug_mode = tcfg.get("augment", "auto")
        use_aug = aug_mode == "always" or (aug_mode == "auto" and ctx.category in MVTEC_TEXTURES)
        loader = make_loader(ctx.train_ds, tcfg.get("batch_size", 16), True, ctx.num_workers, drop_last=True)
        if len(loader) == 0:
            loader = make_loader(ctx.train_ds, len(ctx.train_ds), True, ctx.num_workers)
        val_loader = make_loader(ctx.val_ds, 16, False, ctx.num_workers) if ctx.val_ds is not None else None
        opt = torch.optim.AdamW(self.net.parameters(), lr=tcfg.get("lr", 5e-4), weight_decay=tcfg.get("weight_decay", 1e-5))
        iters = int(tcfg.get("iters", 4000))
        sched = WarmCosineScheduler(opt, iters, tcfg.get("warmup_iters", 100), final_ratio=0.01)
        keeper = BestKeeper(self.net)
        logger = JsonlLogger(ctx.out_dir / f"train_log_{ctx.category}.jsonl")
        val_every = int(tcfg.get("val_every", 250))
        it_data = infinite(loader)
        self.net.train()
        pbar = tqdm(range(iters), desc=f"ssim_cae[{ctx.category}]", leave=False)
        for it in pbar:
            x = next(it_data)["image"].to(dev, non_blocking=True)
            if use_aug:
                x = torch.stack([_augment(xi) for xi in x])
            with autocast(dev, self.amp):
                x_hat = self.net(x)
            loss = self._loss(x.float(), x_hat.float(), tcfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if (it + 1) % 50 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            if val_loader is not None and ((it + 1) % val_every == 0 or it + 1 == iters):
                val = self._val_loss(val_loader, tcfg, dev)
                keeper.update(val, it + 1)
                logger(iter=it + 1, loss=loss.item(), val_loss=val, lr=sched.get_last_lr()[0])
                self.net.train()
        keeper.restore()
        self.eval()
        log.info("ssim_cae[%s] best val loss %.5f @ %d", ctx.category, keeper.best, keeper.best_iter)
        return {"iters": iters, "best_val_loss": keeper.best, "best_iter": keeper.best_iter}

    @torch.no_grad()
    def _val_loss(self, loader, tcfg, dev) -> float:
        self.net.eval()
        tot, n = 0.0, 0
        for b in loader:
            x = b["image"].to(dev)
            with autocast(dev, self.amp):
                x_hat = self.net(x)
            tot += self._loss(x.float(), x_hat.float(), tcfg).item() * len(x)
            n += len(x)
        return tot / max(n, 1)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.prepare(images)
        with autocast(x.device, self.amp):
            x_hat = self.net(x)
        x_hat = x_hat.float()
        amap = self.map_ssim_w * ms_ssim_anomaly_map(x_hat, x, scales=self.ssim_scales) + \
            self.map_l1_w * (x - x_hat).abs().mean(dim=1)
        amap = gaussian_blur(amap, self.blur_sigma)
        score = topk_mean(amap, self.topk_ratio) if self.topk_ratio > 0 else amap.flatten(1).amax(1)
        return {"anomaly_map": amap, "image_score": score, "reconstruction": x_hat}
