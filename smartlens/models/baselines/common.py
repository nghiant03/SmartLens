"""Shared training-loop helpers for the baselines."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch

from smartlens.models.base import log


class JsonlLogger:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def __call__(self, **row) -> None:
        row = {"time_s": round(time.time() - self.t0, 2), **row}
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")


class BestKeeper:
    """Keeps a CPU copy of the best (lowest validation loss) module state for early stopping."""

    def __init__(self, module: torch.nn.Module, patience: int | None = None) -> None:
        self.module = module
        self.best = float("inf")
        self.best_state = None
        self.best_iter = -1
        self.patience = patience
        self.bad = 0

    def update(self, value: float, it: int) -> bool:
        """Returns True if training should stop (patience exhausted)."""
        if value < self.best:
            self.best, self.best_iter, self.bad = value, it, 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in self.module.state_dict().items()}
        else:
            self.bad += 1
        return self.patience is not None and self.bad >= self.patience

    def restore(self) -> None:
        if self.best_state is not None:
            self.module.load_state_dict(self.best_state)
            log.info("Restored best checkpoint from iter %d (val=%.5f)", self.best_iter, self.best)


def autocast(device: torch.device, enabled: bool, dtype=torch.bfloat16):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled and device.type == "cuda")


def deepcopy_cfg(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
