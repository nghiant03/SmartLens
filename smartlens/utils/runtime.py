"""Runtime helpers: seeding, logging, device and in-repo weight caches."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

from smartlens.config import PROJECT_ROOT


def setup_env() -> None:
    """Keep every downloaded artefact (pretrained weights) inside the repository."""
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "weights" / "torch"))
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "weights" / "huggingface"))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger("smartlens")


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
