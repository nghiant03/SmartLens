"""Minimal YAML configuration utilities (deep merge + dotted CLI overrides + env expansion)."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
#: datasets live inside the repository (git-ignored) unless SMARTLENS_DATA_ROOT overrides it
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Optional single-level inheritance: ``_base_: other.yaml`` (relative to this file).
    base = cfg.pop("_base_", None)
    if base:
        cfg = deep_merge(load_yaml(path.parent / base), cfg)
    return cfg


def _parse_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    """Apply ``a.b.c=value`` style overrides (values parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


def resolve_path(path: str | Path) -> Path:
    """Expand ``~`` and environment variables. ``$SMARTLENS_DATA_ROOT`` defaults to ``<repo>/datasets``."""
    text = str(path)
    if "SMARTLENS_DATA_ROOT" in text and "SMARTLENS_DATA_ROOT" not in os.environ:
        text = text.replace("${SMARTLENS_DATA_ROOT}", str(DEFAULT_DATA_ROOT))
        text = text.replace("$SMARTLENS_DATA_ROOT", str(DEFAULT_DATA_ROOT))
    p = Path(os.path.expandvars(os.path.expanduser(text)))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def save_yaml(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
