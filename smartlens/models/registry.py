"""Model registry."""

from __future__ import annotations

from smartlens.models.base import AnomalyModel


def _registry() -> dict[str, type[AnomalyModel]]:
    from smartlens.models.baselines.dinomaly import Dinomaly
    from smartlens.models.baselines.patchcore import PatchCore
    from smartlens.models.baselines.ssim_cae import SSIMCAE

    return {m.name: m for m in (SSIMCAE, PatchCore, Dinomaly)}


class _LazyRegistry(dict):
    def _fill(self):
        if not super().__len__():
            self.update(_registry())

    def __getitem__(self, k):
        self._fill()
        return super().__getitem__(k)

    def __contains__(self, k):
        self._fill()
        return super().__contains__(k)

    def keys(self):
        self._fill()
        return super().keys()


MODEL_REGISTRY = _LazyRegistry()


def build_model(cfg: dict) -> AnomalyModel:
    name = cfg["name"]
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](cfg)
