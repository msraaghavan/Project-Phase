"""Model adapters.

All model-specific behaviour -- attention-module naming, GQA head mapping and
RoPE re-indexing -- lives behind :class:`~rcm.adapters.base.ModelAdapter`, so
supporting a new architecture means writing one adapter file rather than
editing the pipeline (design doc, "Maintainability").
"""

from rcm.adapters.base import InterventionSpec, ModelAdapter, TurnOutput
from rcm.adapters.tiny import TinyAdapter, TinyTransformer

__all__ = [
    "ModelAdapter",
    "InterventionSpec",
    "TurnOutput",
    "TinyAdapter",
    "TinyTransformer",
    "build_adapter",
]


def build_adapter(cfg):
    """Construct the adapter named by ``cfg.model.backend``."""
    backend = cfg.model.backend
    if backend == "tiny":
        return TinyAdapter(cfg)
    if backend == "hf":
        from rcm.adapters.hf import HFAdapter  # imported lazily: heavy dependency

        return HFAdapter(cfg)
    raise ValueError(f"unknown model backend {backend!r} (expected 'tiny' or 'hf')")
