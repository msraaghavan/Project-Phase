"""Module 2 -- Memory structure analysis: attractors and bias reservoirs.

Realises Objectives 2 and 3: detect persistent KV structures (**Memory
Attractors**) beyond the known attention-sink class, and localise where social
bias accumulates (**Bias Reservoirs**).
"""

from rcm.module2.attractors import AttractorSets, detect_attractors
from rcm.module2.metrics import (
    PositionMetrics,
    attention_graph_centrality,
    compute_metrics,
    lifetime,
    persistence,
    reuse_score,
)
from rcm.module2.reservoirs import ReservoirResult, localize_reservoirs

__all__ = [
    "PositionMetrics",
    "compute_metrics",
    "reuse_score",
    "lifetime",
    "persistence",
    "attention_graph_centrality",
    "AttractorSets",
    "detect_attractors",
    "ReservoirResult",
    "localize_reservoirs",
]
