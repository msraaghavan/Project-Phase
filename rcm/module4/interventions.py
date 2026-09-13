"""Target-set construction for the three KV-only interventions.

A target set ``S`` is a set of ``(layer, KV head, position)`` triples.  It is
built from the Module 2 output: the heads come from the Bias Reservoir ranking,
the positions from the conversation's *novel* attractor set (sinks excluded, so
the intervention cannot be dismissed as "you deleted the attention sink").

Two negative controls are built the same way and are what stop a positive result
from being an artefact of simply removing attention mass:

``random_head_control``  same number of heads and positions, but heads drawn
                         from the non-reservoir pool;
``sink_only_control``    the reservoir heads, but positions restricted to the
                         known attention-sink class.
"""

from __future__ import annotations

import numpy as np

from rcm.adapters.base import InterventionSpec
from rcm.config import InterventionConfig


def build_intervention(
    heads,
    positions,
    mode: str,
    cfg: InterventionConfig,
    directions: dict[tuple[int, int], np.ndarray] | None = None,
) -> InterventionSpec:
    """Build a spec targeting ``positions`` in each of ``heads``.

    ``heads`` is an iterable of ``(layer, kv_head)`` pairs.
    """
    targets = {(int(l), int(h)): sorted({int(p) for p in positions}) for l, h in heads}
    dirs = {}
    if directions:
        for key in targets:
            if key in directions and directions[key] is not None:
                dirs[key] = np.asarray(directions[key], dtype=np.float64)
    return InterventionSpec(
        mode=mode, targets=targets, directions=dirs,
        epsilon=cfg.epsilon, gate=cfg.gate,
    )


def random_head_control(
    reservoir_heads,
    all_heads,
    positions,
    mode: str,
    cfg: InterventionConfig,
    directions: dict[tuple[int, int], np.ndarray] | None = None,
    seed: int = 0,
) -> InterventionSpec:
    """Same-size intervention on heads that are *not* reservoirs."""
    reservoir = {(int(l), int(h)) for l, h in reservoir_heads}
    pool = [k for k in ((int(l), int(h)) for l, h in all_heads) if k not in reservoir]
    if not pool:
        pool = list(reservoir)
    rng = np.random.default_rng(seed)
    n = min(len(reservoir), len(pool)) or 1
    idx = rng.choice(len(pool), size=n, replace=False)
    chosen = [pool[int(i)] for i in idx]
    return build_intervention(chosen, positions, mode, cfg, directions)


def sink_only_control(
    reservoir_heads,
    sink_positions,
    mode: str,
    cfg: InterventionConfig,
    directions: dict[tuple[int, int], np.ndarray] | None = None,
) -> InterventionSpec:
    """Reservoir heads, but only the already-known attention-sink positions."""
    return build_intervention(reservoir_heads, sink_positions, mode, cfg, directions)


def union_positions(attractor_sets, use_novel: bool = True, limit: int | None = None) -> list[int]:
    """Pool attractor positions across conversations into one target position set.

    A single target set is used for every replay so that the intervention is a
    fixed, pre-declared object rather than something re-tuned per conversation.
    """
    counts: dict[int, int] = {}
    for s in attractor_sets:
        for p in (s.novel if use_novel else s.all_positions):
            counts[p] = counts.get(p, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if limit is not None:
        ranked = ranked[:limit]
    return sorted(p for p, _ in ranked)
