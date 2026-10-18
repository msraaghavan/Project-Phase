"""Memory Attractor detection.

A position is declared an attractor when the reuse, lifetime and persistence
conditions hold **jointly**:

    A = { t : z_R(t) > z*  and  L(t) > lambda*T  and  P(t) > p* }

Early positions (``t <= sink_positions``) are reported as a *separate baseline
class*.  That separation is the whole point of the analysis: attention sinks at
the start of the sequence are already known from Xiao et al. (2024), so a
detector that only rediscovers them has found nothing new.  Only attractors
outside the sink class count as a finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rcm.config import MetricConfig
from rcm.module2.metrics import PositionMetrics
from rcm.probes import ITOS


@dataclass
class AttractorSets:
    """The attractor set of one conversation, split into sink and novel."""

    conv_id: str
    all_positions: list[int] = field(default_factory=list)
    sink: list[int] = field(default_factory=list)
    novel: list[int] = field(default_factory=list)
    seq_len: int = 0

    @property
    def n_novel(self) -> int:
        return len(self.novel)

    @property
    def novel_fraction(self) -> float:
        return self.n_novel / self.seq_len if self.seq_len else 0.0

    def jaccard(self, other: "AttractorSets", use_novel: bool = True) -> float:
        """Attractor-set overlap ``J = |A_i and A_j| / |A_i or A_j|``."""
        a = set(self.novel if use_novel else self.all_positions)
        b = set(other.novel if use_novel else other.all_positions)
        if not a and not b:
            return float("nan")
        return len(a & b) / len(a | b)


def detect_attractors(
    metrics: PositionMetrics, cfg: MetricConfig, conv_id: str = "conv"
) -> AttractorSets:
    """Apply the joint attractor criterion."""
    T = metrics.seq_len
    mask = (
        (metrics.reuse_z > cfg.z_star)
        & (metrics.lifetime > cfg.lam * T)
        & (metrics.persistence > cfg.p_star)
    )
    positions = np.nonzero(mask)[0].tolist()
    sink = [t for t in positions if t <= cfg.sink_positions]
    novel = [t for t in positions if t > cfg.sink_positions]
    return AttractorSets(
        conv_id=conv_id, all_positions=positions, sink=sink, novel=novel, seq_len=T
    )


def attractor_token_profile(sets: AttractorSets, token_ids, label=None) -> dict[str, int]:
    """Which vocabulary items the novel attractors sit on.

    In the controlled testbed the answer is checkable by construction: the
    group and trait tokens carry the bias, so attractors landing on them is the
    expected positive result, and attractors landing on function words would
    signal a detector problem.
    """
    profile: dict[str, int] = {}
    for t in sets.novel:
        if t < len(token_ids):
            tok = label(int(token_ids[t])) if label else ITOS[int(token_ids[t])]
            profile[tok] = profile.get(tok, 0) + 1
    return dict(sorted(profile.items(), key=lambda kv: -kv[1]))


def sink_baseline_comparison(metrics: PositionMetrics, cfg: MetricConfig) -> dict[str, float]:
    """Summary statistics contrasting the sink class with the rest.

    Reported so a reader can judge whether the novel attractors are merely a
    weaker version of the sink phenomenon or a structurally different class.
    """
    T = metrics.seq_len
    sink_mask = np.arange(T) <= cfg.sink_positions
    out = {}
    for name, arr in metrics.as_arrays().items():
        out[f"sink_{name}"] = float(np.mean(arr[sink_mask])) if sink_mask.any() else float("nan")
        out[f"nonsink_{name}"] = (
            float(np.mean(arr[~sink_mask])) if (~sink_mask).any() else float("nan")
        )
    return out
