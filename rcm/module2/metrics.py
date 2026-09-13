"""The four attractor metrics of Stage 2 of the methodology.

Every formula here is the design document's, transcribed directly; the only
change is that positions are 0-indexed in code, so the uniform-attention
baseline for a query at position ``q`` is ``1/(q+1)`` rather than ``1/q``.

Writing ``alpha_bar[q, t]`` for the head- and layer-averaged attention from the
query at position ``q`` to the key at position ``t``:

reuse score
    ``R(t) = 1/(T-1-t) * sum_{q>t} (q+1) * alpha_bar[q, t]``
    -- normalised so that a token attended exactly as much as a uniform
    baseline scores 1.

memory lifetime
    ``L(t) = max { q - t : alpha_tilde_q(t) > kappa/(q+1) }`` where
    ``alpha_tilde_q(t)`` is ``alpha_bar[., t]`` smoothed over a window ``w``.

attention persistence
    ``P(t)`` = fraction of ``(query, layer, head)`` triples for which ``t`` sits
    in the top-k of the attention distribution.

centrality
    PageRank of ``t`` in the directed attention graph with edge weight
    ``alpha_bar[q, t]``.

A position is a **Memory Attractor** when all three of
``z_R(t) > z*``, ``L(t) > lambda*T`` and ``P(t) > p*`` hold jointly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcm.config import MetricConfig


@dataclass
class PositionMetrics:
    """Per-position metrics for one conversation."""

    reuse: np.ndarray          # R(t)
    reuse_z: np.ndarray        # z_R(t), within-conversation z-score of R
    lifetime: np.ndarray       # L(t)
    persistence: np.ndarray    # P(t)
    centrality: np.ndarray     # c(t)
    seq_len: int

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "reuseScore": self.reuse,
            "reuseZ": self.reuse_z,
            "lifetime": self.lifetime,
            "persistence": self.persistence,
            "centrality": self.centrality,
        }


def _uniform_baseline(T: int) -> np.ndarray:
    """``1/(q+1)`` for every query position ``q`` (0-indexed)."""
    return 1.0 / np.arange(1, T + 1, dtype=np.float64)


def query_positions(alpha_bar: np.ndarray) -> np.ndarray:
    """Absolute positions that actually issued a query.

    Entries grafted from a peer cache (shared-KV arm) are attended *to* but
    never attend *from*, so their rows are empty.  Every denominator below
    counts real queries only; otherwise the shared-KV arm's metrics would be
    diluted by exactly the entries the experiment is about.
    """
    return np.nonzero(np.asarray(alpha_bar).sum(axis=1) > 0)[0]


def reuse_score(alpha_bar: np.ndarray) -> np.ndarray:
    """``R(t)``: mean future attention, in units of the uniform baseline."""
    a = np.asarray(alpha_bar, dtype=np.float64)
    T = a.shape[0]
    q_idx = query_positions(a)
    weighted = a[q_idx] * (q_idx + 1)[:, None]                # (q+1) * alpha_bar[q, t]
    out = np.zeros(T)
    for t in range(T):
        later = q_idx > t
        n = int(later.sum())
        if n > 0:
            out[t] = weighted[later, t].sum() / n
    return out


def lifetime(alpha_bar: np.ndarray, window: int = 8, kappa: float = 1.0) -> np.ndarray:
    """``L(t)``: last horizon at which smoothed attention still clears the threshold.

    The attention column ``alpha_bar[., t]`` is smoothed with a trailing window
    of ``window`` *queries* and compared against ``kappa`` times the uniform
    baseline smoothed with the same window, so that uniformly attended
    positions score exactly the threshold (lifetime zero) while an entry that
    keeps drawing more than its share stays "alive".
    """
    a = np.asarray(alpha_bar, dtype=np.float64)
    T = a.shape[0]
    q_idx = query_positions(a)
    nq = len(q_idx)
    if nq == 0:
        return np.zeros(T)
    base = 1.0 / (q_idx + 1)

    def trailing_mean(col: np.ndarray) -> np.ndarray:
        csum = np.cumsum(col)
        idx = np.arange(nq)
        lo = idx - window                                   # window covers (q-w, q]
        prev = np.where(lo >= 0, csum[np.clip(lo, 0, nq - 1)], 0.0)
        return (csum - prev) / np.minimum(idx + 1, window)

    thresh = kappa * trailing_mean(base)
    rows = a[q_idx]
    out = np.zeros(T)
    for t in range(T):
        smoothed = trailing_mean(rows[:, t])
        alive = np.nonzero((smoothed > thresh + 1e-12) & (q_idx > t))[0]
        out[t] = float(q_idx[alive].max() - t) if alive.size else 0.0
    return out


def persistence(topk_count: np.ndarray, T: int, n_layers: int, n_heads: int,
                q_idx: np.ndarray | None = None) -> np.ndarray:
    """``P(t)``: fraction of (query, layer, head) triples keeping ``t`` in the top-k."""
    counts = np.asarray(topk_count, dtype=np.float64)[:T]
    q_idx = np.arange(T) if q_idx is None else np.asarray(q_idx)
    denom = np.array([(q_idx > t).sum() * n_layers * n_heads for t in range(T)], dtype=np.float64)
    out = np.zeros(T)
    nz = denom > 0
    out[nz] = counts[nz] / denom[nz]
    return out


def attention_graph_centrality(
    alpha_bar: np.ndarray, damping: float = 0.85, tol: float = 1e-10, max_iter: int = 200
) -> np.ndarray:
    """PageRank over the directed attention graph.

    Nodes are token positions; an edge ``q -> t`` carries weight
    ``alpha_bar[q, t]``.  Rows with no outgoing mass (positions that were never
    used as a query -- grafted peer entries, for instance) are treated as
    dangling nodes and redistribute uniformly, the standard convention.
    """
    a = np.asarray(alpha_bar, dtype=np.float64)
    T = a.shape[0]
    if T == 0:
        return np.zeros(0)
    row = a.sum(axis=1, keepdims=True)
    dangling = (row[:, 0] <= 0)
    norm = np.divide(a, np.where(row > 0, row, 1.0))
    c = np.full(T, 1.0 / T)
    for _ in range(max_iter):
        contrib = norm.T @ c
        contrib += c[dangling].sum() / T
        nxt = (1.0 - damping) / T + damping * contrib
        if np.abs(nxt - c).sum() < tol:
            c = nxt
            break
        c = nxt
    return c


def compute_metrics(stats: dict, cfg: MetricConfig) -> PositionMetrics:
    """Compute all four metrics from a tap's reduced statistics."""
    alpha_bar = np.asarray(stats["alpha_bar"], dtype=np.float64)
    T = int(stats["seq_len"])
    alpha_bar = alpha_bar[:T, :T]
    R = reuse_score(alpha_bar)
    L = lifetime(alpha_bar, cfg.smooth_window, cfg.kappa)
    P = persistence(stats["topk_count"], T, int(stats["n_layers"]), int(stats["n_heads"]),
                    query_positions(alpha_bar))
    c = attention_graph_centrality(alpha_bar, cfg.pagerank_damping)
    sd = R.std()
    Rz = (R - R.mean()) / sd if sd > 1e-12 else np.zeros_like(R)
    return PositionMetrics(reuse=R, reuse_z=Rz, lifetime=L, persistence=P, centrality=c, seq_len=T)
