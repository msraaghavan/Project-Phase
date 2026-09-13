"""Memory Contagion: the component of bias transfer that text cannot explain.

With the three arms shared-KV ``S``, text-only ``T`` and isolated control ``C``,
and ``B_j`` the receiving agent's bias index,

    MC_{i->j} = B_j^(S) - B_j^(T)        memory-mediated transfer
    TC_{i->j} = B_j^(T) - B_j^(C)        text-mediated transfer

Defining contagion as a *residual* is what makes the claim falsifiable: if
everything the receiver picks up is explainable by the text it saw, ``MC`` is
zero by construction, and no amount of correlation elsewhere can rescue it.

Directed dependence is corroborated two ways: transfer entropy between the two
agents' per-turn bias series, and the Jaccard overlap of their attractor sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

from rcm.config import ARM_CONTROL, ARM_SHARED, ARM_TEXT


@dataclass
class ContagionResult:
    memory_contagion: float
    text_contagion: float
    bias_shared: float
    bias_text: float
    bias_control: float
    mc_ci: tuple[float, float]
    mc_p_value: float
    n_shared: int
    n_text: int
    n_control: int

    def as_dict(self) -> dict[str, float]:
        return {
            "memoryContagion": self.memory_contagion,
            "textContagion": self.text_contagion,
            "biasShared": self.bias_shared,
            "biasText": self.bias_text,
            "biasControl": self.bias_control,
            "mcCiLow": self.mc_ci[0],
            "mcCiHigh": self.mc_ci[1],
            "mcPValue": self.mc_p_value,
            "nShared": self.n_shared,
            "nText": self.n_text,
            "nControl": self.n_control,
        }


def _bootstrap_diff_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``mean(a) - mean(b)``."""
    rng = np.random.default_rng(seed)
    if len(a) == 0 or len(b) == 0:
        return (float("nan"), float("nan"))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def contagion_decomposition(
    bias_by_arm: dict[str, Sequence[float]], n_boot: int = 5000, seed: int = 0
) -> ContagionResult:
    """Split observed transfer into its memory-mediated and text-mediated parts."""
    s = np.asarray(bias_by_arm.get(ARM_SHARED, []), dtype=float)
    t = np.asarray(bias_by_arm.get(ARM_TEXT, []), dtype=float)
    c = np.asarray(bias_by_arm.get(ARM_CONTROL, []), dtype=float)
    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")  # noqa: E731
    mc = mean(s) - mean(t)
    tc = mean(t) - mean(c)
    ci = _bootstrap_diff_ci(s, t, n_boot=n_boot, seed=seed)
    if len(s) > 1 and len(t) > 1:
        p = float(stats.mannwhitneyu(s, t, alternative="two-sided").pvalue)
    else:
        p = float("nan")
    return ContagionResult(
        memory_contagion=mc, text_contagion=tc,
        bias_shared=mean(s), bias_text=mean(t), bias_control=mean(c),
        mc_ci=ci, mc_p_value=p, n_shared=len(s), n_text=len(t), n_control=len(c),
    )


# --------------------------------------------------------------------------
# Transfer entropy
# --------------------------------------------------------------------------
def _discretize(x: np.ndarray, n_bins: int, edges: np.ndarray | None = None) -> np.ndarray:
    if edges is None:
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]) if len(x) else np.array([0.0])
    return np.clip(np.searchsorted(edges, x, side="right"), 0, n_bins - 1)


def transfer_entropy(
    source_series: Sequence[Sequence[float]],
    target_series: Sequence[Sequence[float]],
    n_bins: int = 2,
    smoothing: float = 0.5,
) -> float:
    """Plug-in transfer entropy ``TE_{X->Y}`` in bits, pooled over conversations.

    ``TE = sum p(y', y, x) log2 [ p(y'|y, x) / p(y'|y) ]``

    Per-conversation series are only a handful of turns long, so the triples
    ``(y_{t+1}, y_t, x_t)`` are pooled across conversations of the same arm and
    the joint table is Laplace-smoothed.  A positive value means the sender's
    bias history carries information about the receiver's next bias state beyond
    the receiver's own history.
    """
    xs, ys_prev, ys_next = [], [], []
    for sx, sy in zip(source_series, target_series):
        sx = np.asarray(sx, dtype=float)
        sy = np.asarray(sy, dtype=float)
        n = min(len(sx), len(sy)) - 1
        for i in range(max(0, n)):
            xs.append(sx[i])
            ys_prev.append(sy[i])
            ys_next.append(sy[i + 1])
    if len(xs) < 4:
        return float("nan")
    xs, ys_prev, ys_next = map(np.asarray, (xs, ys_prev, ys_next))
    pooled = np.concatenate([ys_prev, ys_next])
    y_edges = np.quantile(pooled, np.linspace(0, 1, n_bins + 1)[1:-1])
    x_edges = np.quantile(xs, np.linspace(0, 1, n_bins + 1)[1:-1])
    X = _discretize(xs, n_bins, x_edges)
    Yp = _discretize(ys_prev, n_bins, y_edges)
    Yn = _discretize(ys_next, n_bins, y_edges)

    joint = np.full((n_bins, n_bins, n_bins), smoothing)   # [y_next, y_prev, x]
    for a, b, c in zip(Yn, Yp, X):
        joint[a, b, c] += 1.0
    joint /= joint.sum()

    p_yp_x = joint.sum(axis=0)                              # [y_prev, x]
    p_yn_yp = joint.sum(axis=2)                             # [y_next, y_prev]
    p_yp = joint.sum(axis=(0, 2))                           # [y_prev]

    te = 0.0
    for a in range(n_bins):
        for b in range(n_bins):
            for c in range(n_bins):
                pj = joint[a, b, c]
                if pj <= 0:
                    continue
                cond_full = pj / p_yp_x[b, c] if p_yp_x[b, c] > 0 else 0.0
                cond_own = p_yn_yp[a, b] / p_yp[b] if p_yp[b] > 0 else 0.0
                if cond_full > 0 and cond_own > 0:
                    te += pj * np.log2(cond_full / cond_own)
    return float(te)


def attractor_overlap(sets_a, sets_b, use_novel: bool = True) -> float:
    """Mean Jaccard overlap between two aligned lists of attractor sets."""
    vals = [a.jaccard(b, use_novel=use_novel) for a, b in zip(sets_a, sets_b)]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")
