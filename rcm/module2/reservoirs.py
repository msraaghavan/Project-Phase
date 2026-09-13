"""Bias Reservoir localisation.

Stage 3 of the methodology.  For each head the value vectors sitting at that
conversation's attractor positions are pooled,

    u^n_{l,h} = (1/|A_n|) * sum_{t in A_n} v_{l,h,t}

and an L2-regularised logistic probe is cross-validated to predict the
conversation's bias polarity ``y_n`` from ``u^n_{l,h}``.

Predictive evidence alone is not enough -- a probe can read a correlate it does
not control -- so the **Bias Reservoir score** multiplies held-out AUC by the
*measured* effect of ablating that head:

    S(l, h) = AUC_{l,h} * max(0, (B - B^{abl(l,h)}) / B)

Reservoirs are the top-m heads by ``S`` that survive a Benjamini-Hochberg
correction against a random-head null, so a head is only called a reservoir if
it both predicts bias and causally carries it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rcm.config import MetricConfig


@dataclass
class HeadResult:
    layer: int
    kv_head: int
    auc: float
    p_value: float
    q_value: float = 1.0
    ablation_effect: float = 0.0
    score: float = 0.0
    direction: np.ndarray | None = None
    n_samples: int = 0

    def key(self) -> tuple[int, int]:
        return (self.layer, self.kv_head)


@dataclass
class ReservoirResult:
    heads: list[HeadResult] = field(default_factory=list)
    reservoirs: list[HeadResult] = field(default_factory=list)
    baseline_bias: float = 0.0
    used_ablation: bool = False

    def ranking(self) -> list[dict]:
        return [
            {
                "layer": h.layer,
                "head": h.kv_head,
                "auc": h.auc,
                "pValue": h.p_value,
                "qValue": h.q_value,
                "ablationEffect": h.ablation_effect,
                "score": h.score,
                "isReservoir": any(h.key() == r.key() for r in self.reservoirs),
            }
            for h in sorted(self.heads, key=lambda x: (-x.score, -x.auc))
        ]

    def directions(self) -> dict[tuple[int, int], np.ndarray]:
        return {h.key(): h.direction for h in self.heads if h.direction is not None}


def pool_attractor_values(values: np.ndarray, positions: list[int]) -> np.ndarray:
    """Pool value vectors over a conversation's attractor set.

    ``values`` is ``[n_layers, n_kv_heads, T, head_dim]``; the result is
    ``[n_layers, n_kv_heads, head_dim]``.  With an empty attractor set the
    conversation contributes a zero vector, which the probe then cannot use.
    """
    L, H, T, D = values.shape
    pos = [p for p in positions if 0 <= p < T]
    if not pos:
        return np.zeros((L, H, D), dtype=np.float64)
    return values[:, :, pos, :].mean(axis=2).astype(np.float64)


def _probe():
    """L2-regularised logistic probe on standardised features.

    Standardising is what makes the probe well conditioned: raw value vectors
    have very different scales per dimension and lbfgs needs thousands of
    iterations on them.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0))


def _direction(pipe) -> np.ndarray:
    """Probe weight vector expressed in the *raw* value space, for perturbation."""
    scaler, clf = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]
    return (clf.coef_[0] / np.where(scaler.scale_ > 0, scaler.scale_, 1.0)).astype(np.float64)


def _cv_auc(X: np.ndarray, y: np.ndarray, folds: int, seed: int = 0) -> tuple[float, np.ndarray]:
    """Cross-validated AUC and the probe direction fitted on all data."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return 0.5, np.zeros(X.shape[1])
    n_splits = int(min(folds, np.bincount(y).min()))
    scores = np.zeros(len(y), dtype=float)
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            clf = _probe()
            clf.fit(X[tr], y[tr])
            scores[te] = clf.decision_function(X[te])
        auc = float(roc_auc_score(y, scores))
    else:  # too few samples in a class for CV; fall back to in-sample
        auc = 0.5
    full = _probe().fit(X, y)
    return auc, _direction(full)


def _permutation_p(X: np.ndarray, y: np.ndarray, observed: float, folds: int,
                   n_perm: int, seed: int) -> float:
    """Random-head null: how often a label permutation beats the observed AUC."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2 or n_perm <= 0:
        return 1.0
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(y)
        auc, _ = _cv_auc(X, perm, folds, seed=int(rng.integers(1 << 30)))
        if auc >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """BH step-up procedure.  Returns ``(rejected, q_values)``."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]        # enforce monotonicity
    q = np.clip(q, 0, 1)
    out_q = np.empty(n)
    out_q[order] = q
    return out_q <= alpha, out_q


def localize_reservoirs(
    pooled: np.ndarray,                 # [n_conv, n_layers, n_kv_heads, head_dim]
    labels: np.ndarray,                 # [n_conv] bias polarity in {0, 1}
    cfg: MetricConfig,
    ablation_fn: Callable[[int, int], float] | None = None,
    baseline_bias: float = 0.0,
    n_perm: int = 50,
    seed: int = 0,
) -> ReservoirResult:
    """Rank every (layer, KV head) by the Bias Reservoir score."""
    n_conv, L, H, D = pooled.shape
    labels = np.asarray(labels).astype(int)
    heads: list[HeadResult] = []
    for l in range(L):
        for h in range(H):
            X = pooled[:, l, h, :]
            auc, direction = _cv_auc(X, labels, cfg.probe_folds, seed=seed + 17 * l + h)
            p = _permutation_p(X, labels, auc, cfg.probe_folds, n_perm, seed + 101 * l + h)
            heads.append(
                HeadResult(layer=l, kv_head=h, auc=auc, p_value=p,
                           direction=direction, n_samples=n_conv)
            )

    rejected, qvals = benjamini_hochberg(np.array([x.p_value for x in heads]), cfg.fdr_alpha)
    for head, q, rej in zip(heads, qvals, rejected):
        head.q_value = float(q)

    used_ablation = ablation_fn is not None
    for head in heads:
        effect = 0.0
        if used_ablation and abs(baseline_bias) > 1e-9:
            b_abl = ablation_fn(head.layer, head.kv_head)
            effect = max(0.0, (baseline_bias - b_abl) / abs(baseline_bias))
        head.ablation_effect = float(effect)
        head.score = float(head.auc * effect) if used_ablation else float(head.auc)

    survivors = [h for h, rej in zip(heads, rejected) if rej]
    # ties in S (e.g. no measurable ablation effect) are broken by probe AUC
    ranked = sorted(survivors or [], key=lambda x: (-x.score, -x.auc))[: cfg.n_reservoirs]
    return ReservoirResult(
        heads=heads, reservoirs=ranked, baseline_bias=baseline_bias, used_ablation=used_ablation
    )
