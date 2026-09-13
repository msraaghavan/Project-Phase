"""Module 2 metrics on synthetic caches with analytically known structure."""

import networkx as nx
import numpy as np
import pytest

from rcm.config import MetricConfig
from rcm.module2.attractors import detect_attractors
from rcm.module2.metrics import (
    attention_graph_centrality,
    compute_metrics,
    lifetime,
    persistence,
    reuse_score,
)


def uniform_alpha(T):
    """Every query attends uniformly over its causal window."""
    a = np.zeros((T, T))
    for q in range(T):
        a[q, : q + 1] = 1.0 / (q + 1)
    return a


def attractor_alpha(T, star, mass=0.6):
    """Every query after `star` puts `mass` on it, the rest uniformly."""
    a = uniform_alpha(T)
    for q in range(star + 1, T):
        a[q, :] = (1 - mass) / (q + 1)
        a[q, star] += mass
    return a


def test_reuse_uniform_baseline_scores_one():
    R = reuse_score(uniform_alpha(40))
    assert np.allclose(R[:-1], 1.0)
    assert R[-1] == 0.0                      # last position has no future queries


def test_reuse_attractor_above_one():
    T, star = 40, 10
    R = reuse_score(attractor_alpha(T, star))
    assert R[star] > 5.0
    assert np.all(R[star + 1:-1] < 1.0)


def test_lifetime_uniform_is_zero_and_attractor_spans_sequence():
    T, star = 60, 8
    assert np.all(lifetime(uniform_alpha(T), window=4, kappa=1.0) == 0)
    L = lifetime(attractor_alpha(T, star), window=4, kappa=1.0)
    assert L[star] == T - 1 - star


def test_persistence_denominator():
    T, L_, H = 10, 2, 4
    counts = np.zeros(T)
    counts[3] = (T - 1 - 3) * L_ * H       # in top-k for every later (q, l, h)
    P = persistence(counts, T, L_, H)
    assert P[3] == pytest.approx(1.0)
    assert P[-1] == 0.0


def test_pagerank_matches_networkx():
    T = 25
    a = attractor_alpha(T, 5)
    ours = attention_graph_centrality(a, damping=0.85)
    G = nx.DiGraph()
    G.add_nodes_from(range(T))
    for q in range(T):
        for t in range(T):
            if a[q, t] > 0:
                G.add_edge(q, t, weight=float(a[q, t]))
    ref = nx.pagerank(G, alpha=0.85, weight="weight", tol=1e-12)
    ref = np.array([ref[i] for i in range(T)])
    assert np.allclose(ours, ref, atol=1e-6)
    # position 0 is a self-loop absorbing node -- the attention sink in PageRank
    # form -- which is exactly why the design reports the sink class separately;
    # outside it, the planted attractor dominates.
    assert np.argmax(ours[1:]) + 1 == 5


def test_joint_criterion_finds_planted_attractor_only():
    T, L_, H, star = 80, 4, 8, 20
    a = attractor_alpha(T, star, mass=0.5)
    counts = np.zeros(T)
    counts[star] = (T - 1 - star) * L_ * H
    counts[0] = (T - 1) * L_ * H * 0.02      # weak early presence, below p*
    stats = {"alpha_bar": a, "topk_count": counts, "seq_len": T, "n_layers": L_, "n_heads": H}
    cfg = MetricConfig()
    m = compute_metrics(stats, cfg)
    sets = detect_attractors(m, cfg)
    assert sets.novel == [star]
    assert sets.sink == []
