"""Modules 3 and 4: contagion decomposition, transfer entropy, phase fit,
changepoint, BH-FDR and the intervention acceptance rule."""

from types import SimpleNamespace

import numpy as np

from rcm.adapters.base import InterventionSpec
from rcm.config import ARM_CONTROL, ARM_SHARED, ARM_TEXT, InterventionConfig
from rcm.module2.reservoirs import benjamini_hochberg
from rcm.module3.contagion import contagion_decomposition, transfer_entropy
from rcm.module3.phase import changepoint_1d, fit_phase_transition, logistic_curve
from rcm.module4.evaluate import CellOutcome, evaluate_intervention


def test_contagion_is_a_residual():
    r = contagion_decomposition({ARM_SHARED: [0.8, 0.9], ARM_TEXT: [0.5, 0.6], ARM_CONTROL: [0.2, 0.2]},
                                n_boot=200)
    assert np.isclose(r.memory_contagion, 0.30)
    assert np.isclose(r.text_contagion, 0.35)
    assert np.isclose(r.memory_contagion + r.text_contagion, r.bias_shared - r.bias_control)


def test_transfer_entropy_detects_coupling():
    rng = np.random.default_rng(0)
    src, tgt_coupled, tgt_indep = [], [], []
    for _ in range(60):
        x = rng.normal(size=8)
        src.append(x)
        tgt_coupled.append(np.concatenate([[0.0], x[:-1]]) + 0.05 * rng.normal(size=8))
        tgt_indep.append(rng.normal(size=8))
    assert transfer_entropy(src, tgt_coupled) > transfer_entropy(src, tgt_indep) + 0.1


def test_phase_fit_recovers_planted_transition():
    pis = np.linspace(0, 1, 9)
    by_p = {}
    rng = np.random.default_rng(1)
    for p in pis:
        by_p[float(p)] = list(logistic_curve(p, -0.2, 0.8, 25.0, 0.55) + 0.01 * rng.normal(size=5))
    res = fit_phase_transition(pis, by_p, n_boot=100, n_models=2)
    assert abs(res.pi_c - 0.55) < 0.05
    assert res.criteria["logistic_beats_linear_aic"] and res.criteria["logistic_beats_linear_bic"]
    assert res.criteria["kappa_ci_excludes_linear"]
    assert res.transition_detected


def test_phase_fit_rejects_linear_data():
    pis = np.linspace(0, 1, 9)
    rng = np.random.default_rng(2)
    by_p = {float(p): list(0.1 + 0.5 * p + 0.01 * rng.normal(size=5)) for p in pis}
    res = fit_phase_transition(pis, by_p, n_boot=100)
    assert not res.transition_detected


def test_changepoint_exact():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    idx, cost = changepoint_1d(y)
    assert idx == 4 and cost == 0.0


def test_benjamini_hochberg():
    rejected, q = benjamini_hochberg(np.array([0.001, 0.01, 0.2, 0.9]), alpha=0.05)
    assert rejected.tolist() == [True, True, False, False]
    assert np.all(np.diff(q) >= 0)


class FakeReplay:
    """Stands in for the orchestrator: every cell has a fixed bias, and each edit
    shifts it (and optionally perplexity / factual accuracy) by a known amount."""

    def __init__(self, edits):
        self.edits = edits                    # id(spec) -> (d_bias, ppl, n_wrong of 10)

    def run(self, arm, pressure, seed, intervention=None):
        d_bias, ppl, wrong = self.edits.get(id(intervention), (0.0, 2.0, 0))
        return SimpleNamespace(final_bias=0.4 + 0.03 * seed + d_bias,
                               turns=[SimpleNamespace(role="receiver", ppl=ppl)],
                               hallucination_flags=[i < wrong for i in range(10)])


def _judge(edit, random_edit=(-0.02, 2.0, 0), sink_edit=(-0.02, 2.0, 0)):
    spec = InterventionSpec("ablate", {(0, 0): [5]})
    rnd, sink = InterventionSpec("ablate", {(1, 1): [5]}), InterventionSpec("ablate", {(0, 0): [0]})
    fake = FakeReplay({id(spec): edit, id(rnd): random_edit, id(sink): sink_edit})
    cells = [("shared-kv", 1.0, s) for s in range(10)]
    return evaluate_intervention(fake, cells, spec, {"random": rnd, "sink": sink}, InterventionConfig())


def test_acceptance_rule_accepts_a_clean_effect():
    out = _judge((-0.2, 2.0, 0))
    assert out.accepted and out.reasons == []
    assert np.isclose(out.ace, -0.2) and np.isclose(out.ace_random, -0.02)
    assert np.isclose(out.relative_reduction, 0.2 / np.mean([0.4 + 0.03 * s for s in range(10)]))
    assert out.p_vs_zero < 0.05 and out.p_vs_random < 0.05 and out.coherence_ok


def test_acceptance_rule_rejections():
    """Each failure mode is caught and named, as reported on the results page."""
    forgets = _judge((-0.2, 2.0, 1))                       # accuracy 1.0 -> 0.9: over the 2-point limit
    assert not forgets.accepted and np.isclose(forgets.delta_acc, -0.1)
    assert any("coherence budget violated: dAcc" in r for r in forgets.reasons)
    garbles = _judge((-0.2, 2.6, 0))                       # perplexity +30 %: over the 5 % limit
    assert not garbles.accepted and np.isclose(garbles.delta_ppl, 0.3)
    assert any("dPPL" in r for r in garbles.reasons)
    placebo = _judge((-0.2, 2.0, 0), random_edit=(-0.2, 2.0, 0))
    assert not placebo.accepted and any("random-head control" in r for r in placebo.reasons)
    nothing = _judge((0.0, 2.0, 0))
    assert not nothing.accepted and any("not significant vs zero" in r for r in nothing.reasons)
    c = CellOutcome("shared-kv", 1.0, 0, bias_base=0.6, bias_int=0.3, ppl_base=2.0, ppl_int=2.0,
                    acc_base=1.0, acc_int=1.0)
    assert np.isclose(c.delta_bias, -0.3)
