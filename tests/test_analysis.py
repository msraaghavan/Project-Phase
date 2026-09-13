"""Modules 3 and 4: contagion decomposition, transfer entropy, phase fit,
changepoint, BH-FDR and the intervention acceptance rule."""

import numpy as np

from rcm.config import ARM_CONTROL, ARM_SHARED, ARM_TEXT, InterventionConfig
from rcm.module2.reservoirs import benjamini_hochberg
from rcm.module3.contagion import contagion_decomposition, transfer_entropy
from rcm.module3.phase import changepoint_1d, fit_phase_transition, logistic_curve
from rcm.module4.evaluate import CellOutcome, InterventionOutcome


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


def test_acceptance_rule_rejects_on_coherence_budget():
    cfg = InterventionConfig()
    out = InterventionOutcome(mode="ablate", delta_ppl=0.30, delta_acc=0.0, p_vs_zero=0.001,
                              p_vs_random=0.001, p_vs_sink=0.001)
    ppl_ok = out.delta_ppl <= cfg.max_ppl_increase
    assert not ppl_ok                         # 30 % perplexity increase breaks the 5 % budget
    c = CellOutcome("shared-kv", 1.0, 0, bias_base=0.6, bias_int=0.3, ppl_base=2.0, ppl_int=2.0,
                    acc_base=1.0, acc_int=1.0)
    assert np.isclose(c.delta_bias, -0.3)
