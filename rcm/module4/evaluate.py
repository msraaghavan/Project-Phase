"""Causal evaluation of a KV-only intervention.

Each evaluation cell ``(arm, pressure, seed)`` is replayed twice under the same
seed -- once untouched and once with the KV edit applied -- so the two runs
differ *only* in the edit.  The average causal effect is

    ACE(S) = E[B | do(S)] - E[B]

estimated by paired replay.  An intervention is accepted only when all of:

* ACE is significantly below zero (paired bootstrap);
* ACE is significantly stronger than the random-head control;
* ACE is significantly stronger than the sink-only control;
* the coherence budget holds -- relative perplexity increase within
  ``max_ppl_increase`` and factual accuracy within ``max_acc_drop``.

The coherence budget is a **rejection criterion, not a reported metric**: an
intervention that lowers bias by degrading the model fails the workflow instead
of passing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rcm.config import InterventionConfig


@dataclass
class CellOutcome:
    arm: str
    pressure: float
    seed: int
    bias_base: float
    bias_int: float
    ppl_base: float
    ppl_int: float
    acc_base: float
    acc_int: float

    @property
    def delta_bias(self) -> float:
        return self.bias_int - self.bias_base


@dataclass
class InterventionOutcome:
    mode: str
    n_cells: int = 0
    n_heads: int = 0
    n_entries: int = 0
    ace: float = float("nan")
    ace_ci: tuple[float, float] = (float("nan"), float("nan"))
    relative_reduction: float = float("nan")
    delta_ppl: float = float("nan")
    delta_acc: float = float("nan")
    p_vs_zero: float = float("nan")
    p_vs_random: float = float("nan")
    p_vs_sink: float = float("nan")
    ace_random: float = float("nan")
    ace_sink: float = float("nan")
    coherence_ok: bool = False
    accepted: bool = False
    reasons: list[str] = field(default_factory=list)
    cells: list[CellOutcome] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode, "nCells": self.n_cells, "nHeads": self.n_heads,
            "nEntries": self.n_entries, "ace": self.ace,
            "aceCiLow": self.ace_ci[0], "aceCiHigh": self.ace_ci[1],
            "relativeReduction": self.relative_reduction,
            "deltaPpl": self.delta_ppl, "deltaAcc": self.delta_acc,
            "pVsZero": self.p_vs_zero, "pVsRandom": self.p_vs_random,
            "pVsSink": self.p_vs_sink, "aceRandom": self.ace_random,
            "aceSink": self.ace_sink, "coherenceOk": self.coherence_ok,
            "accepted": self.accepted, "reasons": self.reasons,
        }


def _paired_bootstrap_p(diff: np.ndarray, n_boot: int, seed: int, alternative: str = "less") -> float:
    """One-sided paired bootstrap p-value for the mean of ``diff``."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(n_boot)])
    if alternative == "less":
        return float((means >= 0).mean())
    return float((means <= 0).mean())


def _bootstrap_ci(x: np.ndarray, n_boot: int, alpha: float, seed: int) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def _cell_stats(trace) -> tuple[float, float, float]:
    """``(bias, mean turn perplexity, factual accuracy)`` of one replay."""
    bias = float(trace.final_bias)
    ppls = [t.ppl for t in trace.turns if t.role == "receiver" and np.isfinite(t.ppl)]
    ppl = float(np.mean(ppls)) if ppls else float("nan")
    flags = trace.hallucination_flags
    acc = float(1.0 - np.mean(flags)) if flags else float("nan")
    return bias, ppl, acc


def replay_cells(orchestrator, cells, intervention=None) -> list:
    """Replay every ``(arm, pressure, seed)`` cell, optionally with an edit."""
    return [
        orchestrator.run(arm=arm, pressure=pressure, seed=seed, intervention=intervention)
        for arm, pressure, seed in cells
    ]


def evaluate_intervention(
    orchestrator,
    cells,
    spec,
    control_specs: dict[str, object] | None = None,
    cfg: InterventionConfig | None = None,
    baseline_traces=None,
    seed: int = 0,
) -> InterventionOutcome:
    """Run the paired replay and apply the acceptance rule."""
    cfg = cfg or InterventionConfig()
    control_specs = control_specs or {}
    base = list(baseline_traces) if baseline_traces is not None else replay_cells(orchestrator, cells)
    inter = replay_cells(orchestrator, cells, spec)

    out = InterventionOutcome(
        mode=spec.mode, n_cells=len(cells),
        n_heads=len(spec.targets), n_entries=spec.n_entries(),
    )
    for (arm, pressure, s), tb, ti in zip(cells, base, inter):
        bias_b, ppl_b, acc_b = _cell_stats(tb)
        bias_i, ppl_i, acc_i = _cell_stats(ti)
        out.cells.append(CellOutcome(arm, pressure, s, bias_b, bias_i, ppl_b, ppl_i, acc_b, acc_i))

    d_bias = np.array([c.delta_bias for c in out.cells])
    out.ace = float(np.nanmean(d_bias))
    out.ace_ci = _bootstrap_ci(d_bias, cfg.n_bootstrap, cfg.alpha, seed)
    base_mean = float(np.nanmean([c.bias_base for c in out.cells]))
    out.relative_reduction = (
        float(-out.ace / abs(base_mean)) if abs(base_mean) > 1e-9 else float("nan")
    )
    out.p_vs_zero = _paired_bootstrap_p(d_bias, cfg.n_bootstrap, seed)

    ppl_b = np.nanmean([c.ppl_base for c in out.cells])
    ppl_i = np.nanmean([c.ppl_int for c in out.cells])
    out.delta_ppl = float((ppl_i - ppl_b) / ppl_b) if np.isfinite(ppl_b) and ppl_b > 0 else float("nan")
    acc_b = np.nanmean([c.acc_base for c in out.cells])
    acc_i = np.nanmean([c.acc_int for c in out.cells])
    out.delta_acc = float(acc_i - acc_b)

    # -- negative controls ------------------------------------------------
    for name, ctrl in control_specs.items():
        ctrl_traces = replay_cells(orchestrator, cells, ctrl)
        d_ctrl = np.array([
            _cell_stats(tc)[0] - c.bias_base for tc, c in zip(ctrl_traces, out.cells)
        ])
        ace_ctrl = float(np.nanmean(d_ctrl))
        p = _paired_bootstrap_p(d_bias - d_ctrl, cfg.n_bootstrap, seed + 7)
        if name == "random":
            out.ace_random, out.p_vs_random = ace_ctrl, p
        elif name == "sink":
            out.ace_sink, out.p_vs_sink = ace_ctrl, p

    # -- acceptance rule ---------------------------------------------------
    reasons: list[str] = []
    if not (np.isfinite(out.p_vs_zero) and out.p_vs_zero < cfg.alpha):
        reasons.append(f"effect not significant vs zero (p={out.p_vs_zero:.3f})")
    if control_specs.get("random") is not None and not (
        np.isfinite(out.p_vs_random) and out.p_vs_random < cfg.alpha
    ):
        reasons.append(f"not separable from random-head control (p={out.p_vs_random:.3f})")
    if control_specs.get("sink") is not None and not (
        np.isfinite(out.p_vs_sink) and out.p_vs_sink < cfg.alpha
    ):
        reasons.append(f"not separable from sink-only control (p={out.p_vs_sink:.3f})")
    ppl_ok = (not np.isfinite(out.delta_ppl)) or out.delta_ppl <= cfg.max_ppl_increase
    acc_ok = (not np.isfinite(out.delta_acc)) or out.delta_acc >= -cfg.max_acc_drop
    out.coherence_ok = bool(ppl_ok and acc_ok)
    if not ppl_ok:
        reasons.append(f"coherence budget violated: dPPL={out.delta_ppl:+.3f}")
    if not acc_ok:
        reasons.append(f"coherence budget violated: dAcc={out.delta_acc:+.3f}")
    out.reasons = reasons
    out.accepted = not reasons
    return out
