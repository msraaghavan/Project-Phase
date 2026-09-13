"""Memory phase transitions.

Sweeping semantic pressure ``pi`` and fitting

    B(pi) = B0 + (B1 - B0) / (1 + exp(-kappa * (pi - pi_c)))
    chi(pi) = dB/dpi

gives the critical pressure ``pi_c`` and the susceptibility ``chi``.  A
transition is claimed only when **all** of the following hold -- the criteria are
pre-registered, not chosen after looking at the curve:

1. the logistic fit beats a linear null on **both** AIC and BIC;
2. the bootstrap confidence interval on ``kappa`` excludes the linear regime
   (``kappa <= KAPPA_LINEAR``, where the logistic is indistinguishable from a
   straight line over the unit interval);
3. an *independent* changepoint detector locates a break inside the confidence
   interval of ``pi_c``;
4. criteria 1-3 agree across at least ``MIN_SEEDS`` seeds.

The design also requires agreement across two models; in a single-model run that
component is reported as not satisfied rather than quietly dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

KAPPA_LINEAR = 2.0     # kappa at or below this is effectively linear over pi in [0, 1]
MIN_SEEDS = 3
MIN_MODELS = 2


def logistic_curve(pi, b0: float, b1: float, kappa: float, pi_c: float):
    """``B(pi)`` of the design document."""
    pi = np.asarray(pi, dtype=float)
    return b0 + (b1 - b0) / (1.0 + np.exp(-kappa * (pi - pi_c)))


def susceptibility(pi, b0: float, b1: float, kappa: float, pi_c: float):
    """``chi(pi) = dB/dpi``, maximal at the critical pressure."""
    pi = np.asarray(pi, dtype=float)
    e = np.exp(-kappa * (pi - pi_c))
    return (b1 - b0) * kappa * e / (1.0 + e) ** 2


def _ic(sse: float, n: int, k: int) -> tuple[float, float]:
    """Gaussian AIC and BIC from a residual sum of squares."""
    sse = max(float(sse), 1e-15)
    ll_term = n * np.log(sse / n)
    return ll_term + 2 * k, ll_term + k * np.log(n)


def changepoint_1d(y: np.ndarray) -> tuple[int, float]:
    """Exact single-changepoint search with an L2 (piecewise-constant) cost.

    An exhaustive scan is optimal for one changepoint and needs no extra
    dependency, so the detector stays independent of the logistic fit it is
    meant to corroborate.  Returns ``(index, cost)`` where ``index`` is the
    first point of the second segment.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return 0, float("inf")
    best_idx, best_cost = 1, float("inf")
    for k in range(1, n):
        left, right = y[:k], y[k:]
        cost = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
        if cost < best_cost:
            best_idx, best_cost = k, cost
    return best_idx, float(best_cost)


@dataclass
class PhaseResult:
    b0: float = float("nan")
    b1: float = float("nan")
    kappa: float = float("nan")
    pi_c: float = float("nan")
    kappa_ci: tuple[float, float] = (float("nan"), float("nan"))
    pi_c_ci: tuple[float, float] = (float("nan"), float("nan"))
    aic_logistic: float = float("nan")
    bic_logistic: float = float("nan")
    aic_linear: float = float("nan")
    bic_linear: float = float("nan")
    r2_logistic: float = float("nan")
    changepoint_pi: float = float("nan")
    max_susceptibility: float = float("nan")
    criteria: dict[str, bool] = field(default_factory=dict)
    n_seeds_supporting: int = 0
    n_models: int = 1
    transition_detected: bool = False
    pressures: list[float] = field(default_factory=list)
    bias_mean: list[float] = field(default_factory=list)
    bias_sem: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "b0": self.b0, "b1": self.b1, "kappa": self.kappa, "piC": self.pi_c,
            "kappaCiLow": self.kappa_ci[0], "kappaCiHigh": self.kappa_ci[1],
            "piCCiLow": self.pi_c_ci[0], "piCCiHigh": self.pi_c_ci[1],
            "aicLogistic": self.aic_logistic, "aicLinear": self.aic_linear,
            "bicLogistic": self.bic_logistic, "bicLinear": self.bic_linear,
            "r2Logistic": self.r2_logistic,
            "changepointPi": self.changepoint_pi,
            "maxSusceptibility": self.max_susceptibility,
            "criteria": self.criteria,
            "nSeedsSupporting": self.n_seeds_supporting,
            "nModels": self.n_models,
            "transitionDetected": self.transition_detected,
        }


def _fit_logistic(pi: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares logistic fit; returns ``(params, sse)``."""
    p0 = [float(b.min()), float(b.max()), 8.0, 0.5]
    bounds = ([-2.0, -2.0, 0.0, -0.5], [2.0, 2.0, 200.0, 1.5])
    if np.ptp(b) < 1e-9:          # perfectly flat curve: the logistic degenerates to a constant
        return np.array([b[0], b[0], 0.0, 0.5]), 0.0
    popt, _ = curve_fit(logistic_curve, pi, b, p0=p0, bounds=bounds, max_nfev=2000)
    sse = float(((b - logistic_curve(pi, *popt)) ** 2).sum())
    return popt, sse


def fit_phase_transition(
    pressures,
    bias_by_pressure: dict[float, list[float]],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    n_models: int = 1,
) -> PhaseResult:
    """Fit ``B(pi)``, test it against a linear null and apply all four criteria.

    ``bias_by_pressure`` maps each pressure level to the per-seed bias values
    observed at it.
    """
    pressures = np.asarray(sorted(pressures), dtype=float)
    means = np.array([np.mean(bias_by_pressure.get(float(p), [np.nan])) for p in pressures])
    sems = np.array([
        (np.std(bias_by_pressure.get(float(p), [0.0]), ddof=1)
         / max(1.0, np.sqrt(len(bias_by_pressure.get(float(p), [0.0])))))
        if len(bias_by_pressure.get(float(p), [])) > 1 else 0.0
        for p in pressures
    ])
    res = PhaseResult(
        pressures=pressures.tolist(), bias_mean=means.tolist(), bias_sem=sems.tolist(),
        n_models=n_models,
    )
    ok = np.isfinite(means)
    if ok.sum() < 4:
        res.criteria = {"enough_points": False}
        return res
    pi, b = pressures[ok], means[ok]
    n = len(pi)

    # -- logistic fit vs linear null ------------------------------------
    try:
        popt, sse_log = _fit_logistic(pi, b)
    except Exception:
        res.criteria = {"logistic_converged": False}
        return res
    res.b0, res.b1, res.kappa, res.pi_c = (float(x) for x in popt)
    slope, intercept = np.polyfit(pi, b, 1)
    sse_lin = float(((b - (slope * pi + intercept)) ** 2).sum())
    res.aic_logistic, res.bic_logistic = _ic(sse_log, n, 4)
    res.aic_linear, res.bic_linear = _ic(sse_lin, n, 2)
    sst = float(((b - b.mean()) ** 2).sum())
    res.r2_logistic = 1.0 - sse_log / sst if sst > 0 else float("nan")
    res.max_susceptibility = float(np.max(np.abs(susceptibility(pi, *popt))))

    # -- bootstrap over seeds -------------------------------------------
    rng = np.random.default_rng(seed)
    kappas, pics = [], []
    for _ in range(n_boot):
        resampled = []
        for p in pi:
            vals = bias_by_pressure.get(float(p), [])
            resampled.append(np.mean(rng.choice(vals, len(vals), replace=True)) if len(vals) else np.nan)
        yb = np.asarray(resampled)
        if not np.all(np.isfinite(yb)):
            continue
        try:
            pb, _ = _fit_logistic(pi, yb)
            kappas.append(pb[2])
            pics.append(pb[3])
        except Exception:
            continue
    if kappas:
        res.kappa_ci = tuple(np.percentile(kappas, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
        res.pi_c_ci = tuple(np.percentile(pics, [100 * alpha / 2, 100 * (1 - alpha / 2)]))

    # -- independent changepoint detector --------------------------------
    cp_idx, _ = changepoint_1d(b)
    cp_idx = min(max(cp_idx, 1), len(pi) - 1)
    # the break lies between two grid points; report the midpoint and treat the
    # half grid step as the detector's resolution when comparing with the CI
    res.changepoint_pi = float(0.5 * (pi[cp_idx - 1] + pi[cp_idx]))
    half_step = float(0.5 * np.median(np.diff(pi))) if len(pi) > 1 else 0.0

    # -- per-seed agreement ----------------------------------------------
    n_seeds = max((len(v) for v in bias_by_pressure.values()), default=0)
    supporting = 0
    for s in range(n_seeds):
        ys = []
        for p in pi:
            vals = bias_by_pressure.get(float(p), [])
            ys.append(vals[s] if s < len(vals) else np.nan)
        ys = np.asarray(ys, dtype=float)
        if not np.all(np.isfinite(ys)):
            continue
        try:
            ps, sse_s = _fit_logistic(pi, ys)
        except Exception:
            continue
        sl, ic = np.polyfit(pi, ys, 1)
        sse_ls = float(((ys - (sl * pi + ic)) ** 2).sum())
        aic_s, bic_s = _ic(sse_s, len(pi), 4)
        aic_ls, bic_ls = _ic(sse_ls, len(pi), 2)
        if aic_s < aic_ls and bic_s < bic_ls and ps[2] > KAPPA_LINEAR:
            supporting += 1
    res.n_seeds_supporting = supporting

    # -- the pre-registered criteria --------------------------------------
    lo_k = res.kappa_ci[0]
    lo_pi, hi_pi = res.pi_c_ci
    res.criteria = {
        "logistic_beats_linear_aic": bool(res.aic_logistic < res.aic_linear),
        "logistic_beats_linear_bic": bool(res.bic_logistic < res.bic_linear),
        "kappa_ci_excludes_linear": bool(np.isfinite(lo_k) and lo_k > KAPPA_LINEAR),
        "changepoint_inside_pic_ci": bool(
            np.isfinite(lo_pi) and np.isfinite(hi_pi)
            and (lo_pi - half_step) <= res.changepoint_pi <= (hi_pi + half_step)
        ),
        "seed_agreement": bool(supporting >= MIN_SEEDS),
        "model_replication": bool(n_models >= MIN_MODELS),
    }
    core = [k for k in res.criteria if k != "model_replication"]
    res.transition_detected = all(res.criteria[k] for k in core)
    return res
