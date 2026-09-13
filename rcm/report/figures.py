"""Publication figures (Matplotlib) from ``results.json``."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rcm.module3.phase import logistic_curve  # noqa: E402
from rcm.probes import ITOS  # noqa: E402

ARM_COLORS = {"shared-kv": "#b3123f", "text-only": "#2c6fbb", "isolated": "#6c757d"}


def make_figures(results: dict | str | Path, out_dir: str | Path) -> list[Path]:
    if not isinstance(results, dict):
        results = json.loads(Path(results).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        _fig_bias_vs_pressure(results, out / "fig1_bias_vs_pressure.png"),
        _fig_attention_heatmap(results, out / "fig2_attention_graph.png"),
        _fig_attractor_metrics(results, out / "fig3_attractor_metrics.png"),
        _fig_reservoir_map(results, out / "fig4_reservoir_map.png"),
        _fig_contagion(results, out / "fig5_contagion.png"),
        _fig_interventions(results, out / "fig6_interventions.png"),
    ]
    return paths


def _fig_bias_vs_pressure(res, path):
    curves = res["module3"]["curves"]
    phase = res["module3"]["phase"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for arm, c in curves.items():
        ax.errorbar(c["pressures"], c["mean"], yerr=c["sem"], marker="o", capsize=3,
                    color=ARM_COLORS.get(arm, None), label=arm)
        ph = phase[arm]
        if np.isfinite(ph.get("kappa", np.nan)):
            xs = np.linspace(0, 1, 200)
            ax.plot(xs, logistic_curve(xs, ph["b0"], ph["b1"], ph["kappa"], ph["piC"]),
                    ls="--", lw=1, color=ARM_COLORS.get(arm, None), alpha=0.7)
    ps = phase["shared-kv"]
    if np.isfinite(ps.get("piC", np.nan)):
        ax.axvline(ps["piC"], color=ARM_COLORS["shared-kv"], ls=":", lw=1)
        ax.text(ps["piC"], ax.get_ylim()[1], f" $\\pi_c$={ps['piC']:.2f}", va="top", fontsize=8)
    ax.set_xlabel("semantic pressure $\\pi$")
    ax.set_ylabel("receiver bias index $B$")
    ax.set_title("Bias vs pressure by memory arm (dashed: logistic fit)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _fig_attention_heatmap(res, path):
    ex = res["example_conversation"]
    a = np.asarray(ex["alpha_bar"])
    T = a.shape[0]
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(np.log10(a + 1e-6), cmap="magma", aspect="auto")
    for t in ex["novel"]:
        ax.axvline(t, color="cyan", lw=0.5, alpha=0.6)
    for t in ex["sink"]:
        ax.axvline(t, color="lime", lw=0.5, alpha=0.6)
    ax.set_xlabel("key position $t$")
    ax.set_ylabel("query position $q$")
    ax.set_title(f"Head-averaged attention $\\bar\\alpha_{{q,t}}$  (T={T})\n"
                 "cyan: novel attractors, green: sink class")
    fig.colorbar(im, ax=ax, label="$\\log_{10}$ attention")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _fig_attractor_metrics(res, path):
    ex = res["example_conversation"]
    m = ex["metrics"]
    T = len(m["reuseScore"])
    x = np.arange(T)
    fig, axes = plt.subplots(4, 1, figsize=(7.5, 7.0), sharex=True)
    for ax, (name, label) in zip(axes, [("reuseZ", "$z_R(t)$ reuse z-score"),
                                        ("lifetime", "$L(t)$ lifetime"),
                                        ("persistence", "$P(t)$ persistence"),
                                        ("centrality", "$c(t)$ PageRank")]):
        ax.plot(x, m[name], lw=0.9, color="#333")
        for t in ex["novel"]:
            ax.axvline(t, color="#b3123f", lw=0.6, alpha=0.5)
        for t in ex["sink"]:
            ax.axvline(t, color="green", lw=0.6, alpha=0.5)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
    ch = ex["channels"]
    for start in range(1, T):
        if ch[start] != ch[start - 1]:
            axes[-1].axvline(start, color="#2c6fbb", lw=0.4, ls=":")
    axes[-1].set_xlabel("cache position $t$  (red: novel attractors, green: sink class)")
    fig.suptitle("Attractor metrics along one shared-KV conversation", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _fig_reservoir_map(res, path):
    rk = res["module2"]["reservoirs"]
    L = max(r["layer"] for r in rk) + 1
    H = max(r["head"] for r in rk) + 1
    auc = np.zeros((L, H))
    score = np.zeros((L, H))
    for r in rk:
        auc[r["layer"], r["head"]] = r["auc"]
        score[r["layer"], r["head"]] = r["score"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    for ax, mat, title in zip(axes, [auc, score], ["probe AUC", "reservoir score $S(\\ell,h)$"]):
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_xlabel("KV head")
        ax.set_ylabel("layer")
        ax.set_title(title)
        ax.set_xticks(range(H))
        ax.set_yticks(range(L))
        for r in rk:
            if r["isReservoir"]:
                ax.add_patch(plt.Rectangle((r["head"] - 0.5, r["layer"] - 0.5), 1, 1,
                                           fill=False, ec="red", lw=2))
        fig.colorbar(im, ax=ax)
    fig.suptitle("Bias Reservoir map (red boxes: heads surviving BH-FDR)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _fig_contagion(res, path):
    byp = res["module3"]["contagion_by_pressure"]
    ps = sorted(float(p) for p in byp)
    mc = [byp[str(p) if str(p) in byp else p]["memoryContagion"] for p in ps]
    tc = [byp[str(p) if str(p) in byp else p]["textContagion"] for p in ps]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    w = 0.04
    ax.bar(np.array(ps) - w / 2, mc, width=w, color=ARM_COLORS["shared-kv"], label="Memory contagion $B^{(S)}-B^{(T)}$")
    ax.bar(np.array(ps) + w / 2, tc, width=w, color=ARM_COLORS["text-only"], label="Text contagion $B^{(T)}-B^{(C)}$")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("semantic pressure $\\pi$")
    ax.set_ylabel("bias transfer to receiver")
    c = res["module3"]["contagion"]
    ax.set_title(f"Contagion decomposition  (overall MC={c['memoryContagion']:+.3f}, "
                 f"TC={c['textContagion']:+.3f})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _fig_interventions(res, path):
    m4 = res["module4"]
    modes = list(m4)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    x = np.arange(len(modes))
    w = 0.25
    axes[0].bar(x - w, [m4[m]["ace"] for m in modes], w, color=ARM_COLORS["shared-kv"], label="reservoir target")
    axes[0].bar(x, [m4[m]["aceRandom"] for m in modes], w, color="#999", label="random-head control")
    axes[0].bar(x + w, [m4[m]["aceSink"] for m in modes], w, color="#5a5", label="sink-only control")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modes)
    axes[0].set_ylabel("ACE = $E[B|do(S)] - E[B]$")
    axes[0].set_title("Causal effect vs negative controls", fontsize=10)
    axes[0].legend(fontsize=7)
    axes[1].bar(x - w / 2, [100 * m4[m]["deltaPpl"] for m in modes], w, color="#c77", label="$\\Delta$PPL (%)")
    axes[1].bar(x + w / 2, [100 * m4[m]["deltaAcc"] for m in modes], w, color="#77c", label="$\\Delta$Acc (points)")
    axes[1].axhline(5, color="#c77", ls="--", lw=0.8)
    axes[1].axhline(-2, color="#77c", ls="--", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{m}\n{'ACCEPT' if m4[m]['accepted'] else 'REJECT'}" for m in modes])
    axes[1].set_title("Coherence budget (dashed: limits)", fontsize=10)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def token_label(tid: int) -> str:
    return ITOS.get(int(tid), "?")
