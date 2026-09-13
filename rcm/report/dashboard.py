"""Self-contained interactive dashboard (Plotly, single HTML file).

Attention-flow heatmap, attractor-lifetime timeline, reservoir map, the
bias-versus-pressure curves and the intervention summary, all embedded in one
file that opens in any browser with no server.  The FastAPI service in
:mod:`rcm.report.api` exposes the same data as JSON for the Streamlit front end.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from rcm.probes import ITOS


def build_dashboard(results: dict | str | Path, path: str | Path) -> Path:
    if not isinstance(results, dict):
        results = json.loads(Path(results).read_text(encoding="utf-8"))
    ex = results["example_conversation"]
    a = np.asarray(ex["alpha_bar"])
    T = a.shape[0]
    toks = [ITOS.get(int(t), "?") for t in ex["tokens"]]
    hover = [f"t={i} '{toks[i]}' ({ex['channels'][i]}, turn {ex['turnIdx'][i]})" for i in range(T)]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Attention flow (log10 head-averaged attention)",
            "Bias vs semantic pressure by memory arm",
            "Attractor metrics along the cache (red = novel attractors)",
            "Bias Reservoir map: probe AUC per (layer, KV head)",
            "Per-turn bias, sender vs receiver",
            "Interventions: ACE vs negative controls",
        ),
        vertical_spacing=0.10, horizontal_spacing=0.09,
    )
    fig.add_trace(go.Heatmap(z=np.log10(a + 1e-6), colorscale="Magma", showscale=False,
                             hovertext=[[f"q={q} -> {hover[t]}: {a[q, t]:.3f}" for t in range(T)] for q in range(T)],
                             hoverinfo="text"), row=1, col=1)

    curves = results["module3"]["curves"]
    for arm, c in curves.items():
        fig.add_trace(go.Scatter(x=c["pressures"], y=c["mean"], mode="lines+markers", name=arm,
                                 error_y=dict(type="data", array=c["sem"], visible=True)), row=1, col=2)

    m = ex["metrics"]
    x = list(range(T))
    for name in ("reuseZ", "persistence", "centrality"):
        y = np.asarray(m[name], dtype=float)
        if name == "centrality":
            y = y / (y.max() or 1.0)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, hovertext=hover), row=2, col=1)
    for t in ex["novel"]:
        fig.add_vline(x=t, line=dict(color="red", width=1), opacity=0.4, row=2, col=1)

    rk = results["module2"]["reservoirs"]
    L = max(r["layer"] for r in rk) + 1
    H = max(r["head"] for r in rk) + 1
    auc = np.zeros((L, H))
    txt = [["" for _ in range(H)] for _ in range(L)]
    for r in rk:
        auc[r["layer"], r["head"]] = r["auc"]
        txt[r["layer"]][r["head"]] = f"AUC {r['auc']:.2f}<br>S {r['score']:.2f}<br>q {r['qValue']:.2f}" + (
            "<br><b>RESERVOIR</b>" if r["isReservoir"] else "")
    fig.add_trace(go.Heatmap(z=auc, colorscale="Viridis", text=txt, hoverinfo="text", showscale=False,
                             zmin=0.3, zmax=1.0), row=2, col=2)

    fig.add_trace(go.Scatter(y=ex["sender_bias"], mode="lines+markers", name="sender B"), row=3, col=1)
    fig.add_trace(go.Scatter(y=ex["receiver_bias"], mode="lines+markers", name="receiver B"), row=3, col=1)

    m4 = results["module4"]
    modes = list(m4)
    fig.add_trace(go.Bar(x=modes, y=[m4[k]["ace"] for k in modes], name="reservoir target"), row=3, col=2)
    fig.add_trace(go.Bar(x=modes, y=[m4[k]["aceRandom"] for k in modes], name="random-head control"), row=3, col=2)
    fig.add_trace(go.Bar(x=modes, y=[m4[k]["aceSink"] for k in modes], name="sink-only control"), row=3, col=2)

    c = results["module3"]["contagion"]
    ph = results["module3"]["phase"]["shared-kv"]
    fig.update_layout(
        height=1150, width=1250, barmode="group",
        title=(f"Runtime Computational Memory - {results['model']['name']}  |  "
               f"{results['n_conversations']} conversations  |  "
               f"MC={c['memoryContagion']:+.3f}  TC={c['textContagion']:+.3f}  |  "
               f"pi_c={ph.get('piC', float('nan')):.2f}  transition={ph.get('transitionDetected')}"),
        template="plotly_white", legend=dict(orientation="h", y=-0.04),
    )
    fig.update_xaxes(title_text="key position", row=1, col=1)
    fig.update_yaxes(title_text="query position", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="pressure pi", row=1, col=2)
    fig.update_yaxes(title_text="receiver bias B", row=1, col=2)
    fig.update_xaxes(title_text="cache position", row=2, col=1)
    fig.update_xaxes(title_text="KV head", row=2, col=2)
    fig.update_yaxes(title_text="layer", row=2, col=2)
    fig.update_xaxes(title_text="turn", row=3, col=1)
    fig.update_yaxes(title_text="ACE", row=3, col=2)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, path, include_plotlyjs=True, full_html=True)
    return path
