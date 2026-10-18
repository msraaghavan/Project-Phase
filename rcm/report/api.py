"""Web app and JSON service over completed results directories.

The page at ``/`` is the user-facing product: an interactive demo of the
two-agent experiment and the main findings for every model that has been run.
The page carries its own data (written by :mod:`rcm.report.site_data`), so it
also works when ``index.html`` is opened straight from disk.  The JSON
endpoints follow the API design of the document:

    GET  /conversations
    GET  /conversations/{id}/attractors
    GET  /heads/reservoir-ranking
    GET  /contagion?arm=
    POST /replay            (queued intervention run; returns the accepted spec)
    GET  /api/demo/options  (the five demo conversations and who they are about)
    POST /api/demo          (runs one conversation live, optionally with a KV edit)
    GET  /api/summary       (headline numbers of this results directory)
    GET  /api/runs          (headline numbers of every results*/ directory found)

Start it with ``python -m rcm.cli serve`` (or ``uvicorn rcm.report.api:app``)
after ``python run_all.py``.  The service binds to localhost only.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rcm.store.schema import TraceStore

PROJECT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"


def _resolve(path: str | Path) -> Path:
    """Relative paths work from the project folder as well as the current one."""
    path = Path(path)
    return path if path.is_absolute() or path.exists() else PROJECT / path


RESULTS_DIR = _resolve(os.environ.get("RCM_RESULTS", "results"))

FIGURES = [
    ("fig1_bias_vs_pressure.png", "Bias vs semantic pressure by memory arm"),
    ("fig2_attention_graph.png", "Attention-flow graph"),
    ("fig3_attractor_metrics.png", "Attractor metrics along the cache"),
    ("fig4_reservoir_map.png", "Bias reservoir map"),
    ("fig5_contagion.png", "Memory vs text contagion"),
    ("fig6_interventions.png", "Interventions vs negative controls"),
]

app = FastAPI(title="RCM trace service", version="1.1.0",
              description="Runtime-memory traces, analysis results and a live demo.")


def _results(root: Path | None = None) -> dict:
    path = (root or RESULTS_DIR) / "results.json"
    if not path.exists():
        raise HTTPException(404, f"{path} not found; run the pipeline first")
    return json.loads(path.read_text(encoding="utf-8"))


def _store() -> TraceStore:
    return TraceStore(RESULTS_DIR / "traces")


def _readable(text: str, backend: str) -> str:
    """Testbed symbols shown with the readable names of :data:`rcm.probes.SURFACE`."""
    from rcm.probes import render

    if backend != "tiny":
        return text.strip()
    out = " answer ".join(render(part) for part in f" {text} ".split(" answer ")).strip()
    for mark in ".,?":
        out = out.replace(" " + mark, mark)
    return out


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# pages and files
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    path = RESULTS_DIR / "dashboard.html"
    if not path.exists():
        raise HTTPException(404, "dashboard not built; run the pipeline first")
    return FileResponse(path)


@app.get("/figures/{name}", include_in_schema=False)
def figure(name: str):
    if not re.fullmatch(r"[\w.-]+\.png", name):
        raise HTTPException(400, "bad figure name")
    path = RESULTS_DIR / "figures" / name
    if not path.exists():
        raise HTTPException(404, "no such figure")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# headline numbers
# ---------------------------------------------------------------------------
_EDIT_KEYS = ("ace", "relativeReduction", "aceRandom", "aceSink", "pVsZero", "pVsRandom",
              "pVsSink", "deltaAcc", "deltaPpl", "accepted", "reasons")


def _summary(res: dict, root: Path) -> dict:
    m2, m3, m4 = res["module2"], res["module3"], res["module4"]
    best = max(m4, key=lambda m: m4[m]["relativeReduction"])
    phase = m3["phase"]["shared-kv"]
    crit = phase.get("criteria", {})
    backend = res["model"]["backend"]
    check = res["runtime_memory_check"]
    ex = res["example_conversation"]
    return {
        "dir": root.name,
        "model": res["model"]["name"],
        "backend": backend,
        "layers": res["model"]["n_layers"],
        "kvHeads": res["model"]["n_kv_heads"],
        "headDim": res["model"]["head_dim"],
        "nConversations": res["n_conversations"],
        "seconds": res["total_seconds"],
        "contagion": m3["contagion"],
        "curves": m3["curves"],
        "phase": {"transitionDetected": phase["transitionDetected"],
                  "passed": sum(bool(v) for v in crit.values()), "total": len(crit)},
        # chance of picking the unfair word over the fair one, P = sigmoid(logit gap)
        "memoryCheck": {"fair": _sigmoid(check["mean_gap_anti"]),
                        "none": _sigmoid(check["mean_gap_none"]),
                        "unfair": _sigmoid(check["mean_gap_stereo"]),
                        "items": check["n_items"]},
        "attractors": {arm: v["mean_novel_attractors"] for arm, v in m2["attractors"]["per_arm"].items()},
        "reservoirs": [{k: r[k] for k in ("layer", "head", "auc", "isReservoir")} for r in m2["reservoirs"]],
        "nReservoirs": len(m2["reservoir_heads"]),
        "edits": {mode: {k: o[k] for k in _EDIT_KEYS} for mode, o in m4.items()},
        "best": {"mode": best} | {k: m4[best][k] for k in _EDIT_KEYS},
        "example": [{"round": t["turnIdx"] // 2 + 1, "role": t["role"],
                     "text": _readable(t["text"], backend), "bias": t["biasScore"]}
                    for t in ex["turns"]],
        "figures": [{"file": f, "caption": c} for f, c in FIGURES
                    if (root / "figures" / f).exists()],
    }


@app.get("/api/summary")
def summary():
    return _summary(_results(), RESULTS_DIR)


@app.get("/api/runs")
def runs():
    roots = {RESULTS_DIR.resolve()} | {p.parent.resolve() for base in {Path.cwd(), RESULTS_DIR.parent}
                                       for p in base.glob("results*/results.json")}
    out = [_summary(_results(r), r) for r in sorted(roots) if (r / "results.json").exists()]
    return sorted(out, key=lambda s: (s["backend"] != "tiny", s["model"]))


# ---------------------------------------------------------------------------
# live demo: one conversation on the frozen model, optionally with a KV edit
# ---------------------------------------------------------------------------
_engine: dict = {}
_lock = threading.Lock()
DEMO_SEEDS = range(5)


def _config():
    from rcm.config import ExperimentConfig

    manifest = json.loads((RESULTS_DIR / "manifest.json").read_text(encoding="utf-8"))
    cfg = ExperimentConfig.from_dict(manifest["config"])
    cfg.out_dir = str(RESULTS_DIR)
    return cfg


def _load_engine() -> dict:
    """Load the model this results directory was produced with (once)."""
    if not _engine:
        from rcm.module1.orchestrator import ConversationOrchestrator
        from rcm.testbed import prepare_adapter

        cfg = _config()
        adapter = prepare_adapter(cfg, verbose=False)
        res = _results()
        heads = [tuple(h) for h in next(iter(res["module4"].values()))["target_heads"]]
        positions = res["module4_attempts"][-1]["positions"]
        _engine.update(cfg=cfg, adapter=adapter, orch=ConversationOrchestrator(adapter, cfg),
                       heads=heads, positions=positions, baseline={})
    return _engine


def _topic(seed: int) -> dict:
    from rcm.probes import SURFACE, ProbeBank

    p = ProbeBank(seed=_config().master_seed).pick(seed)
    return {"seed": seed, "group": SURFACE[p.group_a], "otherGroup": SURFACE[p.group_b],
            "unfair": SURFACE[p.stereo_trait], "fair": SURFACE[p.anti_trait]}


@app.get("/api/demo/options")
def demo_options():
    return {"model": _results()["model"]["name"], "statements": 16,
            "topics": [_topic(s) for s in DEMO_SEEDS]}


class DemoRequest(BaseModel):
    arm: str = "shared-kv"
    pressure: float = 1.0
    seed: int = 0
    intervention: str = "none"      # none | ablate | gate


def _demo_run(eng: dict, arm: str, pressure: float, seed: int) -> dict:
    """One conversation as the page shows it: turns, bias per turn and the cache."""
    from rcm.module2.attractors import detect_attractors
    from rcm.module2.metrics import compute_metrics

    cfg, adapter, orch = eng["cfg"], eng["adapter"], eng["orch"]
    base = orch.run(arm, pressure, seed)
    metrics = compute_metrics(base.stats, cfg.metrics)
    sets = detect_attractors(metrics, cfg.metrics, base.convId)
    token_text = [adapter.token_text(p["tokenId"]) for p in base.provenance]
    backend = cfg.model.backend
    novel, sink = set(sets.novel), set(sets.sink)
    cache = [{
        "t": t, "tok": token_text[t].strip(), "readable": _readable(token_text[t], backend),
        "ch": p["channel"], "turn": p["turnIdx"],
        "mark": "novel" if t in novel else "sink" if t in sink else "",
    } for t, p in enumerate(base.provenance)]
    turns = [{"round": tr.turnIdx // 2 + 1, "role": tr.role, "text": tr.text.strip(),
              "readable": _readable(tr.text, backend), "bias": tr.biasScore} for tr in base.turns]
    return {
        "model": getattr(adapter, "name", cfg.model.name),
        "arm": arm, "pressure": pressure, "seed": seed,
        "topic": _topic(seed),
        "nStatements": len(base.context_plan),
        "nUnfair": int(sum(base.context_plan)),
        "turns": turns,
        "senderBias": base.sender_bias,
        "receiverBias": base.receiver_bias,
        "finalBias": base.final_bias,
        "cache": cache,
        "nNovel": len(novel),
        "nSink": len(sink),
        "seqLen": len(cache),
    }


def _edited_run(eng: dict, arm: str, pressure: float, seed: int, mode: str) -> dict:
    """The same conversation replayed with the reservoir heads edited."""
    from rcm.module4.interventions import build_intervention

    spec = build_intervention(eng["heads"], eng["positions"], mode, eng["cfg"].intervention)
    tr = eng["orch"].run(arm, pressure, seed, intervention=spec)
    return {"receiverBias": tr.receiver_bias, "finalBias": tr.final_bias}


def _baseline(eng: dict, seed: int) -> float:
    """The listener's bias when it hears nothing (independent of pressure)."""
    from rcm.config import ARM_CONTROL

    if seed not in eng["baseline"]:
        eng["baseline"][seed] = eng["orch"].run(ARM_CONTROL, 0.0, seed).final_bias
    return eng["baseline"][seed]


@app.post("/api/demo")
def demo(req: DemoRequest):
    from rcm.config import ARM_CONTROL

    if req.arm not in ("shared-kv", "text-only", "isolated"):
        raise HTTPException(400, "arm must be shared-kv, text-only or isolated")
    if not 0.0 <= req.pressure <= 1.0 or not 0 <= req.seed <= 99:
        raise HTTPException(400, "pressure must be in [0, 1] and seed in [0, 99]")
    if req.intervention not in ("none", "ablate", "gate"):
        raise HTTPException(400, "intervention must be none, ablate or gate")

    t0 = time.time()
    with _lock:                      # one model, one conversation at a time
        eng = _load_engine()
        out = _demo_run(eng, req.arm, req.pressure, req.seed)
        edited = (None if req.intervention == "none"
                  else _edited_run(eng, req.arm, req.pressure, req.seed, req.intervention))
        if req.arm == ARM_CONTROL:
            eng["baseline"].setdefault(req.seed, out["finalBias"])
        baseline = _baseline(eng, req.seed)
    out.update(intervention=req.intervention, seconds=time.time() - t0,
               baselineBias=baseline, edited=edited)
    return out


# ---------------------------------------------------------------------------
# trace service (design-document API)
# ---------------------------------------------------------------------------
@app.get("/conversations")
def conversations(limit: int = 50, cursor: str | None = None):
    df = _store().conversations().sort_values("convId")
    if cursor:
        df = df[df.convId > cursor]
    page = df.head(limit)
    return {"items": page.to_dict(orient="records"),
            "next_cursor": page.convId.iloc[-1] if len(page) == limit else None}


@app.get("/conversations/{conv_id}/attractors")
def attractors(conv_id: str):
    df = _store().kv_records(conv_id, columns=["position", "isAttractor", "reuseScore",
                                               "lifetime", "persistence", "centrality", "channel"])
    if df.empty:
        raise HTTPException(404, "unknown conversation")
    pos = df[df.isAttractor].drop_duplicates("position")
    return {"convId": conv_id, "attractors": pos.to_dict(orient="records")}


@app.get("/heads/reservoir-ranking")
def reservoir_ranking():
    return _results()["module2"]["reservoirs"]


@app.get("/contagion")
def contagion(arm: str | None = None):
    m3 = _results()["module3"]
    out = {"contagion": m3["contagion"], "by_pressure": m3["contagion_by_pressure"],
           "transfer_entropy": m3["transfer_entropy"]}
    if arm:
        out["curve"] = m3["curves"].get(arm)
        out["phase"] = m3["phase"].get(arm)
    return out


class ReplayRequest(BaseModel):
    mode: str = "ablate"
    arm: str = "shared-kv"
    pressure: float = 1.0
    seed: int = 0


@app.post("/replay")
def replay(req: ReplayRequest):
    m4 = _results()["module4"]
    if req.mode not in m4:
        raise HTTPException(400, f"unknown mode {req.mode}")
    return {"queued": True, "request": req.model_dump(), "last_outcome": m4[req.mode]}
