"""Read-only FastAPI service over a completed results directory.

Endpoints follow the API design of the document:

    GET  /conversations
    GET  /conversations/{id}/attractors
    GET  /heads/reservoir-ranking
    GET  /contagion?arm=
    POST /replay            (queued intervention run; returns the accepted spec)

Run with ``uvicorn rcm.report.api:app --host 127.0.0.1 --port 8000`` after
``python run_all.py``.  The service binds to localhost only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rcm.store.schema import TraceStore

RESULTS_DIR = Path(os.environ.get("RCM_RESULTS", "results"))

app = FastAPI(title="RCM trace service", version="1.0.0",
              description="Read-only access to runtime-memory traces and analysis results.")


def _results() -> dict:
    path = RESULTS_DIR / "results.json"
    if not path.exists():
        raise HTTPException(404, f"{path} not found; run the pipeline first")
    return json.loads(path.read_text(encoding="utf-8"))


def _store() -> TraceStore:
    return TraceStore(RESULTS_DIR / "traces")


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
