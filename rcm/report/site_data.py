"""Data embedded in the web page (``rcm/report/static/index.html``).

Every conversation the demo can show -- each topic, each number of unfair
statements in the sweep grid, each memory arm, with and without a KV edit -- is
run here once with the same code as ``POST /api/demo``.  Together with the
headline numbers of every ``results*/`` directory it is written into the
page's ``<script id="rcm-data">`` block, so ``index.html`` is self-contained: it
works from ``rcm serve``, opened straight from disk, or sent as a single file.

    python -m rcm.cli site
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from rcm.report import api

PAGE = api.STATIC / "index.html"
_BLOCK = re.compile(r'(<script id="rcm-data">)(.*?)(</script>)', re.S)
EDIT_MODES = ("ablate", "gate")
_CH = {"system": "s", "probe": "p", "agent": "a", "peer": "e"}


def _r(xs, nd: int = 6):
    return [round(float(x), nd) for x in xs]


def _compact(d: dict) -> dict:
    """One demo conversation with the cache stored column-wise."""
    cache = d["cache"]
    return {
        "nUnfair": d["nUnfair"],
        "turns": [[t["role"][0], t["readable"], round(t["bias"], 6)] for t in d["turns"]],
        "senderBias": _r(d["senderBias"]),
        "receiverBias": _r(d["receiverBias"]),
        "finalBias": round(d["finalBias"], 6),
        "tok": [c["readable"] for c in cache],
        "ch": "".join(_CH[c["ch"]] for c in cache),
        "turn": [c["turn"] for c in cache],
        "novel": [c["t"] for c in cache if c["mark"] == "novel"],
        "sink": [c["t"] for c in cache if c["mark"] == "sink"],
    }


def demo_key(seed: int, arm: str, n_unfair: int) -> str:
    return f"{seed}|{arm}|{n_unfair}"


def is_built(page: Path = PAGE) -> bool:
    m = _BLOCK.search(page.read_text(encoding="utf-8"))
    return bool(m and m.group(2).strip())


def load(page: Path = PAGE) -> dict:
    """The data currently embedded in the page."""
    body = _BLOCK.search(page.read_text(encoding="utf-8")).group(2)
    return json.loads(body[body.index("{"):body.rindex("}") + 1].replace("<\\/", "</"))


def write(data: dict, page: Path = PAGE) -> Path:
    text = page.read_text(encoding="utf-8")
    js = "window.RCM_DATA = " + json.dumps(data, separators=(",", ":")).replace("</", "<\\/") + ";"
    if not _BLOCK.search(text):
        raise ValueError(f'{page} has no <script id="rcm-data"> block')
    page.write_text(_BLOCK.sub(lambda m: m.group(1) + js + m.group(3), text, count=1), encoding="utf-8")
    return page


def build(page: Path = PAGE, verbose: bool = True) -> Path:
    t0 = time.time()
    eng = api._load_engine()
    cfg = eng["cfg"]
    n_items = 16
    demos, baseline = {}, {}
    cells = [(s, a, p) for s in api.DEMO_SEEDS for a in cfg.run.arms for p in cfg.run.pressures]
    for i, (seed, arm, pressure) in enumerate(cells, 1):
        d = api._demo_run(eng, arm, pressure, seed)
        n_items = d["nStatements"]
        rec = _compact(d)
        rec["edits"] = {m: {k: (_r(v) if isinstance(v, list) else round(v, 6))
                            for k, v in api._edited_run(eng, arm, pressure, seed, m).items()}
                        for m in EDIT_MODES}
        demos[demo_key(seed, arm, round(pressure * n_items))] = rec
        if verbose and i % 15 == 0:
            print(f"  {i}/{len(cells)} conversations  ({time.time() - t0:.0f} s)", flush=True)
    for seed in api.DEMO_SEEDS:
        baseline[str(seed)] = round(api._baseline(eng, seed), 6)
    data = {
        "model": getattr(eng["adapter"], "name", cfg.model.name),
        "statements": n_items,
        "levels": [round(p * n_items) for p in cfg.run.pressures],
        "topics": [api._topic(s) for s in api.DEMO_SEEDS],
        "hotspots": [list(h) for h in eng["heads"]],
        "baseline": baseline,
        "demos": demos,
        "runs": api.runs(),
    }
    write(data, page)
    if verbose:
        print(f"wrote {len(demos)} conversations into {page} "
              f"({page.stat().st_size / 1e6:.2f} MB, {time.time() - t0:.0f} s)")
    return page


if __name__ == "__main__":
    build()
