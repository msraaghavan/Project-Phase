"""Command-line entry point.

    python -m rcm.cli run   [--config configs/default.yaml] [--quick]
    python -m rcm.cli report [--results results]
    python -m rcm.cli site   [--results results]
    python -m rcm.cli serve  [--results results] [--port 8000]

``run`` executes the full pipeline; ``report`` regenerates figures and the
dashboard from an existing ``results.json``; ``site`` rebuilds the data
embedded in the web page (``rcm/report/static/index.html``); ``serve`` starts the web app
(demo, findings, JSON API) on http://127.0.0.1:8000.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rcm.config import ExperimentConfig


def _quick(cfg: ExperimentConfig) -> ExperimentConfig:
    """Reduced grid for smoke tests (same code path, fewer cells)."""
    cfg.run.seeds = (0, 1, 2)
    cfg.run.pressures = (0.0, 0.5, 1.0)
    cfg.run.train_steps = 200
    cfg.out_dir = str(Path(cfg.out_dir) / "quick")
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rcm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full experiment")
    r.add_argument("--config", default=None)
    r.add_argument("--quick", action="store_true")
    r.add_argument("--out", default=None)
    p = sub.add_parser("report", help="regenerate figures and dashboard")
    p.add_argument("--results", default="results")
    d = sub.add_parser("site", help="rebuild the data inside the web page")
    d.add_argument("--results", default="results")
    w = sub.add_parser("serve", help="start the web app on localhost")
    w.add_argument("--results", default="results")
    w.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    if args.cmd in ("serve", "site"):
        import os

        os.environ["RCM_RESULTS"] = args.results
        from rcm.report import site_data

        if args.cmd == "site" or not site_data.is_built():
            print("building the web page's data (one-off, a few minutes) ...")
            site_data.build()
        if args.cmd == "site":
            return 0

        import uvicorn

        print(f"open http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
        uvicorn.run("rcm.report.api:app", host="127.0.0.1", port=args.port, log_level="warning")
        return 0

    if args.cmd == "run":
        from rcm.pipeline import run_pipeline
        from rcm.report import build_dashboard, make_figures

        cfg = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
        if args.out:
            cfg.out_dir = args.out
        if args.quick:
            cfg = _quick(cfg)
        results = run_pipeline(cfg)
        figs = make_figures(results, Path(cfg.out_dir) / "figures")
        dash = build_dashboard(results, Path(cfg.out_dir) / "dashboard.html")
        print(f"[6] {len(figs)} figures -> {figs[0].parent}\n[6] dashboard -> {dash}")
        return 0

    from rcm.report import build_dashboard, make_figures

    res = json.loads((Path(args.results) / "results.json").read_text(encoding="utf-8"))
    make_figures(res, Path(args.results) / "figures")
    build_dashboard(res, Path(args.results) / "dashboard.html")
    print("report regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
