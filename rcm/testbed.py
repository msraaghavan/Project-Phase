"""The validation testbed: a reference transformer with a known bias structure.

The design document validates every metric "against synthetic caches with
analytically known attractor structure" before it touches an 8B model.  This
module is that testbed end to end: it trains the reference transformer once on
the controlled corpus, caches the checkpoint, and runs a diagnostic that
verifies the model actually carries bias *in runtime memory* -- i.e. that its
prediction for "the <group> person is ..." moves with the polarity of the
sentences already in its context.  If that diagnostic failed, none of the later
modules would have anything to measure, so it is reported in the results.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from rcm.adapters import build_adapter
from rcm.config import ExperimentConfig
from rcm.module1.scoring import option_scores
from rcm.probes import ProbeBank, build_corpus

# Corpus parameters (pre-registered; part of the manifest through the config hash)
STEREOTYPE_RATE = 0.65     # corpus-level prior that a document is stereotype-consistent
CONSISTENCY = 1.0          # within-document polarity is deterministic
DOC_LEN = 96
N_DOCS = 4000
REPETITION_FRAC = 0.25


def prepare_adapter(cfg: ExperimentConfig, force: bool = False, verbose: bool = True):
    """Return the frozen testbed adapter, training it on first use."""
    adapter = build_adapter(cfg)
    if cfg.model.backend != "tiny":
        return adapter
    ckpt = Path(cfg.out_dir) / "testbed_model.pt"
    if ckpt.exists() and not force:
        adapter.load(ckpt)
        if verbose:
            print(f"  loaded cached testbed model from {ckpt}")
        return adapter
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus(N_DOCS, STEREOTYPE_RATE, seed=cfg.master_seed,
                          doc_len=DOC_LEN, consistency=CONSISTENCY,
                          repetition_frac=REPETITION_FRAC)
    if verbose:
        print(f"  training testbed model: {corpus.shape[0]} docs x {corpus.shape[1]} tokens, "
              f"{cfg.run.train_steps} steps")
    t0 = time.time()
    losses = adapter.train_on_corpus(
        corpus, steps=cfg.run.train_steps, batch=cfg.run.train_batch,
        seed=cfg.master_seed, log_every=100 if verbose else 0,
    )
    adapter.save(ckpt)
    (ckpt.parent / "testbed_training.json").write_text(json.dumps({
        "steps": cfg.run.train_steps, "batch": cfg.run.train_batch,
        "final_loss": float(np.mean(losses[-50:])), "seconds": time.time() - t0,
        "stereotype_rate": STEREOTYPE_RATE, "consistency": CONSISTENCY,
        "n_docs": N_DOCS, "doc_len": DOC_LEN,
    }, indent=2))
    return adapter


def runtime_memory_check(adapter, probe_bank: ProbeBank | None = None,
                         n_items: int = 6, seed: int = 0) -> dict:
    """Does the model's bias move with what is in its context?

    For every probe, the scoring prompt is preceded by ``n_items`` context
    sentences that are all stereotype-consistent, all counter-stereotypical, or
    absent.  The logit gap (stereo minus anti trait) must be ordered
    ``anti < none < stereo`` for runtime memory to be carrying bias at all.
    """
    bank = probe_bank or ProbeBank()
    rng = random.Random(seed)
    rows = []
    for probe in bank.bias_probes:
        def gap(items):
            ctx = adapter.bos_ids()
            for st in items:
                ctx += adapter.encode(probe.context_sentence(st, rng))
            ctx += adapter.encode(probe.scoring_prompt())
            sc = option_scores(adapter, None, ctx, [probe.stereo_trait, probe.anti_trait])
            return float(sc[0] - sc[1])
        rows.append({
            "probe": probe.probe_id,
            "gap_none": gap([]),
            "gap_stereo": gap([True] * n_items),
            "gap_anti": gap([False] * n_items),
        })
    g_none = float(np.mean([r["gap_none"] for r in rows]))
    g_st = float(np.mean([r["gap_stereo"] for r in rows]))
    g_an = float(np.mean([r["gap_anti"] for r in rows]))
    return {
        "n_items": n_items,
        "mean_gap_none": g_none,
        "mean_gap_stereo": g_st,
        "mean_gap_anti": g_an,
        "in_context_effect": g_st - g_an,
        "ordered": bool(g_an < g_none < g_st),
        "per_probe": rows,
    }
