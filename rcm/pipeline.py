"""End-to-end experiment: instrument -> measure -> attribute -> intervene.

This is Figure 2 as code.  The online loop (Module 1) runs every
``(arm, pressure, seed)`` cell of the grid and logs its trace; the offline loop
(Modules 2-4) turns those traces into attractor sets, a reservoir ranking, the
contagion / phase-transition analysis and finally a causal, seed-matched
intervention that is accepted or rejected against negative controls and the
coherence budget.  Everything is written to ``results/`` as JSON, figures and a
dashboard.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rcm.config import ARM_CONTROL, ARM_SHARED, ARM_TEXT, ExperimentConfig
from rcm.module1.logger import ProvenanceLogger
from rcm.module1.orchestrator import ConversationOrchestrator, describe_trace
from rcm.module2.attractors import (
    attractor_token_profile,
    detect_attractors,
    sink_baseline_comparison,
)
from rcm.module2.metrics import compute_metrics
from rcm.module2.reservoirs import localize_reservoirs, pool_attractor_values
from rcm.module3.contagion import contagion_decomposition, transfer_entropy
from rcm.module3.phase import fit_phase_transition
from rcm.module4.evaluate import evaluate_intervention
from rcm.module4.interventions import (
    build_intervention,
    random_head_control,
    sink_only_control,
    union_positions,
)
from rcm.testbed import prepare_adapter, runtime_memory_check


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _nan_to_none(o):
    """Strict JSON has no NaN; write missing values as null."""
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_to_none(v) for v in o]
    return o


def _save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_nan_to_none(json.loads(json.dumps(obj, default=_json_default))),
                      indent=2, allow_nan=False)
    path.write_text(text, encoding="utf-8")


def sender_attractor_overlap(trace, sender_sets, receiver_sets) -> float:
    """Jaccard overlap between the two agents' attractor sets.

    Only meaningful in the shared-KV arm, where the receiver's grafted entries
    are the *same* KV entries as the sender's: a receiver attractor that was
    grafted is mapped back to its source position through the provenance record
    and compared against the sender's attractor set.
    """
    src = {
        p["position"]: p["sourcePosition"]
        for p in trace.provenance if p.get("sourcePosition") is not None
    }
    a = {src[t] for t in receiver_sets.novel if t in src}
    b = set(sender_sets.novel)
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def reservoir_read_positions(traces, keys, target_heads, adapter, sink_positions: int,
                             limit: int = 32) -> list[int]:
    """Refined target set: the non-sink positions the reservoir heads read most.

    Figure 2 loops back to target selection when an intervention is not
    separable from its controls.  The first candidate set is the attractor set;
    this second candidate uses the per-head attention mass recorded by the tap
    to pick the KV entries the reservoir heads themselves attend to.
    """
    acc: dict[int, float] = defaultdict(float)
    for k in keys:
        mass = np.asarray(traces[k].stats["mass"])          # [L, H_q, T]
        for layer, kv_head in target_heads:
            for h in adapter.query_heads_of(kv_head):
                m = mass[layer, h]
                for t in np.argsort(-m)[:limit]:
                    if t > sink_positions:
                        acc[int(t)] += float(m[t])
    return sorted(t for t, _ in sorted(acc.items(), key=lambda kv: -kv[1])[:limit])


def run_pipeline(cfg: ExperimentConfig, verbose: bool = True) -> dict[str, Any]:
    t_start = time.time()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a, **k: print(*a, **k, flush=True)) if verbose else (lambda *a, **k: None)

    # ------------------------------------------------------------------
    # 0. manifest and model
    # ------------------------------------------------------------------
    cfg.write_manifest()
    log(f"[0] manifest written  (config sha256 {cfg.config_hash()[:12]})")
    adapter = prepare_adapter(cfg, verbose=verbose)
    memory_check = runtime_memory_check(adapter)
    log(f"[0] runtime-memory check: stereo ctx gap {memory_check['mean_gap_stereo']:+.2f}, "
        f"none {memory_check['mean_gap_none']:+.2f}, anti {memory_check['mean_gap_anti']:+.2f} "
        f"-> ordered={memory_check['ordered']}")

    orch = ConversationOrchestrator(adapter, cfg)
    logger = ProvenanceLogger(cfg)

    # ------------------------------------------------------------------
    # 1. Module 1 -- the instrumented sweep
    # ------------------------------------------------------------------
    traces: dict[tuple[str, float, int], Any] = {}
    metrics: dict[tuple[str, float, int], Any] = {}
    receiver_sets: dict[tuple[str, float, int], Any] = {}
    sender_sets: dict[tuple[str, float, int], Any] = {}
    n_cells = cfg.n_cells
    log(f"[1] Module 1: running {n_cells} conversations "
        f"({len(cfg.run.arms)} arms x {len(cfg.run.pressures)} pressures x {len(cfg.run.seeds)} seeds)")
    t1 = time.time()
    i = 0
    for arm in cfg.run.arms:
        for pressure in cfg.run.pressures:
            for seed in cfg.run.seeds:
                key = (arm, float(pressure), int(seed))
                tr = orch.run(arm, float(pressure), int(seed))
                m = compute_metrics(tr.stats, cfg.metrics)
                sets = detect_attractors(m, cfg.metrics, tr.convId)
                ms = compute_metrics(tr.sender_stats, cfg.metrics)
                s_sets = detect_attractors(ms, cfg.metrics, tr.convId + ":a1")
                logger.log_conversation(tr, m, sets)
                traces[key], metrics[key], receiver_sets[key], sender_sets[key] = tr, m, sets, s_sets
                i += 1
                if verbose and (i % 15 == 0 or i == n_cells):
                    log(f"    {i:4d}/{n_cells}  {describe_trace(tr)}")
    sweep_seconds = time.time() - t1
    log(f"[1] sweep done in {sweep_seconds:.1f}s "
        f"({sweep_seconds / n_cells:.2f}s per conversation)")

    # ------------------------------------------------------------------
    # 2. Module 2 -- attractors and reservoirs
    # ------------------------------------------------------------------
    log("[2] Module 2: attractors and bias reservoirs")
    attractor_summary = {"per_arm": {}, "token_profile": {}, "sink_vs_nonsink": {}}
    profile_total: dict[str, int] = defaultdict(int)
    sink_cmp = defaultdict(list)
    for arm in cfg.run.arms:
        keys = [k for k in traces if k[0] == arm]
        n_novel = [receiver_sets[k].n_novel for k in keys]
        n_sink = [len(receiver_sets[k].sink) for k in keys]
        attractor_summary["per_arm"][arm] = {
            "mean_novel_attractors": float(np.mean(n_novel)),
            "mean_sink_attractors": float(np.mean(n_sink)),
            "mean_seq_len": float(np.mean([receiver_sets[k].seq_len for k in keys])),
            "frac_conversations_with_novel": float(np.mean([n > 0 for n in n_novel])),
        }
    for k, tr in traces.items():
        for tok, c in attractor_token_profile(receiver_sets[k], tr.token_ids(),
                                                adapter.token_text).items():
            profile_total[tok] += c
        for name, v in sink_baseline_comparison(metrics[k], cfg.metrics).items():
            sink_cmp[name].append(v)
    attractor_summary["token_profile"] = dict(
        sorted(profile_total.items(), key=lambda kv: -kv[1])
    )
    attractor_summary["sink_vs_nonsink"] = {k: float(np.nanmean(v)) for k, v in sink_cmp.items()}
    # channel profile: where do novel attractors come from?
    ch_counts: dict[str, int] = defaultdict(int)
    for k, tr in traces.items():
        for t in receiver_sets[k].novel:
            ch_counts[tr.provenance[t]["channel"]] += 1
    attractor_summary["channel_profile"] = dict(ch_counts)
    log(f"    novel attractors / conversation: "
        + ", ".join(f"{a}={v['mean_novel_attractors']:.1f}" for a, v in attractor_summary['per_arm'].items()))
    log(f"    top attractor tokens: {list(attractor_summary['token_profile'].items())[:6]}")

    # -- reservoirs: probes over attractor-pooled value vectors ------------
    keys_all = list(traces)
    pooled = np.stack([
        pool_attractor_values(traces[k].values, receiver_sets[k].all_positions) for k in keys_all
    ])
    final_bias = np.array([traces[k].final_bias for k in keys_all])
    labels = (final_bias > 0).astype(int)
    minority = min(labels.mean(), 1 - labels.mean())
    label_rule = "sign(B)"
    if minority < 0.2:           # too unbalanced for a probe: fall back to a median split
        labels = (final_bias > np.median(final_bias)).astype(int)
        label_rule = "B > median(B)"

    # the ablation effect is *measured*: replay a fixed cell set with the head ablated
    abl_cells = [(ARM_SHARED, float(max(cfg.run.pressures)), int(s)) for s in cfg.run.seeds[:3]]
    # the target set is built from the arm that will be intervened on
    shared_keys = [k for k in keys_all if k[0] == ARM_SHARED]
    novel_positions = union_positions([receiver_sets[k] for k in shared_keys], limit=32)
    positions_fallback = False
    if not novel_positions:      # no novel attractor cleared all three thresholds:
        positions_fallback = True  # fall back to the most-reused non-sink positions
        score = defaultdict(float)
        for k in shared_keys:
            z = metrics[k].reuse_z
            for t in np.argsort(-z)[:8]:
                if t > cfg.metrics.sink_positions:
                    score[int(t)] += float(z[t])
        novel_positions = sorted(t for t, _ in sorted(score.items(), key=lambda kv: -kv[1])[:16])
    abl_baseline = float(np.mean([traces[c].final_bias for c in abl_cells]))

    def ablation_fn(layer: int, kv_head: int) -> float:
        spec = build_intervention([(layer, kv_head)], novel_positions, "ablate", cfg.intervention)
        return float(np.mean([
            orch.run(arm, p, s, intervention=spec).final_bias for arm, p, s in abl_cells
        ]))

    t2 = time.time()
    reservoirs = localize_reservoirs(
        pooled, labels, cfg.metrics, ablation_fn=ablation_fn,
        baseline_bias=abl_baseline, n_perm=40, seed=cfg.master_seed,
    )
    ranking = reservoirs.ranking()
    log(f"    reservoir ranking computed in {time.time() - t2:.1f}s "
        f"(labels: {label_rule}, n={len(labels)}, balance={labels.mean():.2f})")
    for r in ranking[:4]:
        log(f"      L{r['layer']}H{r['head']}: AUC={r['auc']:.3f} q={r['qValue']:.3f} "
            f"abl={r['ablationEffect']:.3f} S={r['score']:.3f} reservoir={r['isReservoir']}")

    # ------------------------------------------------------------------
    # 3. Module 3 -- contagion and phase transitions
    # ------------------------------------------------------------------
    log("[3] Module 3: contagion and phase transitions")
    bias_by_arm = {arm: [traces[k].final_bias for k in traces if k[0] == arm] for arm in cfg.run.arms}
    contagion_all = contagion_decomposition(bias_by_arm, seed=cfg.master_seed)
    contagion_by_pressure = {}
    for p in cfg.run.pressures:
        by_arm = {arm: [traces[k].final_bias for k in traces if k[0] == arm and k[1] == float(p)]
                  for arm in cfg.run.arms}
        contagion_by_pressure[float(p)] = contagion_decomposition(by_arm, n_boot=1000,
                                                                  seed=cfg.master_seed).as_dict()
    te = {}
    overlap = {}
    for arm in cfg.run.arms:
        keys = [k for k in traces if k[0] == arm]
        te[arm] = transfer_entropy([traces[k].sender_bias for k in keys],
                                   [traces[k].receiver_bias for k in keys])
        js = [sender_attractor_overlap(traces[k], sender_sets[k], receiver_sets[k]) for k in keys]
        js = [j for j in js if j == j]
        overlap[arm] = float(np.mean(js)) if js else 0.0
    phase = {}
    for arm in cfg.run.arms:
        by_p = {float(p): [traces[(arm, float(p), int(s))].final_bias for s in cfg.run.seeds]
                for p in cfg.run.pressures}
        phase[arm] = fit_phase_transition(cfg.run.pressures, by_p, n_boot=400,
                                          seed=cfg.master_seed, n_models=1)
    ps = phase[ARM_SHARED]
    log(f"    MC={contagion_all.memory_contagion:+.3f} "
        f"(95% CI {contagion_all.mc_ci[0]:+.3f}..{contagion_all.mc_ci[1]:+.3f}, p={contagion_all.mc_p_value:.3g})  "
        f"TC={contagion_all.text_contagion:+.3f}")
    log(f"    TE sender->receiver: " + ", ".join(f"{a}={v:.3f}" for a, v in te.items()))
    log(f"    shared-KV phase fit: pi_c={ps.pi_c:.3f} kappa={ps.kappa:.2f} "
        f"AIC log/lin={ps.aic_logistic:.1f}/{ps.aic_linear:.1f} transition={ps.transition_detected}")

    # ------------------------------------------------------------------
    # 4. Module 4 -- causal intervention
    # ------------------------------------------------------------------
    log("[4] Module 4: causal KV-only interventions")
    target_heads = [(r["layer"], r["head"]) for r in ranking if r["isReservoir"]]
    fallback = not target_heads
    if fallback:      # nothing survived FDR: intervene on the top-m by score, flagged as such
        target_heads = [(r["layer"], r["head"]) for r in ranking[: cfg.metrics.n_reservoirs]]
    all_heads = [(l, h) for l in range(adapter.n_layers) for h in range(adapter.n_kv_heads)]
    sink_positions = list(range(cfg.metrics.sink_positions + 1))
    directions = reservoirs.directions()

    eval_pressures = sorted(cfg.run.pressures)[-3:]
    eval_cells = [(ARM_SHARED, float(p), int(s)) for p in eval_pressures for s in cfg.run.seeds]
    baseline_traces = [traces[c] for c in eval_cells]

    candidates = [("novel-attractors", novel_positions)]
    read_positions = reservoir_read_positions(traces, shared_keys, target_heads, adapter,
                                              cfg.metrics.sink_positions)
    if read_positions and set(read_positions) != set(novel_positions):
        candidates.append(("reservoir-read", read_positions))

    attempts = []
    interventions = {}
    t4 = time.time()
    for set_name, positions in candidates:
        log(f"    target set '{set_name}': {len(target_heads)} heads x {len(positions)} positions")
        outcomes = {}
        for mode in cfg.intervention.modes:
            spec = build_intervention(target_heads, positions, mode, cfg.intervention, directions)
            controls = {
                "random": random_head_control(target_heads, all_heads, positions, mode,
                                              cfg.intervention, directions, seed=cfg.master_seed),
                "sink": sink_only_control(target_heads, sink_positions, mode,
                                          cfg.intervention, directions),
            }
            outcome = evaluate_intervention(orch, eval_cells, spec, controls, cfg.intervention,
                                            baseline_traces=baseline_traces, seed=cfg.master_seed)
            outcome.target_summary = {"heads": target_heads, "positions": positions,
                                      "target_set": set_name, "fallback_heads": fallback}
            outcome.epsilon = cfg.intervention.epsilon
            outcomes[mode] = outcome
            logger.log_intervention(f"{set_name}-{mode}-{cfg.config_hash()[:8]}", outcome)
            log(f"      {mode:<8} ACE={outcome.ace:+.3f} ({outcome.relative_reduction * 100:+.1f}% rel) "
                f"random={outcome.ace_random:+.3f} sink={outcome.ace_sink:+.3f} "
                f"dPPL={outcome.delta_ppl:+.3f} dAcc={outcome.delta_acc:+.3f} "
                f"p0={outcome.p_vs_zero:.3f} -> {'ACCEPT' if outcome.accepted else 'REJECT'}"
                + (f"  [{'; '.join(outcome.reasons)}]" if outcome.reasons else ""))
        attempts.append({"target_set": set_name, "n_positions": len(positions),
                         "positions": positions,
                         "modes": {m: o.as_dict() for m, o in outcomes.items()},
                         "any_accepted": any(o.accepted for o in outcomes.values())})
        interventions = outcomes
        if any(o.accepted for o in outcomes.values()):
            break          # Figure 2: exit the refinement loop on the first accepted edit
    log(f"    interventions evaluated in {time.time() - t4:.1f}s over {len(attempts)} target set(s)")

    # ------------------------------------------------------------------
    # 5. persist
    # ------------------------------------------------------------------
    store_manifest = logger.close()
    curves = {
        arm: {
            "pressures": [float(p) for p in cfg.run.pressures],
            "mean": [float(np.mean([traces[(arm, float(p), int(s))].final_bias for s in cfg.run.seeds]))
                     for p in cfg.run.pressures],
            "sem": [float(np.std([traces[(arm, float(p), int(s))].final_bias for s in cfg.run.seeds], ddof=1)
                          / np.sqrt(len(cfg.run.seeds))) if len(cfg.run.seeds) > 1 else 0.0
                    for p in cfg.run.pressures],
        }
        for arm in cfg.run.arms
    }
    example_key = (ARM_SHARED, float(max(cfg.run.pressures)), int(cfg.run.seeds[0]))
    ex = traces[example_key]
    example = {
        "convId": ex.convId,
        "tokens": [tr for tr in (ex.provenance[t]["tokenId"] for t in range(ex.stats["seq_len"]))],
        "token_text": [adapter.token_text(ex.provenance[t]["tokenId"])
                       for t in range(ex.stats["seq_len"])],
        "channels": [ex.provenance[t]["channel"] for t in range(ex.stats["seq_len"])],
        "turnIdx": [ex.provenance[t]["turnIdx"] for t in range(ex.stats["seq_len"])],
        "metrics": {k: v.tolist() for k, v in metrics[example_key].as_arrays().items()},
        "attractors": receiver_sets[example_key].all_positions,
        "sink": receiver_sets[example_key].sink,
        "novel": receiver_sets[example_key].novel,
        "alpha_bar": np.asarray(ex.stats["alpha_bar"]).tolist(),
        "head_mass": np.asarray(ex.stats["mass"]).sum(axis=2).tolist(),
        "receiver_bias": ex.receiver_bias,
        "sender_bias": ex.sender_bias,
        "turns": [t.__dict__ for t in ex.turns],
    }
    results = {
        "config_sha256": cfg.config_hash(),
        "model": adapter.describe() | {"name": getattr(adapter, "name", cfg.model.name),
                                      "backend": cfg.model.backend},
        "runtime_memory_check": {k: v for k, v in memory_check.items() if k != "per_probe"},
        "n_conversations": n_cells,
        "sweep_seconds": sweep_seconds,
        "seconds_per_conversation": sweep_seconds / n_cells,
        "module2": {
            "attractors": attractor_summary,
            "reservoirs": ranking,
            "label_rule": label_rule,
            "reservoir_heads": [(r["layer"], r["head"]) for r in ranking if r["isReservoir"]],
            "target_positions": novel_positions,
            "target_positions_fallback": positions_fallback,
        },
        "module3": {
            "contagion": contagion_all.as_dict(),
            "contagion_by_pressure": contagion_by_pressure,
            "transfer_entropy": te,
            "attractor_overlap": overlap,
            "phase": {arm: phase[arm].as_dict() for arm in cfg.run.arms},
            "curves": curves,
        },
        "module4": {mode: o.as_dict() | {"target_heads": target_heads,
                                          "fallback_heads": fallback,
                                          "target_set": attempts[-1]["target_set"],
                                          "n_positions": attempts[-1]["n_positions"]}
                    for mode, o in interventions.items()},
        "module4_attempts": attempts,
        "eval_cells": len(eval_cells),
        "store": {"n_files": len(store_manifest.get("files", {})),
                  "root": str(logger.root)},
        "example_conversation": example,
        "total_seconds": time.time() - t_start,
    }
    _save(out_dir / "results.json", results)
    log(f"[5] results written to {out_dir / 'results.json'}  "
        f"(total {results['total_seconds']:.0f}s)")
    return results
