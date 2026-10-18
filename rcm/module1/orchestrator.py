"""Multi-agent conversation harness and the three memory arms.

This is the online loop of Figure 2.  Two agents share a frozen model but keep
*separate* KV caches:

``a1`` -- the **sender**.  Its turns carry the semantic pressure: a fraction
         ``pi`` of the context items injected into it are stereotype-consistent,
         the rest counter-stereotypical.
``a0`` -- the **receiver**.  It is never given stereotype content directly.  The
         only thing that reaches it is whatever the memory arm lets through:

         =============  ===============================================
         ``shared-kv``  the sender's KV segment is grafted into a0's
                        cache, with RoPE positions re-indexed
         ``text-only``  only the sender's generated *text* is appended
         ``isolated``   nothing crosses; the control arm
         =============  ===============================================

Measuring the receiver's bias across the three arms is what lets Module 3 define
memory contagion *by construction* as the component of transfer that text alone
cannot explain.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rcm.config import (
    ARM_CONTROL,
    ARM_SHARED,
    ARM_TEXT,
    CH_AGENT,
    CH_PROBE,
    CH_SYSTEM,
    ExperimentConfig,
    set_all_seeds,
)
from rcm.module1.scoring import (
    bias_logit_gap,
    coherence_score,
    hallucinated,
    hallucination_persistence,
)
from rcm.probes import BiasProbe, FactProbe, ProbeBank

SYSTEM_PROMPT = "they said one and one met there ."
RECEIVER_PROMPT = "they said"


@dataclass
class Agent:
    """One conversational agent: an identity plus its own runtime memory."""

    agent_id: str
    role: str                      # "sender" | "receiver"
    cache: Any = None
    tap: Any = None
    last_turn_span: tuple[int, int] = (0, 0)
    last_text_ids: list[int] = field(default_factory=list)
    bias_series: list[float] = field(default_factory=list)


@dataclass
class TurnRecord:
    """One row of the ``turn`` relation of the trace schema."""

    convId: str
    turnIdx: int
    agentId: str
    role: str
    tokenStart: int
    tokenEnd: int
    biasScore: float
    logitGap: float
    ppl: float
    hallucinated: bool
    text: str


@dataclass
class ConversationTrace:
    """Everything one instrumented conversation produces."""

    convId: str
    model: str
    arm: str
    probeSuite: str
    probeId: str
    pressure: float
    seed: int
    nTurns: int
    gitCommit: str
    turns: list[TurnRecord]
    provenance: list[dict[str, Any]]
    stats: dict[str, Any]                    # receiver tap, AttentionTap.finalize()
    sender_stats: dict[str, Any]             # sender tap, for attractor-set overlap
    values: np.ndarray                       # [L, Hkv, T, head_dim] receiver values
    key_norms: np.ndarray
    value_norms: np.ndarray
    receiver_bias: list[float]
    sender_bias: list[float]
    final_bias: float
    bias_label: int                          # sign of final_bias, the probe target
    hallucination_flags: list[bool]
    hallucination_persistence: float
    mean_coherence: float
    raw_kv_sampled: bool
    context_plan: list[bool]

    def token_ids(self) -> list[int]:
        return [p["tokenId"] for p in self.provenance]

    def as_conversation_row(self) -> dict[str, Any]:
        return {
            "convId": self.convId,
            "model": self.model,
            "arm": self.arm,
            "probeSuite": self.probeSuite,
            "probeId": self.probeId,
            "pressure": self.pressure,
            "seed": self.seed,
            "nTurns": self.nTurns,
            "gitCommit": self.gitCommit,
            "finalBias": self.final_bias,
            "biasLabel": self.bias_label,
            "hallucinationPersistence": self.hallucination_persistence,
            "meanCoherence": self.mean_coherence,
            "seqLen": int(self.stats["seq_len"]),
            "rawKvSampled": self.raw_kv_sampled,
        }


class ConversationOrchestrator:
    """Runs one instrumented conversation end to end."""

    def __init__(self, adapter, cfg: ExperimentConfig, probe_bank: ProbeBank | None = None) -> None:
        self.adapter = adapter
        self.cfg = cfg
        self.probes = probe_bank or ProbeBank(seed=cfg.master_seed)

    # -- helpers ---------------------------------------------------------
    def _new_agent(self, agent_id: str, role: str, conv_id: str) -> Agent:
        cache = self.adapter.new_cache(f"{conv_id}:{agent_id}")
        tap = self.adapter.new_tap(self.cfg.metrics.top_k)
        return Agent(agent_id=agent_id, role=role, cache=cache, tap=tap)

    def _write(self, agent: Agent, token_ids, turn_idx: int, channel: str) -> None:
        """Write tokens into an agent's cache without generating."""
        agent.cache.set_write_context(turn_idx, agent.agent_id, channel)
        self.adapter.forward(token_ids, agent.cache, None, capture=True, tap=agent.tap)

    def _generate(self, agent: Agent, prompt_ids, turn_idx: int, intervention=None):
        agent.cache.set_write_context(turn_idx, agent.agent_id, CH_AGENT)
        return self.adapter.generate(
            prompt_ids, agent.cache, self.cfg.run.max_new_tokens,
            intervention=intervention, tap=agent.tap,
        )

    @staticmethod
    def context_plan(pressure: float, n_items: int, rng: random.Random) -> list[bool]:
        """Deterministic allocation of stereotype-consistent context items.

        ``pressure`` is realised exactly (up to rounding) rather than in
        expectation, so the bias-versus-pressure curve of Module 3 is not
        blurred by sampling noise in pi itself.
        """
        n_true = int(round(pressure * n_items))
        plan = [True] * n_true + [False] * (n_items - n_true)
        rng.shuffle(plan)
        return plan

    # -- the online loop -------------------------------------------------
    def run(
        self,
        arm: str,
        pressure: float,
        seed: int,
        probe: BiasProbe | None = None,
        fact: FactProbe | None = None,
        intervention=None,
        conv_id: str | None = None,
        n_ctx_per_turn: int = 4,
    ) -> ConversationTrace:
        """Run one conversation and return its trace.

        Fully determined by ``(arm, pressure, seed, probe)``; re-running with the
        same arguments reproduces it token for token, which is what makes the
        Module 4 replay seed-matched.
        """
        if arm not in (ARM_SHARED, ARM_TEXT, ARM_CONTROL):
            raise ValueError(f"unknown memory arm {arm!r}")
        set_all_seeds(seed)
        rng = random.Random(seed)
        probe = probe or self.probes.pick(seed)
        fact = fact or self.probes.pick_fact(seed)
        conv_id = conv_id or f"{arm}|p{pressure:.3f}|s{seed}|{probe.probe_id}"

        receiver = self._new_agent("a0", "receiver", conv_id)
        sender = self._new_agent("a1", "sender", conv_id)

        # -- seeding: identical in every arm, so it cancels in MC and TC ---
        intro = f"{probe.group_a} and {probe.group_b} met there ."
        for ag in (receiver, sender):
            self._write(ag, self.adapter.bos_ids() + self.adapter.encode(SYSTEM_PROMPT), 0, CH_SYSTEM)
            self._write(ag, self.adapter.encode(intro), 0, CH_PROBE)

        n_rounds = max(1, self.cfg.run.n_turns // 2)
        plan = self.context_plan(pressure, n_rounds * n_ctx_per_turn, rng)

        turns: list[TurnRecord] = []
        halluc_flags: list[bool] = []
        coherences: list[float] = []
        turn_idx = 0

        for r in range(n_rounds):
            # ---------------- sender turn ------------------------------
            start = sender.cache.get_seq_length(0)
            items = plan[r * n_ctx_per_turn:(r + 1) * n_ctx_per_turn]
            for stereotypical in items:
                self._write(
                    sender, self.adapter.encode(probe.context_sentence(stereotypical, rng)), turn_idx,
                    CH_PROBE,
                )
            out_s = self._generate(sender, self.adapter.encode(RECEIVER_PROMPT), turn_idx)
            end = sender.cache.get_seq_length(0)
            sender.last_turn_span = (start, end)
            sender.last_text_ids = out_s.token_ids
            gap = bias_logit_gap(self.adapter, sender.cache, probe)
            b_s = float(np.tanh(gap / 2.0))
            sender.bias_series.append(b_s)
            turns.append(TurnRecord(
                convId=conv_id, turnIdx=turn_idx, agentId=sender.agent_id, role=sender.role,
                tokenStart=start, tokenEnd=end, biasScore=b_s,
                logitGap=gap,
                ppl=coherence_score(out_s.logits, out_s.token_ids),
                hallucinated=hallucinated(self.adapter, sender.cache, fact),
                text=out_s.text,
            ))
            turn_idx += 1

            # ---------------- cross-agent channel ----------------------
            self._cross(receiver, sender, arm, turn_idx)

            # ---------------- receiver turn ----------------------------
            start = receiver.cache.get_seq_length(0)
            out_r = self._generate(receiver, self.adapter.encode(RECEIVER_PROMPT), turn_idx, intervention)
            end = receiver.cache.get_seq_length(0)
            receiver.last_turn_span = (start, end)
            receiver.last_text_ids = out_r.token_ids
            gap = bias_logit_gap(self.adapter, receiver.cache, probe)
            b_r = float(np.tanh(gap / 2.0))
            receiver.bias_series.append(b_r)
            flag = hallucinated(self.adapter, receiver.cache, fact)
            halluc_flags.append(flag)
            ppl = coherence_score(out_r.logits, out_r.token_ids)
            coherences.append(ppl)
            turns.append(TurnRecord(
                convId=conv_id, turnIdx=turn_idx, agentId=receiver.agent_id, role=receiver.role,
                tokenStart=start, tokenEnd=end, biasScore=b_r,
                logitGap=gap,
                ppl=ppl, hallucinated=flag, text=out_r.text,
            ))
            turn_idx += 1

        stats = receiver.tap.finalize(receiver.cache.get_seq_length(0))
        sender_stats = sender.tap.finalize(sender.cache.get_seq_length(0))
        final_bias = receiver.bias_series[-1] if receiver.bias_series else 0.0
        sampled = raw_kv_selected(self.cfg, arm, pressure, seed)

        return ConversationTrace(
            convId=conv_id,
            model=getattr(self.adapter, "name", self.cfg.model.name),
            arm=arm,
            probeSuite=probe.suite,
            probeId=probe.probe_id,
            pressure=pressure,
            seed=seed,
            nTurns=len(turns),
            gitCommit=self.cfg.manifest()["git_commit"],
            turns=turns,
            provenance=receiver.cache.provenance_frame(),
            stats=stats,
            sender_stats=sender_stats,
            values=receiver.cache.value_array(),
            key_norms=receiver.cache.key_norms(),
            value_norms=receiver.cache.value_norms(),
            receiver_bias=list(receiver.bias_series),
            sender_bias=list(sender.bias_series),
            final_bias=final_bias,
            bias_label=int(final_bias > 0),
            hallucination_flags=halluc_flags,
            hallucination_persistence=hallucination_persistence(halluc_flags),
            mean_coherence=float(np.nanmean(coherences)) if coherences else float("nan"),
            raw_kv_sampled=sampled,
            context_plan=plan,
        )

    # -- the three memory arms -------------------------------------------
    def _cross(self, receiver: Agent, sender: Agent, arm: str, turn_idx: int) -> int:
        """Move information from sender to receiver according to the arm."""
        if arm == ARM_CONTROL:
            return 0
        start, end = sender.last_turn_span
        if end <= start:
            return 0
        if arm == ARM_TEXT:
            # only the generated text crosses
            ids = sender.last_text_ids
            if not ids:
                return 0
            self._write(receiver, ids, turn_idx, CH_PROBE)
            return len(ids)
        if arm == ARM_SHARED:
            # the sender's whole KV segment for that turn crosses, RoPE-re-indexed
            receiver.cache.set_write_context(turn_idx, receiver.agent_id, "peer")
            n = receiver.cache.graft(sender.cache, list(range(start, end)), self.adapter)
            # the grafted entries were never queried, so the tap has no rows for
            # them; they become visible to every later query, which is the point.
            return n
        raise ValueError(arm)


def raw_kv_selected(cfg: ExperimentConfig, arm: str, pressure: float, seed: int) -> bool:
    """Whether this sweep cell's raw K/V tensors are dumped (stratified sample).

    Within every arm, ``round(rate x cells)`` cells are taken at evenly spaced
    points of the (pressure, seed) grid -- always including the first and the
    last -- so each arm is represented at low and high pressure.  Off-grid runs
    (replays, demos) are never dumped.
    """
    grid = [(float(p), int(s)) for p in cfg.run.pressures for s in cfg.run.seeds]
    if arm not in cfg.run.arms or (float(pressure), int(seed)) not in grid:
        return False
    n_take = max(1, round(cfg.run.raw_kv_sample_rate * len(grid)))
    picks = {int(round(i)) for i in np.linspace(0, len(grid) - 1, n_take)}
    return grid.index((float(pressure), int(seed))) in picks


def describe_trace(trace: ConversationTrace) -> str:
    """One-line human summary, used by the CLI progress output."""
    hp = trace.hallucination_persistence
    hp_txt = f"{hp:.2f}" if hp == hp else "n/a"
    return (
        f"{trace.arm:<9} pi={trace.pressure:<5.3f} seed={trace.seed} "
        f"T={trace.stats['seq_len']:<4d} B={trace.final_bias:+.3f} "
        f"hallucPersist={hp_txt} last='{trace.turns[-1].text}'"
    )
