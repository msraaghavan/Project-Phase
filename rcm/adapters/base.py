"""The :class:`ModelAdapter` interface and the intervention specification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class InterventionSpec:
    """A KV-only edit applied during a seed-matched replay (Module 4).

    ``targets`` maps ``(layer, kv_head)`` to the absolute cache positions the
    edit applies to.  No parameter is ever written; the edit lives entirely in
    the attention computation over the cache.

    mode
        ``"ablate"``  -- set the pre-softmax logit of every target entry to
        ``-inf`` and renormalise.
        ``"perturb"`` -- project the probe-identified bias direction out of the
        target value vectors, ``v <- v - eps * (w_hat . v) w_hat``.
        ``"gate"``    -- rescale attention to the target set by ``g`` in [0, 1]
        before renormalising.
        ``"none"``    -- baseline replay.
    """

    mode: str = "none"
    targets: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    directions: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    epsilon: float = 1.0
    gate: float = 0.1

    def is_active(self) -> bool:
        return self.mode != "none" and bool(self.targets)

    def positions(self, layer: int, kv_head: int) -> list[int]:
        return self.targets.get((layer, kv_head), [])

    def n_entries(self) -> int:
        return sum(len(v) for v in self.targets.values())

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "n_heads": len(self.targets),
            "n_entries": self.n_entries(),
            "epsilon": self.epsilon,
            "gate": self.gate,
        }


@dataclass
class TurnOutput:
    """What one instrumented generation step returns."""

    token_ids: list[int]
    text: str
    logits: np.ndarray            # [n_new, vocab] logits at each generated step
    n_prompt_tokens: int


class ModelAdapter(ABC):
    """Uniform view of a frozen decoder-only transformer.

    The adapter never trains the model under study during an experiment: the
    weights are frozen and only the runtime KV state is observed and edited.
    """

    # -- geometry --------------------------------------------------------
    @property
    @abstractmethod
    def n_layers(self) -> int: ...

    @property
    @abstractmethod
    def n_heads(self) -> int:
        """Number of query heads."""

    @property
    @abstractmethod
    def n_kv_heads(self) -> int:
        """Number of key/value heads (equal to ``n_heads`` without GQA)."""

    @property
    @abstractmethod
    def head_dim(self) -> int: ...

    @property
    def gqa_group(self) -> int:
        """How many query heads share one KV head."""
        return self.n_heads // self.n_kv_heads

    def query_heads_of(self, kv_head: int) -> list[int]:
        """Query heads that read from ``kv_head`` (the GQA mapping)."""
        g = self.gqa_group
        return list(range(kv_head * g, (kv_head + 1) * g))

    # -- execution -------------------------------------------------------
    @abstractmethod
    def new_cache(self, **kwargs):
        """Return a fresh :class:`~rcm.module1.cache.TracingCache`."""

    @abstractmethod
    def forward(self, token_ids, cache=None, intervention=None, capture=True):
        """Run a forward pass over ``token_ids``, appending to ``cache``."""

    @abstractmethod
    def generate(self, prompt_ids, cache, max_new_tokens, intervention=None, **kw) -> TurnOutput:
        """Greedy, deterministic decoding with instrumentation."""

    @abstractmethod
    def sequence_logprob(self, token_ids) -> float:
        """Sum of token log-probabilities, used for perplexity."""

    @abstractmethod
    def next_token_logits(self, prompt_ids) -> np.ndarray:
        """Logits for the token following ``prompt_ids`` (no cache side effects)."""

    @abstractmethod
    def rope_reindex(self, keys: np.ndarray, old_pos, new_pos) -> np.ndarray:
        """Re-index RoPE-rotated keys from ``old_pos`` to ``new_pos``.

        Needed when a peer's cache segment is transplanted in the shared-KV
        arm: the rotary phase of a key encodes its absolute position, so a
        transplanted entry must be rotated by the position delta or it lands at
        the wrong place in the attention geometry.
        """

    # -- convenience -----------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "gqa_group": self.gqa_group,
        }
