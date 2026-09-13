"""Module 1 -- Runtime memory instrumentation and provenance logging.

Realises Objective 1: record per-token KV-cache state with full provenance
(layer, head, absolute position, turn, agent identity, source channel) during
multi-agent conversations, at inference cost and with no parameter updates.
"""

from rcm.module1.cache import KVProvenance, ProvenanceMixin, TracingCache
from rcm.module1.hooks import AttentionTap
from rcm.module1.logger import ProvenanceLogger
from rcm.module1.orchestrator import Agent, ConversationOrchestrator, ConversationTrace
from rcm.module1.scoring import bias_index, coherence_score, hallucinated

__all__ = [
    "KVProvenance",
    "ProvenanceMixin",
    "TracingCache",
    "AttentionTap",
    "ProvenanceLogger",
    "Agent",
    "ConversationOrchestrator",
    "ConversationTrace",
    "bias_index",
    "hallucinated",
    "coherence_score",
]
