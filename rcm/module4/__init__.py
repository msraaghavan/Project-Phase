"""Module 4 -- Causal KV-only intervention, evaluation and reporting.

Realises Objective 5: show that editing *only* the KV cache, during a
seed-matched replay and with no parameter updates, changes behaviour -- and that
the change survives negative controls and a coherence budget.
"""

from rcm.module4.evaluate import InterventionOutcome, evaluate_intervention
from rcm.module4.interventions import (
    build_intervention,
    random_head_control,
    sink_only_control,
)

__all__ = [
    "build_intervention",
    "random_head_control",
    "sink_only_control",
    "InterventionOutcome",
    "evaluate_intervention",
]
