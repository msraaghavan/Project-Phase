"""Module 3 -- Multi-agent contagion and phase-transition analysis.

Realises Objective 4: separate the memory-mediated component of bias transfer
from the text-mediated one, and test whether bias accumulation under rising
semantic pressure is a phase transition rather than a smooth drift.
"""

from rcm.module3.contagion import (
    ContagionResult,
    attractor_overlap,
    contagion_decomposition,
    transfer_entropy,
)
from rcm.module3.phase import (
    PhaseResult,
    changepoint_1d,
    fit_phase_transition,
    logistic_curve,
)

__all__ = [
    "ContagionResult",
    "contagion_decomposition",
    "transfer_entropy",
    "attractor_overlap",
    "PhaseResult",
    "fit_phase_transition",
    "logistic_curve",
    "changepoint_1d",
]
