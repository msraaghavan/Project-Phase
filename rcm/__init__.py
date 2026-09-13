"""Runtime Computational Memory (RCM).

Reference implementation of the design in
"Runtime Computational Memory: Understanding Emergent Behaviour through
KV-Cache Dynamics in Transformer Language Models" (23CSE498, Team D10).

The package is organised as the four modules of the design document:

    rcm.module1 -- runtime memory instrumentation and provenance logging
    rcm.module2 -- memory structure analysis (attractors, bias reservoirs)
    rcm.module3 -- multi-agent contagion and phase-transition analysis
    rcm.module4 -- causal KV-only intervention, evaluation and reporting

Everything is driven by a single :class:`rcm.config.ExperimentConfig`, and every
run is fully reproducible from its recorded manifest.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
