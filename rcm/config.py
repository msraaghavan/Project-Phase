"""Experiment configuration, seeding and the reproducibility manifest.

A run is completely described by an :class:`ExperimentConfig`.  The config is
hashed into a SHA-256 manifest that is written next to the traces, so a result
can never be silently altered after the fact (design doc section 9,
"Security Architecture" / "Reliability").
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

# --------------------------------------------------------------------------
# The three memory arms of the multi-agent protocol (design doc, Module 1).
# --------------------------------------------------------------------------
ARM_SHARED = "shared-kv"   # receiving agent's cache is extended with the peer's KV
ARM_TEXT = "text-only"     # only generated text crosses between agents
ARM_CONTROL = "isolated"   # agents never interact
ARMS = (ARM_SHARED, ARM_TEXT, ARM_CONTROL)

# Source channel of a KV entry (provenance field `channel`).
CH_SYSTEM = "system"
CH_PROBE = "probe"
CH_AGENT = "agent"
CH_PEER = "peer"  # transplanted from another agent in the shared-KV arm


@dataclass
class ModelConfig:
    """Architecture of the instrumented model.

    The defaults describe the CPU-sized reference transformer used for the
    validation testbed.  Switching ``backend`` to ``"hf"`` and setting
    ``hf_model_id`` runs the identical pipeline against a frozen open-weight
    model (Llama-3-8B-Instruct, Mistral-7B-Instruct, Qwen2.5-7B) -- the only
    thing that changes is the :class:`~rcm.adapters.base.ModelAdapter`.
    """

    backend: str = "tiny"            # "tiny" | "hf"
    name: str = "tiny-gqa-4L8H"
    hf_model_id: str | None = None

    # Reference-transformer geometry (decoder-only, GQA, RoPE) ------------
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8                 # query heads
    n_kv_heads: int = 2              # key/value heads  -> GQA group size 4
    d_ff: int = 256
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads


@dataclass
class MetricConfig:
    """Thresholds for the Module 2 attractor definition.

    These are *pre-registered*: they are fixed in the config before the main
    runs, never tuned afterwards (design doc, "Other Design Artifacts").
    """

    z_star: float = 1.0        # z* : reuse-score z-score threshold
    lam: float = 0.30          # lambda : lifetime threshold as a fraction of T
    p_star: float = 0.10       # p* : attention-persistence threshold
    top_k: int = 8             # k in the TopK of the persistence metric
    smooth_window: int = 8     # w : smoothing window of the lifetime metric
    kappa: float = 1.0         # kappa : relative threshold of the lifetime metric
    pagerank_damping: float = 0.85
    sink_positions: int = 4    # t <= 4 is reported as the known attention-sink class
    fdr_alpha: float = 0.05    # Benjamini-Hochberg level for reservoir screening
    n_reservoirs: int = 4      # m : number of heads kept as the reservoir set
    probe_folds: int = 5       # cross-validation folds of the logistic probe


@dataclass
class InterventionConfig:
    """Module 4 intervention and coherence budget."""

    modes: tuple[str, ...] = ("ablate", "perturb", "gate")
    epsilon: float = 1.0       # perturbation strength
    gate: float = 0.10         # g in [0, 1] for the gating mode
    max_ppl_increase: float = 0.05   # coherence budget: <= 5 % perplexity degradation
    max_acc_drop: float = 0.02       # coherence budget: within 2 accuracy points
    n_bootstrap: int = 2000
    alpha: float = 0.05


@dataclass
class RunConfig:
    """The experimental grid."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    pressures: tuple[float, ...] = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
    arms: tuple[str, ...] = ARMS
    n_turns: int = 8                 # turns per conversation
    max_new_tokens: int = 10         # tokens generated per turn
    n_agents: int = 2
    raw_kv_sample_rate: float = 0.05  # stratified 5 % raw-tensor dump
    train_steps: int = 400            # testbed corpus training steps
    train_batch: int = 32


@dataclass
class ExperimentConfig:
    """Top-level configuration.  One object fully determines a run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    run: RunConfig = field(default_factory=RunConfig)

    out_dir: str = "results"
    probe_suite: str = "synthetic-bbq"
    master_seed: int = 1234

    # -- serialisation --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        def _mk(klass, sub):
            fields = {f.name for f in dataclasses.fields(klass)}
            kwargs = {}
            for k, v in (sub or {}).items():
                if k not in fields:
                    continue
                # dataclasses.asdict turns tuples into lists; restore them.
                default = getattr(klass(), k)
                kwargs[k] = tuple(v) if isinstance(default, tuple) else v
            return klass(**kwargs)

        return cls(
            model=_mk(ModelConfig, d.get("model")),
            metrics=_mk(MetricConfig, d.get("metrics")),
            intervention=_mk(InterventionConfig, d.get("intervention")),
            run=_mk(RunConfig, d.get("run")),
            out_dir=d.get("out_dir", "results"),
            probe_suite=d.get("probe_suite", "synthetic-bbq"),
            master_seed=int(d.get("master_seed", 1234)),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "ExperimentConfig":
        """Load from YAML if PyYAML is present, otherwise from JSON."""
        text = Path(path).read_text(encoding="utf-8")
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
        except Exception:
            data = json.loads(text)
        return cls.from_dict(data or {})

    # -- reproducibility -------------------------------------------------
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def manifest(self) -> dict[str, Any]:
        """Everything needed to reproduce the run, byte for byte."""
        return {
            "config": self.to_dict(),
            "config_sha256": self.config_hash(),
            "git_commit": git_commit(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "master_seed": self.master_seed,
        }

    def write_manifest(self, out_dir: str | os.PathLike[str] | None = None) -> Path:
        d = Path(out_dir or self.out_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "manifest.json"
        path.write_text(json.dumps(self.manifest(), indent=2, default=str), encoding="utf-8")
        return path

    @property
    def n_cells(self) -> int:
        """Number of (arm, pressure, seed) cells in the sweep."""
        return len(self.run.arms) * len(self.run.pressures) * len(self.run.seeds)


def git_commit() -> str:
    """Best-effort git commit id; ``"nogit"`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        commit = out.stdout.strip()
        return commit or "nogit"
    except Exception:
        return "nogit"


def set_all_seeds(seed: int) -> None:
    """Pin every source of randomness.

    Without this, seed-matched replay -- and therefore every causal claim in
    Module 4 -- would be meaningless.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - older torch builds
        pass


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
