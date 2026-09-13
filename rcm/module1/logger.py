"""Provenance and trace writer.

Writes the five-relation store of :mod:`rcm.store.schema`: a Parquet fact table
for the per-(layer, head, position) KV records, plus a SQLite index over the
conversation, turn and intervention metadata.

Two design decisions from the document are implemented literally here:

* **statistics for every run, raw tensors for a sample.**  Every conversation
  contributes provenanced statistics; the raw key/value tensors are dumped only
  for a stratified ``raw_kv_sample_rate`` fraction of runs.
* **integrity by manifest.**  Every file written is SHA-256 hashed into
  ``checksums.json`` alongside the config and git commit that produced it, so a
  published result cannot be silently altered afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rcm.config import ExperimentConfig, sha256_file
from rcm.store.schema import TraceStore

KV_SCHEMA = pa.schema([
    ("convId", pa.string()),
    ("model", pa.string()),
    ("layer", pa.int16()),
    ("head", pa.int16()),
    ("position", pa.int32()),
    ("turnIdx", pa.int16()),
    ("agentId", pa.string()),
    ("channel", pa.string()),
    ("tokenId", pa.int32()),
    ("attnMassRecv", pa.float32()),
    ("reuseScore", pa.float32()),
    ("lifetime", pa.float32()),
    ("persistence", pa.float32()),
    ("centrality", pa.float32()),
    ("kNorm", pa.float32()),
    ("vNorm", pa.float32()),
    ("isAttractor", pa.bool_()),
])


class ProvenanceLogger:
    """Streaming writer for one experiment's traces."""

    def __init__(self, cfg: ExperimentConfig, out_dir: str | Path | None = None,
                 flush_every: int = 20) -> None:
        self.cfg = cfg
        self.root = Path(out_dir or cfg.out_dir) / "traces"
        self.parquet_dir = self.root / "parquet"
        self.raw_dir = self.root / "raw_kv"
        for d in (self.root, self.parquet_dir, self.raw_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.store = TraceStore(self.root)
        self.con = TraceStore.initialise(self.store.db_path)
        self._kv_buffer: list[dict[str, Any]] = []
        self._edges: list[dict[str, Any]] = []
        self._part_idx = 0
        self._flush_every = flush_every
        self._n_conversations = 0
        self._files: list[Path] = []

    # -- writing ---------------------------------------------------------
    def log_conversation(self, trace, metrics, attractors) -> None:
        """Record one conversation: metadata, turns, KV facts and edges."""
        row = trace.as_conversation_row()
        self.con.execute(
            "INSERT OR REPLACE INTO conversation VALUES "
            "(:convId,:model,:arm,:probeSuite,:probeId,:pressure,:seed,:nTurns,:gitCommit,"
            ":finalBias,:biasLabel,:hallucinationPersistence,:meanCoherence,:seqLen,:rawKvSampled)",
            {**row, "rawKvSampled": int(row["rawKvSampled"])},
        )
        self.con.executemany(
            "INSERT OR REPLACE INTO turn VALUES "
            "(:convId,:turnIdx,:agentId,:role,:tokenStart,:tokenEnd,:biasScore,:logitGap,"
            ":ppl,:hallucinated,:text)",
            [{**t.__dict__, "hallucinated": int(t.hallucinated)} for t in trace.turns],
        )
        self._kv_buffer.extend(self._kv_rows(trace, metrics, attractors))
        if trace.raw_kv_sampled:
            self._dump_raw_kv(trace)
            self._edges.extend(self._edge_rows(trace))
        self._n_conversations += 1
        if self._n_conversations % self._flush_every == 0:
            self.flush()

    def _kv_rows(self, trace, metrics, attractors) -> list[dict[str, Any]]:
        prov = trace.provenance
        T = int(trace.stats["seq_len"])
        mass = trace.stats["mass"]                       # [L, H_q, T]
        L, H, _ = mass.shape
        group = max(1, H // trace.key_norms.shape[1])
        attractor_set = set(attractors.all_positions)
        arrays = metrics.as_arrays()
        rows = []
        for l in range(L):
            for h in range(H):
                kv_h = h // group
                k_norm = trace.key_norms[l, kv_h]
                v_norm = trace.value_norms[l, kv_h]
                for t in range(min(T, len(prov))):
                    p = prov[t]
                    rows.append({
                        "convId": trace.convId,
                        "model": trace.model,
                        "layer": l,
                        "head": h,
                        "position": t,
                        "turnIdx": p["turnIdx"],
                        "agentId": p["agentId"],
                        "channel": p["channel"],
                        "tokenId": p["tokenId"],
                        "attnMassRecv": float(mass[l, h, t]),
                        "reuseScore": float(arrays["reuseScore"][t]),
                        "lifetime": float(arrays["lifetime"][t]),
                        "persistence": float(arrays["persistence"][t]),
                        "centrality": float(arrays["centrality"][t]),
                        "kNorm": float(k_norm[t]) if t < len(k_norm) else 0.0,
                        "vNorm": float(v_norm[t]) if t < len(v_norm) else 0.0,
                        "isAttractor": t in attractor_set,
                    })
        return rows

    @staticmethod
    def _edge_rows(trace, threshold: float = 0.02) -> list[dict[str, Any]]:
        """Attention-graph edges, retained only for sampled conversations."""
        a = np.asarray(trace.stats["alpha_bar"])
        T = int(trace.stats["seq_len"])
        a = a[:T, :T]
        qs, ks = np.nonzero(a > threshold)
        return [
            {"convId": trace.convId, "layer": -1, "head": -1,
             "qPos": int(q), "kPos": int(k), "weight": float(a[q, k])}
            for q, k in zip(qs, ks)
        ]

    def _dump_raw_kv(self, trace) -> None:
        """Raw tensor dump for the stratified sample."""
        path = self.raw_dir / f"{_safe(trace.convId)}.npz"
        np.savez_compressed(
            path,
            values=trace.values.astype(np.float16),
            key_norms=trace.key_norms.astype(np.float32),
            value_norms=trace.value_norms.astype(np.float32),
            token_ids=np.asarray(trace.token_ids(), dtype=np.int32),
        )
        self._files.append(path)

    def log_intervention(self, run_id: str, outcome, conv_id: str = "") -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO intervention VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                run_id, conv_id, outcome.mode,
                json.dumps({str(k): v for k, v in getattr(outcome, "target_summary", {}).items()}),
                float(getattr(outcome, "epsilon", 0.0)),
                float(outcome.ace), float(outcome.delta_ppl), float(outcome.delta_acc),
                float(outcome.p_vs_zero), int(outcome.accepted),
            ),
        )
        self.con.commit()

    # -- flushing and closing --------------------------------------------
    def flush(self) -> None:
        if self._kv_buffer:
            path = self.parquet_dir / f"kvRecord-{self._part_idx:04d}.parquet"
            table = pa.Table.from_pylist(self._kv_buffer, schema=KV_SCHEMA)
            pq.write_table(table, path, compression="snappy")
            counts: dict[tuple[str, str], int] = {}
            for r in self._kv_buffer:
                counts[(r["convId"], r["model"])] = counts.get((r["convId"], r["model"]), 0) + 1
            self.con.executemany(
                "INSERT OR REPLACE INTO kvPartition VALUES (?,?,?,?)",
                [(cid, model, path.name, n) for (cid, model), n in counts.items()],
            )
            self._files.append(path)
            self._part_idx += 1
            self._kv_buffer.clear()
        self.con.commit()

    def close(self) -> dict[str, Any]:
        """Flush, write the edge table and the integrity manifest."""
        self.flush()
        if self._edges:
            path = self.parquet_dir / "attnEdge.parquet"
            pd.DataFrame(self._edges).to_parquet(path, index=False)
            self._files.append(path)
        self.con.commit()
        self.con.close()

        checksums = {}
        for p in sorted(set(self._files)) + [self.store.db_path]:
            if Path(p).exists():
                checksums[Path(p).name] = sha256_file(p)
        manifest = self.cfg.manifest()
        manifest["files"] = checksums
        manifest["nConversations"] = self._n_conversations
        (self.root / "checksums.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        return manifest


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
