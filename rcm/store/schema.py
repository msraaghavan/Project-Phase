"""The five-relation trace schema.

Storing every raw KV tensor is infeasible -- a Llama-3-8B cache costs about
130 KB per token in bf16, so one 4096-token conversation exceeds 500 MB.  The
schema therefore separates cheap *provenanced statistics*, kept for every run,
from expensive raw tensors, kept only for a stratified 5 % sample.

    conversation(convId PK, model, arm, probeSuite, pressure, seed, nTurns,
                 gitCommit, ...)                         1 : N  turn
    turn(convId FK, turnIdx, agentId, role, tokenStart,
         tokenEnd, biasScore, ppl, ...)                  1 : N  kvRecord
    kvRecord(convId FK, layer, head, position, turnIdx, agentId, channel,
             attnMassRecv, reuseScore, lifetime, persistence, centrality,
             kNorm, vNorm)                               -- the fact table
    attnEdge(convId, layer, head, qPos, kPos, weight)    -- sampled layers only
    intervention(runId PK, convId FK, mode, targetSet, epsilon,
                 deltaBias, deltaPpl, deltaAcc, pValue)

The fact table lives in Parquet, partitioned by ``(model, convId)``; SQLite
holds the metadata index so a query can find the right partition without
scanning.  DuckDB, when installed, reads the Parquet directly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

RELATIONS = ("conversation", "turn", "kvRecord", "attnEdge", "intervention")

DDL = """
CREATE TABLE IF NOT EXISTS conversation (
    convId                   TEXT PRIMARY KEY,
    model                    TEXT NOT NULL,
    arm                      TEXT NOT NULL,
    probeSuite               TEXT,
    probeId                  TEXT,
    pressure                 REAL NOT NULL,
    seed                     INTEGER NOT NULL,
    nTurns                   INTEGER NOT NULL,
    gitCommit                TEXT,
    finalBias                REAL,
    biasLabel                INTEGER,
    hallucinationPersistence REAL,
    meanCoherence            REAL,
    seqLen                   INTEGER,
    rawKvSampled             INTEGER
);

CREATE TABLE IF NOT EXISTS turn (
    convId     TEXT NOT NULL REFERENCES conversation(convId),
    turnIdx    INTEGER NOT NULL,
    agentId    TEXT NOT NULL,
    role       TEXT NOT NULL,
    tokenStart INTEGER NOT NULL,
    tokenEnd   INTEGER NOT NULL,
    biasScore  REAL,
    logitGap   REAL,
    ppl        REAL,
    hallucinated INTEGER,
    text       TEXT,
    PRIMARY KEY (convId, turnIdx)
);

CREATE TABLE IF NOT EXISTS kvPartition (
    convId  TEXT NOT NULL REFERENCES conversation(convId),
    model   TEXT NOT NULL,
    path    TEXT NOT NULL,
    nRows   INTEGER NOT NULL,
    PRIMARY KEY (convId, path)
);

CREATE TABLE IF NOT EXISTS intervention (
    runId     TEXT PRIMARY KEY,
    convId    TEXT,
    mode      TEXT NOT NULL,
    targetSet TEXT,
    epsilon   REAL,
    deltaBias REAL,
    deltaPpl  REAL,
    deltaAcc  REAL,
    pValue    REAL,
    accepted  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_conv_arm      ON conversation(arm, pressure, seed);
CREATE INDEX IF NOT EXISTS idx_turn_conv     ON turn(convId);
CREATE INDEX IF NOT EXISTS idx_part_conv     ON kvPartition(convId);
"""


class TraceStore:
    """Read-only convenience access to a completed trace directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.db_path = self.root / "index.sqlite"
        self.parquet_dir = self.root / "parquet"

    # -- connection ------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def initialise(db_path: str | Path) -> sqlite3.Connection:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(db_path)
        con.executescript(DDL)
        con.commit()
        return con

    # -- queries ---------------------------------------------------------
    def conversations(self) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql_query("SELECT * FROM conversation", con)

    def turns(self, conv_id: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM turn"
        params: tuple = ()
        if conv_id:
            q += " WHERE convId = ?"
            params = (conv_id,)
        with self.connect() as con:
            return pd.read_sql_query(q, con, params=params)

    def interventions(self) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql_query("SELECT * FROM intervention", con)

    def kv_records(self, conv_id: str | None = None, columns=None) -> pd.DataFrame:
        """Read the Parquet fact table, optionally one conversation's partition."""
        import pyarrow.parquet as pq

        files = sorted(self.parquet_dir.glob("kvRecord-*.parquet"))
        if not files:
            return pd.DataFrame()
        filters = [("convId", "=", conv_id)] if conv_id else None
        table = pq.ParquetDataset(files, filters=filters).read(columns=columns)
        return table.to_pandas()

    def attn_edges(self, conv_id: str | None = None) -> pd.DataFrame:
        path = self.parquet_dir / "attnEdge.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        return df[df.convId == conv_id] if conv_id else df

    def duckdb_query(self, sql: str):
        """Run analytical SQL straight over the Parquet, if DuckDB is installed."""
        import duckdb

        con = duckdb.connect()
        con.execute(
            f"CREATE VIEW kvRecord AS SELECT * FROM "
            f"read_parquet('{(self.parquet_dir / 'kvRecord-*.parquet').as_posix()}')"
        )
        return con.execute(sql).fetchdf()
