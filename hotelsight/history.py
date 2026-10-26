"""Saved prediction history: a small SQLite file next to the package (one row per scored booking).

The location can be changed with the HOTELSIGHT_HISTORY_DB environment variable (the tests use a
temporary file). Nothing here touches the frozen experiment.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from .paths import ROOT

COLUMNS = ["id", "created_at", "source", "batch_id", "batch_name", "reference", "hotel", "arrival",
           "probability", "predicted_cancel", "threshold", "top_raising", "top_lowering", "actual_cancelled",
           "inputs_json", "model_sha256"]


def db_path() -> Path:
    return Path(os.environ.get("HOTELSIGHT_HISTORY_DB") or ROOT / "history" / "predictions.db")


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, source TEXT NOT NULL,
        batch_id TEXT NOT NULL, batch_name TEXT, reference TEXT, hotel TEXT, arrival TEXT,
        probability REAL NOT NULL, predicted_cancel INTEGER NOT NULL, threshold REAL NOT NULL,
        top_raising TEXT, top_lowering TEXT, actual_cancelled INTEGER, inputs_json TEXT, model_sha256 TEXT)""")
    return con


def new_batch_id() -> str:
    return uuid.uuid4().hex[:12]


def save(entries: list, source: str, batch_name: str, threshold: float, model_sha256: str,
         batch_id: str | None = None) -> str:
    """entries: dicts with reference, hotel, arrival, probability, top_raising, top_lowering, record, actual."""
    batch_id = batch_id or new_batch_id()
    now = datetime.now().isoformat(timespec="seconds")
    rows = [(now, source, batch_id, batch_name, e.get("reference"), e.get("hotel"), e.get("arrival"),
             float(e["probability"]), int(e["probability"] >= threshold), threshold, e.get("top_raising"),
             e.get("top_lowering"), e.get("actual"), json.dumps(e.get("record"), default=float), model_sha256)
            for e in entries]
    with _connect() as con:
        con.executemany(f"INSERT INTO predictions ({', '.join(COLUMNS[1:])}) VALUES ({', '.join('?' * (len(COLUMNS) - 1))})",
                        rows)
    return batch_id


def load() -> pd.DataFrame:
    with _connect() as con:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", con)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def with_stay(df: pd.DataFrame) -> pd.DataFrame:
    """Add each saved reservation's nights and stay value (room rate x nights, EUR) from its stored inputs."""
    recs = [json.loads(x or "{}") for x in df["inputs_json"]]
    nights = [(r.get("stays_in_weekend_nights") or 0) + (r.get("stays_in_week_nights") or 0) for r in recs]
    value = [r["adr"] * n if r.get("adr") is not None else None for r, n in zip(recs, nights)]
    return df.assign(nights=nights, stay_value=pd.to_numeric(pd.Series(value, index=df.index, dtype="object")))


def delete_batch(batch_id: str) -> int:
    with _connect() as con:
        return con.execute("DELETE FROM predictions WHERE batch_id = ?", (batch_id,)).rowcount


def clear() -> int:
    with _connect() as con:
        return con.execute("DELETE FROM predictions").rowcount
