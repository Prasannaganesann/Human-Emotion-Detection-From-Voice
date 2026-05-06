"""
database.py
===========
SQLite-backed session history storage.
Stores every prediction with metadata for trend analysis.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DB_PATH

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Schema
# ─────────────────────────────────────────────

CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    filename    TEXT,
    emotion     TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    probabilities TEXT  NOT NULL,   -- JSON
    model_name  TEXT    NOT NULL,
    duration_s  REAL,
    session_id  TEXT
);
"""

CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    note        TEXT
);
"""


# ─────────────────────────────────────────────
#  Connection Helper
# ─────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    """Return a configured SQLite connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the database schema (idempotent)."""
    with _get_conn() as conn:
        conn.execute(CREATE_PREDICTIONS_TABLE)
        conn.execute(CREATE_SESSIONS_TABLE)
    logger.info(f"Database initialized at {DB_PATH}")


# ─────────────────────────────────────────────
#  Prediction CRUD
# ─────────────────────────────────────────────

def save_prediction(emotion: str,
                    confidence: float,
                    probabilities: dict,
                    model_name: str,
                    filename: str = None,
                    duration_s: float = None,
                    session_id: str = None) -> int:
    """
    Persist a single prediction record.

    Returns
    -------
    int : Row ID of the inserted record.
    """
    ts = datetime.now().isoformat(timespec="seconds")
    probs_json = json.dumps({k: round(float(v), 4)
                             for k, v in probabilities.items()})

    sql = """
    INSERT INTO predictions
        (timestamp, filename, emotion, confidence, probabilities, model_name, duration_s, session_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            ts, filename, emotion, round(float(confidence), 4),
            probs_json, model_name, duration_s, session_id
        ))
        row_id = cur.lastrowid

    logger.debug(f"Saved prediction #{row_id}: {emotion} ({confidence:.1%})")
    return row_id


def get_predictions(limit: int = 100, session_id: str = None) -> list[dict]:
    """
    Retrieve recent predictions.

    Parameters
    ----------
    limit : int
        Max rows to return.
    session_id : str, optional
        Filter to a specific session.

    Returns
    -------
    list[dict]
    """
    sql = "SELECT * FROM predictions"
    params = []
    if session_id:
        sql += " WHERE session_id = ?"
        params.append(session_id)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with _get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    results = []
    for row in rows:
        d = dict(row)
        d["probabilities"] = json.loads(d["probabilities"])
        results.append(d)
    return results


def get_emotion_stats() -> dict:
    """
    Aggregate statistics across all predictions.

    Returns
    -------
    dict : emotion → {count, avg_confidence}
    """
    sql = """
    SELECT emotion,
           COUNT(*)       AS count,
           AVG(confidence) AS avg_confidence
    FROM predictions
    GROUP BY emotion
    ORDER BY count DESC
    """
    with _get_conn() as conn:
        rows = conn.execute(sql).fetchall()

    return {
        row["emotion"]: {
            "count": row["count"],
            "avg_confidence": round(row["avg_confidence"], 4)
        }
        for row in rows
    }


def get_trend_data(n: int = 50) -> list[dict]:
    """
    Return the last N predictions ordered by time (ascending) for trend charts.

    Returns
    -------
    list[dict] with keys: timestamp, emotion, confidence
    """
    sql = """
    SELECT timestamp, emotion, confidence
    FROM (
        SELECT * FROM predictions ORDER BY id DESC LIMIT ?
    ) sub
    ORDER BY id ASC
    """
    with _get_conn() as conn:
        rows = conn.execute(sql, (n,)).fetchall()
    return [dict(r) for r in rows]


def clear_history(session_id: str = None) -> None:
    """
    Delete prediction records.

    Parameters
    ----------
    session_id : str, optional
        If provided, only deletes records for that session.
        If None, deletes ALL records (global clear).
    """
    with _get_conn() as conn:
        if session_id:
            conn.execute(
                "DELETE FROM predictions WHERE session_id = ?", (session_id,)
            )
            logger.info(f"History cleared for session {session_id}.")
        else:
            conn.execute("DELETE FROM predictions")
            logger.warning("All prediction history cleared.")

