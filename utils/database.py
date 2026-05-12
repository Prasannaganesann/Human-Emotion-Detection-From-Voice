"""
database.py
===========
SQLite-backed session history storage.
Stores every prediction with metadata for trend analysis, search, and filtering.

v2 additions:
- search_predictions() — full-text search on emotion + filename
- get_predictions() now supports emotion_filter, min_confidence, max_confidence
- get_session_stats() — per-session aggregated stats
- export_csv() helper
"""

import csv
import io
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DB_PATH

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Schema
# ─────────────────────────────────────────────

CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    filename      TEXT,
    emotion       TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    probabilities TEXT    NOT NULL,  -- JSON blob
    model_name    TEXT    NOT NULL,
    duration_s    REAL,
    session_id    TEXT,
    inference_ms  REAL               -- added v2
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

# Migration: add inference_ms column if it doesn't exist yet
MIGRATE_INFERENCE_MS = """
ALTER TABLE predictions ADD COLUMN inference_ms REAL;
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
        # Attempt migration — silently skip if column already exists
        try:
            conn.execute(MIGRATE_INFERENCE_MS)
        except sqlite3.OperationalError:
            pass
    logger.info(f"DB initialized at {DB_PATH}")


# ─────────────────────────────────────────────
#  Write
# ─────────────────────────────────────────────

def save_prediction(
    emotion: str,
    confidence: float,
    probabilities: dict,
    model_name: str,
    filename: str = None,
    duration_s: float = None,
    session_id: str = None,
    inference_ms: float = None,
) -> int:
    """
    Persist a single prediction.

    Returns
    -------
    int : Row ID of the inserted record.
    """
    ts         = datetime.now().isoformat(timespec="seconds")
    probs_json = json.dumps({k: round(float(v), 4) for k, v in probabilities.items()})

    sql = """
    INSERT INTO predictions
        (timestamp, filename, emotion, confidence, probabilities,
         model_name, duration_s, session_id, inference_ms)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            ts, filename, emotion, round(float(confidence), 4),
            probs_json, model_name, duration_s, session_id,
            round(float(inference_ms), 1) if inference_ms else None,
        ))
        row_id = cur.lastrowid

    logger.debug(f"Saved prediction #{row_id}: {emotion} ({confidence:.1%})")
    return row_id


# ─────────────────────────────────────────────
#  Read
# ─────────────────────────────────────────────

def get_predictions(
    limit: int = 100,
    session_id: str = None,
    emotion_filter: str = None,
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
    search_text: str = None,
    offset: int = 0,
) -> list[dict]:
    """
    Retrieve predictions with optional filtering and pagination.

    Parameters
    ----------
    emotion_filter : str, optional
        Exact emotion label to filter on.
    min_confidence / max_confidence : float
        Confidence range 0.0–1.0.
    search_text : str, optional
        Substring match on filename or emotion.
    offset : int
        Pagination offset.
    """
    sql    = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if session_id:
        sql += " AND session_id = ?"
        params.append(session_id)
    if emotion_filter:
        sql += " AND emotion = ?"
        params.append(emotion_filter)
    if min_confidence > 0.0:
        sql += " AND confidence >= ?"
        params.append(min_confidence)
    if max_confidence < 1.0:
        sql += " AND confidence <= ?"
        params.append(max_confidence)
    if search_text:
        sql += " AND (emotion LIKE ? OR filename LIKE ?)"
        term = f"%{search_text}%"
        params += [term, term]

    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params += [limit, offset]

    with _get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    results = []
    for row in rows:
        d = dict(row)
        d["probabilities"] = json.loads(d["probabilities"])
        results.append(d)
    return results


def get_predictions_count(
    session_id: str = None,
    emotion_filter: str = None,
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
    search_text: str = None,
) -> int:
    """Return total count matching the same filters as get_predictions()."""
    sql    = "SELECT COUNT(*) FROM predictions WHERE 1=1"
    params = []
    if session_id:
        sql += " AND session_id = ?"
        params.append(session_id)
    if emotion_filter:
        sql += " AND emotion = ?"
        params.append(emotion_filter)
    if min_confidence > 0.0:
        sql += " AND confidence >= ?"
        params.append(min_confidence)
    if max_confidence < 1.0:
        sql += " AND confidence <= ?"
        params.append(max_confidence)
    if search_text:
        sql += " AND (emotion LIKE ? OR filename LIKE ?)"
        term = f"%{search_text}%"
        params += [term, term]

    with _get_conn() as conn:
        return conn.execute(sql, params).fetchone()[0]


def get_emotion_stats() -> dict:
    """
    Aggregate statistics across all predictions.

    Returns
    -------
    dict : emotion → {count, avg_confidence, max_confidence}
    """
    sql = """
    SELECT emotion,
           COUNT(*)           AS count,
           AVG(confidence)    AS avg_confidence,
           MAX(confidence)    AS max_confidence
    FROM predictions
    GROUP BY emotion
    ORDER BY count DESC
    """
    with _get_conn() as conn:
        rows = conn.execute(sql).fetchall()

    return {
        row["emotion"]: {
            "count":          row["count"],
            "avg_confidence": round(row["avg_confidence"], 4),
            "max_confidence": round(row["max_confidence"], 4),
        }
        for row in rows
    }


def get_session_stats(session_id: str) -> dict:
    """Per-session statistics."""
    sql = """
    SELECT COUNT(*) as total,
           AVG(confidence) as avg_conf,
           MAX(confidence) as max_conf
    FROM predictions WHERE session_id = ?
    """
    with _get_conn() as conn:
        row = conn.execute(sql, (session_id,)).fetchone()
    return {
        "total":    row["total"],
        "avg_conf": round(row["avg_conf"] or 0, 4),
        "max_conf": round(row["max_conf"] or 0, 4),
    }


def get_trend_data(n: int = 50) -> list[dict]:
    """
    Return the last N predictions ordered by time (ascending) for trend charts.
    """
    sql = """
    SELECT timestamp, emotion, confidence
    FROM (SELECT * FROM predictions ORDER BY id DESC LIMIT ?) sub
    ORDER BY id ASC
    """
    with _get_conn() as conn:
        rows = conn.execute(sql, (n,)).fetchall()
    return [dict(r) for r in rows]


def get_distinct_emotions() -> list[str]:
    """Return the list of unique emotions in the database."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT emotion FROM predictions ORDER BY emotion"
        ).fetchall()
    return [r[0] for r in rows]


# ─────────────────────────────────────────────
#  Delete
# ─────────────────────────────────────────────

def clear_history(session_id: str = None) -> None:
    """Delete prediction records."""
    with _get_conn() as conn:
        if session_id:
            conn.execute(
                "DELETE FROM predictions WHERE session_id = ?", (session_id,)
            )
            logger.info(f"History cleared for session {session_id}.")
        else:
            conn.execute("DELETE FROM predictions")
            logger.warning("All prediction history cleared.")


# ─────────────────────────────────────────────
#  Export
# ─────────────────────────────────────────────

def export_predictions_csv(session_id: str = None) -> str:
    """
    Export predictions as a CSV string.

    Returns
    -------
    str : CSV content ready for st.download_button.
    """
    rows = get_predictions(limit=10000, session_id=session_id)
    if not rows:
        return ""

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[
        "id", "timestamp", "filename", "emotion",
        "confidence", "model_name", "duration_s", "session_id", "inference_ms"
    ])
    writer.writeheader()
    for row in rows:
        row_copy = {k: row.get(k, "") for k in writer.fieldnames}
        writer.writerow(row_copy)
    return buf.getvalue()
