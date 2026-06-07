"""
database/db_manager.py
----------------------
Handles all SQLite database operations: schema creation, CRUD for
evaluations, responses, and scores. Acts as the single source of truth
for persistence across sessions.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# Always store the DB file in the project root so exports can find it easily
DB_PATH = Path(__file__).parent.parent / "evaluations.db"


def get_connection() -> sqlite3.Connection:
    """Return a connection with row_factory set so rows behave like dicts."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db() -> None:
    """
    Create all tables if they don't already exist.
    Called once at application startup.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # --- evaluations: one row per user prompt session ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt      TEXT    NOT NULL,
            created_at  TEXT    NOT NULL
        )
    """)

    # --- responses: one row per model response within an evaluation ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_id   INTEGER NOT NULL REFERENCES evaluations(id),
            model_name      TEXT    NOT NULL,
            response_text   TEXT    NOT NULL,
            tokens_used     INTEGER DEFAULT 0,
            latency_ms      REAL    DEFAULT 0.0
        )
    """)

    # --- scores: human + automated scores per response ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            response_id     INTEGER NOT NULL REFERENCES responses(id),
            scorer_type     TEXT    NOT NULL,   -- 'human' | 'automated'
            accuracy        REAL,
            completeness    REAL,
            clarity         REAL,
            creativity      REAL,
            helpfulness     REAL,
            overall_quality REAL,
            weighted_score  REAL,
            justification   TEXT,               -- JSON string for automated
            scored_at       TEXT    NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# ─── Evaluation CRUD ────────────────────────────────────────────────────────

def create_evaluation(prompt: str) -> int:
    """Insert a new evaluation row and return its generated ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO evaluations (prompt, created_at) VALUES (?, ?)",
        (prompt, datetime.now().isoformat())
    )
    eval_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return eval_id  # type: ignore[return-value]


def get_all_evaluations() -> list[dict]:
    """Return all evaluations ordered newest-first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM evaluations ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_evaluations(query: str) -> list[dict]:
    """Full-text search over prompt text (case-insensitive)."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM evaluations WHERE prompt LIKE ? ORDER BY created_at DESC",
        (f"%{query}%",)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Response CRUD ───────────────────────────────────────────────────────────

def save_response(
    evaluation_id: int,
    model_name: str,
    response_text: str,
    tokens_used: int = 0,
    latency_ms: float = 0.0,
) -> int:
    """Persist a model's response and return the new row ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO responses
           (evaluation_id, model_name, response_text, tokens_used, latency_ms)
           VALUES (?, ?, ?, ?, ?)""",
        (evaluation_id, model_name, response_text, tokens_used, latency_ms),
    )
    resp_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return resp_id  # type: ignore[return-value]


def get_responses_for_evaluation(evaluation_id: int) -> list[dict]:
    """Fetch all model responses belonging to one evaluation."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM responses WHERE evaluation_id = ?",
        (evaluation_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Score CRUD ──────────────────────────────────────────────────────────────

def save_score(
    response_id: int,
    scorer_type: str,
    accuracy: Optional[float],
    completeness: Optional[float],
    clarity: Optional[float],
    creativity: Optional[float],
    helpfulness: Optional[float],
    overall_quality: Optional[float],
    weighted_score: Optional[float],
    justification: Optional[dict | str] = None,
) -> int:
    """Persist a score row (human or automated) and return its ID."""
    just_str = json.dumps(justification) if isinstance(justification, dict) else justification
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO scores
           (response_id, scorer_type, accuracy, completeness, clarity,
            creativity, helpfulness, overall_quality, weighted_score,
            justification, scored_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (response_id, scorer_type, accuracy, completeness, clarity,
         creativity, helpfulness, overall_quality, weighted_score,
         just_str, datetime.now().isoformat()),
    )
    score_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return score_id  # type: ignore[return-value]


def get_scores_for_response(response_id: int) -> list[dict]:
    """Return all score rows (human + automated) for a given response."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM scores WHERE response_id = ?",
        (response_id,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        # Deserialize JSON justification back to dict if possible
        if d.get("justification"):
            try:
                d["justification"] = json.loads(d["justification"])
            except (json.JSONDecodeError, TypeError):
                pass
        result.append(d)
    return result


def get_full_evaluation_data(evaluation_id: int) -> dict:
    """
    Return a fully-hydrated dict for one evaluation including all responses
    and their scores. Useful for export and history display.
    """
    conn = get_connection()
    eval_row = conn.execute(
        "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
    ).fetchone()
    conn.close()

    if not eval_row:
        return {}

    data = dict(eval_row)
    data["responses"] = []

    for resp in get_responses_for_evaluation(evaluation_id):
        resp["scores"] = get_scores_for_response(resp["id"])
        data["responses"].append(resp)

    return data
