"""
storage.py
SQLite-backed persistence layer for emotion analysis history.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from models import EmotionResult

DEFAULT_DB_PATH = Path(__file__).parent / "emotion_history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    emotion TEXT NOT NULL,
    confidence REAL NOT NULL,
    timestamp TEXT NOT NULL
);
"""


class Storage:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def save(self, result: EmotionResult) -> int:
        """Persist an EmotionResult, returning its new row id."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO analyses (text, emotion, confidence, timestamp) VALUES (?, ?, ?, ?)",
                (result.text, result.emotion, result.confidence, result.timestamp),
            )
            return cur.lastrowid

    def get_recent(self, limit: int = 10) -> list[EmotionResult]:
        """Return the most recent `limit` analyses, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, text, emotion, confidence, timestamp FROM analyses "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [EmotionResult.from_row(row) for row in rows]

    def most_common_emotion(self) -> str | None:
        """Return the most frequently logged emotion, or None if history is empty."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT emotion, COUNT(*) as c FROM analyses "
                "GROUP BY emotion ORDER BY c DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()
        return row[0]

    def clear(self):
        """Wipe all history. Used mainly for tests."""
        with self._connect() as conn:
            conn.execute("DELETE FROM analyses")
