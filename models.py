"""
models.py
Data models used across the emotion detection system.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class EmotionResult:
    """Represents the outcome of running emotion detection on a piece of text."""

    text: str
    emotion: str
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise ValueError("text must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "emotion": self.emotion,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "EmotionResult":
        """Build an EmotionResult from a SQLite row: (id, text, emotion, confidence, timestamp)."""
        _, text, emotion, confidence, timestamp = row
        return cls(text=text, emotion=emotion, confidence=confidence, timestamp=timestamp)

    def __str__(self) -> str:
        return f"[{self.timestamp}] '{self.text[:40]}...' -> {self.emotion} ({self.confidence:.0%})" \
            if len(self.text) > 40 else f"[{self.timestamp}] '{self.text}' -> {self.emotion} ({self.confidence:.0%})"
