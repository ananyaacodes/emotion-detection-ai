"""
emotion_engine.py
Wraps a pretrained transformer model to classify text into emotions.

Model: j-hartmann/emotion-english-distilroberta-base
Labels: anger, disgust, fear, joy, neutral, sadness, surprise

The heavy ML dependencies (torch, transformers) are imported lazily inside
EmotionEngine.__init__ so that the rest of the codebase (models, storage, cli)
can be imported and unit-tested without requiring them to be installed.
"""

from models import EmotionResult

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


class ModelLoadError(RuntimeError):
    """Raised when the underlying ML model can't be loaded (missing deps, no internet, etc.)."""


class EmotionEngine:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._pipeline = None  # lazy-loaded

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ModelLoadError(
                "transformers/torch are not installed. Run: pip install -r requirements.txt"
            ) from e

        try:
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=1,
            )
        except Exception as e:
            raise ModelLoadError(
                f"Could not load model '{self.model_name}'. "
                "Check your internet connection (first run downloads the model)."
            ) from e
        return self._pipeline

    def analyze(self, text: str) -> EmotionResult:
        """Run emotion detection on a single string and return an EmotionResult."""
        if not text or not text.strip():
            raise ValueError("Cannot analyze empty text.")

        clf = self._load_pipeline()
        raw = clf(text)

        # pipeline with top_k=1 returns [[{"label": ..., "score": ...}]] on newer
        # versions, or [{"label": ..., "score": ...}] on older ones — handle both.
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]
        top = raw[0]

        return EmotionResult(
            text=text,
            emotion=top["label"],
            confidence=float(top["score"]),
        )

    def analyze_batch(self, texts: list[str]) -> list[EmotionResult]:
        """Analyze multiple lines of text, skipping blank ones."""
        results = []
        for line in texts:
            line = line.strip()
            if not line:
                continue
            results.append(self.analyze(line))
        return results
