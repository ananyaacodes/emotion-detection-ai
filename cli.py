"""
cli.py
Interactive command-line interface for the emotion detection system.
"""

from pathlib import Path

from emotion_engine import EmotionEngine, ModelLoadError
from storage import Storage

BANNER = """
============================================
   Emotion Detection AI
   Type a sentence to analyze its emotion.
   Commands: :history  :stats  :batch <file>  :quit
============================================
"""

EMOJI = {
    "joy": "🙂",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "disgust": "🤢",
    "surprise": "😲",
    "neutral": "😐",
}


class EmotionCLI:
    def __init__(self, engine: EmotionEngine = None, storage: Storage = None):
        self.engine = engine or EmotionEngine()
        self.storage = storage or Storage()

    def run(self):
        print(BANNER)
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input == ":quit":
                print("Goodbye!")
                break
            elif user_input == ":history":
                self._show_history()
            elif user_input == ":stats":
                self._show_stats()
            elif user_input.startswith(":batch"):
                self._handle_batch(user_input)
            else:
                self._handle_single(user_input)

    def _handle_single(self, text: str):
        try:
            result = self.engine.analyze(text)
        except ModelLoadError as e:
            print(f"[Error] {e}")
            return
        except ValueError as e:
            print(f"[Error] {e}")
            return

        self.storage.save(result)
        emoji = EMOJI.get(result.emotion, "")
        print(f"  -> {result.emotion} {emoji}  (confidence: {result.confidence:.0%})")

    def _handle_batch(self, command: str):
        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            print("Usage: :batch <path-to-txt-file>")
            return

        path = Path(parts[1].strip())
        if not path.exists():
            print(f"[Error] File not found: {path}")
            return

        lines = path.read_text(encoding="utf-8").splitlines()
        try:
            results = self.engine.analyze_batch(lines)
        except ModelLoadError as e:
            print(f"[Error] {e}")
            return

        for result in results:
            self.storage.save(result)
            print(f"  {result}")
        print(f"Analyzed {len(results)} lines from {path.name}.")

    def _show_history(self, limit: int = 10):
        recent = self.storage.get_recent(limit)
        if not recent:
            print("No history yet.")
            return
        print(f"Last {len(recent)} analyses:")
        for r in recent:
            print(f"  {r}")

    def _show_stats(self):
        total = self.storage.count()
        common = self.storage.most_common_emotion()
        print(f"Total analyses: {total}")
        print(f"Most common emotion: {common or 'N/A'}")


if __name__ == "__main__":
    EmotionCLI().run()
