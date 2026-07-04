# Emotion Detection AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-DistilRoBERTa-yellow)
![SQLite](https://img.shields.io/badge/Storage-SQLite-07405e?logo=sqlite&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A command-line tool that detects emotion in text using a pretrained transformer model, with persistent history and batch analysis. Originally built in 1st year as a simple TextBlob polarity script (Happy / Sad / Neutral) — rebuilt with a modular architecture, a real multi-class emotion model, SQLite persistence, and a test suite.

## Features

- **Real emotion classification** — joy, sadness, anger, fear, disgust, surprise, and neutral, via [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base), not just positive/negative polarity
- **Confidence scores** on every prediction
- **Persistent history** — every analysis is saved to SQLite; view your last N analyses or your most common emotion
- **Batch mode** — analyze an entire `.txt` file of lines in one command
- **Unit tested** — engine, storage, and models are covered by `pytest`, with the ML model mocked so tests run instantly without needing to download anything

## Project structure

```
emotion-detection-ai/
├── main.py              # entry point
├── cli.py               # interactive CLI (single, batch, history, stats)
├── emotion_engine.py     # wraps the HuggingFace pipeline
├── models.py             # EmotionResult data model
├── storage.py             # SQLite persistence layer
├── tests/
│   ├── test_emotion_engine.py
│   ├── test_models.py
│   └── test_storage.py
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone https://github.com/ananyaacodes/emotion-detection-ai.git
cd emotion-detection-ai
pip install -r requirements.txt
python main.py
```

> **Note:** The first run downloads the emotion model (~330MB) from Hugging Face, so it needs an internet connection. After that it's cached locally and works offline.

## Usage

```
============================================
   Emotion Detection AI
   Type a sentence to analyze its emotion.
   Commands: :history  :stats  :batch <file>  :quit
============================================
> I just got the internship offer!
  -> joy 🙂  (confidence: 94%)
> :history
Last 1 analyses:
  [2026-07-04T12:03:11] 'I just got the internship offer!' -> joy (94%)
> :stats
Total analyses: 1
Most common emotion: joy
> :quit
Goodbye!
```

Batch mode reads a plain text file, one sentence per line:

```
> :batch sample_texts.txt
```

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

Tests mock the transformer pipeline directly, so they run in milliseconds with no model download required.

## Why this version is different from v1

| | v1 (1st year) | v2 (this version) |
|---|---|---|
| Detection method | TextBlob polarity threshold | Pretrained transformer (7 emotion classes) |
| Structure | Single file | Modular: engine / storage / CLI / models |
| Persistence | None | SQLite history |
| Batch analysis | No | Yes |
| Tests | No | Yes (pytest, mocked model) |
| Error handling | Minimal | Explicit `ModelLoadError`, input validation |

## License

MIT
