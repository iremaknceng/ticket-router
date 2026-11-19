"""Loading the trained ladder + the shared normalizer from models/."""
from __future__ import annotations

import json
from pathlib import Path

from config import MODEL_DIR, MODEL_KEYS
from src.preprocess import Normalizer, SpellCorrector, tokenize, turkish_lower


def build_vocab(raw_texts) -> list[str]:
    return sorted({w for t in raw_texts for w in tokenize(turkish_lower(t))})


def save_preprocessing(labels, vocab, d: Path = MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
    (d / "labels.json").write_text(json.dumps(labels, ensure_ascii=False))
    (d / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))


def load_preprocessing(d: Path = MODEL_DIR):
    labels = json.loads((d / "labels.json").read_text())
    vocab = set(json.loads((d / "vocab.json").read_text()))
    return labels, Normalizer(SpellCorrector(vocab=vocab))


def model_path(key: str, d: Path = MODEL_DIR) -> Path:
    return d / key


def is_trained(key: str, d: Path = MODEL_DIR) -> bool:
    p = model_path(key, d)
    marker = {"tfidf_lr": "tfidf_lr.pkl", "w2v_ffnn": "ffnn.npz", "student_int8": "model_int8.pt"}
    return (p / marker.get(key, "config.json")).exists()


def load_model(key: str, labels, d: Path = MODEL_DIR, dev: str = "cpu"):
    p = model_path(key, d)
    if key == "tfidf_lr":
        from src.classical import TfidfLR
        m = TfidfLR.load(p)
        m.files = lambda: TfidfLR.files(p)
    elif key == "w2v_ffnn":
        from src.classical import W2VFFNN
        m = W2VFFNN.load(p)
        m.files = lambda: W2VFFNN.files(p)
    else:
        from src.transformer import TransformerClassifier
        m = TransformerClassifier(key, p, labels, int8=(key == "student_int8"), dev=dev)
    m.key = key
    return m


def load_available(labels, keys=MODEL_KEYS, d: Path = MODEL_DIR, dev="cpu"):
    out = {}
    for k in keys:
        if is_trained(k, d):
            out[k] = load_model(k, labels, d, dev)
        else:
            print(f"[models] '{k}' not trained/downloaded yet - skipped")
    return out


def files_size_mb(files) -> float:
    return sum(Path(f).stat().st_size for f in files) / 2 ** 20
