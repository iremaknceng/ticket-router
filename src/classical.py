"""The two non-Transformer rungs of the model ladder, behind one interface:
`predict_proba(list_of_normalized_texts) -> (N, n_labels) array`, `labels`, `save/load`.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from src.ffnn import FFNN
from src.preprocess import tokenize
from src.word2vec import Word2Vec


class TfidfLR:
    """Area B baseline: word 1-2-gram + char 2-5-gram tf-idf -> multinomial logistic regression."""
    key = "tfidf_lr"

    def __init__(self, labels, C=10.0):
        self.labels = labels
        feats = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", tokenizer=tokenize, token_pattern=None,
                                     ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                                     min_df=2, max_features=100_000)),
        ])
        self.pipe = make_pipeline(feats, LogisticRegression(C=C, max_iter=3000))

    def fit(self, texts, labels):
        self.pipe.fit(texts, labels)
        # reorder probability columns to the shared label order
        cls = list(self.pipe.classes_)
        self._order = [cls.index(l) if l in cls else None for l in self.labels]
        return self

    def predict_proba(self, texts):
        p = self.pipe.predict_proba(texts)
        out = np.zeros((len(texts), len(self.labels)), np.float32)
        for j, k in enumerate(self._order):
            if k is not None:
                out[:, j] = p[:, k]
        return out

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "tfidf_lr.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(d: Path):
        with open(d / "tfidf_lr.pkl", "rb") as f:
            return pickle.load(f)

    @staticmethod
    def files(d: Path):
        return [d / "tfidf_lr.pkl"]


class W2VFFNN:
    """Area A: averaged own-trained word2vec vectors -> NumPy FFNN (SGD + backprop)."""
    key = "w2v_ffnn"

    def __init__(self, labels, w2v_kwargs=None, ffnn_kwargs=None):
        self.labels = labels
        self.lab2id = {l: i for i, l in enumerate(labels)}
        self.w2v_kwargs = w2v_kwargs or {}
        self.ffnn_kwargs = ffnn_kwargs or {}
        self.w2v = self.net = None

    def features(self, texts):
        return np.stack([self.w2v.sentence_vector(tokenize(t)) for t in texts])

    def fit(self, texts, labels, val_texts=None, val_labels=None, extra_corpus=()):
        sents = [tokenize(t) for t in list(texts) + list(extra_corpus)]
        print("Training word2vec (skip-gram, negative sampling) ...")
        self.w2v = Word2Vec(**self.w2v_kwargs).fit(sents)
        X = self.features(texts)
        y = np.array([self.lab2id[l] for l in labels])
        Xv = yv = None
        if val_texts is not None:
            Xv = self.features(val_texts)
            yv = np.array([self.lab2id.get(l, 0) for l in val_labels])
        print("Training feedforward network with SGD + backpropagation ...")
        self.net = FFNN(X.shape[1], len(self.labels), **self.ffnn_kwargs).fit(X, y, Xv, yv)
        return self

    def predict_proba(self, texts):
        return self.net.predict_proba(self.features(texts))

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        self.w2v.save(d / "word2vec.npz")
        self.net.save(d / "ffnn.npz")
        (d / "labels.json").write_text(json.dumps(self.labels, ensure_ascii=False))

    @classmethod
    def load(cls, d: Path):
        m = cls(json.loads((d / "labels.json").read_text()))
        m.w2v = Word2Vec.load(d / "word2vec.npz")
        m.net = FFNN.load(d / "ffnn.npz")
        return m

    @staticmethod
    def files(d: Path):
        return [d / "word2vec.npz", d / "ffnn.npz"]
