"""Area B/A - our own word2vec: skip-gram with negative sampling (SGNS), NumPy only.

Trained on the training messages. For every (center, context) pair inside a dynamic
window we maximise log σ(c·o) + Σ_k log σ(-c·n_k) with k negatives drawn from the
unigram^0.75 distribution, using mini-batch SGD with a linearly decaying learning rate.
Frequent words are sub-sampled (Mikolov et al., 2013).
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class Word2Vec:
    def __init__(self, dim=100, window=4, negative=5, min_count=2, epochs=10, lr=0.05,
                 batch=256, subsample=1e-3, seed=42):
        self.dim, self.window, self.negative = dim, window, negative
        self.min_count, self.epochs, self.lr, self.batch = min_count, epochs, lr, batch
        self.subsample, self.seed = subsample, seed
        self.vocab: dict[str, int] = {}
        self.W_in = self.W_out = None

    # -------------------------------------------------------------- training
    def _pairs(self, corpus, rng, keep_prob):
        centers, contexts = [], []
        for sent in corpus:
            ids = [i for i in sent if rng.random() < keep_prob[i]]
            for pos, c in enumerate(ids):
                w = rng.integers(1, self.window + 1)          # dynamic window
                for j in range(max(0, pos - w), min(len(ids), pos + w + 1)):
                    if j != pos:
                        centers.append(c); contexts.append(ids[j])
        return np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)

    def fit(self, sentences: list[list[str]], verbose=True):
        rng = np.random.default_rng(self.seed)
        counts = Counter(w for s in sentences for w in s)
        words = [w for w, c in counts.most_common() if c >= self.min_count]
        self.vocab = {w: i for i, w in enumerate(words)}
        V = len(words)
        freq = np.array([counts[w] for w in words], dtype=np.float64)
        noise = freq ** 0.75
        noise /= noise.sum()
        f = freq / freq.sum()
        keep_prob = np.minimum(1.0, np.sqrt(self.subsample / f) + self.subsample / f)
        corpus = [[self.vocab[w] for w in s if w in self.vocab] for s in sentences]

        self.W_in = (rng.random((V, self.dim)) - 0.5) / self.dim
        self.W_out = np.zeros((V, self.dim))
        total_steps = None
        step = 0
        for ep in range(self.epochs):
            cen, ctx = self._pairs(corpus, rng, keep_prob)
            order = rng.permutation(len(cen))
            cen, ctx = cen[order], ctx[order]
            if total_steps is None:
                total_steps = self.epochs * (len(cen) // self.batch + 1)
            losses = []
            for b in range(0, len(cen), self.batch):
                lr = max(self.lr * (1 - step / total_steps), self.lr * 1e-3)
                step += 1
                c_ids, o_ids = cen[b:b + self.batch], ctx[b:b + self.batch]
                n_ids = rng.choice(V, size=(len(c_ids), self.negative), p=noise)
                c = self.W_in[c_ids]                     # (B, d)
                o = self.W_out[o_ids]                    # (B, d)
                n = self.W_out[n_ids]                    # (B, k, d)
                s_pos = _sigmoid(np.sum(c * o, axis=1))              # σ(c·o)
                s_neg = _sigmoid(np.einsum("bkd,bd->bk", n, c))      # σ(c·n)
                losses.append(-np.mean(np.log(s_pos + 1e-9) + np.log(1 - s_neg + 1e-9).sum(1)))
                # backprop of the SGNS loss
                g_pos = (s_pos - 1.0)[:, None]                       # dL/d(c·o)
                g_neg = s_neg[:, :, None]                            # dL/d(c·n)
                grad_c = g_pos * o + np.sum(g_neg * n, axis=1)
                grad_o = g_pos * c
                grad_n = g_neg * c[:, None, :]
                np.add.at(self.W_in, c_ids, -lr * grad_c)
                np.add.at(self.W_out, o_ids, -lr * grad_o)
                np.add.at(self.W_out, n_ids.reshape(-1), -lr * grad_n.reshape(-1, self.dim))
            if verbose:
                print(f"  word2vec epoch {ep + 1}/{self.epochs}  pairs={len(cen)}  loss={np.mean(losses):.4f}")
        return self

    # -------------------------------------------------------------- inference
    def sentence_vector(self, tokens: list[str]) -> np.ndarray:
        ids = [self.vocab[t] for t in tokens if t in self.vocab]
        if not ids:
            return np.zeros(self.dim, dtype=np.float32)
        return self.W_in[ids].mean(axis=0).astype(np.float32)

    def most_similar(self, word: str, topn=5):
        if word not in self.vocab:
            return []
        E = self.W_in / (np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-9)
        sims = E @ E[self.vocab[word]]
        inv = {i: w for w, i in self.vocab.items()}
        return [(inv[i], float(sims[i])) for i in np.argsort(-sims)[1:topn + 1]]

    def save(self, path):
        words = sorted(self.vocab, key=self.vocab.get)
        np.savez_compressed(path, W_in=self.W_in.astype(np.float32), words=np.array(words))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        m = cls(dim=z["W_in"].shape[1])
        m.W_in = z["W_in"]
        m.vocab = {w: i for i, w in enumerate(z["words"].tolist())}
        return m
