"""Area A - feedforward neural network written from scratch in NumPy.

input (averaged word2vec, standardized) -> Linear -> ReLU -> Dropout -> Linear -> Softmax
Trained with mini-batch SGD + momentum and hand-written backpropagation of the
cross-entropy loss. The epoch with the best validation accuracy is kept (early stopping).
"""
from __future__ import annotations

import numpy as np


class FFNN:
    def __init__(self, n_in, n_classes, hidden=512, dropout=0.3, lr=0.05, momentum=0.9,
                 weight_decay=1e-4, epochs=60, batch=64, seed=42):
        self.rng = np.random.default_rng(seed)
        self.hp = dict(hidden=hidden, dropout=dropout, lr=lr, momentum=momentum,
                       weight_decay=weight_decay, epochs=epochs, batch=batch)
        # He initialisation for the ReLU layer
        self.W1 = self.rng.normal(0, np.sqrt(2 / n_in), (n_in, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.W2 = self.rng.normal(0, np.sqrt(1 / hidden), (hidden, n_classes)).astype(np.float32)
        self.b2 = np.zeros(n_classes, np.float32)
        self.mu = np.zeros(n_in, np.float32)
        self.sd = np.ones(n_in, np.float32)

    @property
    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    # ------------------------------------------------------------------ forward
    def _forward(self, X, train=False):
        X = (X - self.mu) / self.sd
        z1 = X @ self.W1 + self.b1
        h = np.maximum(z1, 0)
        mask = None
        if train and self.hp["dropout"] > 0:     # inverted dropout
            keep = 1 - self.hp["dropout"]
            mask = (self.rng.random(h.shape) < keep).astype(np.float32) / keep
            h = h * mask
        logits = h @ self.W2 + self.b2
        return X, z1, h, mask, logits

    @staticmethod
    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        return self.softmax(self._forward(np.asarray(X, np.float32))[-1])

    # ------------------------------------------------------------------ training
    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        X = np.asarray(X, np.float32); y = np.asarray(y)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        hp = self.hp
        vel = [np.zeros_like(p) for p in self.params]
        best_acc, best = -1, None
        n = len(X)
        for ep in range(hp["epochs"]):
            lr = hp["lr"] * 0.5 * (1 + np.cos(np.pi * ep / hp["epochs"]))   # cosine decay
            order = self.rng.permutation(n)
            loss_sum = 0.0
            for b in range(0, n, hp["batch"]):
                idx = order[b:b + hp["batch"]]
                Xb, yb = X[idx], y[idx]
                Xn, z1, h, mask, logits = self._forward(Xb, train=True)
                p = self.softmax(logits)
                loss_sum += -np.log(p[np.arange(len(yb)), yb] + 1e-9).sum()
                # ---- backpropagation
                d_logits = p.copy()
                d_logits[np.arange(len(yb)), yb] -= 1
                d_logits /= len(yb)                       # dL/dlogits for mean CE
                dW2 = h.T @ d_logits + hp["weight_decay"] * self.W2
                db2 = d_logits.sum(0)
                dh = d_logits @ self.W2.T
                if mask is not None:
                    dh *= mask
                dz1 = dh * (z1 > 0)                       # ReLU derivative
                dW1 = Xn.T @ dz1 + hp["weight_decay"] * self.W1
                db1 = dz1.sum(0)
                # ---- SGD with momentum
                for i, (prm, g) in enumerate(zip(self.params, [dW1, db1, dW2, db2])):
                    vel[i] = hp["momentum"] * vel[i] - lr * g
                    prm += vel[i]
            msg = f"  ffnn epoch {ep + 1:>3}/{hp['epochs']}  train_loss={loss_sum / n:.4f}"
            if X_val is not None:
                acc = float((self.predict_proba(X_val).argmax(1) == np.asarray(y_val)).mean())
                msg += f"  val_acc={acc:.4f}"
                if acc > best_acc:
                    best_acc, best = acc, [p.copy() for p in self.params]
            if verbose and (ep % 5 == 4 or ep == hp["epochs"] - 1):
                print(msg)
        if best is not None:
            self.W1, self.b1, self.W2, self.b2 = best
        return self

    def save(self, path):
        np.savez_compressed(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, mu=self.mu, sd=self.sd)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls(z["W1"].shape[0], z["W2"].shape[1], hidden=z["W1"].shape[1])
        m.W1, m.b1, m.W2, m.b2, m.mu, m.sd = (z[k] for k in ["W1", "b1", "W2", "b2", "mu", "sd"])
        return m
