"""Areas C + D - Transformer rungs of the ladder.

C: fine-tune multilingual BERT (WordPiece tokenizer, 12 self-attention layers) with a
   sequence-classification head  ->  "teacher".
D: knowledge distillation into a 6-layer student initialised from every 2nd teacher
   layer (DistilBERT-style), trained on  alpha * CE(hard labels)
   + (1 - alpha) * T^2 * KL(softmax(teacher/T) || softmax(student/T)).
D: post-training dynamic INT8 quantization of the student (Linear weights int8 with
   int8 activations computed on the fly; embedding tables int8 weight-only).
"""
from __future__ import annotations

import copy
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from config import MAX_LEN, SEED, STUDENT_LAYERS, TEACHER_NAME


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batches(tok, texts, labels=None, bs=32, shuffle=False):
    idx = list(range(len(texts)))
    if shuffle:
        random.shuffle(idx)
    for b in range(0, len(idx), bs):
        sel = idx[b:b + bs]
        enc = tok([texts[i] for i in sel], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")
        if labels is not None:
            enc["labels"] = torch.tensor([labels[i] for i in sel])
        yield enc


@torch.no_grad()
def predict_logits(model, tok, texts, bs=64):
    dev = next(model.parameters()).device
    model.eval()
    out = []
    for enc in _batches(tok, texts, bs=bs):
        enc = {k: v.to(dev) for k, v in enc.items()}
        out.append(model(**enc).logits.float().cpu())
    return torch.cat(out).numpy()


def _accuracy(model, tok, texts, y):
    return float((predict_logits(model, tok, texts).argmax(1) == np.array(y)).mean())


def _train_loop(model, tok, texts, y, val_texts, val_y, epochs, lr, bs, loss_fn, out_dir, tag):
    dev = device()
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(texts) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best = -1.0
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for enc in _batches(tok, texts, y, bs=bs, shuffle=True):
            enc = {k: v.to(dev) for k, v in enc.items()}
            labels = enc.pop("labels")
            logits = model(**enc).logits
            loss = loss_fn(logits, labels, enc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item() * len(labels); n += len(labels)
        acc = _accuracy(model, tok, val_texts, val_y)
        print(f"  [{tag}] epoch {ep + 1}/{epochs} loss={tot / n:.4f} val_acc={acc:.4f} ({time.time() - t0:.0f}s)")
        if acc > best:
            best = acc
            model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    return best


def _label_maps(labels):
    return {i: l for i, l in enumerate(labels)}, {l: i for i, l in enumerate(labels)}


def finetune_teacher(train, val, labels, out_dir: Path, epochs=4, lr=3e-5, bs=32):
    set_seed()
    id2label, label2id = _label_maps(labels)
    tok = AutoTokenizer.from_pretrained(TEACHER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id)
    y = [label2id[l] for l in train.labels]
    vy = [label2id.get(l, 0) for l in val.labels]
    ce = lambda logits, labels, enc: F.cross_entropy(logits, labels)
    best = _train_loop(model, tok, train.texts, y, val.texts, vy, epochs, lr, bs, ce, out_dir, "teacher")
    print(f"Teacher saved to {out_dir} (best val acc {best:.4f})")


def make_student(teacher, n_layers=STUDENT_LAYERS):
    """Copy embeddings, classifier and every k-th encoder layer of the teacher."""
    cfg = copy.deepcopy(teacher.config)
    n_teacher = cfg.num_hidden_layers
    keep = [round(i * (n_teacher - 1) / (n_layers - 1)) for i in range(n_layers)] if n_layers > 1 else [0]
    keep = sorted(set(keep))
    cfg.num_hidden_layers = len(keep)
    student = AutoModelForSequenceClassification.from_config(cfg)
    prefix = teacher.base_model_prefix          # "bert"
    pat = re.compile(rf"^{prefix}\.encoder\.layer\.(\d+)\.(.*)$")
    new_sd = {}
    for k, v in teacher.state_dict().items():
        m = pat.match(k)
        if m:
            i = int(m.group(1))
            if i in keep:
                new_sd[f"{prefix}.encoder.layer.{keep.index(i)}.{m.group(2)}"] = v.clone()
        else:
            new_sd[k] = v.clone()
    missing, unexpected = student.load_state_dict(new_sd, strict=False)
    missing = [k for k in missing if not k.endswith("position_ids")]
    assert not missing and not unexpected, (missing, unexpected)
    print(f"Student initialised from teacher layers {keep}")
    return student


def distill_student(train, val, labels, teacher_dir: Path, out_dir: Path,
                    epochs=5, lr=5e-5, bs=32, T=2.0, alpha=0.5):
    set_seed()
    dev = device()
    tok = AutoTokenizer.from_pretrained(teacher_dir)
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_dir).to(dev).eval()
    student = make_student(teacher)
    _, label2id = _label_maps(labels)
    y = [label2id[l] for l in train.labels]
    vy = [label2id.get(l, 0) for l in val.labels]

    def kd_loss(s_logits, labels, enc):
        with torch.no_grad():
            t_logits = teacher(**enc).logits
        hard = F.cross_entropy(s_logits, labels)
        soft = F.kl_div(F.log_softmax(s_logits / T, -1), F.softmax(t_logits / T, -1),
                        reduction="batchmean") * T * T
        return alpha * hard + (1 - alpha) * soft

    best = _train_loop(student, tok, train.texts, y, val.texts, vy, epochs, lr, bs, kd_loss, out_dir, "student")
    print(f"Student saved to {out_dir} (best val acc {best:.4f})")


# --------------------------------------------------------------------- quantization
def quantize_dynamic_int8(model, embeddings=True):
    """Dynamic INT8 post-training quantization (CPU, fbgemm/qnnpack kernels)."""
    from torch.ao import quantization as tq
    model = model.cpu().eval()
    spec = {nn.Linear: tq.default_dynamic_qconfig}
    if embeddings:
        spec[nn.Embedding] = tq.float_qparams_weight_only_qconfig
    return tq.quantize_dynamic(copy.deepcopy(model), spec, dtype=torch.qint8)


def quantize_student(student_dir: Path, out_dir: Path):
    tok = AutoTokenizer.from_pretrained(student_dir)
    fp32 = AutoModelForSequenceClassification.from_pretrained(student_dir).eval()
    enc = tok(["test message"], return_tensors="pt")
    meta = {"method": "torch dynamic quantization, qint8", "embeddings_int8": True}
    try:
        q = quantize_dynamic_int8(fp32, embeddings=True)
        q(**enc)
    except Exception as e:   # some torch versions cannot run int8 embeddings in BERT
        print(f"  int8 embeddings not supported here ({e}); quantizing Linear layers only")
        q = quantize_dynamic_int8(fp32, embeddings=False)
        meta["embeddings_int8"] = False
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(q.state_dict(), out_dir / "model_int8.pt")
    fp32.config.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "quant_meta.json").write_text(json.dumps(meta))
    print(f"Quantized student saved to {out_dir}: {meta}")


# ----------------------------------------------------------------- inference wrapper
class TransformerClassifier:
    """Same interface as the classical models: predict_proba(texts) -> (N, n_labels)."""

    def __init__(self, key, model_dir: Path, labels, int8=False, dev="cpu"):
        from transformers import AutoConfig
        self.key, self.labels, self.dir = key, labels, Path(model_dir)
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        if int8:
            meta = json.loads((self.dir / "quant_meta.json").read_text())
            cfg = AutoConfig.from_pretrained(self.dir)
            base = AutoModelForSequenceClassification.from_config(cfg).eval()
            self.model = quantize_dynamic_int8(base, embeddings=meta["embeddings_int8"])
            sd = torch.load(self.dir / "model_int8.pt", map_location="cpu", weights_only=False)
            self.model.load_state_dict(sd)
            dev = "cpu"                                   # int8 kernels are CPU-only
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.dir)
        self.dev = torch.device(dev)
        self.model.to(self.dev).eval()
        id2label = self.model.config.id2label
        model_labels = [id2label[i] for i in range(len(id2label))]
        assert model_labels == list(labels), "label order mismatch between model and data"

    @torch.no_grad()
    def predict_proba(self, texts, bs=64):
        out = []
        for enc in _batches(self.tok, list(texts), bs=bs):
            enc = {k: v.to(self.dev) for k, v in enc.items()}
            out.append(torch.softmax(self.model(**enc).logits.float(), -1).cpu())
        return torch.cat(out).numpy()

    def to(self, dev):
        if not (self.dir / "model_int8.pt").exists():
            self.dev = torch.device(dev)
            self.model.to(self.dev)
        return self

    def files(self):
        if (self.dir / "model_int8.pt").exists():
            return [self.dir / "model_int8.pt"]
        return list(self.dir.glob("*.safetensors")) + list(self.dir.glob("pytorch_model*.bin"))
