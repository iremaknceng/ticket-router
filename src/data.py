"""Dataset loading: CLINC150 (oos_plus) + hand-written TR/EN home-appliance examples.

Splits are fixed: CLINC's own train/val/test (+ oos_train/oos_val/oos_test) and the
`split` column of data/appliance_intents.csv. Every model is trained on `train`,
tunes its out-of-scope confidence threshold on `val`, and is scored on the SAME `test`.
"""
from __future__ import annotations

import csv
import json
import urllib.request
from dataclasses import dataclass

from config import CLINC_URL, DATA_DIR, OOS, RAW_DIR


@dataclass
class Split:
    texts: list[str]
    labels: list[str]
    source: list[str]   # "clinc" or "appliance"

    def __len__(self):
        return len(self.texts)


def _load_clinc() -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / "data_oos_plus.json"
    if not path.exists():
        print(f"Downloading CLINC150 from {CLINC_URL}")
        try:
            urllib.request.urlretrieve(CLINC_URL, path)
        except Exception as e:  # fall back to the Hugging Face copy
            print(f"  GitHub download failed ({e}); trying Hugging Face datasets ...")
            return _load_clinc_hf()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_clinc_hf() -> dict:
    from datasets import load_dataset
    ds = load_dataset("clinc/clinc_oos", "plus")
    names = ds["train"].features["intent"].names
    out = {}
    for split, key in [("train", "train"), ("validation", "val"), ("test", "test")]:
        rows = [[r["text"], names[r["intent"]]] for r in ds[split]]
        out[key] = [r for r in rows if r[1] != OOS]
        out["oos_" + key] = [[t, OOS] for t, l in rows if l == OOS]
    return out


def _load_appliance() -> dict[str, list[list[str]]]:
    out = {"train": [], "val": [], "test": []}
    with open(DATA_DIR / "appliance_intents.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["split"]].append([row["text"], row["intent"]])
    return out


def load_splits(smoke: bool = False) -> dict[str, Split]:
    """smoke=True uses only the small appliance file (no download) for quick tests."""
    appl = _load_appliance()
    clinc = {} if smoke else _load_clinc()
    splits = {}
    for name in ["train", "val", "test"]:
        rows, src = [], []
        for t, l in clinc.get(name, []) + clinc.get("oos_" + name, []):
            rows.append((t, OOS if l == "oos" else l)); src.append("clinc")
        for t, l in appl[name]:
            rows.append((t, l)); src.append("appliance")
        splits[name] = Split([r[0] for r in rows], [r[1] for r in rows], src)
    return splits


def label_list(train: Split) -> list[str]:
    return sorted(set(train.labels))
