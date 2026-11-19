"""The application pipeline: normalize -> small-talk rules -> classify -> threshold -> team."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace

import numpy as np

from config import AUTO_REPLY, HUMAN_AGENT, MODEL_DIR, MODEL_KEYS, OOS, RESULTS_DIR, domain_for, team_for
from src.eliza import Eliza
from src.models import load_available, load_preprocessing

DEFAULT_THRESHOLD = 0.5
PRIMARY = "student_int8"          # the application runs the quantized Transformer (R5)
SMALL_TALK_FALLBACK = ("I'm the automatic support assistant. Please describe the problem with "
                       "your appliance. / Cihazınızdaki sorunu yazarsanız yardımcı olabilirim.")


@dataclass
class Prediction:
    model: str
    intent: str
    confidence: float
    team: str
    latency_ms: float


@dataclass
class RouteResult:
    normalized: str
    model_codes: list
    corrections: list
    small_talk_reply: str | None
    primary: Prediction | None
    all_models: list = field(default_factory=list)


def decide(probs: np.ndarray, labels, threshold: float):
    """argmax intent; out-of-scope if the model says 'oos' or is not confident enough."""
    j = int(np.argmax(probs))
    conf = float(probs[j])
    intent = labels[j]
    if intent == OOS or conf < threshold:
        return OOS, conf
    return intent, conf


class Router:
    def __init__(self, keys=MODEL_KEYS, model_dir=MODEL_DIR):
        self.labels, self.normalizer = load_preprocessing(model_dir)
        self.models = load_available(self.labels, keys, model_dir)
        self.eliza = Eliza()
        tpath = next((p for p in [model_dir / "thresholds.json", RESULTS_DIR / "thresholds.json"]
                      if p.exists()), None)
        self.thresholds = json.loads(tpath.read_text()) if tpath else {}

    def threshold(self, key):
        return self.thresholds.get(key, DEFAULT_THRESHOLD)

    def predict_one(self, key, text) -> Prediction:
        t0 = time.perf_counter()
        probs = self.models[key].predict_proba([text])[0]
        ms = (time.perf_counter() - t0) * 1000
        intent, conf = decide(probs, self.labels, self.threshold(key))
        return Prediction(key, intent, conf, team_for(intent), ms)

    def route(self, raw: str) -> RouteResult:
        norm = self.normalizer(raw)
        rule = self.eliza.respond(norm.text)
        reply = rule.reply if rule else None
        preds = [self.predict_one(k, norm.text) for k in self.models]
        primary = next((p for p in preds if p.model == PRIMARY), preds[-1] if preds else None)
        if reply is None and primary is not None and domain_for(primary.intent) == "small_talk":
            # the classifier recognised small talk that no rule matched -> generic rule reply
            reply = SMALL_TALK_FALLBACK
        if reply is not None and primary is not None:
            primary = replace(primary, intent=f"small talk ({rule.rule if rule else primary.intent})",
                              team=AUTO_REPLY)
        return RouteResult(norm.text, norm.model_codes, norm.corrections, reply, primary, preds)
