"""Evaluate every trained model on the SAME held-out test set and build the results table.

    python evaluate.py [--smoke] [--runs 200]

For each model:
  * the out-of-scope confidence threshold is tuned on the validation split
  * test metrics: accuracy (all 157 classes incl. oos), in-scope accuracy, OOS recall
  * size on disk (MB) of the weight files the application loads
  * CPU latency per message (batch size 1, mean over --runs >= 100 messages, after warm-up)
Outputs: results/results.json, results/results.md, results/thresholds.json,
         results/errors_<model>.csv, results/hardware.json
"""
import argparse
import csv
import json
import os
import platform
import time

import numpy as np

from config import MODEL_DIR, MODEL_KEYS, MODEL_TITLES, OOS, RESULTS_DIR
from src.data import load_splits
from src.models import files_size_mb, load_available, load_preprocessing
from src.router import decide


def hardware_info():
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    info = {"cpu": cpu, "logical_cores": os.cpu_count(), "python": platform.python_version(),
            "os": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["gpu_used_for_training"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


def metrics(pred, gold):
    pred, gold = np.array(pred), np.array(gold)
    ins, oos = gold != OOS, gold == OOS
    return {
        "accuracy": float((pred == gold).mean()),
        "in_scope_accuracy": float((pred[ins] == gold[ins]).mean()) if ins.any() else None,
        "oos_recall": float((pred[oos] == OOS).mean()) if oos.any() else None,
    }


def apply_threshold(P, labels, th):
    return [decide(p, labels, th)[0] for p in P]


def tune_threshold(P, labels, gold):
    best_th, best_score = 0.0, -1
    for th in np.round(np.arange(0.0, 0.96, 0.05), 2):
        m = metrics(apply_threshold(P, labels, th), gold)
        score = (m["in_scope_accuracy"] or 0) + (m["oos_recall"] if m["oos_recall"] is not None else 0)
        if score > best_score:
            best_th, best_score = float(th), score
    return best_th


def latency_ms(model, texts, runs):
    sample = [texts[i % len(texts)] for i in range(runs)]
    for t in sample[:10]:                        # warm-up
        model.predict_proba([t])
    times = []
    for t in sample:
        t0 = time.perf_counter()
        model.predict_proba([t])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def fmt(x, pct=True):
    return "-" if x is None else (f"{100 * x:.2f}%" if pct else f"{x:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs", type=int, default=200)
    args = ap.parse_args()
    assert args.runs >= 100, "homework requires latency averaged over >= 100 runs"
    RESULTS_DIR.mkdir(exist_ok=True)

    labels, normalizer = load_preprocessing()
    splits = load_splits(smoke=args.smoke)
    raw_test = list(splits["test"].texts)
    val_x = [normalizer(t).text for t in splits["val"].texts]
    test_x = [normalizer(t).text for t in raw_test]
    val_y, test_y = splits["val"].labels, splits["test"].labels
    appl_idx = [i for i, s in enumerate(splits["test"].source) if s == "appliance"]

    try:
        import torch
        gpu = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        gpu = "cpu"
    models = load_available(labels, MODEL_KEYS)
    results, thresholds = {}, {}
    for key, model in models.items():
        print(f"\n=== {MODEL_TITLES[key]} ===")
        is_fp32_tf = key in ("bert_teacher", "student_fp32")
        if is_fp32_tf:
            model.to(gpu)                         # fast batched scoring; latency is measured on CPU
        Pv, Pt = model.predict_proba(val_x), model.predict_proba(test_x)
        th = tune_threshold(Pv, labels, val_y)
        pred = apply_threshold(Pt, labels, th)
        m = metrics(pred, test_y)
        m["appliance_accuracy"] = float(np.mean([pred[i] == test_y[i] for i in appl_idx])) if appl_idx else None
        if is_fp32_tf:
            model.to("cpu")
        m["latency_ms_mean"], m["latency_ms_p95"] = latency_ms(model, test_x, args.runs)
        m["size_mb"] = files_size_mb(model.files())
        m["threshold"] = th
        results[key], thresholds[key] = m, th
        print(json.dumps(m, indent=2))
        with open(RESULTS_DIR / f"errors_{key}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["raw_text", "normalized", "gold", "pred", "confidence"])
            for i, (p, g) in enumerate(zip(pred, test_y)):
                if p != g:
                    w.writerow([raw_test[i], test_x[i], g, p, f"{Pt[i].max():.3f}"])

    hw = hardware_info()
    hw["latency_runs"] = args.runs
    hw["test_size"] = len(test_y)
    hw["test_oos"] = int(sum(y == OOS for y in test_y))
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    (RESULTS_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (MODEL_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))  # shipped with weights
    (RESULTS_DIR / "hardware.json").write_text(json.dumps(hw, indent=2))

    lines = [f"Test set: {len(test_y)} messages ({hw['test_oos']} out-of-scope). "
             f"Latency: CPU, batch 1, mean of {args.runs} runs on {hw['cpu']} "
             f"({hw['logical_cores']} logical cores).", "",
             "| Model | Accuracy | In-scope acc. | OOS recall | Appliance acc. | Size (MB) | Latency mean (ms) | p95 (ms) | Threshold |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, m in results.items():
        lines.append(f"| {MODEL_TITLES[key]} | {fmt(m['accuracy'])} | {fmt(m['in_scope_accuracy'])} | "
                     f"{fmt(m['oos_recall'])} | {fmt(m['appliance_accuracy'])} | {m['size_mb']:.1f} | "
                     f"{m['latency_ms_mean']:.2f} | {m['latency_ms_p95']:.2f} | {m['threshold']:.2f} |")
    table = "\n".join(lines)
    (RESULTS_DIR / "results.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
