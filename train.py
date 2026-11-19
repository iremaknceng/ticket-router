"""Train the whole model ladder.

    python train.py                 # everything (GPU recommended for the Transformer stages)
    python train.py --stage classical
    python train.py --stage teacher|student|quantize
    python train.py --smoke         # tiny appliance-only run to test the code (no download)
"""
import argparse
import json
import time

from config import MODEL_DIR
from src.data import label_list, load_splits
from src.models import build_vocab, load_preprocessing, model_path, save_preprocessing

STAGES = ["classical", "teacher", "student", "quantize"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES + ["all", "transformer"], default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher-epochs", type=int, default=4)
    ap.add_argument("--student-epochs", type=int, default=5)
    ap.add_argument("--appliance-oversample", type=int, default=10,
                    help="repeat each hand-written appliance TRAIN example k times (class balance)")
    args = ap.parse_args()
    stages = STAGES if args.stage == "all" else STAGES[1:] if args.stage == "transformer" else [args.stage]

    splits = load_splits(smoke=args.smoke)
    for k, s in splits.items():
        print(f"{k:>5}: {len(s):>6} messages")
    labels = label_list(splits["train"])
    save_preprocessing(labels, build_vocab(splits["train"].texts))
    _, normalizer = load_preprocessing()
    norm = {k: [normalizer(t).text for t in s.texts] for k, s in splits.items()}
    for k, s in splits.items():          # normalized copies used by every model
        s.texts = norm[k]
    train, val = splits["train"], splits["val"]
    if args.appliance_oversample > 1:
        # 9 training examples per appliance intent vs ~100 per CLINC intent -> repeat them so
        # every model sees a balanced number. Only the TRAIN split is changed; val/test are not.
        extra = [(t, l, s) for t, l, s in zip(train.texts, train.labels, train.source) if s == "appliance"]
        for _ in range(args.appliance_oversample - 1):
            for t, l, s in extra:
                train.texts.append(t); train.labels.append(l); train.source.append(s)
        print(f"Oversampled {len(extra)} appliance training examples x{args.appliance_oversample} "
              f"-> train size {len(train)}")
    log = {}

    if "classical" in stages:
        from src.classical import TfidfLR, W2VFFNN
        t0 = time.time()
        TfidfLR(labels).fit(train.texts, train.labels).save(model_path("tfidf_lr"))
        log["tfidf_lr_train_s"] = round(time.time() - t0, 1)
        t0 = time.time()
        kw = dict(epochs=3, hidden=64) if args.smoke else {}
        W2VFFNN(labels, w2v_kwargs=dict(epochs=2 if args.smoke else 10),
                ffnn_kwargs=kw).fit(train.texts, train.labels, val.texts, val.labels) \
            .save(model_path("w2v_ffnn"))
        log["w2v_ffnn_train_s"] = round(time.time() - t0, 1)

    if any(s in stages for s in ["teacher", "student", "quantize"]):
        from src import transformer as T
        if args.smoke:
            T.MAX_LEN = 32
        if "teacher" in stages:
            t0 = time.time()
            T.finetune_teacher(train, val, labels, model_path("bert_teacher"),
                               epochs=1 if args.smoke else args.teacher_epochs)
            log["teacher_train_s"] = round(time.time() - t0, 1)
        if "student" in stages:
            t0 = time.time()
            T.distill_student(train, val, labels, model_path("bert_teacher"), model_path("student_fp32"),
                              epochs=1 if args.smoke else args.student_epochs)
            log["student_train_s"] = round(time.time() - t0, 1)
        if "quantize" in stages:
            T.quantize_student(model_path("student_fp32"), model_path("student_int8"))

    log_path = MODEL_DIR / "train_log.json"
    old = json.loads(log_path.read_text()) if log_path.exists() else {}
    log_path.write_text(json.dumps({**old, **log}, indent=2))
    print("Done:", log)


if __name__ == "__main__":
    main()
