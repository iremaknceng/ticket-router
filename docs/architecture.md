# Architecture

```mermaid
flowchart LR
    U["User types a message<br/>(Gradio web UI, app.py)"] --> N

    subgraph PRE["Preprocessing - src/preprocess.py (Area B)"]
        N["Regex normalization<br/>Turkish-aware lowercase, URL/e-mail/phone,<br/>repeated letters"] --> MC["Regex model-code extraction<br/>DW-6534 → modelcode"]
        MC --> ED["Min. edit distance correction<br/>of product names (TR+EN lexicon)"]
    end

    ED --> EL{"ELIZA rules<br/>src/eliza.py (B)<br/>pure small talk?"}
    EL -- yes --> R1["Rule-based reply<br/>team: Auto-reply"]
    EL -- no --> M

    subgraph M["Model ladder - all on CPU in the app"]
        M1["TF-IDF + LogReg<br/>scikit-learn, FP64 (B)"]
        M2["own word2vec (SGNS) → avg → FFNN<br/>NumPy, FP32, SGD + backprop (A)"]
        M3["mBERT teacher, 12 layers<br/>FP32 (C)"]
        M4["Distilled student, 6 layers<br/>FP32 (C + D: distillation)"]
        M5["Distilled student, 6 layers<br/>INT8 dynamic quantization (D)<br/>★ primary model"]
    end

    M5 --> TH{"max softmax ≥ threshold<br/>and intent ≠ oos?<br/>(threshold tuned on val)"}
    TH -- no --> HA["Human agent"]
    TH -- yes --> TM["intent → team table<br/>config.py"]
    M1 & M2 & M3 & M4 --> SBS["Side-by-side table<br/>intent, confidence, team, latency"]
    TM --> OUT["UI: routed team, intent, confidence,<br/>normalized text, model codes, corrections"]
    HA --> OUT
    R1 --> OUT
    SBS --> OUT
```

**Where it runs.** Training: Google Colab (T4 GPU for the Transformers, CPU for the rest).
Inference / app / latency benchmark: CPU only (Colab CPU or a laptop). Weights are downloaded
from the Hugging Face Hub into `models/` on first start.

| Component | File | Model / precision |
|---|---|---|
| Normalization, model codes, edit distance | `src/preprocess.py` | rules |
| Small-talk replies | `src/eliza.py` | rules |
| Baseline | `src/classical.py::TfidfLR` | tf-idf (word 1-2 + char 2-5) + logistic regression |
| Neural network | `src/word2vec.py`, `src/ffnn.py` | SGNS word2vec 100-d → FFNN 512 hidden, FP32 |
| Transformer | `src/transformer.py::finetune_teacher` | bert-base-multilingual-cased, FP32 |
| Distilled student | `src/transformer.py::distill_student` | 6-layer mBERT, FP32 |
| Quantized student | `src/transformer.py::quantize_student` | 6-layer mBERT, INT8 (dynamic) |
| Routing | `src/router.py`, `config.py` | threshold + lookup table |
| UI | `app.py` | Gradio |
