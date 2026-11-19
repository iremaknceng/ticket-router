# Customer Support Ticket Router (SWE015 – Use Case 2)

A home-appliance brand receives support messages in **Turkish and English**. This app
normalizes each message, predicts its intent with a confidence score and routes it to the
right team. Low-confidence / out-of-scope messages go to a **human agent**; greetings and
small talk get an **ELIZA-style rule-based reply**. Four model types are shown side by side,
and the routing decision is made by the **distilled + INT8-quantized mBERT**.

Architecture diagram: [docs/architecture.md](docs/architecture.md) (Mermaid, renders on GitHub).

## Quick start (run the app)

```bash
git clone https://github.com/<your-user>/ticket-router.git
cd ticket-router
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                              # → http://localhost:7860
```

`python app.py` is the single run command. On the first start it downloads the trained
weights from the Hugging Face Hub repo set in `config.py` (`HF_WEIGHTS_REPO`, or env var
`ROUTER_HF_REPO`). Options: `--share` (public link), `--light` (skip the FP32 Transformers).

> CPU-only machine? `pip install torch --index-url https://download.pytorch.org/whl/cpu`
> before `pip install -r requirements.txt` gives a much smaller download.

## Reproduce training and the results table

Open `notebooks/colab_train.ipynb` in Google Colab (T4 GPU) and *Run all*, or locally:

```bash
python train.py            # all 4 rungs + distillation + quantization  (GPU recommended)
python evaluate.py         # same test set for every model → results/results.md
python scripts/push_weights.py --repo <hf-user>/ticket-router-weights
```

`python train.py --smoke && python evaluate.py --smoke` runs the whole pipeline on the 98
hand-written examples in ~2 minutes (no dataset download) to check the setup.

## Data

| Source | Size | Use |
|---|---|---|
| [CLINC150](https://github.com/clinc/oos-eval) `data_oos_plus.json` (Larson et al., 2019, CC BY 3.0) | 15,250 train / 3,100 val / 5,500 test, 150 intents + out-of-scope | main benchmark, OOS evaluation |
| `data/appliance_intents.csv` (hand-written, TR + EN) | 98 messages: 6 appliance intents × 15 + 8 OOS; split 56 / 14 / 28 | target domain, Turkish |

The splits are fixed and identical for all models: train → fit, val → early stopping and
the OOS confidence threshold, test → every number in `results/`.
The 56 hand-written training messages are repeated 10× in the training split only
(`--appliance-oversample 10`), because each appliance intent has 9 examples vs. ~100 per CLINC intent.

## Models (the ladder)

| # | Model | Area | Code |
|---|---|---|---|
| 1 | TF-IDF (word 1-2-grams + char 2-5-grams) + logistic regression | B | `src/classical.py` |
| 2 | Own word2vec (skip-gram, negative sampling, NumPy) → averaged → FFNN (ReLU, dropout) trained with SGD + momentum and hand-written backprop | A | `src/word2vec.py`, `src/ffnn.py` |
| 3 | `bert-base-multilingual-cased` fine-tuned with a classification head (teacher) | C | `src/transformer.py` |
| 4 | 6-layer student distilled from the teacher (CE + T²·KL, T = 2, α = 0.5), then **dynamic INT8 quantization** (PyTorch) | D | `src/transformer.py` |

Preprocessing (area B, `src/preprocess.py`): regex normalization, regex model-code
extraction (`DW-6534`, `rf 4410`), minimum-edit-distance correction of product names,
ELIZA small-talk rules (`src/eliza.py`).

## Metrics

* **accuracy** over all test messages (151 CLINC classes + 6 appliance classes, `oos` included)
* **in-scope accuracy** and **OOS recall** (share of out-of-scope messages sent to the human agent)
* **size** of the weight files on disk, **latency** per message on CPU (batch 1, mean of 200 runs)

## Repository layout

```
app.py              Gradio application (runs the INT8 student + all others side by side)
train.py            trains every model          evaluate.py   results table
config.py           paths, model names, intent → team routing table
src/                preprocess, eliza, data, word2vec, ffnn, classical, transformer, models, router
data/               hand-written appliance examples (CLINC150 is downloaded to data/raw/)
notebooks/          colab_train.ipynb
docs/               architecture diagram
results/            metrics produced by evaluate.py
```

Weights are **not** stored in this repository (see `.gitignore`); they are on the Hugging Face Hub.

## AI usage

Claude (Anthropic) was used to draft the code, the hand-written example messages and the
slides; all code was reviewed, run and evaluated by me, and all numbers come from my own runs.
