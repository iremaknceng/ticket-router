"""Gradio web app: type any support message (TR or EN) and see where it is routed.

    python app.py            # downloads weights from the HF Hub on first run, then serves on :7860
    python app.py --share    # also create a public gradio.live link
    python app.py --light    # only the quantized student + classical models (less RAM)
"""
import argparse

import gradio as gr

from config import HF_WEIGHTS_REPO, MODEL_DIR, MODEL_KEYS, MODEL_TITLES
from src.models import is_trained
from src.router import PRIMARY, Router

EXAMPLES = [
    "bulaşık makinem su boşaltmıyor, model DW-6534",
    "Merhaba!",
    "my fridge stoped cooling, its a RF 4410",
    "buzdolabim garantisi bitti mi acaba??",
    "I was charged twice for the repair!!!",
    "how many calories are in a banana",
    "who won the champions league last night",
    "çamaşır makinasi için yeni kapı lastiği lazım",
    "thanks a lot",
]


def ensure_weights():
    if is_trained(PRIMARY) and (MODEL_DIR / "labels.json").exists():
        return
    if HF_WEIGHTS_REPO.startswith("YOUR_"):
        raise SystemExit("No trained models in ./models. Run `python train.py` or set ROUTER_HF_REPO.")
    from huggingface_hub import snapshot_download
    print(f"Downloading weights from https://huggingface.co/{HF_WEIGHTS_REPO} ...")
    snapshot_download(repo_id=HF_WEIGHTS_REPO, local_dir=MODEL_DIR, allow_patterns=None)


def build_ui(router: Router):
    def run(message):
        if not message or not message.strip():
            return "Please type a message.", "", []
        r = router.route(message)
        p = r.primary
        if r.small_talk_reply:
            head = f"### 💬 {r.small_talk_reply}\n**Routed to:** {p.team if p else 'Auto-reply'}"
        else:
            head = (f"### ➡️ Routed to: **{p.team}**\n"
                    f"Intent: `{p.intent}` · confidence **{p.confidence:.1%}** "
                    f"(threshold {router.threshold(p.model):.2f}) · model: {MODEL_TITLES[p.model]} "
                    f"· {p.latency_ms:.1f} ms")
        details = (f"**Normalized text:** `{r.normalized}`  \n"
                   f"**Model codes found:** {', '.join(r.model_codes) or '-'}  \n"
                   f"**Spelling corrections:** "
                   f"{', '.join(f'{a} → {b}' for a, b in r.corrections) or '-'}")
        rows = [[MODEL_TITLES[x.model], x.intent, f"{x.confidence:.3f}", x.team, f"{x.latency_ms:.1f}"]
                for x in r.all_models]
        return head, details, rows

    with gr.Blocks(title="Customer Support Ticket Router") as demo:
        gr.Markdown("# 🧺 Customer Support Ticket Router\nType a customer message in Turkish or English. "
                    "The **quantized (INT8) distilled mBERT** decides the team; low-confidence or "
                    "out-of-scope messages go to a **human agent**; small talk gets a rule-based reply.")
        msg = gr.Textbox(label="Customer message", lines=2, placeholder="e.g. bulaşık makinem su boşaltmıyor")
        btn = gr.Button("Route message", variant="primary")
        head = gr.Markdown()
        details = gr.Markdown()
        table = gr.Dataframe(headers=["Model", "Intent", "Confidence", "Team", "Latency (ms)"],
                             label="All models side by side", interactive=False)
        gr.Examples(EXAMPLES, inputs=msg)
        btn.click(run, msg, [head, details, table])
        msg.submit(run, msg, [head, details, table])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    ensure_weights()
    keys = ["tfidf_lr", "w2v_ffnn", "student_int8"] if args.light else MODEL_KEYS
    router = Router(keys)
    build_ui(router).launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
