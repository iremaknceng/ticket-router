"""Upload models/ (weights, labels, thresholds) to a Hugging Face Hub model repo.

    huggingface-cli login            # or: export HF_TOKEN=...
    python scripts/push_weights.py --repo <hf-username>/ticket-router-weights
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--private", action="store_true")
args = ap.parse_args()

from huggingface_hub import HfApi  # noqa: E402

api = HfApi()
api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
api.upload_folder(folder_path=str(MODEL_DIR), repo_id=args.repo, repo_type="model",
                  ignore_patterns=["*/checkpoint-*", "*.tmp"])
print(f"Uploaded {MODEL_DIR} to https://huggingface.co/{args.repo}")
print(f"Now set HF_WEIGHTS_REPO = \"{args.repo}\" in config.py and push the change to GitHub.")
