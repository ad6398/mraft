#!/usr/bin/env python
import argparse
from huggingface_hub import HfApi

# ─── Edit these ────────────────────────────────────────────────────────────────
REPO_ID  = "ad6398/thesis-experiments-data"      # <— your dataset repo
HF_TOKEN = "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"      # <— or set to None if you've already done `huggingface-cli login`
# ────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Upload a local folder into a Hugging Face dataset repo"
    )
    parser.add_argument(
        "local_folder",
        help="Path to the local folder you want to upload"
    )
    parser.add_argument(
        "path_in_repo",
        help="Target folder path inside the dataset repo"
    )
    args = parser.parse_args()

    api = HfApi()
    api.upload_large_folder(
        folder_path=args.local_folder,
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=args.path_in_repo,
        token=HF_TOKEN,
    )
    print(f"✔️ Uploaded '{args.local_folder}' → '{REPO_ID}/{args.path_in_repo}'")

if __name__ == "__main__":
    main()
