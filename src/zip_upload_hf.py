#!/usr/bin/env python
import argparse
import os
import shutil
from huggingface_hub import HfApi

# ─── EDIT THESE ───────────────────────────────────────────────────────────────
REPO_ID  = "ad6398/thesis-experiments-data"      # <— your dataset repo
HF_TOKEN = "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"     # or None if you've done `huggingface-cli login`
# ───────────────────────────────────────────────────────────────────────────────

def zip_folder(folder_path: str) -> str:
    """
    Creates a .zip archive of folder_path in the same parent directory.
    Returns the path to the .zip file.
    """
    base_name = os.path.basename(os.path.normpath(folder_path))
    archive_name = os.path.join(os.path.dirname(folder_path), base_name)
    zip_path = shutil.make_archive(archive_name, 'zip', folder_path)
    return zip_path

def main():
    parser = argparse.ArgumentParser(
        description="Zip a folder and upload the ZIP to a Hugging Face dataset repo"
    )
    parser.add_argument(
        "local_folder",
        help="Path to the folder you want to zip & upload"
    )
    parser.add_argument(
        "zip_in_repo",
        help="Target path (including filename.zip) inside the repo, e.g. data/my_archive.zip"
    )
    args = parser.parse_args()

    # 1️⃣ Zip it
    zip_path = zip_folder(args.local_folder)
    # zip_path = "/home/ubuntu/sft-colqwen-mpdocvqa-21-4-1k.zip"
    print(f"🗜  Created ZIP: {zip_path}")

    # 2️⃣ Upload it
    api = HfApi()
    api.upload_file(
        path_or_fileobj=zip_path,
        path_in_repo=args.zip_in_repo,
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message=f"Add {os.path.basename(zip_path)}"
    )
    print(f"✔️ Uploaded '{zip_path}' → '{REPO_ID}/{args.zip_in_repo}'")

if __name__ == "__main__":
    main()



