from huggingface_hub import hf_hub_download

# Replace these with your values
repo_id = "ad6398/thesis-experiments-data"  # e.g., "ad6398/mpdocvqa"
filename = "sft-colqwen-image-embeddings-21-4-1k-ckpt" # e.g., "train.json"

# Optional: If your dataset is private
token = "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"  # if needed, else set token=None

# Download the file
local_file_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",  # Important: since it's a dataset
    token=token,          # Optional: Only needed for private repos
)

print(f"File downloaded at: {local_file_path}")
