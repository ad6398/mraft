from huggingface_hub import snapshot_download
import os
import zipfile
import tarfile

def download_mpdocvqa_data(hf_token: str):
    """
    Download MP DocVQA dataset from a private Hugging Face repository.
    
    Args:
        hf_token (str): Hugging Face API token for accessing private repository
        
    Returns:
        str: Path to the downloaded data directory
    """
    # 2. Specify your dataset repo
    repo_id = "ad6398/mpdocvqa"  # e.g. "facebook/imagenet-mini"

    # 3. Download the full repository snapshot with authentication
    #    This will cache all files (images.tar.zip, train.json, etc.) under ~/.cache/huggingface/hub/
    local_dir = snapshot_download(
        repo_id=repo_id,
        token=hf_token,
        local_dir_use_symlinks=False
    )

    # 4. Locate and unzip images.tar.zip
    zip_path = os.path.join(local_dir, "images.tar.zip")
    if os.path.isfile(zip_path) and zipfile.is_zipfile(zip_path):
        print(f"Extracting ZIP archive: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(local_dir)

    # 5. If that yielded a .tar, extract it too
    tar_path = os.path.join(local_dir, "images.tar")
    if os.path.isfile(tar_path) and tarfile.is_tarfile(tar_path):
        extract_folder = os.path.join(local_dir, "images")
        os.makedirs(extract_folder, exist_ok=True)
        print(f"Extracting TAR archive: {tar_path}")
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(path=extract_folder)
    
    return local_dir

if __name__ == "__main__":
    # Get token from environment variable or prompt user
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        hf_token = input("Please enter your Hugging Face API token: ")
    
    data_dir = download_mpdocvqa_data(hf_token)
    print(f"Data downloaded to: {data_dir}") 