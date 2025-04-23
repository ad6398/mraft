

from huggingface_hub import snapshot_download
import os
import tarfile
import time
import torch

from typing import Optional
from transformers import BitsAndBytesConfig

from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


def extract_tar_gz_with_progress(tar_path: str, dest_dir: str, update_interval: int = 10000) -> None:
    """
    Extract a .tar.gz archive with progress reporting.

    Args:
        tar_path (str): Path to the .tar.gz file.
        dest_dir (str): Directory where files will be extracted.
        update_interval (int): How many files to extract before printing progress.
    """
    # Open the archive to list members
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        total = len(members)

    os.makedirs(dest_dir, exist_ok=True)

    start_time = time.time()
    extracted = 0

    # Re-open to extract one by one
    with tarfile.open(tar_path, "r:gz") as tf:
        for i, member in enumerate(members, 1):
            tf.extract(member, path=dest_dir)
            extracted += 1
            # Print progress every `update_interval` files or on last file
            if i % update_interval == 0 or i == total:
                elapsed = time.time() - start_time
                print(f"[Extraction] {i}/{total} files extracted — Elapsed: {elapsed:.2f}s")


def download_mpdocvqa_data(hf_token: str, cache_dir: str = None) -> str:
    """
    Download MP DocVQA dataset from a private Hugging Face repository and extract images with progress.

    Args:
        hf_token (str): Hugging Face API token for accessing private repository
        cache_dir (str, optional): Directory to cache the snapshot. Defaults to None.

    Returns:
        str: Path to the downloaded data directory
    """
    repo_id = "ad6398/mpdocvqa"

    # Download the full repository snapshot
    local_dir = snapshot_download(
        repo_type="dataset",
        repo_id=repo_id,
        token=hf_token,
        cache_dir=cache_dir,
        # resume_download=True
    )

    # Locate and extract images.tar.gz
    gz_filename = "images.tar.gz"
    gz_path = os.path.join(local_dir, gz_filename)
    extract_folder = os.path.join(local_dir)

    if os.path.isfile(gz_path) and tarfile.is_tarfile(gz_path):
        print(f"Found archive: {gz_path}\nStarting extraction...")
        extract_tar_gz_with_progress(gz_path, extract_folder)
        print("Extraction complete.")
    else:
        print(f"No {gz_filename} found in {local_dir}")

    return local_dir # equal to cache_dir


def setup_quantization(quantization_startegy = None, device="auto") -> Optional[BitsAndBytesConfig]:
    if quantization_startegy is None:
        return None
    if device == "cpu":
        raise ValueError("Quantization requires CUDA GPU.")
    if quantization_startegy == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization_startegy == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    raise ValueError(f"Unknown quantization: {quantization_startegy}")

def print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")


def load_model_processor(model_name, quantization = None, dtype=torch.bfloat16):
    # Load processor and model
    # lora_cfg = LoraConfig.from_pretrained(model_name)
    bnb_config = setup_quantization(quantization)
    model = ColQwen2_5.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="flash_attention_2"
            if is_flash_attn_2_available() else None,
    )
    
    processor = ColQwen2_5_Processor.from_pretrained(model_name, use_fast=True)
    # 
    return model, processor


def load_model_processor_inference(model_name, quantization ,devic="cpu"):
    model, processor = load_model_processor(model_name, quantization)
    model.eval()
    return model, processor

def load_model_processor_train(model_name, quantization, device="cpu"):
    model, processor = load_model_processor(model_name, quantization)
    for n, p in model.named_parameters():
        if "lora" in n:
            p.requires_grad = True
    print_trainable_parameters(model)
    # c) Processor & collator
    return model, processor

