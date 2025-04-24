#!/usr/bin/env python
import os
import argparse
from typing import List

import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file

from utils import load_model_processor_inference
from mpdocvqa_datasets import MPDocQueryEmbeddDataset


def embed_queries_text(
    queries: List[str],
    processor,
    model,
    device: torch.device
) -> torch.Tensor:
    """
    Tokenize and encode a batch of text queries into token-level embeddings.
    Returns [batch_size, seq_len, emb_dim] on CPU in bfloat16.
    """
    inputs = processor(
        text=queries,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embeds = model(**inputs)  # expect get_text_features API
        embeds = embeds.cpu().to(torch.bfloat16)
    return embeds


def parse_args():
    p = argparse.ArgumentParser(
        description="Build and save per-question safetensors for MPDocVQA."
    )
    p.add_argument(
        "--data_dir", type=str, default="data",
        help="Root folder containing question_answers/{split}.json"
    )
    p.add_argument(
        "--split", type=str, choices=["train","val","test"],
        required=True, help="Which split to process"
    )
    p.add_argument(
        "--output_dir", type=str,  default="./question_embeddings",
         help="Where to write .safetensors files"
    )
    
    p.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for embedding forward pass"
    )
    p.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="Path or HF name of your pretrained model"
    )
    p.add_argument(
        "--quantization", type=str, default=None,
        help="Quantization config (e.g. '4bit', '8bit', or None for bf16)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # load model + processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model_processor_inference(
        model_name=args.model_name_or_path,
        quantization=args.quantization,
        devic=device
    )
    model.to(device).eval()

    # dataset & loader
    ds = MPDocQueryEmbeddDataset(root_dir=args.data_dir, split=args.split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: {
            "question_ids": [b["question_id"] for b in batch],
            "queries":      [b["query"] for b in batch]
        }
    )

    # iterate, embed, and save
    for batch in loader:
        qids   = batch["question_ids"]
        texts  = batch["queries"]
        embeds = embed_queries_text(texts, processor, model, device)
        for i, qid in enumerate(qids):
            filepath = os.path.join(output_dir, f"{qid}.safetensors")
            save_file({"embeddings": embeds[i]}, filepath)
            print(f"✔ Saved {qid} → {filepath}")


if __name__ == "__main__":
    main()


# python make_embeddings.py \
#   --data_dir /path/to/mpdocvqa \
#   --split train \
#   --output_dir ./question_embeddings \
#   --batch_size 16 \
#   --model_name_or_path your/model/path \
#   --quantization int8
