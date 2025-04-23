#!/usr/bin/env python
"""
Script to read per-image embeddings (BF16 safetensors) from disk and build a FAISS index.

Usage:
    python index_image_embeddings_faiss.py \
        --embeddings_dir <path_to_saved_embeddings> \
        --output_dir <path_to_store_index> \
        [--faiss_index_type flatip|ivfflat|ivfpq] \
        [--nlist N] [--m M]
"""
import argparse
from pathlib import Path
import faiss
import numpy as np
import safetensors.torch
import torch
# from loguru import logger
from tqdm.auto import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FAISS index from image embeddings"
    )
    parser.add_argument(
        "--embeddings_dir", type=str, required=True,
        help="Directory containing .safetensors embedding files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save FAISS index and metadata"
    )
    parser.add_argument(
        "--faiss_index_type", type=str, default="flatip",
        choices=["flatip", "ivfflat", "ivfpq"],
        help="Type of FAISS index to build"
    )
    parser.add_argument(
        "--nlist", type=int, default=100,
        help="Number of IVF lists (for ivfflat and ivfpq)"
    )
    parser.add_argument(
        "--m", type=int, default=8,
        help="Number of PQ subquantizers (for ivfpq)"
    )
    return parser.parse_args()

def main(args=None):
    if not args:
        args = parse_args()
    embeddings_path = Path(args.embeddings_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all embedding files
    files = sorted(embeddings_path.glob("*.safetensors"))
    if not files:
        print(f"No .safetensors files found in {embeddings_path}")
        return

    # Load embeddings into array
    vectors = []
    keys = []
    for f in tqdm(files, desc="Loading embeddings"):
        data = safetensors.torch.load_file(str(f))
        emb = data.get("embedding")
        # ensure CPU numpy float32
        emb_np = emb.to(torch.float32).cpu().numpy()
        # flatten if necessary
        if emb_np.ndim > 1:
            emb_np = emb_np.reshape(-1, emb_np.shape[-1])
        # if multiple vectors per file, handle each row separately
        for vec in emb_np:
            vectors.append(vec)
            keys.append(f.stem)

    all_vectors = np.vstack(vectors).astype(np.float32)
    d = all_vectors.shape[1]
    print(f"Loaded {all_vectors.shape[0]} vectors of dimension {d}")

    # Build FAISS index
    if args.faiss_index_type == "flatip":
        index = faiss.IndexFlatIP(d)
    elif args.faiss_index_type == "ivfflat":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, args.nlist)
    else:  # ivfpq
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, args.nlist, args.m, 8)

    # Train if necessary
    if not isinstance(index, faiss.IndexFlat):
        index.train(all_vectors)
    index.add(all_vectors)

    # Save index
    index_file = output_path / "image_embeddings.index"
    faiss.write_index(index, str(index_file))
    print(f"Saved FAISS index to {index_file}")

    # Save metadata (keys)
    meta_file = output_path / "keys.json"
    with open(meta_file, "w") as f:
        json.dump(keys, f)
    print(f"Saved keys mapping to {meta_file}")

    # Example query demonstration
    query_vec = all_vectors[:1]  # using first vector as example query
    k = 5
    D, I = index.search(query_vec, k)
    print(f"Example search for key {keys[0]} -> neighbors:")
    for dist, idx in zip(D[0], I[0]):
        print(f"{keys[idx]} (score={dist})")

if __name__ == "__main__":
    main()
    
# from dataclasses import dataclass
# @dataclass
# class indexer_args:
#     embeddings_dir: str = "index-dir"
#     output_dir: str = "faiss-out"
#     faiss_index_type = "flatip"

# main(indexer_args)