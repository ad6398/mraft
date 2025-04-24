#!/usr/bin/env python
import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from embedding_loader_helpers import (
    load_query_embedding,
    load_all_image_embeddings,
    load_image_index
)
from metrics import precision_recall_at_k


def evaluate_mpdocvqa(
    questions_json: str,
    query_embeddings_dir: str,
    image_embeddings_dir: str,
    image_index_dir: str,
    top_k: int
):
    """
    Returns:
      - results: dict[qid] = list of (page_id, score) sorted descending
      - avg_p, avg_r: overall P@k and R@k
    """
    # 1) load ground truth
    with open(questions_json, "r") as f:
        meta = json.load(f)
    gt_map = {
        ex["questionId"]: ex["page_ids"][ex["answer_page_idx"]]
        for ex in meta["data"]
    }

    # 2) load image-token embeddings + faiss index
    all_token_embs, _ = None, None
    all_token_embs = load_all_image_embeddings(image_embeddings_dir)   # np.ndarray [N,dim]
    index, token2pageuid = load_image_index(image_index_dir)

    results = {}
    precisions, recalls = [], []

    # 3) iterate
    for qid in tqdm(gt_map, desc=f"Evaluating (k={top_k})"):
        # load query embedding
        q_emb = load_query_embedding(query_embeddings_dir, qid)      # torch.Tensor [seq_len,dim]
        q_np  = q_emb.to(torch.float32).numpy()

        # kNN per token
        D, I = index.search(q_np, top_k)  # [seq_len, k]

        # MaxSim aggregation
        page_scores = {}
        for t in range(q_np.shape[0]):
            per_tok = {}
            for j in range(top_k):
                idx = I[t, j]
                uid = token2pageuid[idx]
                sim = float(np.dot(q_np[t], all_token_embs[idx]))
                if uid not in per_tok or sim > per_tok[uid]:
                    per_tok[uid] = sim
            for uid, s in per_tok.items():
                page_scores[uid] = page_scores.get(uid, 0.0) + s

        # sort descending, take top_k
        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        top_pages = sorted_pages[:top_k]
        results[qid] = top_pages

        # metrics
        p, r = precision_recall_at_k(
            [uid for uid, _ in top_pages],
            [gt_map[qid]],
            k=top_k
        )
        precisions.append(p)
        recalls.append(r)
        print(f"{qid} → P@{top_k}={p:.3f}, R@{top_k}={r:.3f}")

    avg_p = float(np.mean(precisions))
    avg_r = float(np.mean(recalls))
    return results, avg_p, avg_r


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate MPDocVQA retrieval and dump JSON results."
    )
    p.add_argument("--data_dir", type=str, default="data",
                   help="MPDocVQA root (contains question_answers/, train/, etc.)")
    p.add_argument("--split", type=str, choices=["train","val","test"], required=True,
                   help="Which split to evaluate")
    p.add_argument("--query_embeddings_dir", type=str, required=True,
                   help="Folder of query safetensors (e.g. data/train/queries_embedding)")
    p.add_argument("--image_embeddings_dir", type=str, required=True,
                   help="Folder containing image embeddings")
    p.add_argument("--image_index_dir", type=str, required=True,
                   help="Folder containing the FAISS index")
    p.add_argument("--top_k", type=int, default=1,
                   help="Number of pages to retrieve per query")
    p.add_argument("--output_json", type=str, default=None,
                   help="Where to write results JSON (default: data/{split}/retrieval_results.json)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    questions_json = os.path.join(
        args.data_dir, "question_answers", f"{args.split}.json"
    )
    out_json = args.output_json or os.path.join(
        args.data_dir, args.split, "retrieval_results.json"
    )

    results, avg_p, avg_r = evaluate_mpdocvqa(
        questions_json=questions_json,
        query_embeddings_dir=args.query_embeddings_dir,
        image_embeddings_dir=args.image_embeddings_dir,
        image_index_dir=args.image_index_dir,
        top_k=args.top_k
    )

    # write JSON: { qid: [ [page_id, score], ... ], ... }
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results for split='{args.split}' to {out_json}")
    print(f"Average P@{args.top_k}: {avg_p:.4f}")
    print(f"Average R@{args.top_k}: {avg_r:.4f}")
