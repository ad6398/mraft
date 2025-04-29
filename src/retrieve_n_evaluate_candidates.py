#!/usr/bin/env python

import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import safetensors.torch

from metrics import precision_recall_at_k

from mpdocvqa_datasets import MPDocVQANonFAISSRetrievalEvalDataset


def evaluate_candidate_pages(dataset: MPDocVQANonFAISSRetrievalEvalDataset, top_k: int):
    results = {}
    precisions, recalls = [], []

    for qid, page_ids, q_emb, answer_page in tqdm(dataset, desc=f"Evaluating (k={top_k})"):
        q_np = q_emb.numpy()  # (L, D)
        page_scores = {}

        for pid in page_ids:
            # load the page embedding (assumed shape (N, D) of patches/tokens)
            p_path = dataset.page_dir / f"{pid}.safetensors"
            pagedata = safetensors.torch.load_file(str(p_path))
            p_emb = pagedata["embedding"].to(torch.float32).cpu().numpy()  # (N, D)

            # MaxSim: for each query token find max over page tokens, then sum
            sim_mat = q_np @ p_emb.T              # (L, N)
            per_token_max = sim_mat.max(axis=1)   # (L,)
            page_scores[pid] = float(per_token_max.sum())  # Convert to Python float

        # sort pages by score desc
        ranked = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        top_pages = [(pid, float(score)) for pid, score in ranked[:top_k]]  # Convert scores to Python floats
        results[qid] = top_pages

        # compute metrics
        preds = [pid for pid, _ in top_pages]
        p, r = precision_recall_at_k(preds, [answer_page], k=top_k)
        precisions.append(p)
        recalls.append(r)
        # print(f"{qid} →  P@{top_k}={p:.3f},  R@{top_k}={r:.3f}")

    avg_p = float(np.mean(precisions))
    avg_r = float(np.mean(recalls))
    return results, avg_p, avg_r


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exact retrieval over candidate pages for MPDocVQA."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="MPDocVQA root (contains question_answers/, etc.)")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], required=True,
                        help="Which split to evaluate")
    parser.add_argument("--query_embeddings_dir", type=str, required=True,
                        help="Folder of query .safetensors")
    parser.add_argument("--image_embeddings_dir", type=str, required=True,
                        help="Folder of per-page .safetensors")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Number of pages to retrieve per query")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Where to write results JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    questions_json = os.path.join(
        args.data_dir, "question_answers", f"{args.split}.json"
    )
    out_json = args.output_json or os.path.join(
        args.data_dir, args.split, "retrieval_results.json"
    )

    # build dataset
    ds = MPDocVQANonFAISSRetrievalEvalDataset(
        questions_json=questions_json,
        query_embeddings_dir=args.query_embeddings_dir,
        image_embeddings_dir=args.image_embeddings_dir
    )

    # run evaluation
    results, avg_p, avg_r = evaluate_candidate_pages(ds, top_k=args.top_k)

    # save results
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote results for split='{args.split}' to {out_json}")
    print(f"Average P@{args.top_k}: {avg_p:.4f}")
    print(f"Average R@{args.top_k}: {avg_r:.4f}")
