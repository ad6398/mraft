#!/usr/bin/env python

import os
import json
import argparse
import numpy as np
import torch
from multiprocessing import Pool
from tqdm import tqdm
import safetensors.torch

from embedding_loader_helpers import load_query_embedding, load_image_index
from metrics import precision_recall_at_k

# ——————————————————————————————————————————————————————————
# Globals that each worker will inherit
index = None
token2pageuid = None
token2fileidx = None
gt_map = None
query_embeddings_dir = None
image_embeddings_dir = None
top_k = None


def init_worker(image_index_dir, questions_json, query_embeddings_dir_arg,
                image_embeddings_dir_arg, top_k_arg):
    global index, token2pageuid, token2fileidx, gt_map
    global query_embeddings_dir, image_embeddings_dir, top_k

    # 1) load FAISS index + token2pageuid mapping
    index, token2pageuid = load_image_index(image_index_dir)

    # 2) build token → (filename, local_idx)
    file2count = {}
    token2fileidx = {}
    for idx, fname in enumerate(token2pageuid):
        file2count.setdefault(fname, 0)
        token2fileidx[idx] = (fname, file2count[fname])
        file2count[fname] += 1

    # 3) load ground truth map
    with open(questions_json, "r") as f:
        meta = json.load(f)
    gt_map = {
        ex["questionId"]: ex["page_ids"][ex["answer_page_idx"]]
        for ex in meta["data"]
    }

    # 4) stash directories and k in globals
    query_embeddings_dir = query_embeddings_dir_arg
    image_embeddings_dir = image_embeddings_dir_arg
    top_k = top_k_arg


def eval_one(qid):
    # per-query cache for loaded image tensors
    loaded_files = {}

    # 1) load query embedding
    q_emb = load_query_embedding(query_embeddings_dir, qid)  # [seq_len, dim]
    q_np = q_emb.to(torch.float32).numpy()

    # 2) FAISS search (D has inner-product scores)
    D, I = index.search(q_np, top_k)  # shapes: [seq_len, top_k]

    # 3) MaxSim aggregation across tokens
    page_scores = {}
    for t in range(D.shape[0]):
        best = {}
        for j in range(top_k):
            sim = float(D[t, j])
            idx = I[t, j]
            fname, loc = token2fileidx[idx]

            if fname not in loaded_files:
                loaded_files[fname] = safetensors.torch.load_file(
                    os.path.join(image_embeddings_dir, f"{fname}.safetensors")
                )["embedding"].to(torch.float32).cpu()

            uid = token2pageuid[idx]
            best[uid] = max(best.get(uid, -1e9), sim)

        for uid, s in best.items():
            page_scores[uid] = page_scores.get(uid, 0.0) + s

    # 4) take top_k pages by aggregated score
    top_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # 5) compute precision and recall
    p, r = precision_recall_at_k(
        [uid for uid, _ in top_pages],
        [gt_map[qid]],
        k=top_k
    )
    return qid, top_pages, p, r


def evaluate_mpdocvqa(
    questions_json: str,
    query_embeddings_dir: str,
    image_embeddings_dir: str,
    image_index_dir: str,
    top_k: int
):
    global gt_map

    # load ground truth in main process
    with open(questions_json, "r") as f:
        meta = json.load(f)
    gt_map = {
        ex["questionId"]: ex["page_ids"][ex["answer_page_idx"]]
        for ex in meta["data"]
    }
    qids = list(gt_map.keys())

    # parallel settings
    nprocs = 8
    with Pool(
        processes=nprocs,
        initializer=init_worker,
        initargs=(
            image_index_dir,
            questions_json,
            query_embeddings_dir,
            image_embeddings_dir,
            top_k
        )
    ) as pool:
        results = {}
        precisions, recalls = [], []
        for qid, pages, p, r in tqdm(
            pool.imap_unordered(eval_one, qids),
            total=len(qids),
            desc=f"Parallel k={top_k}"
        ):
            results[qid] = pages
            precisions.append(p)
            recalls.append(r)

    avg_p = float(np.mean(precisions))
    avg_r = float(np.mean(recalls))
    return results, avg_p, avg_r


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MPDocVQA retrieval and dump JSON results."
    )
    parser.add_argument("--data_dir", type=str, default="data",
                        help="MPDocVQA root (contains question_answers/, train/, etc.)")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], required=True,
                        help="Which split to evaluate")
    parser.add_argument("--query_embeddings_dir", type=str, required=True,
                        help="Folder of query safetensors")
    parser.add_argument("--image_embeddings_dir", type=str, required=True,
                        help="Folder containing image embeddings")
    parser.add_argument("--image_index_dir", type=str, required=True,
                        help="Folder containing the FAISS index and key.json")
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

    results, avg_p, avg_r = evaluate_mpdocvqa(
        questions_json,
        args.query_embeddings_dir,
        args.image_embeddings_dir,
        args.image_index_dir,
        args.top_k
    )

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote results for split='{args.split}' to {out_json}")
    print(f"Average P@{args.top_k}: {avg_p:.4f}")
    print(f"Average R@{args.top_k}: {avg_r:.4f}")
