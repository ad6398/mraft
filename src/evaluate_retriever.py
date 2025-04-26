import os
import json
import numpy as np
import torch
from tqdm import tqdm
import safetensors.torch

from embedding_loader_helpers import (
    load_query_embedding,
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
    Evaluate retrieval for MPDocVQA without loading all image embeddings into memory.
    """
    # 1) Load ground truth
    with open(questions_json, "r") as f:
        meta = json.load(f)
    gt_map = {
        ex["questionId"]: ex["page_ids"][ex["answer_page_idx"]]
        for ex in meta["data"]
    }

    # 2) Load FAISS index + token2pageuid
    index, token2pageuid = load_image_index(image_index_dir)

    # 3) Load key.json and build token_idx → (file_name, local_idx)
    with open(os.path.join(image_index_dir, "key.json")) as f:
        key_list = json.load(f)  # list: filename per token idx

    token2fileidx = {}
    file2count = {}

    for idx, fname in enumerate(key_list):
        if fname not in file2count:
            file2count[fname] = 0
        token2fileidx[idx] = (fname, file2count[fname])
        file2count[fname] += 1

    results = {}
    precisions, recalls = [], []

    # -- Main evaluation loop --
    for qid in tqdm(gt_map, desc=f"Evaluating (k={top_k})"):
        # Load query embedding
        q_emb = load_query_embedding(query_embeddings_dir, qid)  # [seq_len, dim]
        q_np  = q_emb.to(torch.float32).numpy()

        # Search in FAISS
        D, I = index.search(q_np, top_k)  # [seq_len, k]

        page_scores = {}
        loaded_files = {}  # filename → loaded embeddings

        # MaxSim aggregation
        for t in range(q_np.shape[0]):
            per_tok = {}
            for j in range(top_k):
                idx = I[t, j]  # token index

                file_name, local_idx = token2fileidx[idx]

                # Load the file lazily
                if file_name not in loaded_files:
                    loaded_files[file_name] = safetensors.torch.load_file(
                        os.path.join(image_embeddings_dir, f"{file_name}.safetensors")
                    )["embedding"].to(torch.float32).cpu()

                file_embeddings = loaded_files[file_name]
                token_emb = file_embeddings[local_idx]  # [dim]

                sim = float(np.dot(q_np[t], token_emb.numpy()))
                uid = token2pageuid[idx]

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
