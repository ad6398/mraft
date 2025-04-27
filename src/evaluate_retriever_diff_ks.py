import os
from metrics import precision_recall_at_k
from mpdocvqa_datasets import MPDocQuestionAndTruthDataset
import os
import json
import argparse

def evaluate(root_dir: str, retrieval_json: str, top_k_max: int = 5):
    # 1) load retrieved candidates
    with open(retrieval_json, "r") as f:
        retrievals = json.load(f)
    # map qid→[pid1, pid2, …]
    retrieved = {
        str(qid): [entry[0] for entry in entries]
        for qid, entries in retrievals.items()
    }

    # 2) load val split and build GT lists
    ds = MPDocQuestionAndTruthDataset(root_dir, split="val")
    gt_docs, retrieved_docs = [], []
    for ex in ds:
        qid = ex["question_id"]
        # if qid not in retrieved:
        #     continue  # skip if no retrievals for this qid
        gt_docs.append([ex["ground_truth"]])
        retrieved_docs.append([retrieved[qid]])

    # 3) compute & print metrics for k=1..top_k_max
    print(f"Evaluating on {len(gt_docs)} queries\n")
    for k in range(1, top_k_max + 1):
        p, r = precision_recall_at_k(retrieved_docs, gt_docs, k)
        print(f"@{k:>2} — precision: {p:.4f}, recall: {r:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MPDocVQA retrieval on val split"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to MPDocVQA root (contains question_answers/val.json & images/)"
    )
    parser.add_argument(
        "--retrieval_json", type=str, required=True,
        help="Path to your retrieved‐candidates JSON file"
    )
    parser.add_argument(
        "--top_k_max", type=int, default=5,
        help="Compute metrics for k=1…top_k_max"
    )
    args = parser.parse_args()
    evaluate(args.data_dir, args.retrieval_json, args.top_k_max)