#!/usr/bin/env python
import json
import argparse
from typing import List

from metrics import evaluate_all  

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def main():
    p = argparse.ArgumentParser(
        description="Evaluate MPDocVQA predictions against gold val split"
    )
    p.add_argument(
        "--preds_json", required=True,
        help="JSON file with model outputs: list of {questionId, answer, …}"
    )
    p.add_argument(
        "--val_json", required=True,
        help="MPDocVQA val split JSON (with gold answers)"
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="ANLS threshold to pass to evaluate_all"
    )
    args = p.parse_args()

    # 1) load model predictions
    preds_data = load_json(args.preds_json)
    # build a map: qid → predicted answer
    pred_map = {
        str(item["questionId"]): item["answer"]
        for item in preds_data
    }

    # 2) load gold validation examples
    val_split = load_json(args.val_json)
    gold_examples = val_split.get("data", val_split)

    preds: List[str] = []
    golds: List[str] = []
    for ex in gold_examples:
        qid = str(ex.get("questionId"))
        if qid not in pred_map:
            raise KeyError(f"No prediction found for questionId={qid}")
        # assume each gold example has a single-string "answer"
        gold_answer = ex.get("answers")[0]
        if gold_answer is None:
            raise KeyError(f"No gold answer found in example {qid}")
        preds.append(pred_map[qid])
        golds.append(gold_answer)

    # 3) call your evaluation function
    results = evaluate_all(preds, golds, anls_threshold=args.threshold)

    # 4) report
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
