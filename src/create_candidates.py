import json
import random
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Generate RAFT-style train dataset for MPDocVQA."
    )
    parser.add_argument(
        "--train_json", type=str, required=True,
        help="Path to the original train.json file."
    )
    parser.add_argument(
        "--candidate_json", type=str, required=True,
        help="Path to the candidate.json with top retrieved pages per question."
    )
    parser.add_argument(
        "--output_json", type=str, required=True,
        help="Path where the RAFT-processed JSON will be saved."
    )
    parser.add_argument(
        "--ratio", type=float, default=0.6,
        help="Fraction of samples with gold answer retained (default: 0.6)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    # Load original dataset and candidate pages
    with open(args.train_json, "r") as f:
        train_meta = json.load(f)
    with open(args.candidate_json, "r") as f:
        candidate_map = json.load(f)

    data = train_meta.get("data", [])
    total = len(data)
    num_with_answer = int(total * args.ratio)

    # Shuffle indices to select which examples keep a valid answer
    random.seed(args.seed)
    all_indices = list(range(total))
    random.shuffle(all_indices)
    answer_indices = set(all_indices[:num_with_answer])

    new_data = []
    for idx, ex in enumerate(data):
        qid = ex["questionId"]
        # Original fields
        question = ex.get("question")
        doc_id = ex.get("doc_id")
        original_pages = ex.get("page_ids", [])
        answer_page_idx = ex.get("answer_page_idx")
        original_answers = ex.get("answers", [])

        # Gold page identifier
        gold_page = None
        if answer_page_idx is not None and 0 <= answer_page_idx < len(original_pages):
            gold_page = original_pages[answer_page_idx]

        # Retrieved candidate pages (top 4)
        retrieved = candidate_map.get(str(qid), [])
        # Exclude gold from retrieved list
        filtered = [p for p,_ in retrieved if p != gold_page]

        if idx in answer_indices:
            # 60%: gold + top 3 distractors
            pages = [gold_page] + filtered[:min(3, len(filtered))]
            
            # print(gold_page, qid, retrieved, filtered)
            answers = original_answers
            new_answer_idx = 0
        else:
            # 40%: top 4 distractors only
            # print(qid)
            pages = filtered[:min(4, len(filtered))]
            answers = ["No sufficient evidence to answer the query."]
            new_answer_idx = None

        new_data.append({
            "questionId": qid,
            "question": question,
            "doc_id": doc_id,
            "page_ids": pages,
            "answers": answers,
            "answer_page_idx": new_answer_idx
        })

    # Write out RAFT-processed file
    output = {"data": new_data}
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"RAFT dataset generated: {num_with_answer} answer-checked and {total - num_with_answer} no-evidence examples.")


if __name__ == "__main__":
    main()
