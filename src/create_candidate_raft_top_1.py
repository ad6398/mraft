import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate RAFT-style dataset where gold page must be top-1.")
    parser.add_argument("--train_json", type=str, required=True, help="Path to the original train.json file.")
    parser.add_argument("--candidate_json", type=str, required=True, help="Path to the candidate.json with top retrieved pages per question.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the updated JSON.")
    args = parser.parse_args()

    # Load original dataset and candidate pages
    with open(args.train_json, "r") as f:
        train_meta = json.load(f)
    with open(args.candidate_json, "r") as f:
        candidate_map = json.load(f)

    data = train_meta.get("data", [])
    new_data = []

    for ex in data:
        qid = ex["questionId"]
        question = ex.get("question")
        doc_id = ex.get("doc_id")
        original_pages = ex.get("page_ids", [])
        answer_page_idx = ex.get("answer_page_idx")
        original_answers = ex.get("answers", [])

        # Get gold page ID
        gold_page = None
        if answer_page_idx is not None and 0 <= answer_page_idx < len(original_pages):
            gold_page = original_pages[answer_page_idx]

        # Get top-4 retrieved pages
        retrieved = candidate_map.get(str(qid), [])
        top_pages = [p for p, _ in retrieved[:4]]

        # Check if gold page is top-1
        if gold_page is not None and len(top_pages) > 0 and gold_page == top_pages[0]:
            answers = original_answers
            new_answer_idx = 0
        else:
            answers = ["No sufficient evidence to answer the query."]
            new_answer_idx = None

        new_data.append({
            "questionId": qid,
            "question": question,
            "doc_id": doc_id,
            "page_ids": top_pages,
            "answers": answers,
            "answer_page_idx": new_answer_idx
        })

    # Save updated dataset
    output = {"data": new_data}
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Updated dataset saved with {len(new_data)} examples at {args.output_json}")

if __name__ == "__main__":
    main()
