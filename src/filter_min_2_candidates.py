import json
import argparse
from pathlib import Path

def filter_questions_with_min_candidates(input_path: str, output_path: str, min_candidates: int = 2):
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Ensure input format is valid
    if "data" not in data:
        raise ValueError(f"Expected a top-level 'data' key in {input_path}")

    filtered_data = [
        ex for ex in data["data"]
        if "page_ids" in ex and len(ex["page_ids"]) >= min_candidates and ex['answer_page_idx'] < min_candidates
    ]

    print(f"Filtered {len(filtered_data)} out of {len(data['data'])} total examples with ≥{min_candidates} candidate pages.")

    with open(output_path, 'w') as f:
        json.dump({"data": filtered_data}, f, indent=2)

    print(f"Saved filtered dataset to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter MPDocVQA questions with at least 2 candidate pages")
    parser.add_argument("--input_path", type=str, required=True, help="Path to train.json")
    parser.add_argument("--output_path", type=str, default="page.json", help="Path to save filtered page.json")
    args = parser.parse_args()

    filter_questions_with_min_candidates(args.input_path, args.output_path)
