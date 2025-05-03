#!/usr/bin/env python
import os
import json
import argparse
from PIL import Image
import torch
from unsloth import FastVisionModel 
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
from unsloth.data.collators import UnslothVisionDataCollator
from tqdm import tqdm

SYSTEM_MESSAGE = """
You are a vision-language assistant specialized in answering questions based on document page images.
Given a question about the document, use the provided page images to only generate accurate, short and concise answers.
"""

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def predict_answers(split_json, cands_json, images_dir, top_k, output_path, quant, batch_size=4):
    # 1) load data
    split = load_json(split_json)
    examples = split.get("data", split)
    cands_map = load_json(cands_json)

    model, processor = FastVisionModel.from_pretrained(
        "unsloth/Qwen2.5-VL-7B-Instruct",
        load_in_4bit = True if quant=="4bit" else False,
        use_gradient_checkpointing = "unsloth",
    )

    model.eval()
    collator = UnslothVisionDataCollator(model=model, tokenizer=processor)

    batch_inputs = []
    batch_metadata = []
    results = []

    for ex in tqdm(examples):
        qid = ex.get("questionId")
        question = ex.get("question") or ex.get("content") or ""
        cands = cands_map.get(str(qid))[:top_k]
        if not cands:
            continue

        vision_inputs = [{"type": "text", "text": f"{SYSTEM_MESSAGE}\nQuestion: {question}"}]
        for cand in cands:
            img_path = os.path.join(images_dir, f"{cand[0]}.jpg")
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image for candidate {cand!r} not found at {img_path}")
            vision_inputs.append({"type": "image", "image": f"file://{img_path}"})

        messages = [{"role": "user", "content": vision_inputs}]
        

        batch_inputs.append(messages)
        batch_metadata.append((qid, cands))

        if len(batch_inputs) >= batch_size:
            inputs = collator(batch_inputs)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    use_cache=True,
                    temperature=1.5,
                    min_p=0.1
                )

            trimmed = [
                output[len(inp):]
                for inp, output in zip(inputs["input_ids"], out_ids)
            ]
            answers = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            for (qid, cands), answer in zip(batch_metadata, answers):
                results.append({
                    "questionId": qid,
                    "answer": answer.strip(),
                    "answer_page": int(cands[0][0].rsplit("_p", 1)[1]),
                })

            batch_inputs = []
            batch_metadata = []

    # process remaining examples
    if batch_inputs:
        inputs = collator(batch_inputs)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                use_cache=True,
                temperature=1.5,
                min_p=0.1
            )

        trimmed = [
            output[len(inp):]
            for inp, output in zip(inputs["input_ids"], out_ids)
        ]
        answers = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        for (qid, cands), answer in zip(batch_metadata, answers):
            results.append({
                "questionId": qid,
                "answer": answer.strip(),
                "answer_page": int(cands[0][0].rsplit("_p", 1)[1]),
            })

    # 6) write out
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Wrote {len(results)} predictions to {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MPDocVQA answer predictor")
    p.add_argument("--split_json",  required=True,
                   help="Path to MPDocVQA split JSON (with questions)")
    p.add_argument("--cands_json",  required=True,
                   help="JSON mapping qid → ranked candidate page IDs")
    p.add_argument("--images_dir",  required=True,
                   help="Directory containing page images named <cand>.jpg")
    p.add_argument("--top_k", type=int, default=1,
                   help="How many top candidates to feed into the model")
    p.add_argument("--output", "--output_path", required=True,
                   help="Where to write the predictions JSON")
    p.add_argument("--quantization",
                   choices=["none","4bit","8bit","bf16"],
                   default="bf16",
                   help="Whether to quantize the model weights")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size for inference")
    args = p.parse_args()

    predict_answers(
        args.split_json,
        args.cands_json,
        args.images_dir,
        args.top_k,
        args.output,
        args.quantization,
        args.batch_size
    )
