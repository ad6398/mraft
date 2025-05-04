#!/usr/bin/env python
import os
import json
import argparse
from PIL import Image
import torch
from unsloth import FastVisionModel 
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
from tqdm import tqdm
import json
SYSTEM_MESSAGE = """
Output only answer.
"""
# SYSTEM_MESSAGE = """
# You are a vision-language assistant specialized in answering questions based on document page images.
# Given a question about the document, use the provided page images to only generate accurate, short and concise answers.
# """

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def predict_answers(split_json, cands_json, images_dir, top_k, output_path, quant, model_id):
    # 1) load data
    split = load_json(split_json)
    # assume split has key "data" which is a list of examples
    examples = split.get("data", split)
    cands_map = load_json(cands_json)
    
    model, processor = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit = True if quant=="4bit" else False , # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )
    FastVisionModel.for_inference(model) 

    results = []
    model.eval()
    for ex in tqdm(examples):
        qid = ex.get("questionId")
        question = ex.get("question") or ex.get("content") or ""
        # get top-K candidates (they should be strings like "doc123_p_5")
        cands = cands_map.get(str(qid))[:top_k]
        # if not cands: we don't do this here - Spartans
        #     continue

        # build the vision+text message for Qwen
        # first, load all page images for these candidates
        vision_inputs = []
        vision_inputs.append({"type": "text", "text": f"{SYSTEM_MESSAGE}\nQuestion: {question}"})
        for cand in cands:
            img_path = os.path.join(images_dir, f"{cand[0]}.jpg")
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image for candidate {cand!r} not found at {img_path}")
            vision_inputs.append({"type": "image", "image": f"file://{img_path}"})

        

        # assemble into the chat format
        messages = [
            {
                "role": "user",
                "content": vision_inputs
            }
        ]
        # print(messages)

        # 3) prepare inputs
        text = processor.apply_chat_template(messages,
                                             tokenize=False,
                                             add_generation_prompt=True)
        image_feats, video_feats = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_feats,
            # videos=video_feats,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 4) generate
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                use_cache=True,
                temperature=1.5,
                min_p=0.1
            )


        # strip off the prompt
        trimmed = [
            output[len(inp):]
            for inp, output in zip(inputs["input_ids"], out_ids)
        ]
        answers = processor.batch_decode(
            out_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        answer = answers[0].strip()

        # 5) record result
        print("answer",answer)
        results.append({
            "questionId": qid,
            "answer": answer,
            "answer_page": int(cands[0][0].rsplit("_p", 1)[1]),
            # "candidates": cands
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
    p.add_argument('--model_id',    type=str, required=True, help="Model ID to load (e.g., Qwen/Qwen2-VL-7B-Instruct)")

    args = p.parse_args()

    predict_answers(
        args.split_json,
        args.cands_json,
        args.images_dir,
        args.top_k,
        args.output,
        args.quantization,
        args.model_id
        
    )

# python generate_answers_qwen_unsloth.py --split_json --cands_json --images_dir --top_k --output
