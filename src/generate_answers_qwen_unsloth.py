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
import gc

SYSTEM_MESSAGE = """
Output only answer.
"""

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
def load_and_resize_image(image_path, image_size=512): # unsloth default
    image = Image.open(image_path).convert("RGB")
    if image_size is None:
        return image  # no resizing
    elif isinstance(image_size, int):
        image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    elif isinstance(image_size, (tuple, list)):
        image = image.resize(image_size, Image.Resampling.LANCZOS)
    return image
    
def predict_answers(split_json, cands_json, images_dir, top_k, output_path, quant, model_id="unsloth/Qwen2.5-VL-7B-Instruct"):
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
    # model.eval()
    batch = []
    batch_size = 1
    batch_meta = []
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
            image = load_and_resize_image(img_path)
            vision_inputs.append({"type": "image", "image": image})

        # assemble into the chat format
        messages = [
            {
                "role": "user",
                "content": vision_inputs
            }
        ]

        # 3) prepare inputs
        batch.append(messages)
        batch_meta.append({'qid': qid, 'cands': cands})

        if len(batch) == batch_size or ex == examples[-1]:
            # print("processing ", batch_meta)
            texts = []
            images = []
            videos = []

            for message in batch:
                text = processor.apply_chat_template(message,
                                                    tokenize=False,
                                                    add_generation_prompt=True)
                image_feats, video_feats = process_vision_info(message)
                texts.append(text)
                images.append(image_feats)
                videos.append(video_feats)


            
            inputs = processor(
                text=texts,
                images=images,
                # videos=videos,
                padding="longest",
                return_tensors="pt"
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            # print([(k, v.shape) for k, v in inputs.items()])

            # 4) generate
            # with torch.no_grad():
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
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            for raw_ans, meta in zip(answers, batch_meta):
                answer = raw_ans.strip()
                cands = meta['cands']
                # print("ans", answer)

                # 5) record result
                results.append({
                    "questionId": meta['qid'],
                    "answer": answer,
                    "answer_page": int(cands[0][0].rsplit("_p", 1)[1]),
                    # "candidates": cands
                })
            batch = []
            batch_meta = []
            del inputs, out_ids, trimmed, answers
            torch.cuda.empty_cache()
            gc.collect()

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