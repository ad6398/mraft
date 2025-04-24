#!/usr/bin/env python
"""
Script to extract image embeddings using ColQwen2.5.

This script reads all images from a given folder, computes embeddings
with a ColQwen2_5 vision-language model, and saves each embedding to disk
using the image filename (without extension) as identifier.

Usage:
    python extract_image_embeddings_colqwen2_5.py \
        --model_name_or_path <model_path> \
        --input_dir <path_to_images> \
        --output_dir <path_to_save_embeddings> \
        [--batch_size N]
"""
import argparse
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import safetensors.torch
from utils import load_model_processor_inference

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract image embeddings using ColQwen2.5"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="Pretrained ColQwen2.5 model name or local path"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing image files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory where embeddings will be saved"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Number of images to process per batch"
    )
    parser.add_argument(
        "--quantization", type=str, default=None,
        help="Quantization config (e.g. '4bit', '8bit', or None for bf16)"
    )
    return parser.parse_args()



def batch_list(items, batch_size):
    """Yield successive batches from items list."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main(args=None):
    if not args:
        args = parse_args()

    # Prepare paths
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    quantization = args.quantization


    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Load model and processor
    model, processor = load_model_processor_inference(
        model_name=args.model_name_or_path, 
        quantization=quantization,
        devic=device, 
    )

    # Gather image files
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_paths = sorted([
        p for p in input_path.iterdir() if p.suffix.lower() in extensions
    ])

    # Process in batches
    for batch_paths in tqdm(
        list(batch_list(image_paths, args.batch_size)),
        desc="Encoding batches"
    ):
        # Load and preprocess images
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor.process_images(
            images=images,
            # return_tensors="pt"
        )
        # Move tensors to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Compute embeddings
        with torch.no_grad():
            # model.get_image_features returns embeddings of shape [batch, dim]
            embeds = model(**inputs)
            # Convert to CPU and bfloat16 for saving
            embeds = embeds.cpu().to(torch.bfloat16)

        # Save each embedding file
        for img_path, emb in zip(batch_paths, embeds):
            save_name = f"{img_path.stem}.safetensors"
            save_file = output_path / save_name
            safetensors.torch.save_file(
                {"embedding": emb}, str(save_file)
            )

    print(f"Image Embeddings saved to {output_path}")


if __name__ == "__main__":
    main()

# from dataclasses import dataclass
# @dataclass
# class embedder_args:
#     input_dir = "dummy-images"
#     output_dir = "index-dir"
#     model_name_or_path = "vidore/colqwen2.5-v0.2"
#     batch_size = 1

# main(embedder_args)