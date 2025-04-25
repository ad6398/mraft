#!/usr/bin/env python3
"""
Fast script to extract image embeddings using ColQwen2.5,
with DataLoader prefetching and threaded I/O.
"""

import argparse
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset, DataLoader
import safetensors.torch
from tqdm import tqdm

from utils import load_model_processor_inference


class ImageFolderDataset(Dataset):
    """Dataset that returns (PIL.Image, stem) for each image file."""
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        return img, p.stem


def save_one(stem: str, emb: torch.Tensor, output_dir: str):
    """Save a single embedding to disk."""
    safetensors.torch.save_file(
        {"embedding": emb},
        str(Path(output_dir) / f"{stem}.safetensors")
    )


def parse_args():
    p = argparse.ArgumentParser(description="Fast extract image embeddings")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Images per GPU forward pass")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Workers for DataLoader & ThreadPool")
    p.add_argument("--quantization", type=str, default=None,
                   help="e.g. '4bit' or '8bit'; None for bf16")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load model + processor onto `device`
    model, processor = load_model_processor_inference(
        model_name=args.model_name_or_path,
        quantization=args.quantization,
        devic=device,
    )

    # find images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    all_paths = sorted([
        p for p in Path(args.input_dir).iterdir()
        if p.suffix.lower() in exts
    ])
    dataset = ImageFolderDataset(all_paths)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: (
            [img for img, _ in batch],
            [stem for _, stem in batch]
        )
    )

    # thread‐pool for overlapping I/O
    executor = ThreadPoolExecutor(max_workers=args.num_workers)

    for images, stems in tqdm(loader, desc="Encoding batches"):
        # preprocess + move to device
        inputs = processor.process_images(images=images)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # forward + to CPU/bf16
        with torch.no_grad():
            embeds = model(**inputs).cpu().to(torch.bfloat16)

        # dispatch saves
        for stem, emb in zip(stems, embeds):
            # clone to decouple from batch tensor
            executor.submit(save_one, stem, emb.clone(), args.output_dir)

    executor.shutdown(wait=True)
    print(f"All embeddings saved to {args.output_dir}")


if __name__ == "__main__":
    main()
