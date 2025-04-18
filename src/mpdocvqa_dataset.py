import os
import json
from PIL import Image
from torch.utils.data import Dataset

class MPDocVQADataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Path to snapshot_download(...) directory.
            split (str): One of "train", "val", or "test".
            transform (callable, optional): A torchvision-style transform to apply to each image.
        """
        assert split in {"train", "val", "test"}
        self.split = split
        self.transform = transform
        # load the JSON file for this split
        json_path = os.path.join(root_dir, f"{split}.json")
        with open(json_path, "r") as f:
            meta = json.load(f)
        # meta["data"] is a list of examples
        self.examples = meta["data"]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        qid = ex["questionId"]
        question = ex["question"]
        answers = ex.get("answers", [])            # empty in test set
        doc_id = ex["doc_id"]
        page_ids = ex["page_ids"]
        answer_page_idx = ex.get("answer_page_idx", None)

        # load all pages for this question
        images = []
        for pid in page_ids:
            img_path = os.path.join(self.root_dir, "images", pid + ".jpg")
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        return {
            "question_id": qid,
            "question": question,
            "images": images,
            "answers": answers,
            "doc_id": doc_id,
            "page_ids": page_ids,
            "answer_page_idx": answer_page_idx
        }
