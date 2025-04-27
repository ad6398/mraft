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


class MPDocRetrievalDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train"):
        """
        Args:
            root_dir: path to the folder containing
                      - train.json / val.json / test.json
                      - an 'images/' subfolder with all the .jpg pages.
            split:    one of "train", "val", or "test"
        """
        assert split in {"train", "val", "test"}
        self.root_dir = root_dir
        # load the JSON for this split
        json_path = os.path.join(root_dir, f"question_answers/{split}.json")
        with open(json_path, "r") as f:
            meta = json.load(f)
        self.examples = meta["data"]
        # if split == "val":
        #     self.examples = self.examples[:64]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        qid   = ex["questionId"]
        text  = ex["question"]
        pid   = ex["page_ids"][ ex["answer_page_idx"] ]
        img_p = os.path.join(self.root_dir, "images", f"{pid}.jpg")

        # load the single positive page
        img = Image.open(img_p).convert("RGB")

        return {
            "question_id": qid,
            "query":    text,
            "image":       img,
        }


class MPDocQueryEmbeddDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train"):
        """
        Args:
            root_dir: path to folder with question_answers/{split}.json
            split:    one of "train", "val", or "test"
        """
        assert split in {"train", "val", "test"}
        self.root_dir = root_dir
        json_path = os.path.join(root_dir, "question_answers", f"{split}.json")
        with open(json_path, "r") as f:
            meta = json.load(f)
        self.examples = meta["data"]
        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex    = self.examples[idx]
        qid   = ex["questionId"]
        query = ex["question"]
        return {"question_id": qid, "query": query}


class MPDocQuestionAndTruthDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "val"):
        assert split in {"train", "val", "test"}
        json_path = os.path.join(root_dir, f"question_answers/{split}.json")
        with open(json_path, "r") as f:
            meta = json.load(f)
        self.examples = meta["data"]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        qid = ex["questionId"]
        # ground‐truth single positive
        pid = ex["page_ids"][ex["answer_page_idx"]]
        return {
            "question_id": str(qid),   # cast to str to match JSON keys
            "ground_truth": [pid],
        }