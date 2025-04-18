import os
import json
from PIL import Image
from torch.utils.data import Dataset

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
        json_path = os.path.join(root_dir, f"{split}.json")
        with open(json_path, "r") as f:
            meta = json.load(f)
        self.examples = meta["data"]

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
            "question":    text,
            "image":       img,
        }
