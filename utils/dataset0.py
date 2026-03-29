import torch
from torch.utils.data import Dataset
import json
import os

class BaseSet(Dataset):
    def __init__(self, type="train", max_length=100, text_path=None, use_np=False,
                 img_path=None, edge_path=None, desc_path="image_intent.jsonl"):
        self.type = type
        self.max_length = max_length
        self.text_path = text_path
        self.img_path = img_path
        self.egde_path = edge_path
        self.use_np = use_np
        self.knowledge = int(knowledge)

        with open(self.text_path) as f:
            self.dataset = json.load(f)

        self.img_set = torch.load(self.img_path)
        self.edge_set = torch.load(self.egde_path)

        self.desc_dict = self._load_jsonl_intent(desc_path)
        self._integrate_intent()

    def _load_jsonl_intent(self, desc_path):
        desc_dict = {}
        if os.path.exists(desc_path):
            with open(desc_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if isinstance(data, list) and len(data) >= 2:
                        img_id = str(data[0])
                        desc = data[1]
                        desc_dict[img_id] = desc
        return desc_dict

    def _integrate_intent(self):
        for item in self.dataset:
            img_id = str(item[0])
            desc = self.desc_dict.get(img_id, "")
            text_idx = 3 if self.type == "train" else 4


            if "text" not in item[text_idx]:
                if "token_cap" in item[text_idx]:
                    item[text_idx]["text"] = " ".join(item[text_idx]["token_cap"])
                else:
                    item[text_idx]["text"] = ""

            item[text_idx]["desc"] = desc  

    def __getitem__(self, index):
        sample = self.dataset[index]

        if self.type == "train":
            label = sample[2]
            text = sample[3]
        else:
            label = sample[3] 
            text = sample[4]

        twitter = text["token_cap"]                        # token list of original text
        desc = text.get("desc", "")                         # raw string of image description
        dep = text["token_dep"]                             # dependency edges

        img = self.img_set[index]                           # preprocessed image tensor
        edge = self.edge_set[index]                         # image patch graph edges

        return img, edge, twitter, desc, dep, label

    def __len__(self):
        return len(self.dataset)
