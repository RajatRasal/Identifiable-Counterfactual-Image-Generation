"""
https://github.com/JindongJiang/GNM/blob/766f535689593fcc2e55a805e523deca559520cb/data.py
"""
import os
from typing import Callable, Literal, Optional

from PIL import Image
from torch.utils.data import Dataset


class MultiMNIST(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: Literal["train", "test"] = "train",
        transform: Optional[Callable] = None,
        cache: bool = False,
    ):
        super(MultiMNIST, self).__init__()

        self.transform = transform

        if split == "train":
            img_fn = [
                os.path.join(data_dir, s)
                for s in os.listdir(data_dir)
                if s.startswith("train-image")
            ]
        elif split == "test":
            img_fn = [
                os.path.join(data_dir, s)
                for s in os.listdir(data_dir)
                if s.startswith("test-image") or s.startswith("val-image")
            ]
        else:
            raise NotImplementedError

        self.image_list = []
        for dir in img_fn:
            image_list_i = [
                os.path.join(dir, fn) for fn in os.listdir(dir) if fn.endswith("png")
            ]
            self.image_list.extend(image_list_i)

    def _load_image(self, file):
        return Image.open(file).convert("RGB")

    def __getitem__(self, index):
        file = self.image_list[index]
        image = self._load_image(file)
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_list)
