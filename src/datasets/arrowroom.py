"""
https://github.com/JindongJiang/GNM/blob/766f535689593fcc2e55a805e523deca559520cb/data.py
"""
import os
from typing import Callable, Literal, Optional

from PIL import Image
from torch.utils.data import Dataset


class ArrowRoom(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        cache: bool = False,
        split: Literal["train", "test"] = "train",
        test_size: int = 10000,
    ):
        super(ArrowRoom, self).__init__()
        self.transform = transform

        dirs = os.listdir(data_dir)
        image_dir_list = [os.path.join(data_dir, d, "images") for d in dirs]

        self.image_list = []
        for dir in image_dir_list:
            image_list_i = [
                os.path.join(dir, fn) for fn in os.listdir(dir) if fn.endswith(".png")
            ]
            self.image_list.extend(image_list_i)

        self.image_list = (
            self.image_list[:-test_size]
            if split == "train"
            else self.image_list[-test_size:]
        )

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
