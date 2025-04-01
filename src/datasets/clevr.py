import os
from typing import Callable, Literal, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.clevr import CLEVRClassification
from tqdm import tqdm


class ClevrN(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: Literal["train", "test"],
        num_obj: int = 6,
        transform: Optional[Callable] = None,
        cache: bool = True,
        test_size: int = 10000,
    ):
        super().__init__()

        self.dataset = CLEVRClassification(root=data_dir, split="train", download=True)
        self.cache = cache
        assert num_obj >= 3 and num_obj <= 10
        self.filter_idx = [
            i for i, y in enumerate(self.dataset._labels) if y <= num_obj
        ]
        self._image_files = [self.dataset._image_files[i] for i in self.filter_idx]
        self._image_files = (
            self._image_files[:-test_size]
            if split == "train"
            else self._image_files[-test_size:]
        )
        self.transform = transform

        if self.cache:
            from concurrent.futures import ThreadPoolExecutor

            self._images = []
            with ThreadPoolExecutor() as executor:
                self._images = list(
                    tqdm(
                        executor.map(self._load_image, self._image_files),
                        total=len(self._image_files),
                        desc=f"Caching CLEVR {self.dataset._split}",
                        mininterval=(0.1 if os.environ.get("IS_NOHUP") is None else 90),
                    )
                )

    def _load_image(self, file):
        return Image.open(file).convert("RGB")

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, idx: int):
        if self.cache:
            image = self._images[idx]
        else:
            image = self._load_image(self._image_files[idx])

        if self.transform:
            image = self.transform(image)

        return image
