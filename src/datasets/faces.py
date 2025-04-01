import os
from typing import Callable, Literal, Optional

from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class Faces(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        cache: bool = True,
        split: Literal["train", "test"] = "train",
        test_size: int = 10000,
    ):
        super().__init__()

        self.cache = cache
        self._image_files = [
            os.path.join(data_dir, file) for file in os.listdir(data_dir)
        ]
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
                        desc="Caching Faces",
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
