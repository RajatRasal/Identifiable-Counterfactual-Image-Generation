from typing import Callable, Literal, Optional

import h5py
from torchvision.datasets import VisionDataset


class H5Dataset(VisionDataset):
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        cache: bool = False,
        split: Literal["train", "test"] = "train",
        test_size: int = 10000,
        key: str = "image",
    ):
        super().__init__(data_dir, transform=transform)
        self.hf = h5py.File(self.root, "r")
        self.images = self.hf[key][:]
        self.images = (
            self.images[:-test_size] if split == "train" else self.images[-test_size:]
        )

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img
