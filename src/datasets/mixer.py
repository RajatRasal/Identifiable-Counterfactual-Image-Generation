from functools import partial
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from datasets import ArrowRoom, ClevrN, Faces, H5Dataset

test_size = 10000
dataset_map = {
    "arrowroom": (ArrowRoom, "data/point-out-the-wrong-guy-split"),
    "clevr": (H5Dataset, "data/clevr_10-full.hdf5"),
    "clevr6": (partial(ClevrN, num_obj=6), "data"),
    "shapes3d": (partial(H5Dataset, key="images"), "data/3dshapes.h5"),
    "objectsroom": (H5Dataset, "data/objects_room_train-full.hdf5"),
    "clevrtex": (H5Dataset, "data/clevrtex-full.hdf5"),
    "bitmoji": (Faces, "data/bitmoji"),
    "celeba": (Faces, "data/img_align_celeba/img_align_celeba"),
    "shapestack": (H5Dataset, "data/shapestacks-full.hdf5"),
}


def get_transforms(resolution: int, normalise: bool) -> v2.Compose:
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((resolution, resolution)),
    ]
    if normalise:
        transforms.append(v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    return v2.Compose(transforms)


def mixer(
    dataset: str,
    batch_size: int,
    num_workers: int,
    resolution: int,
    cache: bool,
    normalise: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    transform = get_transforms(resolution, normalise)
    ds_class, data_dir = dataset_map[dataset]
    ds_train = ds_class(
        data_dir=data_dir,
        transform=transform,
        cache=cache,
        split="train",
        test_size=test_size,
    )
    ds_test = ds_class(
        data_dir=data_dir,
        transform=transform,
        cache=cache,
        split="test",
        test_size=test_size,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return dl_train, dl_test


if __name__ == "__main__":
    datasets = [
        "arrowroom",
        "shapes3d",
        "bitmoji",
        "celeba",
        "shapestack",
        "clevr6",
        "arrowroom",
        "clevrtex",
        "objectsroom",
    ]
    for dataset in datasets:
        print(dataset)
        dl_train, dl_test = mixer(
            dataset=dataset,
            batch_size=16,
            num_workers=8,
            resolution=256,
            cache=False,
        )
        print(f"train: {len(dl_train.dataset)}, test: {len(dl_test.dataset)}")
