import os
import random

import numpy as np
import torch


def seed_everything(seed: int, workers: bool = True):
    # Set Python seed
    random.seed(seed)

    # Set NumPy seed
    np.random.seed(seed)

    # Set PyTorch seed for CPU and GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN (at the cost of performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # If workers is True, set the seed for DataLoader workers
    if workers:
        # In PyTorch Lightning, this is handled internally for DataLoader workers
        # But here's how you might pass a worker_init_fn to a DataLoader:
        def worker_init_fn(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        return worker_init_fn


def get_next_log_dir(base_dir="runs", prefix="vae_colourmnist_v"):
    """
    Helper function to determine the next available log directory version.
    """
    os.makedirs(base_dir, exist_ok=True)
    versions = []
    for d in os.listdir(base_dir):
        if d.startswith(prefix):
            try:
                version = int(d[len(prefix) :])
                versions.append(version)
            except ValueError:
                continue
    next_version = max(versions, default=0) + 1
    return os.path.join(base_dir, f"{prefix}{next_version}")
