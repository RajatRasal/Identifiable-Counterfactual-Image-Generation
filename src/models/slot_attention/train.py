import argparse
import datetime
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from datasets.mixer import mixer
from models.slot_attention.model import SlotAttentionAutoEncoder
from models.utils.metrics import all_metrics
from models.utils.wrapper import SlotAEWrapper


def display_slots(image, recon_combined, recons, masks, slots, savefig, num_slots):
    fig, ax = plt.subplots(1, num_slots + 2, figsize=(15, 2))
    image = image.squeeze(0)
    recon_combined = recon_combined.squeeze(0)
    recons = recons.squeeze(0)
    masks = masks.squeeze(0)
    image = image.permute(1, 2, 0).cpu().numpy()
    recon_combined = recon_combined.permute(1, 2, 0)
    recon_combined = recon_combined.cpu().detach().numpy()
    recons = recons.cpu().detach().numpy()
    masks = masks.cpu().detach().numpy()
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[1].imshow(recon_combined)
    ax[1].set_title("Recon.")
    for i in range(num_slots):
        picture = recons[i] * masks[i] + (1 - masks[i])
        ax[i + 2].imshow(picture)
        ax[i + 2].set_title("Slot %s" % str(i + 1))
    for i in range(len(ax)):
        ax[i].grid(False)
        ax[i].axis("off")
    fig.savefig(savefig, bbox_inches="tight")


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", default="model", type=str, help="where to save models"
    )
    parser.add_argument("--dataset", default="clevr6", type=str, help="dataset")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--resolution", default=64, type=int, help="image size")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_slots", default=3, type=int, help="Number of slots.")
    parser.add_argument(
        "--num_iterations", default=3, type=int, help="Number of attention iterations."
    )
    parser.add_argument("--hid_dim", default=64, type=int, help="hidden dimension size")
    parser.add_argument("--learning_rate", default=0.0004, type=float)
    parser.add_argument(
        "--warmup_steps",
        default=10000,
        type=int,
        help="Number of warmup steps for the learning rate.",
    )
    parser.add_argument(
        "--decay_rate",
        default=0.5,
        type=float,
        help="Rate for the learning rate decay.",
    )
    parser.add_argument(
        "--decay_steps",
        default=100000,
        type=int,
        help="Number of steps for the learning rate decay.",
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="number of workers for loading data"
    )
    parser.add_argument(
        "--num_epochs",
        default=1000,
        type=int,
        help="max number of epochs for training",
    )
    parser.add_argument(
        "--max_iters",
        default=100000,
        type=int,
        help="max number of iterations for training",
    )
    opt = parser.parse_args()
    return opt


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opt = argparser()

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    model_dir = opt.model_dir
    os.makedirs(model_dir, exist_ok=True)
    images_dir = f"{model_dir}/images"
    os.makedirs(images_dir, exist_ok=True)
    ckpt_dir = f"{model_dir}/model"
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(f"{model_dir}/hparams.json", "w") as f:
        json.dump(vars(opt), f)

    dl_train, dl_test = mixer(
        opt.dataset,
        opt.batch_size,
        opt.num_workers,
        opt.resolution,
        True,
    )
    model = SlotAttentionAutoEncoder(
        (opt.resolution, opt.resolution),
        opt.num_slots,
        opt.num_iterations,
        opt.hid_dim,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=opt.learning_rate)

    criterion = nn.MSELoss()

    # Train
    start = time.time()
    i = 0
    # TODO: train for 100K iterations?
    for epoch in range(opt.num_epochs):
        model.train()

        total_loss = 0

        for batch, sample in enumerate(tqdm(dl_train)):
            i += 1

            if i < opt.warmup_steps:
                learning_rate = opt.learning_rate * (i / opt.warmup_steps)
            else:
                learning_rate = opt.learning_rate

            lr_decay = opt.decay_rate ** (i / opt.decay_steps)
            learning_rate = learning_rate * lr_decay

            optimizer.param_groups[0]["lr"] = learning_rate

            image = sample.to(device)
            recon_combined, recons, masks, slots = model(image)
            loss = criterion(recon_combined, image)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss /= len(dl_train)

        end = datetime.timedelta(seconds=time.time() - start)
        print("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss, end))

        if not epoch % 5:
            torch.save(
                {"model_state_dict": model.state_dict()}, f"{ckpt_dir}/{epoch}.ckpt"
            )

        if not epoch % 5:
            with torch.no_grad():
                sample = next(iter(dl_test))
                test_img = sample[0].unsqueeze(0).to(device)
                outs = model(test_img)
                display_slots(
                    test_img, *outs, f"{images_dir}/{epoch}.png", opt.num_slots
                )

        if i > opt.max_iter:
            break

    # Test
    wrapper = SlotAEWrapper(model)
    all_metrics(wrapper, dl_test, opt.resolution, device, model_dir)
