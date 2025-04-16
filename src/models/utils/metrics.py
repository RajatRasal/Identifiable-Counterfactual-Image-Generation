import json
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.aggregation import MeanMetric
from torchmetrics.functional.image.lpips import _NoTrainLpips
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from models.utils.clustering import knn
from models.utils.plots import plot
from models.utils.wrapper import SlotWrapper


@torch.no_grad
def reconstruction_metrics(
    model: SlotWrapper,
    dl: DataLoader,
    res: int,
    device: torch.device,
) -> Dict:
    fid = FrechetInceptionDistance(
        feature=2048,
        input_img_size=(3, res, res),
        normalize=True,
        compute_on_cpu=False,
    ).to(device)
    mse_mean = MeanMetric().to(device=device)
    lpips = _NoTrainLpips(net="vgg").to(device=device)
    lpips_mean = MeanMetric().to(device=device)
    origs = []
    recons = []
    for sample in tqdm(dl, desc="Recon"):
        image = sample.to(device)
        recon = model.recon(image)
        # MSE
        mse_score = F.mse_loss(image, recon, reduction="none")
        mse_score = mse_score.view(image.size(0), -1).mean(dim=1)
        mse_mean.update(mse_score)
        # FID
        fid.update(image, real=True)
        fid.update(recon, real=False)
        # LPIPS
        lpips_score = lpips(image, recon).view(image.size(0), 1)
        lpips_mean.update(lpips_score)
        # Accumulate
        origs.append(image.cpu())
        recons.append(recon.cpu())
    origs = torch.cat(origs, dim=0)
    recons = torch.cat(recons, dim=0)
    metrics = {
        "fid": fid.compute().item(),
        "lpips": lpips_mean.compute().item(),
        "mse": mse_mean.compute().item(),
        "origs": origs[:32],
        "recons": recons[:32],
    }
    return metrics


@torch.no_grad
def composition_metrics(
    model: SlotWrapper,
    dl: DataLoader,
    res: int,
    device: torch.device,
) -> Dict:
    fid = FrechetInceptionDistance(
        feature=2048,
        input_img_size=(3, res, res),
        normalize=True,
        compute_on_cpu=False,
    ).to(device)
    comps, attns = knn(model, dl, device)
    comps_dl = DataLoader(TensorDataset(comps), batch_size=dl.batch_size)
    for orig, comp in zip(dl, comps_dl):
        fid.update(orig.to(device), real=True)
        fid.update(comp[0].to(device), real=False)
    return {
        "fid": fid.compute().item(),
        "attns": {k: v[:32] for k, v in attns.items()},
        "comps": comps[:32],
    }


@torch.no_grad
def all_metrics(
    model: SlotWrapper,
    dl: DataLoader,
    image_size: int,
    device: torch.device,
    dir_name: str,
) -> None:
    metrics = reconstruction_metrics(model, dl, image_size, device)
    plot(metrics["origs"], f"{dir_name}/origs")
    plot(metrics["recons"], f"{dir_name}/recons")
    del metrics["origs"]
    del metrics["recons"]
    with open(f"{dir_name}/recon_metrics.csv", "w") as f:
        json.dump(metrics, f)

    metrics = composition_metrics(model, dl, image_size, device)
    with open(f"{dir_name}/comp_metrics.csv", "w") as f:
        json.dump({"fid": metrics["fid"]}, f)
    plot(metrics["comps"], f"{dir_name}/comps")
    for k, v in metrics["attns"].items():
        plot(v, f"{dir_name}/{k}_cluster")


if __name__ == "__main__":
    from argparse import Namespace

    from datasets.mixer import mixer
    from models.slate.slate import SLATE
    from models.utils.wrapper import SLATEWrapper

    # SLATE
    model_name = "slate"
    dataset = "shapes3d"
    batch_size = 512
    # dataset = "bitmoji"
    # batch_size = 128
    # dataset = "celeba"
    # batch_size = 128

    seed = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model
    dir_name = f"runs/{model_name}/{dataset}_{seed}/"
    with open(f"{dir_name}/hparams.json", "r") as f:
        args = Namespace(**json.load(f))
    state = torch.load(f"{dir_name}/best_model.pt", weights_only=True)
    model = SLATE(args).to(device)
    model.load_state_dict(state)

    # wrapper
    wrapper = SLATEWrapper(model, args.tau_final, True, args.image_size)

    # dataset
    _, test_loader = mixer(
        args.dataset,
        batch_size,
        args.num_workers,
        args.image_size,
        False,
    )

    with torch.no_grad():
        metrics = reconstruction_metrics(wrapper, test_loader, args.image_size, device)
        plot(metrics["origs"][:32], f"{dir_name}/origs")
        plot(metrics["recons"][:32], f"{dir_name}/recons")
        del metrics["origs"]
        del metrics["recons"]
        with open(f"{dir_name}/recon_metrics.csv", "w") as f:
            json.dump(metrics, f)

        metrics = composition_metrics(wrapper, test_loader, args.image_size, device)
        with open(f"{dir_name}/comp_metrics.csv", "w") as f:
            json.dump({"fid": metrics["fid"]}, f)
        plot(metrics["comps"][:32], f"{dir_name}/comps")
        for k, v in metrics["attns"].items():
            plot(v[:32], f"{dir_name}/{k}_cluster")
