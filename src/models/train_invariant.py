import argparse
import csv
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils

from datasets.mnist import PairedColourMNIST
from models.ema import EMA
from models.utils import get_next_log_dir, seed_everything
from models.vae import VAE, compute_loss_components, entropy


def train(model, ema, device, train_loader, optimizer, epoch, writer):
    model.train()

    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_align = 0
    total_ent = 0

    for batch_idx, (data1, data2, _) in enumerate(train_loader):
        data1 = data1.to(device)
        data2 = data2.to(device)
        optimizer.zero_grad()
        recon_batch, z1, mu, logvar = model(data1)
        recon_loss, kl_loss = compute_loss_components(recon_batch, data1, mu, logvar)
        z2 = model.reparameterize(*model.encode(data2))
        c_dim = z1.shape[1] // 2
        z1 = z1[:, :c_dim]
        z2 = z2[:, :c_dim]
        align = F.mse_loss(z1, z2, reduction="mean")
        ent = entropy(z1, "mean")
        loss = recon_loss + kl_loss + align + ent
        loss.backward()
        optimizer.step()
        ema.update()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_align += align.item()
        total_ent += ent.item()

        if batch_idx % 100 == 0:
            current_loss = loss.item() / len(data1)
            current_recon = recon_loss.item() / len(data1)
            current_kl = kl_loss.item() / len(data1)
            current_align = align.item() / len(data1)
            current_ent = ent.item() / len(data1)
            print(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] | "
                f"Loss: {current_loss:.4f} | Recon: {current_recon:.4f} | "
                f"KL: {current_kl:.4f} | Align: {current_align:.4f} |"
                f"Entropy: {current_ent:.4f}"
            )

    avg_loss = total_loss / len(train_loader.dataset)
    avg_recon = total_recon / len(train_loader.dataset)
    avg_kl = total_kl / len(train_loader.dataset)
    avg_align = total_align / len(train_loader.dataset)
    avg_entropy = total_ent / len(train_loader.dataset)
    print(
        f"====> Epoch: {epoch} Average Training Loss: {avg_loss:.4f} | "
        f"Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | "
        f"Align: {avg_align:.4f} | Entropy: {avg_entropy:.4f}"
    )
    writer.add_scalar("Loss/Train_Total", avg_loss, epoch)
    writer.add_scalar("Loss/Train_Reconstruction", avg_recon, epoch)
    writer.add_scalar("Loss/Train_KL", avg_kl, epoch)
    writer.add_scalar("Loss/Train_Align", avg_align, epoch)
    writer.add_scalar("Loss/Train_Entropy", avg_entropy, epoch)
    return avg_loss, avg_recon, avg_kl, avg_align, avg_entropy


def evaluate(model, ema, device, loader, epoch, writer, tag="Validation"):
    model.eval()

    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_align = 0
    total_ent = 0

    with ema:
        with torch.no_grad():
            for batch_idx, (data1, data2, _) in enumerate(loader):
                data1 = data1.to(device)
                data2 = data2.to(device)
                recon_batch, z1, mu, logvar = model(data1)
                recon_loss, kl_loss = compute_loss_components(
                    recon_batch, data1, mu, logvar
                )
                z2 = model.reparameterize(*model.encode(data2))
                c_dim = z1.shape[1] // 2
                z1 = z1[:, :c_dim]
                z2 = z2[:, :c_dim]
                align = F.mse_loss(z1, z2, reduction="mean")
                ent = entropy(z1, "mean")
                loss = recon_loss + kl_loss + align + ent

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                total_align += align.item()
                total_ent += ent.item()

                if batch_idx == 0:
                    n = min(data1.size(0), 8)
                    # Log a grid of original vs. reconstruction images to TensorBoard
                    comparison = torch.cat([data1[:n], recon_batch[:n]])
                    writer.add_image(
                        f"{tag}/Reconstruction",
                        utils.make_grid(
                            comparison, nrow=n, normalize=True, value_range=(-1, 1)
                        ),
                        epoch,
                    )

        avg_loss = total_loss / len(loader.dataset)
        avg_recon = total_recon / len(loader.dataset)
        avg_kl = total_kl / len(loader.dataset)
        avg_align = total_align / len(loader.dataset)
        avg_ent = total_ent / len(loader.dataset)
        print(
            f"====> Epoch: {epoch} {tag} Set Loss: {avg_loss:.4f} | "
            f"Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | "
            f"Align: {avg_align:.4f} | Entropy: {avg_ent:.4f}"
        )
        writer.add_scalar(f"Loss/{tag}_Total", avg_loss, epoch)
        writer.add_scalar(f"Loss/{tag}_Reconstruction", avg_recon, epoch)
        writer.add_scalar(f"Loss/{tag}_KL", avg_kl, epoch)
        writer.add_scalar(f"Loss/{tag}_Align", avg_align, epoch)
        writer.add_scalar(f"Loss/{tag}_Entropy", avg_ent, epoch)
    return avg_loss, avg_recon, avg_kl, avg_align, avg_ent


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE on colourMNIST")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--latent_dim", type=int, default=20, help="Dimension of latent space"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    seed_everything(args.seed, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine the next version folder for TensorBoard logs
    log_dir = get_next_log_dir(base_dir="runs", prefix="vae_colourmnist_v")
    print(f"TensorBoard logs will be saved to: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    # Log hyperparameters using add_hparams (with an empty metrics dict)
    writer.add_hparams(vars(args), {})
    print("Hyperparameters logged to TensorBoard using add_hparams.")

    # Transforms: Convert grayscale MNIST to 3 channels, then normalize to [-1, 1]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Load full training dataset and split into train and validation sets
    # (e.g., 90/10 split)
    root = "./data"
    full_train_dataset = PairedColourMNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
        sigma=args.sigma,
        p=args.p,
    )
    test_dataset = PairedColourMNIST(
        root=root,
        train=False,
        download=True,
        transform=transform,
        sigma=args.sigma,
        p=args.p,
    )
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    # Create loaders for training, validation, and testing
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=args.ema_decay)

    # Store dataset examples from before training begins
    examples1, examples2, _ = next(iter(train_loader))
    grid1 = utils.make_grid(examples1, nrow=8, normalize=True, value_range=(-1, 1))
    grid2 = utils.make_grid(examples2, nrow=8, normalize=True, value_range=(-1, 1))
    writer.add_image("Dataset/Pair1", grid1, global_step=0)
    writer.add_image("Dataset/Pair2", grid2, global_step=0)
    print("Dataset examples (before training) logged to TensorBoard.")

    # Training loop with evaluation on the validation set
    final_train_metrics = None
    final_val_metrics = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = train(
            model,
            ema,
            device,
            train_loader,
            optimizer,
            epoch,
            writer,
        )
        val_metrics = evaluate(
            model, ema, device, val_loader, epoch, writer, tag="Validation"
        )
        final_train_metrics = train_metrics
        final_val_metrics = val_metrics

        # Save model weights and EMA weights in a subfolder under log_dir
        checkpoints_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(
                checkpoints_dir,
                f"model_weights_{epoch}_val_loss_{final_val_metrics[0]:.4f}.pth",
            ),
        )
        torch.save(
            ema.shadow, os.path.join(checkpoints_dir, f"ema_weights_{epoch}.pth")
        )
        print(f"Model and EMA weights saved to {checkpoints_dir}")

    # Store the last epoch
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    torch.save(
        model.state_dict(), os.path.join(checkpoints_dir, "_last_model_weights.pth")
    )
    torch.save(ema.shadow, os.path.join(checkpoints_dir, "_last_ema_weights.pth"))
    print(f"Model and EMA weights saved to {checkpoints_dir}")

    # Final evaluation on both the validation and test sets
    final_val = evaluate(
        model, ema, device, val_loader, epoch, writer, tag="Final_Validation"
    )
    final_test = evaluate(model, ema, device, test_loader, epoch, writer, tag="Test")

    # Write final metrics to a CSV file with useful headers
    csv_path = os.path.join(log_dir, "final_metrics.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "Dataset",
            "Total_Loss",
            "Reconstruction_Loss",
            "KL_Loss",
            "Align",
            "Entropy",
        ]
        writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerow(
            {
                "Dataset": "Train",
                "Total_Loss": f"{final_train_metrics[0]:.4f}",
                "Reconstruction_Loss": f"{final_train_metrics[1]:.4f}",
                "KL_Loss": f"{final_train_metrics[2]:.4f}",
                "Align": f"{final_train_metrics[3]:.4f}",
                "Entropy": f"{final_train_metrics[4]:.4f}",
            }
        )
        writer_csv.writerow(
            {
                "Dataset": "Validation",
                "Total_Loss": f"{final_val[0]:.4f}",
                "Reconstruction_Loss": f"{final_val[1]:.4f}",
                "KL_Loss": f"{final_val[2]:.4f}",
                "Align": f"{final_val[3]:.4f}",
                "Entropy": f"{final_val[4]:.4f}",
            }
        )
        writer_csv.writerow(
            {
                "Dataset": "Test",
                "Total_Loss": f"{final_test[0]:.4f}",
                "Reconstruction_Loss": f"{final_test[1]:.4f}",
                "KL_Loss": f"{final_test[2]:.4f}",
                "Align": f"{final_test[3]:.4f}",
                "Entropy": f"{final_test[4]:.4f}",
            }
        )
    print(f"Final metrics saved to {csv_path}")

    writer.close()


if __name__ == "__main__":
    main()
