import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        # Encoder: from 3x28x28 input to latent space
        self.enc_conv1 = nn.Conv2d(
            3, 32, kernel_size=3, stride=2, padding=1
        )  # 32 x 14 x 14
        self.enc_conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )  # 64 x 7 x 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder: from latent space back to 3x28x28 image
        self.fc2 = nn.Linear(latent_dim, 128)
        self.fc3 = nn.Linear(128, 64 * 7 * 7)
        self.dec_deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.dec_deconv2 = nn.ConvTranspose2d(
            32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def encode(self, x):
        h1 = F.relu(self.enc_conv1(x))
        h2 = F.relu(self.enc_conv2(h1))
        h2 = h2.view(h2.size(0), -1)
        h3 = F.relu(self.fc1(h2))
        mu = self.fc_mu(h3)
        logvar = self.fc_logvar(h3)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = F.relu(self.fc2(z))
        h5 = F.relu(self.fc3(h4))
        h5 = h5.view(-1, 64, 7, 7)
        h6 = F.relu(self.dec_deconv1(h5))
        # Use tanh so that output is in [-1, 1]
        recon_x = torch.tanh(self.dec_deconv2(h6))
        return recon_x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, z, mu, logvar


def compute_loss_components(recon_x, x, mu, logvar):
    # MSE reconstruction loss (sum reduction)
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl_loss


def loss_function(recon_x, x, mu, logvar):
    recon_loss, kl_loss = compute_loss_components(recon_x, x, mu, logvar)
    return recon_loss + kl_loss


def entropy(z: torch.Tensor, reduction: str = "none") -> torch.Tensor:
    """
    Compute entropy of embeddings efficiently with reduction options.

    Args:
        embeddings (torch.Tensor): Tensor of shape (Batchsize, 512)
        reduction (str): Specifies the reduction to apply to the output:
        - 'none' (default): Returns entropy per sample (Batchsize,)
        - 'mean': Returns mean entropy over the batch
        - 'sum': Returns total entropy over the batch

    Returns:
        torch.Tensor: Entropy value(s) based on the reduction mode.
    """
    # Compute log-softmax for numerical stability: log_probs = log(p)
    log_probs = F.log_softmax(z, dim=-1)
    # Compute entropy: -sum(p * log(p)) along the last dimension
    entropy = -torch.sum(log_probs.exp() * log_probs, dim=-1)

    # Apply reduction
    if reduction == "mean":
        return entropy.mean()
    elif reduction == "sum":
        return entropy.sum()
    return entropy  # 'none' (default): returns per-sample entropy
