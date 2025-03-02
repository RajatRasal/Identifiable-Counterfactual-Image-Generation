import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from torchvision import datasets, transforms, utils


def set_hue_and_saturation(
    image: Image,
    hue: float,
    saturation: float = 1.0,
) -> Image:
    hsv_image = rgb_to_hsv(image / 255.0)
    hsv_image[..., 0] = hue  # set hue
    hsv_image[..., 1] = saturation
    return np.clip(hsv_to_rgb(hsv_image) * 255.0, 0, 255).astype(np.uint8)


def sample_hue(label: int, sigma: float, p: float) -> float:
    # Sample the binary variable b ~ Bernoulli(p)
    b = torch.bernoulli(torch.tensor([p])).item()  # returns 0.0 or 1.0
    if b == 0:
        # If b = 0: hue ~ Normal(digit / 10 + 0.05, sigma)
        mean = label / 10.0 + 0.05
        hue = torch.normal(mean, sigma).item()
    else:
        # Else: hue ~ Uniform(0, 1)
        hue = torch.rand(1).item()
    hue = max(0.0, min(1.0, hue))  # ensure hue in [0, 1]
    return hue


class ColourMNIST(datasets.MNIST):
    def __init__(self, *args, sigma: float = 0.05, p: float = 0.01, **kwargs):
        """
        Args:
            sigma (float): Standard deviation for Normal distribution.
            p (float): Probability for sampling from Uniform instead of Normal.
        """
        super().__init__(*args, **kwargs)
        self.sigma = torch.tensor(sigma)
        self.p = p

    def __getitem__(self, index):
        # Explicitly retrieve image and label from self.data and self.targets.
        # self.data is a torch tensor of shape [N, H, W] with uint8 values.
        img = self.data[index].numpy()  # Convert to numpy array
        label = self.targets[index]

        # Sample a value for hue in range [0, 1]
        hue = sample_hue(int(label), self.sigma, self.p)

        # Set hue and return image in range [0, 1]
        img = np.repeat(
            np.expand_dims(img, axis=-1),
            repeats=3,
            axis=-1,
        )
        img = set_hue_and_saturation(img, hue)
        img = Image.fromarray(img, mode="RGB")

        # Apply the transform if provided
        # (e.g., convert to tensor, normalization, etc.)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label


if __name__ == "__main__":
    # Define a transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Create an instance of the custom dataset.
    dataset = ColourMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
        sigma=0.05,
        p=0.01,
    )

    # Create a DataLoader.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )

    # Visualize a batch.
    import matplotlib.pyplot as plt

    batch, labels = next(iter(dataloader))
    grid = utils.make_grid(batch, nrow=8, normalize=True, value_range=(-1, 1))
    np_grid = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(np_grid)
    plt.title("ColourMNIST Samples with Modified Hue")
    plt.axis("off")
    plt.savefig("colourmnist.pdf")
    plt.show()
