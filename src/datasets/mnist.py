import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from torchvision import datasets, transforms


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

    def _colourise(self, img: torch.tensor, label: torch.tensor) -> Image:
        img = img.numpy()

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

        return img

    def __getitem__(self, index):
        # Explicitly retrieve image and label from self.data and self.targets.
        # self.data is a torch tensor of shape [N, H, W] with uint8 values.
        img = self.data[index]
        label = self.targets[index]

        # Set image hue
        img = self._colourise(img, label)

        # Apply the transform if provided
        # (e.g., convert to tensor, normalization, etc.)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label


class PairedColourMNIST(ColourMNIST):
    def __init__(self, *args, sigma: float = 0.05, p: float = 0.01, **kwargs):
        """
        Args:
            sigma (float): Standard deviation for Normal distribution.
            p (float): Probability for sampling from Uniform instead of Normal.
        """
        super().__init__(*args, sigma=sigma, p=p, **kwargs)
        # Build mapping from label to list of indices
        self.label_to_indices = {}
        for label in range(0, 10):
            self.label_to_indices[label] = torch.argwhere(
                self.targets == label
            ).flatten()

    def __getitem__(self, index):
        # Explicitly retrieve image and label from self.data and self.targets.
        # self.data is a torch tensor of shape [N, H, W] with uint8 values.
        img1 = self.data[index]
        label = self.targets[index]

        # Form image pairs by getting another image with the same label.
        indices = self.label_to_indices[int(label)]
        img2_idx = int(np.random.choice(indices))
        img2 = self.data[img2_idx]

        # Set image hue
        img1 = self._colourise(img1, label)
        img2 = self._colourise(img2, label)

        # Apply the transform if provided
        # (e.g., convert to tensor, normalization, etc.)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        else:
            img1 = transforms.ToTensor()(img1)
            img2 = transforms.ToTensor()(img2)

        return img1, img2, label
