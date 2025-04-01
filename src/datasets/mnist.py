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


def sample_hue(label: int, sigma: float, p: float, flip_hue: bool) -> float:
    # Sample the binary variable b ~ Bernoulli(p)
    b = torch.bernoulli(torch.tensor([p])).item()  # returns 0.0 or 1.0
    if b == 0:
        # If b = 0: hue ~ Normal(digit / 10 + 0.05, sigma)
        mean = label / 10.0 + 0.05
        hue = torch.normal(mean, sigma).item()
    else:
        # Else: hue ~ Uniform(0, 1)
        hue = torch.rand(1).item()
    if flip_hue:
        hue = 1 - hue
    hue = max(0.0, min(1.0, hue))  # ensure hue in [0, 1]
    return hue


class ColourMNIST(datasets.MNIST):
    def __init__(
        self,
        *args,
        sigma: float = 0.05,
        p: float = 0.01,
        flip_hue: bool = False,
        **kwargs,
    ):
        """
        Args:
            sigma (float): Standard deviation for Normal distribution.
            p (float): Probability for sampling from Uniform instead of Normal.
        """
        super().__init__(*args, **kwargs)
        self.sigma = torch.tensor(sigma)
        self.p = p
        self.flip_hue = flip_hue

    def _colourise(self, img: torch.tensor, label: torch.tensor) -> Image:
        img = img.numpy()

        # Sample a value for hue in range [0, 1]
        hue = sample_hue(int(label), self.sigma, self.p, self.flip_hue)

        # Set hue and return image in range [0, 1]
        img = np.repeat(
            np.expand_dims(img, axis=-1),
            repeats=3,
            axis=-1,
        )
        img = set_hue_and_saturation(img, hue)
        img = Image.fromarray(img, mode="RGB")

        return img, torch.tensor(hue)

    def __getitem__(self, index):
        # Explicitly retrieve image and label from self.data and self.targets.
        # self.data is a torch tensor of shape [N, H, W] with uint8 values.
        img = self.data[index]
        label = self.targets[index]

        # Set image hue
        img, hue = self._colourise(img, label)

        # Apply the transform if provided
        # (e.g., convert to tensor, normalization, etc.)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label, hue


class PairedColourMNIST(ColourMNIST):
    def __init__(self, *args, paired: bool = True, view: bool = False, **kwargs):
        """
        Args:
            sigma (float): Standard deviation for Normal distribution.
            p (float): Probability for sampling from Uniform instead of Normal.
        """
        super().__init__(*args, **kwargs)
        self.paired = paired
        self.view = view
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

        # Set image hue
        cimg1, hue1 = self._colourise(img1, label)
        if self.view:
            # View Augmentation - same image different colour
            cimg2, hue2 = self._colourise(img1, label)
        else:
            # Form image pairs by getting another image with the same label.
            indices = self.label_to_indices[int(label)]
            img2_idx = int(np.random.choice(indices))
            img2 = self.data[img2_idx]
            # Data Augmentation - different image with same label different colour
            cimg2, hue2 = self._colourise(img2, label)

        # Apply the transform if provided
        # (e.g., convert to tensor, normalization, etc.)
        if self.transform is not None:
            cimg1 = self.transform(cimg1)
            cimg2 = self.transform(cimg2)
        else:
            cimg1 = transforms.ToTensor()(cimg1)
            cimg2 = transforms.ToTensor()(cimg2)

        if self.paired:
            return (
                label.reshape(
                    1,
                ),
                (
                    hue1.reshape(
                        1,
                    ),
                    hue2.reshape(
                        1,
                    ),
                ),
                (cimg1, cimg2),
            )
        else:
            return (
                label.reshape(
                    1,
                ),
                hue1.reshape(
                    1,
                ),
                cimg1,
            )
