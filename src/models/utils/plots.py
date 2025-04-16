import matplotlib.pyplot as plt
import torchvision.utils as vutils


def plot(figs, filename):
    fig, ax = plt.subplots(1, 1)
    grid = vutils.make_grid(figs, nrow=4, pad_value=0.2)[:, 2:-2, 2:-2]
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
    ax.axis("off")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    plt.show()
