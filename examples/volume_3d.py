from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opd import transform


def main():
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, 16),
        np.linspace(-1, 1, 32),
        np.linspace(-1, 1, 32),
        indexing="ij",
    )
    volume = (
        1.5 * np.exp(-((x + 0.3) ** 2 + y**2 + (z - 0.2) ** 2) / 0.12)
        - np.exp(-((x - 0.25) ** 2 + (y + 0.25) ** 2 + (z + 0.15) ** 2) / 0.05)
        + 0.15 * np.sin(4 * np.pi * x) * np.cos(3 * np.pi * y)
    )

    levels = [12, 9, 6]
    volumes = [volume] + [transform(volume, level) for level in levels]
    titles = ["Original volume"] + [f"Level {level}" for level in levels]
    middle = volume.shape[0] // 2
    limit = max(abs(volume.min()), abs(volume.max()))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
    for ax, data, title in zip(axes, volumes, titles):
        image = ax.imshow(data[middle], origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes, shrink=0.8, label="Central slice")

    output = Path("figures/opd_3d.png")
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=200)
    print(output)


if __name__ == "__main__":
    main()

