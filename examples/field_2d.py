from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opd import transform


def main():
    data_path = Path(__file__).parent / "data" / "sqg_snapshot.npy"
    field = np.load(data_path)

    levels = [14, 12, 10]
    fields = [field] + [transform(field, level) for level in levels]
    titles = [
        "Original SQG field",
        "Coarse: 4 PDF levels ($j=14$)",
        "Intermediate: 16 PDF levels ($j=12$)",
        "Fine: 64 PDF levels ($j=10$)",
    ]
    limit = np.percentile(np.abs(field), 99.5)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
    for ax, data, title in zip(axes, fields, titles):
        image = ax.imshow(
            data,
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes, shrink=0.82, label="Nondimensional SQG state")

    output = Path("figures/opd_2d.png")
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=200)
    print(output)


if __name__ == "__main__":
    main()
