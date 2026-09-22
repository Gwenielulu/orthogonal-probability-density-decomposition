from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opd import transform


def main():
    x = np.linspace(0, 1, 1024, dtype=np.float32)
    signal = 20 * x**2 * (1 - x) ** 4 * np.cos(12 * np.pi * x)
    levels = [9, 7, 5]

    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(x, signal, color="black", linewidth=1.2)
    axes[0].set_title("Original signal")

    for ax, level in zip(axes[1:], levels):
        ax.plot(x, transform(signal, level), linewidth=1.2)
        ax.set_title(f"OPT approximation, level {level}")

    for ax in axes:
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("x")
    fig.tight_layout()

    output = Path("figures/opd_1d.png")
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=200)
    print(output)


if __name__ == "__main__":
    main()

