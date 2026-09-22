import numpy as np
import torch

from opd import transform as numpy_transform
from opd.torch_opd import transform as torch_transform


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    rng = np.random.default_rng(7)
    field = rng.normal(size=(64, 64)).astype(np.float32)
    numpy_result = numpy_transform(field, level=7)
    tensor = torch.from_numpy(field).to(device)
    torch_result = torch_transform(tensor, level=7).cpu().numpy()

    print(f"device: {device}")
    print(f"maximum difference: {np.max(np.abs(numpy_result - torch_result)):.3e}")


if __name__ == "__main__":
    main()

