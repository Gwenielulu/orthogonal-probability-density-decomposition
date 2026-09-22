import numpy as np


def max_level(field):
    n = np.asarray(field).size
    level = 0
    while n % (2 ** (level + 1)) == 0:
        level += 1
    return level


def transform(field, level):
    values = np.asarray(field)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)

    block_size = 2**level
    if level < 0:
        raise ValueError("level must be nonnegative")
    if values.size % block_size != 0:
        raise ValueError("the number of values must be divisible by 2**level")
    if not np.isfinite(values).all():
        raise ValueError("field must contain only finite values")

    flat = values.reshape(-1)
    order = np.argsort(flat, kind="stable")
    sorted_values = flat[order]
    block_means = sorted_values.reshape(-1, block_size).mean(axis=1)
    sorted_approximation = np.repeat(block_means, block_size)

    approximation = np.empty_like(sorted_approximation)
    approximation[order] = sorted_approximation
    return approximation.reshape(values.shape)


def decompose(field, level=None):
    values = np.asarray(field)
    if level is None:
        level = max_level(values)

    approximations = [transform(values, j) for j in range(level, -1, -1)]
    coarse = approximations[0]
    details = [fine - coarse_level for coarse_level, fine in zip(approximations[:-1], approximations[1:])]
    return coarse, details


def reconstruct(coarse, details):
    field = np.array(coarse, copy=True)
    for detail in details:
        field += detail
    return field

