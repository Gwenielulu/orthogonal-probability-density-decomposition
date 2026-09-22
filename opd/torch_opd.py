import torch


def max_level(field):
    n = field.numel()
    level = 0
    while n % (2 ** (level + 1)) == 0:
        level += 1
    return level


def transform(field, level):
    values = torch.as_tensor(field)
    if not values.is_floating_point():
        values = values.to(torch.get_default_dtype())

    block_size = 2**level
    if level < 0:
        raise ValueError("level must be nonnegative")
    if values.numel() % block_size != 0:
        raise ValueError("the number of values must be divisible by 2**level")
    if not torch.isfinite(values).all():
        raise ValueError("field must contain only finite values")

    flat = values.reshape(-1)
    order = torch.argsort(flat, stable=True)
    sorted_values = flat[order]
    block_means = sorted_values.reshape(-1, block_size).mean(dim=1)
    sorted_approximation = torch.repeat_interleave(block_means, block_size)

    approximation = torch.empty_like(sorted_approximation)
    approximation[order] = sorted_approximation
    return approximation.reshape(values.shape)


def decompose(field, level=None):
    values = torch.as_tensor(field)
    if level is None:
        level = max_level(values)

    approximations = [transform(values, j) for j in range(level, -1, -1)]
    coarse = approximations[0]
    details = [fine - coarse_level for coarse_level, fine in zip(approximations[:-1], approximations[1:])]
    return coarse, details


def reconstruct(coarse, details):
    field = coarse.clone()
    for detail in details:
        field = field + detail
    return field

