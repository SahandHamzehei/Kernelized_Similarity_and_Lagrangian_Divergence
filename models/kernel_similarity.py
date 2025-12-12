# kernel_similarity.py
import torch
import torch.nn.functional as F

def rbf_kernel(x1, x2, gamma=0.5):
    # ||x1 - x2||^2
    diff = x1 - x2
    dist_sq = torch.sum(diff * diff, dim=1)
    return torch.exp(-gamma * dist_sq)

def matern_kernel(x1, x2, nu=1.5, length_scale=1.0):
    # |x1 - x2|
    r = torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1) + 1e-6) / length_scale
    if nu == 0.5:
        k = torch.exp(-r)
    elif nu == 1.5:
        k = (1 + torch.sqrt(3) * r) * torch.exp(-torch.sqrt(3) * r) # type: ignore
    elif nu == 2.5:
        k = (1 + torch.sqrt(5) * r + (5/3) * r**2) * torch.exp(-torch.sqrt(5) * r) # type: ignore
    else:
        k = torch.exp(-r)
    return k

def kernel_distance(z1, z2, kernel='rbf', gamma=0.5, **kwargs):
    if kernel == 'rbf':
        k_xx = rbf_kernel(z1, z1, gamma)
        k_yy = rbf_kernel(z2, z2, gamma)
        k_xy = rbf_kernel(z1, z2, gamma)
    elif kernel == 'matern':
        k_xx = matern_kernel(z1, z1, **kwargs)
        k_yy = matern_kernel(z2, z2, **kwargs)
        k_xy = matern_kernel(z1, z2, **kwargs)
    else:
        raise ValueError("Unsupported kernel type.")

    dist = k_xx + k_yy - 2 * k_xy
    return dist.clamp(min=0)
