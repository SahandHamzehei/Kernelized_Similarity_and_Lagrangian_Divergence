# divergence_penalty.py
import torch

def divergence_penalty(z_source, z_target, kernel='rbf', gamma=0.5):
    def pairwise_kernel(x, y):
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        return torch.exp(-gamma * dist_sq)

    K_ss = pairwise_kernel(z_source, z_source)
    K_tt = pairwise_kernel(z_target, z_target)
    K_st = pairwise_kernel(z_source, z_target)
    m = z_source.size(0)
    n = z_target.size(0)
    loss = (K_ss.mean() + K_tt.mean() - 2 * K_st.mean())
    return loss
