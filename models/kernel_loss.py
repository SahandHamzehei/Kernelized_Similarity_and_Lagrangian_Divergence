# kernel_loss.py
import torch
import torch.nn as nn
from models.kernel_similarity import kernel_distance

class KernelContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, kernel='rbf', gamma=0.5):
        super().__init__()
        self.margin = margin
        self.kernel = kernel
        self.gamma = gamma

    def forward(self, z1, z2, label):
        d_kernel = kernel_distance(z1, z2, kernel=self.kernel, gamma=self.gamma)
        loss = label * torch.pow(d_kernel, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - d_kernel, min=0.0), 2)
        return loss.mean()
