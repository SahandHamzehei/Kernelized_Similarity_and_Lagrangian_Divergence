# siamese_base.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Siamese Encoder ----------
class ECGEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return F.normalize(self.fc(x), p=2, dim=1)

# ---------- Siamese Network ----------
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = ECGEncoder(embedding_dim)
        
        self.classifier_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5) 
        )

        self.morph_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        morph_pred = self.morph_head(z1)
        class_logits = self.classifier_head(z1)
        return z1, z2, morph_pred, class_logits

# ---------- Contrastive Loss ----------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        euclidean_dist = F.pairwise_distance(z1, z2)
        loss = label * torch.pow(euclidean_dist, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_dist, min=0.0), 2)
        return loss.mean()
