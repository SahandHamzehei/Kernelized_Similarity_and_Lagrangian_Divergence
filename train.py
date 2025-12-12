# %%
# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.siamese_base import SiameseNetwork

# %%
# ---------- Dataset for Siamese ----------
class ECGSiameseDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x1 = self.data[idx]
        y1 = self.labels[idx]

        morph_label = map_morph_label(y1)

        if np.random.rand() < 0.5:
            idx2 = np.random.choice(np.where(self.labels == y1)[0])
            pair_label = 1
        else:
            idx2 = np.random.choice(np.where(self.labels != y1)[0])
            pair_label = 0

        x2 = self.data[idx2]

        return (
            torch.tensor(x1).float().unsqueeze(0),
            torch.tensor(x2).float().unsqueeze(0),
            torch.tensor(pair_label).float(),
            torch.tensor(morph_label).long(),
            torch.tensor(y1).long()
        )

# %%
# ---------- Load Data ----------
def load_mitbih(path='data/mitbih_train.csv'):
    df = pd.read_csv(path, header=None)
    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    return data, labels

def map_morph_label(y):
    return 0 if y == 0 else 1

# %%
# ---------- Contrastive Loss with Distance Choice ----------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, distance="euclidean"):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, z1, z2, y):
        if self.distance == "euclidean":
            d = torch.norm(z1 - z2, dim=1)

        elif self.distance == "cosine":
            d = 1.0 - F.cosine_similarity(z1, z2)

        else:
            raise ValueError(f"Unknown distance type: {self.distance}")

        loss = y * d.pow(2) + (1 - y) * torch.clamp(self.margin - d, min=0).pow(2)
        return loss.mean()

# %%
# ---------- Main Training ----------
def train_siamese(distance_type="euclidean"):
    data, labels = load_mitbih()
    x_train, _, y_train, _ = train_test_split(
        data, labels, test_size=0.1, stratify=labels  # type: ignore
    )

    train_ds = ECGSiameseDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    model = SiameseNetwork()

    contrastive_loss = ContrastiveLoss(distance=distance_type)
    morph_loss_fn = nn.CrossEntropyLoss()
    class_loss_fn = nn.CrossEntropyLoss()

    alpha = 0.5
    beta = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        model.train()
        running_loss = 0.0

        for x1, x2, pair_label, morph_label, class_label in train_loader:
            optimizer.zero_grad()

            z1, z2, morph_pred, class_logits = model(x1, x2)

            L_contrast = contrastive_loss(z1, z2, pair_label)
            L_morph = morph_loss_fn(morph_pred, morph_label)
            L_classify = class_loss_fn(class_logits, class_label)

            loss = L_contrast + alpha * L_morph + beta * L_classify
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[{distance_type}] Epoch {epoch+1}/50, Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), f"siamese_base_{distance_type}.pth")
    print(f"Training complete. Model saved as siamese_base_{distance_type}.pth")

# %%
if __name__ == "__main__":
    # Run Euclidean baseline
    #train_siamese(distance_type="euclidean")

    # Run Cosine-distance ablation
    train_siamese(distance_type="cosine")
# %%
