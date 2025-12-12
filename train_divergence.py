# train_divergence.py
# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.siamese_base import SiameseNetwork
from models.kernel_loss import KernelContrastiveLoss
from models.divergence_penalty import divergence_penalty
from train import ECGSiameseDataset, load_mitbih
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# %%
# ------------------ Load PTB Target Domain ------------------
def load_ptb(path_normal='data/ptbdb_normal.csv', path_abnormal='data/ptbdb_abnormal.csv'):
    df_norm = pd.read_csv(path_normal, header=None)
    df_abn = pd.read_csv(path_abnormal, header=None)

    data = np.concatenate([df_norm.iloc[:, :-1].values,
                           df_abn.iloc[:, :-1].values], axis=0)
    labels = np.concatenate([df_norm.iloc[:, -1].values,
                             df_abn.iloc[:, -1].values], axis=0) # type: ignore

    return data, labels


# ------------------ Training with Divergence + Morphology + Classifier ------------------
def train_with_divergence(lambda_div=0.05, alpha=0.5, beta=1.0):
    data_src, labels_src = load_mitbih()
    data_tgt, _ = load_ptb()

    x_train, x_val, y_train, y_val = train_test_split(
        data_src, labels_src, test_size=0.1, stratify=labels_src # type: ignore
    )
    train_ds = ECGSiameseDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    model = SiameseNetwork()

    contrastive_loss_fn = KernelContrastiveLoss(kernel='rbf', gamma=0.5)
    morph_loss_fn = nn.CrossEntropyLoss()
    class_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data_tgt_tensor = torch.tensor(data_tgt).float().unsqueeze(1)

    for epoch in range(50):
        model.train()
        total_loss_epoch = 0
        contrast_epoch = 0
        morph_epoch = 0
        classify_epoch = 0
        div_epoch = 0

        for x1, x2, pair_label, morph_label, class_label in train_loader:
            pair_label = pair_label.float()
            morph_label = morph_label.long()
            class_label = class_label.long()
            optimizer.zero_grad()
            z1, z2, morph_pred, class_logits = model(x1, x2)
            L_contrast = contrastive_loss_fn(z1, z2, pair_label)
            L_morph = morph_loss_fn(morph_pred, morph_label)
            L_classify = class_loss_fn(class_logits, class_label)
            idx_tgt = torch.randint(0, len(data_tgt_tensor), (len(x1),))
            x_tgt = data_tgt_tensor[idx_tgt]
            z_tgt, _, _, _ = model(x_tgt, x_tgt)
            z_src, _, _, _ = model(x1, x1)
            L_div = divergence_penalty(z_src, z_tgt, kernel='rbf', gamma=0.5)

            L_total = (
                L_contrast +
                alpha * L_morph +
                beta * L_classify +
                lambda_div * L_div
            )
            L_total.backward()
            optimizer.step()
            total_loss_epoch += L_total.item()
            contrast_epoch += L_contrast.item()
            morph_epoch += L_morph.item()
            classify_epoch += L_classify.item()
            div_epoch += L_div.item()

        print(f"Epoch [{epoch+1}/50], "
              f"Total: {total_loss_epoch/len(train_loader):.4f}, "
              f"Contrast: {contrast_epoch/len(train_loader):.4f}, "
              f"Morph: {morph_epoch/len(train_loader):.4f}, "
              f"Classify: {classify_epoch/len(train_loader):.4f}, "
              f"Div: {div_epoch/len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "siamese_divergence.pth")
    print("Training complete. Model saved.")

# %%
# ------------------ Run Training ------------------
if __name__ == "__main__":
    train_with_divergence()

# %%
