# train_kernel.py
# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.siamese_base import SiameseNetwork
from models.kernel_loss import KernelContrastiveLoss
from train import ECGSiameseDataset, load_mitbih
from sklearn.model_selection import train_test_split

# %%
def train_siamese_kernel():
    data, labels = load_mitbih()
    x_train, x_val, y_train, y_val = train_test_split(
        data, labels, test_size=0.1, stratify=labels # type: ignore
    )

    train_ds = ECGSiameseDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    # Model and losses
    model = SiameseNetwork()

    contrastive_loss_fn = KernelContrastiveLoss(kernel='rbf', gamma=0.5)
    morph_loss_fn = nn.CrossEntropyLoss()
    class_loss_fn = nn.CrossEntropyLoss()

    alpha = 0.5  
    beta  = 1.0 

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        model.train()
        running_loss = 0.0

        for x1, x2, pair_label, morph_label, class_label in train_loader:

            pair_label = pair_label.float()
            morph_label = morph_label.long()
            class_label = class_label.long()
            optimizer.zero_grad()
            z1, z2, morph_pred, class_logits = model(x1, x2)
            contrast_loss = contrastive_loss_fn(z1, z2, pair_label)
            morph_loss = morph_loss_fn(morph_pred, morph_label)
            class_loss = class_loss_fn(class_logits, class_label)
            loss = contrast_loss + alpha * morph_loss + beta * class_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/50], Total Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "siamese_kernel_rbf.pth")
    print("Training complete. Model saved.")

# %%
if __name__ == "__main__":
    train_siamese_kernel()

# %%
