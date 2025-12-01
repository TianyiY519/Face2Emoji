import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import timm
import os

from fer_dataset import FER2013Images


def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    train_set = FER2013Images("data/fer2013", "train", transform)
    val_set = FER2013Images("data/fer2013", "test",  transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def evaluate(model, val_loader, device, criterion):
    model.eval()
    correct, total = 0, 0
    total_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total, total_loss / len(val_loader)


def train_vit():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Data ----
    train_loader, val_loader = get_data_loaders()

    # ---- Model ----
    model = timm.create_model(
        "vit_tiny_patch16_224",  # ⭐ ViT-Lite：CPU 友好
        pretrained=True,
        num_classes=7
    )
    model.to(device)

    # ---- Loss / Optimizer ----
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    writer = SummaryWriter("runs/vit_training")

    EPOCHS = 10

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"TrainLoss": loss.item()})

        val_acc, val_loss = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        print(f"\nEpoch {epoch+1}/{EPOCHS} — Train Loss: {epoch_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%\n")

        # Save best
        os.makedirs("models_vit", exist_ok=True)
        torch.save(model.state_dict(), f"models_vit/epoch{epoch+1}.pth")


if __name__ == "__main__":
    train_vit()
