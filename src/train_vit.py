import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import timm
import os

from fer_dataset import FER2013Images


# ---------------------------------------------
# Data Loaders
# ---------------------------------------------
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    train_set = FER2013Images("data/fer2013", "train", transform)
    val_set   = FER2013Images("data/fer2013", "test", transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


# ---------------------------------------------
# Evaluation
# ---------------------------------------------
def evaluate(model, loader, device, criterion):
    model.eval()
    correct, total = 0, 0
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)

            loss = criterion(output, labels)
            total_loss += loss.item()

            _, preds = output.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return correct / total, total_loss / len(loader), all_preds, all_labels


# ---------------------------------------------
# Confusion Matrix
# ---------------------------------------------
def save_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_vit.png")
    plt.close()


# ---------------------------------------------
# Plot Loss / Acc curves
# ---------------------------------------------
def save_curves(train_losses, val_losses, val_accs):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("loss_curve_vit.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(val_accs, label="Val Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("accuracy_curve_vit.png")
    plt.close()


# ---------------------------------------------
# Training
# ---------------------------------------------
def train_vit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_loader, val_loader = get_data_loaders()

    # ViT model
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=7)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    writer = SummaryWriter("runs/vit_training")

    EPOCHS = 20
    patience, best_acc = 0, 0
    MAX_PATIENCE = 5

    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"TrainLoss": loss.item()})

        train_loss = running_loss / len(train_loader)
        val_acc, val_loss, preds, labels = evaluate(model, val_loader, device, criterion)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"\nEpoch {epoch+1} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc*100:.2f}%\n")

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            os.makedirs("models_vit", exist_ok=True)
            torch.save(model.state_dict(), "models_vit/best_model.pth")
            print("✨ Saved Best Model!")
        else:
            patience += 1
            if patience >= MAX_PATIENCE:
                print("⛔ Early Stopping Triggered")
                break

    writer.close()

    # Save graphs
    save_curves(train_losses, val_losses, val_accs)

    # Save confusion matrix
    classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    save_confusion_matrix(labels, preds, classes)

    print("🎉 Training Complete! Graphs saved.")


if __name__ == "__main__":
    train_vit()
