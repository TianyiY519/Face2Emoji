import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

import seaborn as sns
import matplotlib.pyplot as plt

from fer_dataset import FER2013Images


# ---------------------------------------------
# Data Loaders with strong augmentation
# ---------------------------------------------
def get_data_loaders(batch_size=16):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    train_set = FER2013Images("data/fer2013", "train", transform_train)
    val_set = FER2013Images("data/fer2013", "test", transform_val)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    return train_loader, val_loader


# ---------------------------------------------
# Evaluation (no mixup here)
# ---------------------------------------------
def evaluate(model, loader, device, criterion):
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    avg_loss = total_loss / len(loader)

    return acc, avg_loss, np.array(all_preds), np.array(all_labels)


# ---------------------------------------------
# Plot & Save Curves
# ---------------------------------------------
def save_curves(train_losses, val_losses, val_accs):
    epochs = range(1, len(train_losses) + 1)

    # Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve (ViT + Mixup)")
    plt.legend()
    plt.savefig("loss_curve_vit_mixup.png")
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [a * 100 for a in val_accs], label="Val Accuracy (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy Curve (ViT + Mixup)")
    plt.legend()
    plt.savefig("accuracy_curve_vit_mixup.png")
    plt.close()


# ---------------------------------------------
# Save Confusion Matrix
# ---------------------------------------------
def save_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (ViT + Mixup)")
    plt.savefig("confusion_matrix_vit_mixup.png")
    plt.close()


# ---------------------------------------------
# Training with ViT + Mixup/CutMix + Cosine LR + Warmup
# ---------------------------------------------
def train_vit_mixup():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_loader, val_loader = get_data_loaders()

    # ---- Model: ViT-small, 比 tiny 更强，CPU 还能承受 ----
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=True,
        num_classes=7,
    )
    model.to(device)

    # ---- Mixup + CutMix ----
    mixup_fn = Mixup(
        mixup_alpha=0.2,
        cutmix_alpha=0.2,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        num_classes=7
    )

    # ---- Loss & Optimizer & Scheduler ----
    train_criterion = SoftTargetCrossEntropy()
    val_criterion = nn.CrossEntropyLoss()

    base_lr = 1e-5
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    EPOCHS = 10
    warmup_epochs = 5
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - warmup_epochs
    )

    writer = SummaryWriter("runs/vit_mixup")

    best_acc = 0.0
    patience = 0
    MAX_PATIENCE = 8

    train_losses, val_losses, val_accs = [], [], []
    last_val_preds, last_val_labels = None, None

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # ---- Warmup + Cosine LR ----
        if epoch < warmup_epochs:
            warmup_lr = base_lr * float(epoch + 1) / warmup_epochs
            for g in optimizer.param_groups:
                g['lr'] = warmup_lr
        else:
            cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (lr={current_lr:.2e})")

        # ---- Training loop ----
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # ⭐ 保证 batch size 为偶数，避免 Mixup 崩掉
            if mixup_fn is not None:
                if images.size(0) % 2 == 1:  # 如果是奇数 batch
                    images = images[:-1]  # 丢掉最后一个样本
                    labels = labels[:-1]

                images, targets = mixup_fn(images, labels)  # 安全 mixup
            else:
                targets = labels

            optimizer.zero_grad()
            outputs = model(images)

            loss = train_criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"TrainLoss": loss.item()})

        avg_train_loss = running_loss / len(train_loader)

        # ---- Validation ----
        val_acc, val_loss, preds, labels_np = evaluate(
            model, val_loader, device, val_criterion
        )

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        last_val_preds, last_val_labels = preds, labels_np

        print(
            f"\nEpoch {epoch+1}/{EPOCHS} "
            f"— Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        # ---- TensorBoard logging ----
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # ---- Early Stopping & Best Model ----
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            os.makedirs("models_vit_mixup", exist_ok=True)
            torch.save(
                model.state_dict(),
                "models_vit_mixup/best_model_vit_mixup.pth"
            )
            print("✨ Saved Best Model!")
        else:
            patience += 1
            if patience >= MAX_PATIENCE:
                print("⛔ Early Stopping triggered.")
                break

    writer.close()

    # ---- Save curves & confusion matrix ----
    save_curves(train_losses, val_losses, val_accs)

    classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    if last_val_labels is not None and last_val_preds is not None:
        save_confusion_matrix(last_val_labels, last_val_preds, classes)

    print("🎉 Training complete. Curves & confusion matrix saved.")


if __name__ == "__main__":
    train_vit_mixup()
