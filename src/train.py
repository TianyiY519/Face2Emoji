import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os

from fer_dataset import FER2013Images


# -----------------------------
# Data Augmentation
# -----------------------------
def get_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])


# -----------------------------
# Weighted Sampler (Class Imbalance)
# -----------------------------
def get_weighted_sampler(dataset):
    labels = [label for _, label in dataset.samples]
    class_sample_count = np.bincount(labels)
    weight_per_class = 1.0 / class_sample_count
    weights = [weight_per_class[label] for label in labels]
    return WeightedRandomSampler(weights, len(weights))


# -----------------------------
# Build Model (you choose speed or accuracy)
# -----------------------------
def build_model(model_type="resnet"):
    if model_type == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.last_channel, 7)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 7)

    return model


# -----------------------------
# Validation Function
# -----------------------------
def evaluate(model, loader, device, criterion):
    model.eval()
    correct, total = 0, 0
    total_loss = 0

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    return accuracy, total_loss / len(loader), all_preds, all_labels


# -----------------------------
# Plot Confusion Matrix
# -----------------------------
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()


# -----------------------------
# Training Loop (Ultimate)
# -----------------------------
def train_super():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    transform = get_transforms()

    train_set = FER2013Images("data/fer2013", "train", transform)
    val_set   = FER2013Images("data/fer2013", "test",  transform)

    sampler = get_weighted_sampler(train_set)
    train_loader = DataLoader(train_set, batch_size=64, sampler=sampler, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4)

    # ------------------
    # Build model
    # ------------------
    model = build_model(model_type="resnet")  # or "mobilenet"
    model = model.to(device)

    # Loss + Label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Scheduler + Warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # TensorBoard
    writer = SummaryWriter("runs/super_training")

    best_acc = 0
    patience = 0
    MAX_PATIENCE = 5

    train_losses, val_losses, val_accs = [], [], []

    EPOCHS = 10

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"TrainLoss": loss.item()})

        # Validation
        val_acc, val_loss, preds, labels = evaluate(model, val_loader, device, criterion)

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        print(f"\nEpoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            os.makedirs("models_best", exist_ok=True)
            torch.save(model.state_dict(), "models_best/best_model.pth")
            print("✨ Saved Best Model!")
        else:
            patience += 1
            if patience >= MAX_PATIENCE:
                print("⛔ Early stopping triggered")
                break

    writer.close()

    # Confusion Matrix
    print("Generating confusion matrix...")
    classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    plot_confusion_matrix(labels, preds, classes)


if __name__ == "__main__":
    train_super()

