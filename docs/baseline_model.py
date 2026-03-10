"""
baseline_model.py — Starter baseline for the Image Sentiment Analysis Challenge.
Fine-tunes a ResNet18 on the training set.

Usage:
    python baseline_model.py --data-dir ./data/train --epochs 10 --output baseline.pth
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


# ─── Model ────────────────────────────────────────────────────────────────────

class BaselineModel(nn.Module):
    """
    ResNet18 fine-tuned for 5-class sentiment classification.
    This is the simplest competitive baseline.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) tensor, values in [0, 1]
        Returns:
            logits: (B, 5)
        """
        return self.backbone(x)


# ─── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset — expects data_dir/class_name/*.jpg structure
    full_dataset = datasets.ImageFolder(args.data_dir)

    # Split 80/20
    n_train = int(len(full_dataset) * 0.8)
    n_val = len(full_dataset) - n_train
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train: {n_train} samples | Val: {n_val} samples")
    print(f"Classes: {full_dataset.classes}")

    # Model
    model = BaselineModel(num_classes=5, pretrained=True).to(device)

    # Optimizer — use lower LR for backbone, higher for head
    optimizer = optim.AdamW([
        {"params": model.backbone.fc.parameters(), "lr": args.lr},
        {"params": [p for n, p in model.backbone.named_parameters() if "fc" not in n], "lr": args.lr / 10},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_loss = train_loss / train_total

        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

        scheduler.step()

    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {args.output}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/train", help="Path to training data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="baseline.pth")
    args = parser.parse_args()
    train(args)
