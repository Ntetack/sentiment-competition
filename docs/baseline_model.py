"""
baseline_model.py — Starter baseline for the Image Emotion Recognition Challenge.

Usage:
    python baseline_model.py --train-dir ./data/train --test-dir ./data/test --output my_submission.csv
"""

import argparse
import csv
import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL_MAP = {e: i for i, e in enumerate(EMOTIONS)}


class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform, with_labels=True):
        self.samples = []
        self.transform = transform
        self.with_labels = with_labels

        if with_labels:
            for emotion in EMOTIONS:
                folder = os.path.join(root_dir, emotion)
                if not os.path.exists(folder):
                    continue
                for fname in os.listdir(folder):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((
                            os.path.join(folder, fname),
                            LABEL_MAP[emotion],
                            fname
                        ))
        else:
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(root_dir, fname), -1, fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, fname = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, fname


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(512, 7)

    def forward(self, x):
        return self.backbone(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", default="my_submission.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Train
    train_ds = EmotionDataset(args.train_dir, train_tf, with_labels=True)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)

    model = BaselineModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        correct, total = 0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            correct += (model(images).argmax(1) == labels).sum().item()
            total += len(labels)
        print(f"Epoch {epoch+1}/{args.epochs} — Acc: {correct/total:.4f}")

    # Predict on test set
    test_ds = EmotionDataset(args.test_dir, test_tf, with_labels=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model.eval()
    rows = []
    with torch.no_grad():
        for images, _, fnames in test_loader:
            preds = model(images.to(device)).argmax(1).cpu().tolist()
            for fname, pred in zip(fnames, preds):
                image_id = os.path.splitext(fname)[0]
                rows.append({"image_id": image_id, "label": pred})

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} predictions to {args.output}")
    print("Submit this file via GitHub Issues!")


if __name__ == "__main__":
    main()
