import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image

DATA_ROOT = r"D:\fruit_project\data"
MODEL_DIR = r"D:\fruit_project\models"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
MAX_SAMPLES = 4000

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def estimate_ripeness(path):
    img = cv2.imread(path)
    if img is None:
        return 1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    green_mask = ((H >= 35) & (H <= 85)) & (S > 60)
    green_ratio = green_mask.mean()
    dark_ratio = (hsv[:, :, 2] < 80).mean()
    if dark_ratio > 0.25:
        return 2
    if green_ratio > 0.45:
        return 0
    return 1

class FruitDataset(Dataset):
    def __init__(self, split_file, transform=None, max_samples=None):
        self.samples = []
        self.transform = transform
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                self.samples.append(parts[0])
        if max_samples and len(self.samples) > max_samples:
            np.random.seed(42)
            idx = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in idx]
        self.label_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        if idx not in self.label_cache:
            self.label_cache[idx] = estimate_ripeness(path)
        label = self.label_cache[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

print("加载数据...")
train_ds = FruitDataset(os.path.join(DATA_ROOT, "split_train.txt"), train_tf, max_samples=MAX_SAMPLES)
val_ds = FruitDataset(os.path.join(DATA_ROOT, "split_val.txt"), val_tf)
print(f"训练集: {len(train_ds)} 张, 验证集: {len(val_ds)} 张")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
device = torch.device("cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, pred = out.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    train_acc = 100. * correct / total

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            _, pred = out.max(1)
            val_correct += pred.eq(labels).sum().item()
            val_total += labels.size(0)
    val_acc = 100. * val_correct / val_total

    print(f"Epoch {epoch}/{EPOCHS}  Train Acc: {train_acc:.2f}%  Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        save_path = os.path.join(MODEL_DIR, "ripeness_mobilenet_full.pth")
        torch.save(model.state_dict(), save_path)
        print(f"  -> 保存最佳: {save_path} ({val_acc:.2f}%)")

print(f"\n完成! 最佳验证准确率: {best_acc:.2f}%")
