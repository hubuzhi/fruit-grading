import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

DATA_ROOT = r"D:\fruit_project\data"
SPLIT_DIR = r"D:\fruit_project\data"
MODEL_DIR = r"D:\fruit_project\models"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FULL = 10

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

def estimate_quality(path):
    img = cv2.imread(path)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = (gray < 90).mean()
    low_sat = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1] < 40).mean()
    if dark > 0.30:
        return 2
    if dark > 0.12 or low_sat > 0.5:
        return 1
    return 0

class FruitDataset(Dataset):
    def __init__(self, split_file, task, transform=None, max_samples=None):
        self.samples = []
        self.task = task
        self.transform = transform
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                path = parts[0]
                full_path = os.path.join(DATA_ROOT, path)
                self.samples.append(full_path)
        if max_samples and len(self.samples) > max_samples:
            import random
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
        self._label_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if idx not in self._label_cache:
            if self.task == "ripeness":
                self._label_cache[idx] = estimate_ripeness(path)
            else:
                self._label_cache[idx] = estimate_quality(path)
        return img, self._label_cache[idx]

def build_model(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def train_one_model(task_name, num_classes, train_loader, val_loader, device):
    print(f"\n{'='*60}")
    print(f"开始训练: {task_name}")
    print(f"{'='*60}")

    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    print("\n阶段1: 只训练分类头...")
    optimizer = optim.AdamW(model.classifier[1].parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(EPOCHS_HEAD):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_HEAD}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_acc = 100. * correct / total
        print(f"  Train Loss: {running_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")

    print("\n阶段2: 解冻主干，一起微调...")
    for param in model.features[-2:].parameters():
        param.requires_grad = True
    optimizer = optim.AdamW([
        {'params': model.classifier[1].parameters(), 'lr': 1e-3},
        {'params': model.features[-2:].parameters(), 'lr': 1e-4},
    ], weight_decay=1e-4)

    best_val_acc = 0.0
    for epoch in range(EPOCHS_FULL):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_FULL}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_acc = 100. * correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100. * val_correct / val_total
        print(f"  Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if task_name == "ripeness":
                save_path = os.path.join(MODEL_DIR, "ripeness_mobilenet_full.pth")
            else:
                save_path = os.path.join(MODEL_DIR, "quality_mobilenet_full.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  保存最佳模型: {save_path} (Val Acc: {val_acc:.2f}%)")

    print(f"\n{task_name} 训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    return best_val_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("\n加载训练集...")
    train_ripe = FruitDataset(os.path.join(SPLIT_DIR, "split_train.txt"), "ripeness", transform=train_tf, max_samples=4000)
    train_qual = FruitDataset(os.path.join(SPLIT_DIR, "split_train.txt"), "quality", transform=train_tf, max_samples=4000)

    print("加载验证集...")
    val_ripe = FruitDataset(os.path.join(SPLIT_DIR, "split_val.txt"), "ripeness", transform=val_tf)
    val_qual = FruitDataset(os.path.join(SPLIT_DIR, "split_val.txt"), "quality", transform=val_tf)

    print(f"成熟度: 训练{len(train_ripe)}张, 验证{len(val_ripe)}张")
    print(f"品质: 训练{len(train_qual)}张, 验证{len(val_qual)}张")

    train_ripe_loader = DataLoader(train_ripe, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_ripe_loader = DataLoader(val_ripe, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    train_qual_loader = DataLoader(train_qual, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_qual_loader = DataLoader(val_qual, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    results = {}
    results["ripeness"] = train_one_model("成熟度分级", 3, train_ripe_loader, val_ripe_loader, device)
    results["quality"] = train_one_model("品质等级判定", 3, train_qual_loader, val_qual_loader, device)

    print("\n" + "="*60)
    print("全部训练完成!")
    print("="*60)
    for task, acc in results.items():
        print(f"  {task}: {acc:.2f}%")

if __name__ == "__main__":
    main()
