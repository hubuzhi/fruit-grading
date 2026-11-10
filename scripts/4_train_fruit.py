# -*- coding: utf-8 -*-
# 水果种类识别模型 (MobileNetV2, 6分类) - 供任务4可视化系统使用
# 用法: D:\anaconda3\python.exe scripts\5_train_fruit.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

FRUIT_NAMES = ["苹果", "香蕉", "番石榴", "青柠", "橙子", "石榴"]
FRUIT_TO_IDX = {n: i for i, n in enumerate(FRUIT_NAMES)}
NUM_CLASSES = 6
EPOCHS = 3
BATCH_SIZE = 32
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("设备:", DEVICE)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
from torchvision import transforms as T
train_tf = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
val_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])


class FruitDS(torch.utils.data.Dataset):
    def __init__(self, split_file, transform=None, cap=200):
        self.samples = []
        self.transform = transform
        from collections import defaultdict
        grp = defaultdict(list)
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                path, fruit, quality = line.split("\t")
                grp[fruit].append((path, fruit))
        # 每类最多 cap 张
        for fruit, lst in grp.items():
            import random
            random.shuffle(lst)
            self.samples += lst[:cap]
        self.samples = [(p, f) for p, f in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, fruit = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, FRUIT_TO_IDX.get(fruit, 0)


def make_loaders(cap=200):
    from PIL import Image
    tr = FruitDS(os.path.join(DATA_ROOT, "split_train.txt"), train_tf, cap=cap)
    va = FruitDS(os.path.join(DATA_ROOT, "split_val.txt"), val_tf, cap=100)
    te = FruitDS(os.path.join(DATA_ROOT, "split_test.txt"), val_tf, cap=100)
    return (torch.utils.data.DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
            torch.utils.data.DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
            torch.utils.data.DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0))


def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model


def train():
    from PIL import Image
    from common import setup_chinese_font
    setup_chinese_font()
    train_loader, val_loader, test_loader = make_loaders(cap=200)
    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    hist = {"train_loss": [], "val_acc": []}
    best = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        tr_loss = running / len(train_loader.dataset)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                pred = out.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        hist["train_loss"].append(tr_loss)
        hist["val_acc"].append(val_acc)
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={tr_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "fruit_mobilenet.pth"))
            print("  -> 保存最佳")

    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            all_pred += out.argmax(1).cpu().tolist()
            all_true += labels.cpu().tolist()
    print("\n== 测试集评估(水果种类) ==")
    print(classification_report(all_true, all_pred, target_names=FRUIT_NAMES, labels=list(range(NUM_CLASSES)), zero_division=0, digits=4))
    cm = confusion_matrix(all_true, all_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(hist["train_loss"], label="train_loss"); ax.plot(hist["val_acc"], label="val_acc")
    ax.set_title("训练过程"); ax.set_xlabel("epoch"); ax.legend()
    ax = axes[1]
    ax.imshow(cm, cmap="Blues")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i][j], ha="center", va="center")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(FRUIT_NAMES, rotation=30)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(FRUIT_NAMES)
    ax.set_title("混淆矩阵(水果种类)")
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, "4_水果模型结果.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("结果图:", out)


if __name__ == "__main__":
    train()
