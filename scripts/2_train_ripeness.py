# -*- coding: utf-8 -*-
# 任务2：新鲜度/成熟度多级分类模型 (MobileNetV2 迁移学习, 3分类: 未成熟/成熟/过熟)
# 用法: D:\anaconda3\python.exe scripts\3_train_ripeness.py
import os, sys

# 必须在导入 torch 前设置，避免 OpenMP 冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import common
from common import make_loaders, setup_chinese_font

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

NUM_CLASSES = 3
CLASS_NAMES = ["未成熟", "成熟", "过熟/腐败"]
EPOCHS = 3
BATCH_SIZE = 32
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("设备:", DEVICE)


def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model


def train():
    setup_chinese_font()
    import io as _io
    _prog = open(os.path.join(PROJECT_ROOT, "report", "train_progress.log"), "a", encoding="utf-8")
    train_loader, val_loader, test_loader = make_loaders(DATA_ROOT, "ripe", batch_size=BATCH_SIZE, train_cap=150)
    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    hist = {"train_loss": [], "val_acc": [], "val_loss": []}
    best_acc = 0.0

    import os as _os
    _live = open(_os.path.join(PROJECT_ROOT, "report", "train_live.log"), "a", encoding="utf-8")
    total_batches = len(train_loader)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for bi, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            if bi % 10 == 0 or bi == total_batches:
                avg = running_loss / (bi * imgs.size(0))
                _live.write(f"epoch {epoch}/{EPOCHS}  batch {bi}/{total_batches}  loss {avg:.4f}\n")
                _live.flush()
                print(f"  [epoch {epoch}/{EPOCHS}] batch {bi}/{total_batches}  loss {avg:.4f}", flush=True)
        scheduler.step()
        tr_loss = running_loss / len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item() * imgs.size(0)
                pred = out.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(val_loss / total)
        hist["val_acc"].append(val_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS}  train_loss={tr_loss:.4f}  val_loss={val_loss/total:.4f}  val_acc={val_acc:.4f}")
        _prog.write(f"{epoch}\t{tr_loss:.4f}\t{val_loss/total:.4f}\t{val_acc:.4f}\n"); _prog.flush()
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "ripeness_mobilenet.pth"))
            print("  -> 保存最佳模型")

    # 测试
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            all_pred += out.argmax(1).cpu().tolist()
            all_true += labels.cpu().tolist()
    print("\n== 测试集评估(成熟度) ==")
    print(classification_report(all_true, all_pred, target_names=CLASS_NAMES, digits=4))
    cm = confusion_matrix(all_true, all_pred)

    # 绘图: loss/acc曲线 + 混淆矩阵
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.plot(hist["train_loss"], label="train_loss")
    ax.plot(hist["val_loss"], label="val_loss")
    ax.set_title("Loss 曲线"); ax.set_xlabel("epoch"); ax.legend()

    ax = axes[1]
    ax.plot(hist["val_acc"], color="#2ca02c")
    ax.set_title("验证集准确率"); ax.set_xlabel("epoch"); ax.set_ylim(0, 1)

    ax = axes[2]
    im = ax.imshow(cm, cmap="Blues")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i][j], ha="center", va="center")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=30)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_title("混淆矩阵(成熟度)")
    fig.colorbar(im)
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, "2_成熟度模型结果.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("结果图:", out)


if __name__ == "__main__":
    train()
