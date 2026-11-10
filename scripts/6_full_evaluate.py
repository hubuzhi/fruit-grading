# -*- coding: utf-8 -*-
"""完整评估脚本：对三个模型生成混淆矩阵/ROC/PR/分水果准确率/ECE等图表"""
import os, sys, json, random
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score, f1_score, accuracy_score)
from sklearn.preprocessing import label_binarize
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (val_tf, load_image, estimate_ripeness, estimate_quality,
                    setup_chinese_font, IMG_SIZE, MEAN, STD)

setup_chinese_font()

DATA_ROOT = r"D:\fruit_project\data"
MODEL_DIR = r"D:\fruit_project\models"
OUT_DIR = r"D:\fruit_project\eval_full"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

FRUITS = ["Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate"]
FRUIT_CN = {"Apple": "苹果", "Banana": "香蕉", "Guava": "番石榴",
            "Lime": "青柠", "Orange": "橙子", "Pomegranate": "石榴"}
RIPENESS_CN = ["未成熟", "成熟", "过熟腐败"]
QUALITY_CN = ["A级(优等)", "B级(合格)", "C级(残次)"]


def load_split(split_file):
    samples = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            path, fruit, quality = parts[0], parts[1], parts[2]
            samples.append((path, fruit, quality))
    return samples


def build_model(num_classes, model_path):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def predict_batch(model, paths, batch_size=32):
    all_probs = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            img = load_image(p)
            imgs.append(val_tf(img))
        batch = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def calc_ece(confidences, predictions, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i+1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc = (predictions[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()
        ece += np.abs(acc - conf) * mask.mean()
    return ece


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{title} - 样本数")
    axes[0].set_ylabel("真实标签")
    axes[0].set_xlabel("预测标签")

    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"{title} - 行归一化")
    axes[1].set_ylabel("真实标签")
    axes[1].set_xlabel("预测标签")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  混淆矩阵 → {save_path}")


def plot_roc_pr(y_true, y_probs, num_classes, class_names, title_prefix, save_prefix):
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    # ROC
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} - ROC曲线")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_roc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC → {save_prefix}_roc.png")

    # PR
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        plt.plot(recall, precision, label=f"{class_names[i]} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} - PR曲线")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_pr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PR → {save_prefix}_pr.png")


def evaluate_model(task_name, num_classes, class_names, model_path, test_samples, label_fn):
    print(f"\n{'='*50}")
    print(f"评估: {task_name}")
    print(f"{'='*50}")

    model = build_model(num_classes, model_path)

    paths = [s[0] for s in test_samples]
    fruits = [s[1] for s in test_samples]

    y_true = np.array([label_fn(p) for p in paths])
    y_probs = predict_batch(model, paths)
    y_pred = y_probs.argmax(axis=1)
    confidences = y_probs.max(axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    ece = calc_ece(confidences, y_pred, y_true)

    print(f"  样本数: {len(y_true)}")
    print(f"  准确率: {acc:.4f}")
    print(f"  宏F1: {f1_macro:.4f}")
    print(f"  ECE: {ece:.4f}")

    # 分类报告
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"  分类报告:\n{report}")

    # 混淆矩阵
    plot_confusion_matrix(y_true, y_pred, class_names, task_name,
                          os.path.join(OUT_DIR, f"{task_name}_confusion_matrix.png"))

    # ROC + PR
    plot_roc_pr(y_true, y_probs, num_classes, class_names, task_name,
                os.path.join(OUT_DIR, task_name))

    # 分水果准确率
    fruit_accs = {}
    for fruit in FRUITS:
        mask = np.array([f == fruit for f in fruits])
        if mask.sum() > 0:
            f_acc = (y_pred[mask] == y_true[mask]).mean()
            fruit_accs[FRUIT_CN[fruit]] = {"n": int(mask.sum()), "acc": float(f_acc)}
    print(f"  分水果准确率:")
    for fn, info in fruit_accs.items():
        print(f"    {fn}: {info['acc']:.4f} ({info['n']}张)")

    plt.figure(figsize=(8, 5))
    names = list(fruit_accs.keys())
    accs = [fruit_accs[n]["acc"] * 100 for n in names]
    bars = plt.bar(names, accs, color=["#4C72B0", "#55A868", "#C44E52",
                                        "#8172B2", "#CCB974", "#64B5CD"])
    plt.ylabel("准确率 (%)")
    plt.title(f"{task_name} - 分水果准确率")
    plt.ylim(0, 105)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{acc:.1f}%", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{task_name}_per_fruit.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  分水果图 → {task_name}_per_fruit.png")

    # 置信度分布
    plt.figure(figsize=(8, 5))
    correct_mask = y_pred == y_true
    plt.hist(confidences[correct_mask], bins=20, alpha=0.6, label="预测正确", color="#55A868")
    plt.hist(confidences[~correct_mask], bins=20, alpha=0.6, label="预测错误", color="#C44E52")
    plt.xlabel("置信度")
    plt.ylabel("样本数")
    plt.title(f"{task_name} - 置信度分布 (ECE={ece:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{task_name}_confidence_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  置信度分布 → {task_name}_confidence_hist.png")

    # 误判图册（取前9个置信度最高的错误）
    wrong_idx = np.where(y_pred != y_true)[0]
    if len(wrong_idx) > 0:
        wrong_conf = confidences[wrong_idx]
        top_wrong = wrong_idx[np.argsort(-wrong_conf)[:9]]
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for i, idx in enumerate(top_wrong):
            ax = axes[i // 3][i % 3]
            img = load_image(paths[idx])
            ax.imshow(img)
            ax.set_title(f"真实:{class_names[y_true[idx]]}\n预测:{class_names[y_pred[idx]]}\n置信度:{confidences[idx]:.2f}",
                         fontsize=9)
            ax.axis("off")
        plt.suptitle(f"{task_name} - 误判图册（置信度最高的错误）", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{task_name}_error_gallery.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  误判图册 → {task_name}_error_gallery.png")

    return {
        "task": task_name,
        "n_samples": int(len(y_true)),
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "ece": float(ece),
        "per_fruit": fruit_accs,
        "classification_report": report,
    }


# ==========================================
# 主流程
# ==========================================
print("加载测试集...")
test_samples = load_split(os.path.join(DATA_ROOT, "split_test.txt"))
print(f"测试集: {len(test_samples)} 张")

results = {}

# 1. 水果种类识别（6分类，标签=目录里的水果名）
fruit_to_idx = {f: i for i, f in enumerate(FRUITS)}
def fruit_label(path):
    # 从路径里找水果名
    for f in FRUITS:
        if f in path:
            return fruit_to_idx[f]
    return 0

results["fruit"] = evaluate_model(
    "水果种类识别", 6, FRUITS,
    os.path.join(MODEL_DIR, "fruit_mobilenet_weighted.pth"),
    test_samples, fruit_label
)

# 2. 成熟度分类（3分类）
results["ripeness"] = evaluate_model(
    "成熟度分级", 3, RIPENESS_CN,
    os.path.join(MODEL_DIR, "ripeness_mobilenet_full.pth"),
    test_samples, estimate_ripeness
)

# 3. 品质分级（3分类）
results["quality"] = evaluate_model(
    "品质等级判定", 3, QUALITY_CN,
    os.path.join(MODEL_DIR, "quality_mobilenet_full.pth"),
    test_samples, estimate_quality
)

# 保存汇总指标
with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n{'='*50}")
print(f"全部评估完成！结果保存在: {OUT_DIR}")
print(f"{'='*50}")
