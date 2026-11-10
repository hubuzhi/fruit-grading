# -*- coding: utf-8 -*-
# 任务1：图像数据治理与类别均衡（适配真实结构）
import os, random
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
os.makedirs(REPORT_DIR, exist_ok=True)
IMG_SIZE = 224
SEED = 42
random.seed(SEED); np.random.seed(SEED)
QUALITY_DIRS = ["Bad Quality_Fruits", "Good Quality_Fruits"]
FRUIT_MAP = {"Apple":"苹果","Banana":"香蕉","Guava":"番石榴","Lime":"青柠","Lemon":"柠檬","Orange":"橙子","Pomegranate":"石榴"}

def collect_samples():
    samples = []
    for qdir_name in QUALITY_DIRS:
        qdir = os.path.join(DATA_ROOT, qdir_name)
        if not os.path.isdir(qdir): continue
        is_bad = "Bad" in qdir_name
        for fruit_dir in sorted(os.listdir(qdir)):
            fd = os.path.join(qdir, fruit_dir)
            if not os.path.isdir(fd): continue
            fruit_key = fruit_dir.rsplit("_",1)[0]
            fruit_zh = FRUIT_MAP.get(fruit_key, fruit_key)
            files = [f for f in os.listdir(fd) if f.lower().endswith((".jpg",".jpeg",".png"))]
            for f in files:
                samples.append((os.path.join(fd,f), fruit_zh, 1 if is_bad else 0))
    return samples

def plot_distribution(samples):
    counter = Counter()
    for _, fruit, quality in samples:
        counter[f"{fruit}-{'坏' if quality else '好'}"] += 1
    labels = list(counter.keys()); vals = [counter[k] for k in labels]
    order = sorted(range(len(labels)), key=lambda i: vals[i])
    labels = [labels[i] for i in order]; vals = [vals[i] for i in order]
    fig, ax = plt.subplots(figsize=(12,6))
    colors = ["#d62728" if "坏" in l else "#2ca02c" for l in labels]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_title("各水果x品质类别分布（可见类别不均衡）"); ax.set_ylabel("样本数")
    ax.tick_params(axis="x", rotation=45)
    for b,v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v+20, str(v), ha="center", fontsize=8)
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, "1_类别分布.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("分布图:", out)
    total=sum(vals); top=max(vals); low=min(vals)
    print(f"总样本 {total}，最多 {top}，最少 {low}，不均衡比 {top/low:.1f}:1")

def preprocess_samples(samples, limit=6):
    out_root = os.path.join(REPORT_DIR, "preprocess")
    for fruit in FRUIT_MAP.values():
        picked = [s for s in samples if s[1]==fruit][:limit]
        for i,(path,_,_) in enumerate(picked):
            img = cv2.imread(path)
            if img is None: continue
            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
            img_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            d = os.path.join(out_root, fruit); os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, f"{i}_eq.png"), img_eq)
    print("预处理示例已保存到 report/preprocess/")

def build_splits(samples, train_ratio=0.7, val_ratio=0.15):
    groups = defaultdict(list)
    for s in samples: groups[(s[1],s[2])].append(s)
    splits = {"train":[],"val":[],"test":[]}
    for key,lst in groups.items():
        random.shuffle(lst); n=len(lst)
        n_tr=int(n*train_ratio); n_va=int(n*val_ratio)
        splits["train"]+=lst[:n_tr]; splits["val"]+=lst[n_tr:n_tr+n_va]; splits["test"]+=lst[n_tr+n_va:]
    for name,lst in splits.items():
        with open(os.path.join(DATA_ROOT, f"split_{name}.txt"),"w",encoding="utf-8") as f:
            for path,fruit,quality in lst: f.write(f"{path}\t{fruit}\t{quality}\n")
        print(f"split_{name}.txt: {len(lst)} 条")
    return splits

def main():
    from common import setup_chinese_font
    setup_chinese_font()
    samples = collect_samples()
    print("总样本数:", len(samples))
    plot_distribution(samples)
    preprocess_samples(samples)
    build_splits(samples)
    print("\n任务1完成。数据已按水果x品质分层划分为 train/val/test。")

if __name__ == "__main__":
    main()