# -*- coding: utf-8 -*-
# 公共模块：数据集加载、预处理、成熟度/品质标签生成
import os, random

# 解决 Anaconda 中 OpenMP 运行时冲突 (torch/numpy 均带 libiomp5md.dll)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


def estimate_ripeness(path):
    """根据颜色特征估计成熟度等级：0未成熟 1成熟 2过熟/腐败
       用HSV的色调(H)和饱和度(S)辅助判断，模拟人工分级，供课程项目演示。
       good样本一般偏成熟，bad样本偏过熟/腐败。"""
    img = cv2.imread(path)
    if img is None:
        return 1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    # 绿色占比高 -> 未熟; 整体饱和/青绿 -> 未熟
    green_mask = ((H >= 35) & (H <= 85)) & (S > 60)
    green_ratio = green_mask.mean()
    # 黑斑/暗沉(过熟腐败) 判据
    dark_ratio = (hsv[:, :, 2] < 80).mean()
    if dark_ratio > 0.25:
        return 2
    if green_ratio > 0.45:
        return 0
    return 1


def estimate_quality(path):
    """品质等级：0 A级优等 1 B级合格 2 C级残次
       以损伤/暗斑面积占比模拟判断：bad一般归为C级。"""
    img = cv2.imread(path)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = (gray < 90).mean()      # 深色损伤/腐烂占比
    low_sat = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1] < 40).mean()
    if dark > 0.30:
        return 2
    if dark > 0.12 or low_sat > 0.5:
        return 1
    return 0


class FruitDataset(Dataset):
    """从 split_*.txt 读取样本，可切换任务目标 ripe/quality"""
    def __init__(self, split_file, task, transform=None, augment_bad=False, cap=None):
        self.samples = []
        self.task = task
        self.transform = transform
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                path, fruit, quality = parts[0], parts[1], parts[2]
                self.samples.append((path, fruit, quality))
        # 训练抽样：按(水果,品质)每类最多取 cap 张，控制训练量以加快训练
        if cap:
            grp = defaultdict(list)
            for s in self.samples:
                grp[(s[1], s[2])].append(s)
            capped = []
            for key, lst in grp.items():
                random.shuffle(lst)
                capped += lst[:cap]
            self.samples = capped
        self._label_cache = {}
        # 简单类别均衡：对少数类做复制(上采样)，用于训练
        self._upsample(augment_bad)
        self._label_cache = {}

    def _upsample(self, augment_bad):
        if not augment_bad:
            return
        # 先批量计算一次标签并缓存（避免每张图被重复读两次）
        self._label_cache = {}
        lab_of = {}
        for i, (path, fruit, quality) in enumerate(self.samples):
            if self.task == "ripe":
                lab = estimate_ripeness(path)
            else:
                lab = estimate_quality(path)
            self._label_cache[i] = lab
            lab_of.setdefault(lab, []).append(i)
        labels = {k: len(v) for k, v in lab_of.items()}
        max_n = max(labels.values())
        new_samples = list(self.samples)
        extra = []
        for lab, cnt in labels.items():
            if cnt < max_n:
                pool = lab_of[lab]
                need = max_n - cnt
                reps = (pool * (need // len(pool) + 1))[:need]
                extra.extend(reps)
        # 追加复制样本，重建缓存
        base = len(new_samples)
        for idx in extra:
            new_samples.append(self.samples[idx])
            self._label_cache[base] = self._label_cache[idx]
            base += 1
        random.shuffle(new_samples)
        # 重排后缓存 key 失效，重建映射
        new_cache = {}
        for ni, si in enumerate(new_samples):
            # 用原始路径匹配缓存（存原始下标→标签）
            new_cache[ni] = self._label_cache.get(si)
        self.samples = new_samples
        self._label_cache = new_cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, fruit, quality = self.samples[idx]
        img = load_image(path)
        if self.transform:
            img = self.transform(img)
        # 缓存标签，避免每 epoch 重复做颜色特征计算
        if idx not in self._label_cache:
            if self.task == "ripe":
                self._label_cache[idx] = estimate_ripeness(path)
            else:
                self._label_cache[idx] = estimate_quality(path)
        return img, self._label_cache[idx]


def make_loaders(data_root, task, batch_size=32, train_cap=None):
    tr = os.path.join(data_root, "split_train.txt")
    va = os.path.join(data_root, "split_val.txt")
    te = os.path.join(data_root, "split_test.txt")
    ds_tr = FruitDataset(tr, task, transform=train_tf, augment_bad=True, cap=train_cap)
    ds_va = FruitDataset(va, task, transform=val_tf)
    ds_te = FruitDataset(te, task, transform=val_tf)
    return (DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0),
            DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0))

def setup_chinese_font():
    import matplotlib
    from matplotlib import font_manager
    candidates = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]
    for c in candidates:
        if os.path.exists(c):
            font_manager.fontManager.addfont(c)
            prop = font_manager.FontProperties(fname=c)
            matplotlib.rcParams["font.family"] = prop.get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False