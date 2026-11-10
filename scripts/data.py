"""数据层：官方划分读取、标签口径校验、增强变换、加权采样。

数据源约定（已定稿）
--------------------
**唯一数据源是 `metadata/{train,val,test}.csv`**，其中 `quality` 列给出 Good/Bad，
`relative_path` 拼 `data_root` 得到本机路径，`label` 统一为 **Bad=0 / Good=1**——
与 `train_quality.py` 和 `app.py` 的 `QUALITY_CLASSES = ["劣质","优质"]` 一致。

`split_*.txt` 已弃用（它是早期导出物，标签写成 0=Good/1=Bad，与上面相反）。
本模块仍保留 `load_split_txt()` 仅作历史兼容，正常流程不会走到它；
如被显式调用，会自动把标签翻转成项目口径并打印审计信息，避免误用。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# 与 app.py 的 QUALITY_CLASSES = ["劣质", "优质"] 严格对应
CLASSES: List[str] = ["Bad", "Good"]          # index 0 = 劣质, 1 = 优质
CLASSES_CN: Dict[str, str] = {"Bad": "劣质", "Good": "优质"}
QUALITY_TO_LABEL: Dict[str, int] = {"Bad": 0, "Good": 1}
LABEL_TO_QUALITY: Dict[int, str] = {0: "Bad", 1: "Good"}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
def _resolve_paths(raw: str, data_root: Optional[str], relative: Optional[str] = None) -> str:
    """把一条记录解析成本机可读的绝对路径。

    优先用 `relative_path` 拼 `data_root`（换机器只需改 data_root），
    退化时用原始绝对路径。
    """
    if data_root and relative:
        candidate = os.path.join(data_root, *relative.replace("/", os.sep).split(os.sep))
        if os.path.exists(candidate):
            return candidate
    return raw


def load_metadata_csv(
    csv_path: str,
    data_root: Optional[str] = None,
    check_files: bool = True,
) -> pd.DataFrame:
    """读取 metadata/{train,val,test}.csv，返回 [filepath, fruit, quality, label, split]。"""
    df = pd.read_csv(csv_path)
    required = {"file_path", "fruit", "quality"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(csv_path)} 缺少列: {sorted(missing)}")

    rel = df["relative_path"] if "relative_path" in df.columns else None
    df["filepath"] = [
        _resolve_paths(fp, data_root, rel.iloc[i] if rel is not None else None)
        for i, fp in enumerate(df["file_path"])
    ]

    # 以 quality 字符串为准重建 label，不信任 CSV 里可能被改过的 label 列
    df["label"] = df["quality"].map(QUALITY_TO_LABEL)
    if df["label"].isna().any():
        bad = sorted(df.loc[df["label"].isna(), "quality"].unique().tolist())
        raise ValueError(f"出现未知 quality 取值: {bad}，应为 {list(QUALITY_TO_LABEL)}")
    df["label"] = df["label"].astype(int)
    df["split"] = df.get("split", os.path.splitext(os.path.basename(csv_path))[0])

    if check_files:
        _check_files(df, csv_path)
    return df[["filepath", "fruit", "quality", "label", "split"]].reset_index(drop=True)


def load_split_txt(
    txt_path: str,
    data_root: Optional[str] = None,
    check_files: bool = True,
    flip_labels: bool = True,
) -> pd.DataFrame:
    """【历史兼容，正常流程不使用】读取早期导出的 `split_*.txt`。

    该文件里 Good=0 / Bad=1，与项目口径相反，所以默认自动翻转。
    新流程请直接用 `load_metadata_csv()`；保留本函数只是为了让旧脚本不至于报错。
    """
    rows = []
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            with open(txt_path, "r", encoding=enc) as f:
                lines = f.read().splitlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError(f"无法解码 {txt_path}")

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"{os.path.basename(txt_path)} 第 {i} 行列数不足: {line[:60]!r}")
        raw_path, fruit, raw_label = parts[0].strip(), parts[1].strip(), int(parts[2])

        parent = os.path.basename(os.path.dirname(raw_path.replace("/", os.sep)))
        folder_quality = "Bad" if parent.endswith("_Bad") else ("Good" if parent.endswith("_Good") else None)
        folder_fruit = parent.rsplit("_", 1)[0] if "_" in parent else ""

        if data_root:
            norm = raw_path.replace("/", os.sep)
            segs = [s for s in norm.split(os.sep) if s]
            # 官方清单里没有 relative_path，这里以 “<类别目录>/<文件名>” 重建
            path = os.path.join(data_root, *segs[-2:])
        else:
            path = raw_path

        rows.append(
            {
                "filepath": path,
                "fruit": folder_fruit or fruit,
                "quality": folder_quality,
                "raw_label": raw_label,
            }
        )

    df = pd.DataFrame(rows)

    # 用目录名反推真实类别，再决定是否需要翻转
    df["quality"] = df["quality"].fillna(df["raw_label"].map(LABEL_TO_QUALITY))
    if flip_labels:
        df["label"] = df["quality"].map(QUALITY_TO_LABEL)
        mismatch = int((df["label"] != df["raw_label"]).sum())
        if mismatch:
            print(
                f"⚠️  {os.path.basename(txt_path)}: 检测到标签口径与项目约定相反，"
                f"已翻转 {mismatch}/{len(df)} 条（0=Good→1，1=Bad→0）"
            )
    else:
        df["label"] = df["raw_label"]

    df["split"] = os.path.splitext(os.path.basename(txt_path))[0].replace("split_", "")
    df = df.drop(columns=["raw_label"])

    if check_files:
        _check_files(df, txt_path)
    return df[["filepath", "fruit", "quality", "label", "split"]].reset_index(drop=True)


def _check_files(df: pd.DataFrame, source: str) -> None:
    exists = df["filepath"].map(os.path.exists)
    n_missing = int((~exists).sum())
    if n_missing:
        sample = "\n  ".join(df.loc[~exists, "filepath"].head(5))
        raise FileNotFoundError(
            f"{os.path.basename(source)}: {n_missing}/{len(df)} 个文件在本机不存在。\n"
            f"请用 --data-root 指向数据集根目录（形如 ...\\Fruit_Project）。示例:\n  {sample}"
        )


def load_splits(
    metadata_dir: Optional[str] = None,
    split_txt_prefix: Optional[str] = None,
    data_root: Optional[str] = None,
    check_files: bool = True,
) -> Dict[str, pd.DataFrame]:
    """一次性读入 train/val/test。

    唯一推荐路径：`metadata_dir`（含 train.csv / val.csv / test.csv）。
    `split_txt_prefix` 仅为历史兼容保留，新流程不要使用。
    """
    if metadata_dir:
        out = {}
        for split in ("train", "val", "test"):
            path = os.path.join(metadata_dir, f"{split}.csv")
            out[split] = load_metadata_csv(path, data_root, check_files)
    elif split_txt_prefix:
        out = {}
        for split in ("train", "val", "test"):
            path = f"{split_txt_prefix}{split}.txt"
            out[split] = load_split_txt(path, data_root, check_files)
    else:
        raise ValueError("必须提供 metadata_dir 或 split_txt_prefix 之一")

    audit_splits(out)
    return out


def audit_splits(splits: Dict[str, pd.DataFrame]) -> None:
    """打印三份划分的规模、类别分布与跨集重复检查结果。"""
    print("\n=== 数据审计 ===")
    for name in ("train", "val", "test"):
        df = splits[name]
        counts = df["label"].value_counts().sort_index().to_dict()
        good = counts.get(1, 0)
        bad = counts.get(0, 0)
        ratio = good / max(len(df), 1)
        print(f"  {name:5s} n={len(df):6d}  Good(1)={good:5d}  Bad(0)={bad:5d}  Good占比={ratio:.1%}")

    def keys(df):
        return {
            f"{os.path.basename(p).lower()}|{os.path.basename(os.path.dirname(p)).lower()}"
            for p in df["filepath"]
        }

    tr, va, te = keys(splits["train"]), keys(splits["val"]), keys(splits["test"])
    overlaps = {"train∩val": len(tr & va), "train∩test": len(tr & te), "val∩test": len(va & te)}
    print(f"  跨集重复: {overlaps}")
    if any(overlaps.values()):
        print("  ⚠️  存在跨集重复，说明数据泄漏，指标会虚高，需回到数据划分阶段排查")


def class_weights(splits: Dict[str, pd.DataFrame], split: str = "train") -> np.ndarray:
    """balanced 类别权重（纯 NumPy，不依赖 torch），顺序与 CLASSES 一致。

    w_c = N / (K * n_c)。Good 样本多，权重更小；仅用于加权 CrossEntropyLoss。
    """
    df = splits[split]
    n = len(df)
    k = len(CLASSES)
    counts = df["label"].value_counts().to_dict()
    return np.asarray([n / (k * counts.get(c, 1)) for c in range(k)], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Dataset 与变换
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    path: str
    label: int
    fruit: str
    quality: str


class FruitDataset:
    """按 DataFrame 读取图像的最小 Dataset（不依赖 torchvision.datasets.ImageFolder）。

    用 DataFrame 而不是 ImageFolder，是为了**严格复用官方划分**，
    避免 ImageFolder 自己按目录顺序重新分配标签（这也是标签错位的常见来源）。
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.samples = [
            Sample(r.filepath, int(r.label), str(r.fruit), str(r.quality))
            for r in self.df.itertuples(index=False)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        s = self.samples[index]
        image = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, s.label

    def labels(self) -> List[int]:
        return [s.label for s in self.samples]

    def fruits(self) -> List[str]:
        return [s.fruit for s in self.samples]


def build_transforms(img_size: int = 224, train: bool = True, strong: bool = False):
    """训练/评估变换。

    训练：RandomResizedCrop + 水平/垂直翻转 + 旋转 + 颜色抖动（颜色抖动对“品质”判定很关键，
    但幅度不能太大，否则会把“发黄/褐变”这类真实品质线索一起抹掉，所以 brightness/contrast
    控制在 0.2 以内，saturation 稍大一点）。
    评估：Resize(256) + CenterCrop(224)，与 ImageNet 惯例一致。
    """
    from torchvision import transforms

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if not train:
        return transforms.Compose(
            [
                transforms.Resize(int(img_size * 256 / 224)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    ops = [
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.25, hue=0.03),
    ]
    if strong:
        ops.append(transforms.RandomVerticalFlip(p=0.15))
        ops.append(transforms.RandomGrayscale(p=0.05))
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


_PIPE_OK: Optional[bool] = None


def pipes_available() -> bool:
    """探测当前环境能否创建多进程管道（DataLoader 的 num_workers>0 依赖它）。

    受限沙箱、以及某些裁剪过的容器里 `CreateFile` 会被拒绝，表现是
    `PermissionError: [WinError 5]`。这里做一次缓存探测，避免每轮都试。
    """
    global _PIPE_OK
    if _PIPE_OK is None:
        try:
            import multiprocessing as mp

            q = mp.get_context("spawn").Queue()
            q.close()
            _PIPE_OK = True
        except Exception:
            _PIPE_OK = False
    return bool(_PIPE_OK)


def resolve_num_workers(requested: int) -> int:
    """把用户给的 num_workers 收敛到本环境真正可用的值。"""
    n = max(0, int(requested or 0))
    if n > 0 and not pipes_available():
        print("⚠️  当前环境不支持多进程数据加载（Pipe 创建被拒），num_workers 自动降为 0")
        return 0
    return n


def make_loader(
    df: pd.DataFrame,
    transform,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    weighted_sampling: bool = False,
):
    """构建 DataLoader；`weighted_sampling=True` 时按类别倒数概率重采样。"""
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    num_workers = resolve_num_workers(num_workers)

    ds = FruitDataset(df, transform)
    sampler = None
    if weighted_sampling and shuffle:
        labels = np.asarray(ds.labels())
        counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64)
        per_class = 1.0 / np.maximum(counts, 1)
        sample_w = per_class[labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(ds),
            replacement=True,
        )
        shuffle = False  # 与 sampler 互斥

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def dataset_report(splits: Dict[str, pd.DataFrame], out_csv: Optional[str] = None) -> pd.DataFrame:
    """生成“划分 × 水果 × 类别”的统计表（报告里直接用）。"""
    frames = []
    for name, df in splits.items():
        t = df.groupby(["fruit", "quality"]).size().reset_index(name="count")
        t["split"] = name
        frames.append(t)
    table = pd.concat(frames, ignore_index=True)[["split", "fruit", "quality", "count"]]
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        table.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return table
