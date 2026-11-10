"""图表生成：学习曲线、混淆矩阵、ROC/PR、模型对比、数据集分布、错误样本。

所有图统一 300 dpi、白底、中文标签，直接可以插进 Word 报告和 PPT。
中文字体按 Windows → Linux → macOS 顺序探测，找不到时退回 DejaVu Sans（中文会显示成方块，
这时脚本会打印警告，提示把字体文件放到 `fonts/` 目录或改用英文标签）。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

import metrics as M

_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "PingFang SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]


def setup_style(dpi: int = 300, font_size: int = 10) -> bool:
    """配置 matplotlib 中文字体，返回是否找到了中文字体。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = next((f for f in _FONT_CANDIDATES if f in available), None)
    if picked:
        plt.rcParams["font.family"] = picked
    else:
        print("⚠️  未找到中文字体，图表中文可能显示为方块；可改用英文标签或安装 fonts-noto-cjk")

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    return picked is not None and picked != "DejaVu Sans"


def _save(fig, out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"  🖼  已保存 {out_path}")
    return out_path


def _read_history_csv(history_csv: str):
    """读取 Keras CSVLogger 输出或 history.json 导出的曲线数据。"""
    import pandas as pd

    df = pd.read_csv(history_csv)
    df.columns = [c.strip() for c in df.columns]
    for col in ("accuracy", "acc"):
        if col in df.columns and "accuracy" not in df.columns:
            df = df.rename(columns={col: "accuracy"})
    return df


# --------------------------------------------------------------------------- #
# 学习曲线 + 过拟合间隙
# --------------------------------------------------------------------------- #
def plot_learning_curves(
    history_csv: str,
    out_path: str = "figs/learning_curves.png",
    title: str = "训练过程：准确率、损失与泛化间隙",
    mark_best: bool = True,
) -> Optional[str]:
    """三联图：准确率曲线 / 损失曲线 / 训练-验证间隙。

    第三个子图是讲“过拟合”的核心证据：间隙在最优轮之后持续拉大，说明模型开始记忆训练集。
    """
    import matplotlib.pyplot as plt

    df = _read_history_csv(history_csv)
    if "epoch" not in df.columns:
        df["epoch"] = np.arange(1, len(df) + 1)
    if "val_accuracy" not in df.columns:
        print("⚠️  历史记录里没有 val_accuracy，跳过学习曲线")
        return None

    epoch = df["epoch"].to_numpy()
    tr_acc = df.get("accuracy")
    va_acc = df["val_accuracy"]
    tr_loss = df.get("loss")
    va_loss = df.get("val_loss")

    best_idx = int(np.argmax(va_acc.to_numpy())) if mark_best else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    if tr_acc is not None:
        ax.plot(epoch, tr_acc, marker="o", ms=3, label="训练集准确率")
    ax.plot(epoch, va_acc, marker="s", ms=3, label="验证集准确率")
    if best_idx is not None:
        ax.axvline(epoch[best_idx], color="tab:red", ls="--", lw=1, alpha=0.7)
        ax.scatter(
            [epoch[best_idx]], [va_acc.iloc[best_idx]], color="tab:red", zorder=5, s=45,
            label=f"最优轮次 E{int(epoch[best_idx])}={va_acc.iloc[best_idx]:.4f}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) 准确率曲线")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.02)

    ax = axes[1]
    if tr_loss is not None:
        ax.plot(epoch, tr_loss, marker="o", ms=3, label="训练集损失")
    if va_loss is not None:
        ax.plot(epoch, va_loss, marker="s", ms=3, label="验证集损失")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Categorical crossentropy")
    ax.set_title("(b) 损失曲线")
    ax.legend(fontsize=8)

    ax = axes[2]
    if tr_acc is not None:
        gap = tr_acc.to_numpy() - va_acc.to_numpy()
        ax.plot(epoch, gap, marker="^", ms=4, color="tab:purple", label="训练-验证准确率差")
        ax.axhline(0, color="gray", lw=1)
        ax.fill_between(epoch, 0, gap, where=gap > 0, color="tab:purple", alpha=0.15,
                        label="过拟合区间（训练>验证）")
        if best_idx is not None:
            ax.axvline(epoch[best_idx], color="tab:red", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap")
    ax.set_title("(c) 泛化间隙（越大越过拟合）")
    ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# 混淆矩阵
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    out_path: str = "figs/confusion_matrix.png",
    title: str = "测试集混淆矩阵",
    show_counts: bool = True,
) -> str:
    """左右两栏：左为原始计数，右为按真实类别归一化后的比例。"""
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    k = cm.shape[0]
    row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm / row_sum

    fig, axes = plt.subplots(1, 2, figsize=(4.6 * k + 2, 4.2 * k / 1.6 + 1.6))

    for ax, data, subtitle, fmt in (
        (axes[0], cm, "(a) 样本数", "d"),
        (axes[1], cm_norm, "(b) 按行归一化比例", ".2f"),
    ):
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=None)
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(class_names, rotation=20, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("预测类别 (Predicted)")
        ax.set_ylabel("真实类别 (True)")
        ax.set_title(subtitle)
        ax.grid(False)

        threshold = data.max() / 2 if data.max() > 0 else 0.5
        for i in range(k):
            for j in range(k):
                value = data[i, j]
                txt = f"{int(value)}" if fmt == "d" else f"{value:.2%}"
                if not show_counts and fmt == "d" and value == 0:
                    txt = ""
                ax.text(
                    j, i, txt, ha="center", va="center", fontsize=11,
                    color="white" if value > threshold else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# ROC / PR
# --------------------------------------------------------------------------- #
def plot_roc_pr(
    probs: np.ndarray,
    y_true: np.ndarray,
    class_names: Sequence[str],
    out_roc: str = "figs/roc_curves.png",
    out_pr: str = "figs/pr_curves.png",
) -> Dict[str, str]:
    """One-vs-Rest 的 ROC 与 PR 曲线（二分类时只有一条，多分类时逐类一条）。"""
    import matplotlib.pyplot as plt

    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    k = probs.shape[1]
    out = {}

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for c in range(k):
        y_bin = (y_true == c).astype(np.int64)
        if y_bin.sum() == 0:
            continue
        fpr, tpr = M.binary_roc_curve(y_bin, probs[:, c])
        auc = M.auc_roc(y_bin, probs[:, c])
        ax.plot(fpr, tpr, lw=1.8, label=f"{class_names[c]} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="随机猜测")
    ax.set_xlabel("False Positive Rate (1-特异度)")
    ax.set_ylabel("True Positive Rate (召回率)")
    ax.set_title("测试集 ROC 曲线（One-vs-Rest）")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    out["roc"] = _save(fig, out_roc)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for c in range(k):
        y_bin = (y_true == c).astype(np.int64)
        if y_bin.sum() == 0:
            continue
        recall, precision = M.binary_pr_curve(y_bin, probs[:, c])
        ap = M.average_precision(y_bin, probs[:, c])
        ax.step(recall, precision, where="post", lw=1.8,
                label=f"{class_names[c]} (AP={ap:.4f})")
    base = float((y_true == y_true.max()).mean())
    ax.axhline(base, color="k", ls="--", lw=1, alpha=0.6, label=f"随机基线={base:.3f}")
    ax.set_xlabel("Recall (召回率)")
    ax.set_ylabel("Precision (精确率)")
    ax.set_title("测试集 PR 曲线（One-vs-Rest）")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    out["pr"] = _save(fig, out_pr)
    return out


# --------------------------------------------------------------------------- #
# 模型 / 超参数对比
# --------------------------------------------------------------------------- #
def plot_model_comparison(
    results_csv: str,
    out_path: str = "figs/model_comparison.png",
    metric_cols: Sequence[str] = ("accuracy", "macro_f1"),
    label_col: str = "run_name",
) -> Optional[str]:
    """调参实验对比条形图（准确率 + 宏 F1 双指标）。"""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(results_csv)
    if df.empty:
        return None
    cols = [c for c in metric_cols if c in df.columns]
    if not cols or label_col not in df.columns:
        print(f"⚠️  对比图表缺少列: 需要 {label_col} 与 {cols}")
        return None

    labels = df[label_col].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.8 / len(cols)

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels)), 4.4))
    for i, col in enumerate(cols):
        vals = df[col].to_numpy(dtype=float)
        bars = ax.bar(x + i * width - (len(cols) - 1) * width / 2, vals, width, label=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, min(1.0, max(0.6, float(df[list(cols)].to_numpy().max()) + 0.12)))
    ax.set_title("不同模型 / 超参数配置的测试集表现对比")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_hyperparam_effect(
    results_csv: str,
    param: str,
    out_path: Optional[str] = None,
    metric: str = "accuracy",
) -> Optional[str]:
    """单超参数扫描曲线（例如 learning_rate 对准确率的影响）。"""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(results_csv)
    if param not in df.columns or metric not in df.columns:
        return None
    out_path = out_path or f"figs/hparam_{param}.png"

    grouped = df.sort_values(param).groupby(param)[metric].agg(["mean", "max", "min"])
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(grouped.index.astype(str), grouped["mean"], marker="o", label=f"{metric} 均值")
    ax.fill_between(
        range(len(grouped)), grouped["min"], grouped["max"], alpha=0.15, label="min-max 区间"
    )
    ax.set_xlabel(param)
    ax.set_ylabel(metric)
    ax.set_title(f"超参数 {param} 对 {metric} 的影响")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# 数据集分布（配合“类别均衡”章节，也方便副组长写数据治理部分）
# --------------------------------------------------------------------------- #
def plot_dataset_distribution(
    frames,
    out_path: str = "figs/dataset_distribution.png",
) -> str:
    """左：三份划分的类别样本量；右：训练集里各水果 × 类别的分布。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    ax = axes[0]
    split_names = ["train", "val", "test"]
    class_names = frames.class_names
    width = 0.8 / len(class_names)
    x = np.arange(len(split_names))
    for i, cname in enumerate(class_names):
        vals = [
            int((frames.as_dict()[s]["label_name"] == cname).sum()) for s in split_names
        ]
        bars = ax.bar(x + i * width - (len(class_names) - 1) * width / 2, vals, width, label=cname)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={len(frames.as_dict()[s])})" for s in split_names])
    ax.set_ylabel("样本数")
    ax.set_title("(a) 各划分的类别分布")
    ax.legend(fontsize=8)

    ax = axes[1]
    train = frames.train
    fruits = sorted(train["group"].unique())
    x = np.arange(len(fruits))
    for i, cname in enumerate(class_names):
        vals = [int(((train["group"] == f) & (train["label_name"] == cname)).sum()) for f in fruits]
        bars = ax.bar(
            x + i * width - (len(class_names) - 1) * width / 2, vals, width, label=cname
        )
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v}", ha="center", va="bottom",
                        fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(fruits, rotation=15, ha="right")
    ax.set_ylabel("样本数")
    ax.set_title("(b) 训练集：水果种类 × 类别")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# 错误样本图册 / 置信度可靠性
# --------------------------------------------------------------------------- #
def plot_error_gallery(
    image_paths: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probs: np.ndarray,
    class_names: Sequence[str],
    out_path: str = "figs/error_gallery.png",
    max_items: int = 12,
    cols: int = 4,
) -> Optional[str]:
    """把分错的图片拼成一张图册，答辩讲“模型到底错在哪”时最有说服力。"""
    import matplotlib.pyplot as plt
    from PIL import Image

    n = min(len(image_paths), max_items)
    if n == 0:
        print("  （没有错误样本，跳过错误图册）")
        return None

    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for i in range(n):
        ax = axes[i]
        try:
            img = Image.open(image_paths[i]).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "读取失败", ha="center")
        conf = float(probs[i, y_pred[i]])
        ax.set_title(
            f"真值: {class_names[int(y_true[i])]}\n预测: {class_names[int(y_pred[i])]} ({conf:.2%})",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle("测试集误判样本（按置信度从低到高挑选）", fontsize=12)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_confidence_hist(
    probs: np.ndarray,
    y_true: np.ndarray,
    out_path: str = "figs/confidence_hist.png",
) -> str:
    """置信度分布：正确样本 vs 错误样本，用来看模型是否“自信地犯错”。"""
    import matplotlib.pyplot as plt

    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == y_true

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bins = np.linspace(0, 1, 21)
    ax.hist(conf[correct], bins=bins, alpha=0.65, label=f"预测正确 (n={int(correct.sum())})")
    ax.hist(conf[~correct], bins=bins, alpha=0.65,
            label=f"预测错误 (n={int((~correct).sum())})")
    ax.set_xlabel("预测置信度 max p(y|x)")
    ax.set_ylabel("样本数")
    ax.set_yscale("log")
    ax.set_title("置信度分布（对数纵轴）：错误样本是否集中在低置信区")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_per_fruit_accuracy(per_fruit: Dict[str, Dict], out_path: str = "figs/per_fruit_accuracy.png") -> str:
    """按水果种类拆分的准确率 —— 回答“哪种水果最难分”。"""
    import matplotlib.pyplot as plt

    fruits = list(per_fruit.keys())
    accs = [per_fruit[f]["accuracy"] for f in fruits]

    fig, ax = plt.subplots(figsize=(max(6, 0.85 * len(fruits)), 4.0))
    bars = ax.bar(fruits, accs, color="tab:green", alpha=0.8)
    for b, a, f in zip(bars, accs, fruits):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.005, f"{a:.4f}", ha="center", fontsize=8)
    mean_acc = float(np.mean(accs))
    ax.axhline(mean_acc, color="tab:red", ls="--", lw=1, label=f"总体均值 {mean_acc:.4f}")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(min(0.5, max(0.0, min(accs) - 0.1)), 1.02)
    ax.set_title("测试集：各水果种类的分类准确率")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_path)
