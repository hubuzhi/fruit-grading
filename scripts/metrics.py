"""纯 NumPy 实现的分类评价指标。

为什么不用 sklearn？
1. Kaggle / Colab 环境里 sklearn 版本漂移会导致 `classification_report` 输出格式变化，
   直接写进报告的数字和代码对不上；
2. 答辩时老师常问“macro-F1 怎么算的”“AUC 怎么来的”，纯 NumPy 实现可以把每一步摊开讲；
3. 本项目指标规模很小（最多 3 类、2 万张），手写实现没有任何性能问题。

约定：
- `y_true` / `y_pred` 是一维整数标签，取值范围 [0, num_classes)。
- 概率矩阵 `probs` 形状 (N, num_classes)，行和为 1（softmax 输出）。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# 混淆矩阵与分类报告
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true, y_pred, num_classes: int | None = None) -> np.ndarray:
    """返回形状 (K, K) 的混淆矩阵，行=真实类别，列=预测类别。

    cm[i, j] = 真实为 i、被预测为 j 的样本数。
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true/y_pred 长度不一致: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("空输入无法计算混淆矩阵")

    k = int(num_classes) if num_classes else int(max(y_true.max(), y_pred.max())) + 1
    if y_true.min() < 0 or y_pred.min() < 0:
        raise ValueError("标签不能为负")
    if y_true.max() >= k or y_pred.max() >= k:
        raise ValueError(f"标签超出 num_classes={k} 的范围")

    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_scores(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """由混淆矩阵推出各类的 TP/FP/FN、precision、recall、F1、support。"""
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)   # 真实为该类的样本数
    predicted = cm.sum(axis=0).astype(np.float64)  # 被判为该类的样本数
    fp = predicted - tp
    fn = support - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def macro_average(values: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """宏平均（各类等权）；传入 weights 时退化为加权平均。"""
    v = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(v.mean())
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        return float(v.mean())
    return float((v * w).sum() / w.sum())


def summarize(cm: np.ndarray, class_names: Sequence[str]) -> Dict:
    """汇总一份可直接写进报告/JSON 的指标字典。"""
    pc = per_class_scores(cm)
    support = pc["support"]
    total = float(support.sum())
    accuracy = float(np.trace(cm) / total)

    per_class: List[Dict] = []
    for i, name in enumerate(class_names):
        per_class.append(
            {
                "class": name,
                "precision": round(float(pc["precision"][i]), 4),
                "recall": round(float(pc["recall"][i]), 4),
                "f1": round(float(pc["f1"][i]), 4),
                "support": int(pc["support"][i]),
            }
        )

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_average(pc["precision"]), 4),
        "macro_recall": round(macro_average(pc["recall"]), 4),
        "macro_f1": round(macro_average(pc["f1"]), 4),
        "weighted_precision": round(macro_average(pc["precision"], support), 4),
        "weighted_recall": round(macro_average(pc["recall"], support), 4),
        "weighted_f1": round(macro_average(pc["f1"], support), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "support_total": int(total),
    }


def classification_report_text(metrics: Dict, class_names: Sequence[str]) -> str:
    """生成与 sklearn 风格对齐的文本报告，方便直接贴进报告附录。"""
    lines = [
        f"{'class':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}",
        "-" * 56,
    ]
    for row in metrics["per_class"]:
        lines.append(
            f"{row['class']:>12} {row['precision']:>10.4f} {row['recall']:>10.4f} "
            f"{row['f1']:>10.4f} {row['support']:>10d}"
        )
    lines.append("-" * 56)
    lines.append(
        f"{'accuracy':>12} {'':>10} {'':>10} {metrics['accuracy']:>10.4f} "
        f"{metrics['support_total']:>10d}"
    )
    lines.append(
        f"{'macro avg':>12} {metrics['macro_precision']:>10.4f} {metrics['macro_recall']:>10.4f} "
        f"{metrics['macro_f1']:>10.4f} {metrics['support_total']:>10d}"
    )
    lines.append(
        f"{'weighted avg':>12} {metrics['weighted_precision']:>10.4f} "
        f"{metrics['weighted_recall']:>10.4f} {metrics['weighted_f1']:>10.4f} "
        f"{metrics['support_total']:>10d}"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# ROC / PR / AUC
# --------------------------------------------------------------------------- #
def binary_roc_curve(y_true_bin: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """二分类 ROC 曲线，返回 (fpr, tpr)，按阈值从高到低排列。"""
    y = np.asarray(y_true_bin, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int((1 - y).sum()), 1)
    tpr = np.concatenate([[0.0], tp / n_pos])
    fpr = np.concatenate([[0.0], fp / n_neg])
    return fpr, tpr


def binary_pr_curve(y_true_bin: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """二分类 PR 曲线，返回 (recall, precision)。"""
    y = np.asarray(y_true_bin, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = max(int(y.sum()), 1)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, EPS)
    return recall, precision


def auc_roc(y_true_bin: np.ndarray, scores: np.ndarray) -> float:
    """ROC 曲线下面积（梯形积分，等价于 Mann-Whitney U 统计量）。"""
    fpr, tpr = binary_roc_curve(y_true_bin, scores)
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def average_precision(y_true_bin: np.ndarray, scores: np.ndarray) -> float:
    """PR 曲线下面积（AP），用 recall 增量加权的阶梯积分，避免插值高估。"""
    recall, precision = binary_pr_curve(y_true_bin, scores)
    ap = 0.0
    prev_r = 0.0
    # 从右往左做单调包络后再积分，与 sklearn 的 step 实现口径一致
    for r, p in zip(recall, precision):
        if r > prev_r:
            ap += (r - prev_r) * p
            prev_r = r
    return float(ap)


def multiclass_auc(probs: np.ndarray, y_true, num_classes: int | None = None) -> Dict:
    """One-vs-Rest 的宏平均 AUC 与 AP（多分类统一走这条路径）。"""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    k = int(num_classes) if num_classes else probs.shape[1]

    macro_auc = 0.0
    macro_ap = 0.0
    per_class = []
    for c in range(k):
        y_bin = (y_true == c).astype(np.int64)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            # 该类在测试集里不存在，跳过不计入宏平均
            per_class.append({"class_index": c, "auc": None, "ap": None})
            continue
        a = auc_roc(y_bin, probs[:, c])
        p = average_precision(y_bin, probs[:, c])
        macro_auc += a
        macro_ap += p
        per_class.append({"class_index": c, "auc": round(a, 4), "ap": round(p, 4)})

    valid = k - sum(1 for r in per_class if r["auc"] is None)
    valid = max(valid, 1)
    return {
        "macro_auc_ovr": round(macro_auc / valid, 4),
        "macro_ap_ovr": round(macro_ap / valid, 4),
        "per_class": per_class,
    }


# --------------------------------------------------------------------------- #
# 诊断辅助：过拟合间隙、置信度分布、错误样本挑选
# --------------------------------------------------------------------------- #
def generalization_gap(history: Dict[str, List[float]]) -> Dict[str, float]:
    """训练/验证曲线间隙，用来说明过拟合程度（数值越大越可能过拟合）。"""
    acc = history.get("accuracy") or history.get("acc")
    val = history.get("val_accuracy") or history.get("val_acc")
    loss = history.get("loss")
    val_loss = history.get("val_loss")
    if not acc or not val:
        return {}

    best_epoch = int(np.argmax(val))
    return {
        "final_train_acc": round(float(acc[-1]), 4),
        "final_val_acc": round(float(val[-1]), 4),
        "best_val_acc": round(float(val[best_epoch]), 4),
        "best_epoch": best_epoch + 1,
        "acc_gap_final": round(float(acc[-1] - val[-1]), 4),
        "acc_gap_at_best": round(float(acc[best_epoch] - val[best_epoch]), 4),
        "final_train_loss": round(float(loss[-1]), 4) if loss else None,
        "final_val_loss": round(float(val_loss[-1]), 4) if val_loss else None,
        "overfit_ratio": round(float(acc[-1] / max(val[-1], EPS)), 4),
    }


def expected_calibration_error(y_true, probs: np.ndarray, n_bins: int = 10) -> float:
    """ECE：模型“说 90% 有把握时是否真有 90% 正确”，答辩讲置信度可靠性时用。"""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def per_group_accuracy(groups: Sequence[str], y_true, y_pred) -> Dict[str, Dict]:
    """按水果种类（或任意分组）统计准确率，用来支撑“哪类水果更难分”的结论。"""
    groups = np.asarray(groups)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    out: Dict[str, Dict] = {}
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        n = int(mask.sum())
        acc = float((y_true[mask] == y_pred[mask]).mean()) if n else 0.0
        out[g] = {"n": n, "accuracy": round(acc, 4)}
    return out


def top_confusions(cm: np.ndarray, class_names: Sequence[str], top_k: int = 5) -> List[Dict]:
    """列出最容易混淆的类别对，按 (i->j) 的错误数排序。"""
    out = []
    k = cm.shape[0]
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            n = int(cm[i, j])
            if n > 0:
                out.append(
                    {
                        "true": class_names[i],
                        "pred": class_names[j],
                        "count": n,
                        "rate_of_true_class": round(n / max(int(cm[i].sum()), 1), 4),
                    }
                )
    out.sort(key=lambda r: r["count"], reverse=True)
    return out[:top_k]
