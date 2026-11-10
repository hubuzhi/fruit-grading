"""成熟度分档：在“没有成熟度标签”的前提下，能诚实交付的东西。

背景（这是答辩必须讲清的一点）
--------------------------------
`metadata/README.md` 第 7 节明确写了：本数据集**只有 Good/Bad 品质标签，没有
“未熟 / 成熟 / 过熟”的真实成熟度标注**，并且明确要求不要用 HSV 均值规则伪造标签。

因此本模块**不训练所谓的“成熟度分类模型”**，而是提供一个可解释的颜色-纹理特征分档器：

    score = w1 * 平均饱和度 + w2 * (1 - 平均明度) + w3 * 褐变像素占比 + w4 * 纹理粗糙度

- 权重与分档阈值都是**在训练集上按分位数标定**（33% / 67%），不使用任何伪标签；
- 输出三档概率，形状与 `app.py` 的 `_simulate_maturity()` 完全一致，可直接替换；
- 由于没有真实标签，它**不能报告准确率**，只报告“分数分布”和“与品质标签的一致性”
  （劣质组的成熟度分数应显著高于优质组，这是可检验的弱证据）。

这样定位的好处：既满足了系统里有“成熟度”输出的需求，又不会在答辩时被问
“你的成熟度模型准确率多少”时给出一个编造的数字。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

LABELS: List[str] = ["未成熟", "成熟", "过熟"]

# 与 app.py 保持一致的默认中心（app 里 MATURITY_CENTERS 用的就是这三档）
DEFAULT_CENTERS: Dict[str, float] = {"未成熟": 0.30, "成熟": 0.55, "过熟": 0.78}
DEFAULT_THRESHOLDS: Tuple[float, float] = (0.42, 0.62)  # score < t1 未熟；t1~t2 成熟；> t2 过熟


def extract_features(image) -> Dict[str, float]:
    """从 PIL 图像提取与“成熟/褐变”相关的低层特征。

    不用深度学习特征，是因为这里要的是一个**可解释、可复核、零训练成本**的分档器；
    特征本身也符合果蔬采后生理的常识：随着成熟度升高，颜色饱和度与明度下降、
    褐变像素增多、表面纹理变粗糙。
    """
    import numpy as np

    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # 褐变近似：低明度 + 中低饱和度（黑斑、腐烂、褐变区域）
    brown_mask = (v < 0.35) & (s < 0.55)
    brown_ratio = float(brown_mask.mean())

    # 纹理粗糙度：Sobel 梯度均值
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    texture = float(grad.mean())

    # 色彩丰富度（Hasler & Süsstrunk 指标，简化版）
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    colorfulness = float(
        (np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 1] - rgb[..., 2])).mean()
    )

    return {
        "saturation": float(s.mean()),
        "value": float(v.mean()),
        "brown_ratio": brown_ratio,
        "texture": texture,
        "colorfulness": colorfulness,
        "hue_circular_mean": _circular_mean(h),
    }


def _circular_mean(hue: np.ndarray) -> float:
    """色相是 0~1 的环形变量，直接取均值会错（红 0.95 与 0.05 应接近而不是相差 0.9）。"""
    import numpy as np

    angle = hue * 2 * np.pi
    return float((np.arctan2(np.sin(angle).mean(), np.cos(angle).mean()) / (2 * np.pi)) % 1.0)


def maturity_score(features: Dict[str, float]) -> float:
    """把特征压成一个 0~1 的成熟度分数（越大越熟）。

    权重来源：以 app.py 里 `0.62*S + 0.38*(1-V)` 为基线，再补两项对“过熟”敏感的信号
    （褐变占比、纹理粗糙度），权重保持整数比例便于讲解：0.45 / 0.25 / 0.20 / 0.10。
    """
    score = (
        0.45 * features["saturation"]
        + 0.25 * (1.0 - features["value"])
        + 0.20 * min(features["brown_ratio"] / 0.25, 1.0)
        + 0.10 * min(features["texture"] / 0.12, 1.0)
    )
    return float(min(max(score, 0.0), 1.0))


def _softmax_over_centers(score: float, centers: Dict[str, float], sigma: float = 0.18) -> Dict[str, float]:
    """与 app.py 相同的“以中心做高斯响应再归一化”，保证接口行为一致。"""
    import numpy as np

    raw = {name: float(np.exp(-((score - c) ** 2) / (2 * sigma ** 2))) for name, c in centers.items()}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def classify(score: float, thresholds: Tuple[float, float] = DEFAULT_THRESHOLDS,
             centers: Optional[Dict[str, float]] = None) -> Tuple[str, Dict[str, float]]:
    """按阈值给出三档结论与概率（阈值法给结论，高斯给概率，二者口径独立但一致）。"""
    t1, t2 = thresholds
    label = LABELS[0] if score < t1 else (LABELS[1] if score < t2 else LABELS[2])
    probs = _softmax_over_centers(score, centers or DEFAULT_CENTERS)
    return label, probs


# --------------------------------------------------------------------------- #
# 标定：在训练集上按分位数定阈值（不需要标签）
# --------------------------------------------------------------------------- #
def calibrate(image_paths: List[str], max_images: int = 2000, out_json: Optional[str] = None) -> Dict:
    import numpy as np
    from PIL import Image

    if not image_paths:
        raise ValueError("没有输入图像，无法标定")
    paths = image_paths[:max_images]
    scores, feats = [], []
    for i, p in enumerate(paths, 1):
        try:
            image = Image.open(p).convert("RGB")
        except Exception:
            continue
        f = extract_features(image)
        feats.append(f)
        scores.append(maturity_score(f))
        if i % 200 == 0:
            print(f"  已处理 {i}/{len(paths)}")

    arr = np.asarray(scores, dtype=np.float64)
    t1, t2 = np.percentile(arr, [33, 67]).tolist()
    centers = {
        LABELS[0]: float(np.percentile(arr, 16)),
        LABELS[1]: float(np.percentile(arr, 50)),
        LABELS[2]: float(np.percentile(arr, 84)),
    }
    calib = {
        "n_images": int(len(arr)),
        "thresholds": [round(t1, 6), round(t2, 6)],
        "centers": {k: round(v, 6) for k, v in centers.items()},
        "score_stats": {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "p16": round(float(np.percentile(arr, 16)), 6),
            "p50": round(float(np.percentile(arr, 50)), 6),
            "p84": round(float(np.percentile(arr, 84)), 6),
        },
        "feature_means": {
            k: round(float(np.mean([f[k] for f in feats])), 6) for k in feats[0]
        },
        "note": "阈值与中心由训练集分位数标定，未使用任何伪成熟度标签；无监督分档，不可报告准确率",
    }
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(calib, f, ensure_ascii=False, indent=2)
    return calib


def load_calibration(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 弱证据：分数与品质标签的一致性（这是唯一可以“验证”的部分）
# --------------------------------------------------------------------------- #
def consistency_with_quality(csv_path: str, data_root: Optional[str], max_images: int = 1500) -> Dict:
    """检验“劣质组成熟度分数是否更高”。

    这不是准确率，只是单变量区分度：报告 AUC（以 Bad=1 为正类），
    AUC 明显 >0.5 才说明这个分数确实携带了与品质相关的信息，否则应重新设计特征。
    """
    import numpy as np
    import pandas as pd
    from PIL import Image

    df = pd.read_csv(csv_path)
    path_col = "file_path"
    if data_root and "relative_path" in df.columns:
        df["_p"] = [
            os.path.join(data_root, *str(r).replace("/", os.sep).split(os.sep))
            for r in df["relative_path"]
        ]
    else:
        df["_p"] = df[path_col]

    df = df.head(max_images)
    scores, labels = [], []
    for _, row in df.iterrows():
        try:
            image = Image.open(row["_p"]).convert("RGB")
        except Exception:
            continue
        scores.append(maturity_score(extract_features(image)))
        labels.append(1 if str(row.get("quality", "")).lower() == "bad" else 0)

    s = np.asarray(scores)
    y = np.asarray(labels)
    if y.min() == y.max():
        return {"note": "样本里只有一个品质类别，无法计算一致性"}

    # 以 Bad=1 为正类算 AUC：分数越高越可能被判为 Bad
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = tp / max(y_sorted.sum(), 1)
    fpr = fp / max((1 - y_sorted).sum(), 1)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))

    return {
        "n": int(len(s)),
        "mean_score_good": round(float(s[y == 0].mean()), 4) if (y == 0).any() else None,
        "mean_score_bad": round(float(s[y == 1].mean()), 4) if (y == 1).any() else None,
        "auc_bad_vs_good": round(auc, 4),
        "interpretation": (
            "AUC>0.5 说明劣质组分数偏高，分数方向正确；"
            "但这不是成熟度准确率，成熟度仍缺少真实标注"
        ),
    }


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="成熟度三档：特征分档与阈值标定（无监督，不是分类模型）")
    p.add_argument("--fit-csv", default=None, help="标定用的 CSV（如 metadata/train.csv）")
    p.add_argument("--data-root", default=None)
    p.add_argument("--max-images", type=int, default=1500)
    p.add_argument("--out", default="maturity_calibration.json")
    p.add_argument("--verify-consistency", action="store_true", help="顺带检验与品质标签的一致性")
    p.add_argument("--predict", nargs="*", default=None, help="对若干图片输出成熟度分档")
    args = p.parse_args(argv)

    if args.fit_csv:
        import pandas as pd

        df = pd.read_csv(args.fit_csv)
        if args.data_root and "relative_path" in df.columns:
            paths = [
                os.path.join(args.data_root, *str(r).replace("/", os.sep).split(os.sep))
                for r in df["relative_path"]
            ]
        else:
            paths = df["file_path"].tolist()
        calib = calibrate(paths, args.max_images, args.out)
        print(json.dumps(calib, ensure_ascii=False, indent=2))
        if args.verify_consistency:
            print(json.dumps(consistency_with_quality(args.fit_csv, args.data_root, 800),
                             ensure_ascii=False, indent=2))

    if args.predict:
        from PIL import Image

        thresholds = DEFAULT_THRESHOLDS
        centers = DEFAULT_CENTERS
        if os.path.exists(args.out):
            c = load_calibration(args.out)
            thresholds = tuple(c["thresholds"])
            centers = c["centers"]
        for path in args.predict:
            image = Image.open(path).convert("RGB")
            feats = extract_features(image)
            score = maturity_score(feats)
            label, probs = classify(score, thresholds, centers)
            print(f"{os.path.basename(path)}: score={score:.4f} -> {label} "
                  f"({', '.join(f'{k}={v:.1%}' for k, v in probs.items())})")


if __name__ == "__main__":
    main()
