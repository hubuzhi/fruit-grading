"""不装 torch/GPU 也能跑的自检：验证数据层、指标层、成熟度层的正确性。

    python smoke_test.py --metadata-dir "C:\\...\\metadata" --split-prefix "D:\\...\\split_"

做四件事：
1. 校验 metadata CSV 的标签口径（Good→1、Bad→0）；
2. 校验 split_*.txt 的自动翻转（文件里 0=Good，读进来必须变成 1）；
3. 用手算结果校验混淆矩阵/P/R/F1/macro-F1/AUC；
4. 用合成图片校验成熟度特征与分档（不依赖真实数据集）。
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import List, Optional

import numpy as np

PASS, FAIL = "✅ PASS", "❌ FAIL"


class Checker:
    def __init__(self) -> None:
        self.failures: List[str] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        print(f"  {PASS if condition else FAIL}  {name}" + (f"  [{detail}]" if detail else ""))
        if not condition:
            self.failures.append(name)


def test_metadata_labels(c: Checker, metadata_dir: str) -> None:
    import pandas as pd

    import data as D

    print("\n[1] metadata 标签口径")
    csv_path = os.path.join(metadata_dir, "train.csv")
    df = D.load_metadata_csv(csv_path, data_root=None, check_files=False)
    mapping = df.groupby(["quality", "label"]).size().reset_index(name="n")
    good_label = set(mapping.loc[mapping["quality"] == "Good", "label"].tolist())
    bad_label = set(mapping.loc[mapping["quality"] == "Bad", "label"].tolist())
    c.check("Good 映射为 1", good_label == {1}, f"实际 {good_label}")
    c.check("Bad 映射为 0", bad_label == {0}, f"实际 {bad_label}")
    c.check("标签只有 {0,1}", sorted(df["label"].unique().tolist()) == [0, 1])
    c.check("样本数 12911", len(df) == 12911, f"实际 {len(df)}")
    print(f"     分布: {df['label'].value_counts().sort_index().to_dict()}")

    w = D.class_weights({"train": df}, "train")
    print(f"     类别权重 (Bad, Good) = {w.tolist()}")
    c.check("Good 权重 < Bad 权重（Good 更多）", float(w[1]) < float(w[0]))


def test_split_txt_flip(c: Checker, split_prefix: str) -> None:
    import data as D

    print("\n[2] split_*.txt 标签自动翻转")
    path = f"{split_prefix}train.txt"
    if not os.path.exists(path):
        c.check(f"{path} 存在", False)
        return

    raw_bad = 0
    raw_good = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            parent = os.path.basename(os.path.dirname(parts[0].replace("/", os.sep)))
            if parent.endswith("_Bad"):
                raw_bad += 1
                assert parts[2] == "1", "预期文件里 Bad=1"
            elif parent.endswith("_Good"):
                raw_good += 1
                assert parts[2] == "0", "预期文件里 Good=0"
    c.check("文件内 Bad 目录标注为 1", raw_bad > 0, f"{raw_bad} 条")
    c.check("文件内 Good 目录标注为 0", raw_good > 0, f"{raw_good} 条")

    df = D.load_split_txt(path, data_root=None, check_files=False, flip_labels=True)
    bad_labels = set(df.loc[df["quality"] == "Bad", "label"].tolist())
    good_labels = set(df.loc[df["quality"] == "Good", "label"].tolist())
    c.check("翻转后 Bad=0", bad_labels == {0}, f"实际 {bad_labels}")
    c.check("翻转后 Good=1", good_labels == {1}, f"实际 {good_labels}")

    noflip = D.load_split_txt(path, data_root=None, check_files=False, flip_labels=False)
    c.check(
        "不翻转时可以复现原始（错误）口径",
        set(noflip.loc[noflip["quality"] == "Bad", "label"].tolist()) == {1},
    )


def test_metrics(c: Checker) -> None:
    import metrics as M

    print("\n[3] 指标计算（与手算对照）")
    # 真值: 3 个 Bad(0) + 3 个 Good(1)；预测: Bad 全对, Good 错 1 个
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 0])
    cm = M.confusion_matrix(y_true, y_pred, num_classes=2)
    c.check("混淆矩阵", cm.tolist() == [[3, 0], [1, 2]], f"{cm.tolist()}")

    s = M.summarize(cm, ["Bad", "Good"])
    # Bad: precision=3/4=0.75 recall=1.0 f1=6/7≈0.8571
    # Good: precision=1.0 recall=2/3≈0.6667 f1=0.8
    c.check("Bad precision=0.75", abs(s["per_class"][0]["precision"] - 0.75) < 1e-6)
    c.check("Bad recall=1.0", abs(s["per_class"][0]["recall"] - 1.0) < 1e-6)
    c.check("Good precision=1.0", abs(s["per_class"][1]["precision"] - 1.0) < 1e-6)
    c.check("Good recall=0.6667", abs(s["per_class"][1]["recall"] - 0.6667) < 1e-4)
    c.check("accuracy=5/6", abs(s["accuracy"] - 0.8333) < 1e-4, str(s["accuracy"]))
    c.check("macro_f1≈0.8286", abs(s["macro_f1"] - 0.8286) < 1e-4, str(s["macro_f1"]))
    # 加权 F1 = (3*0.8571 + 3*0.8)/6 = 0.8286（两类支持数都是 3，故与宏平均相等）
    c.check("weighted_f1≈0.8286", abs(s["weighted_f1"] - 0.8286) < 1e-4, str(s["weighted_f1"]))

    # AUC：完美分离时 = 1；正类分数整体偏低时 = 0（方向反了）
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7])
    c.check("完美分离 AUC=1.0", abs(M.auc_roc(np.array([0, 0, 0, 1, 1, 1]), scores) - 1.0) < 1e-9)
    c.check("方向相反 AUC=0.0",
            abs(M.auc_roc(np.array([1, 1, 1, 0, 0, 0]), scores) - 0.0) < 1e-9)

    probs = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]]
    )
    ece = M.expected_calibration_error(np.array([0, 0, 0, 1, 1, 0]), probs, n_bins=10)
    c.check("ECE 在 [0,1] 区间", 0.0 <= ece <= 1.0, f"{ece:.4f}")

    groups = ["Apple", "Apple", "Banana", "Banana"]
    pf = M.per_group_accuracy(groups, np.array([0, 1, 0, 1]), np.array([0, 1, 0, 0]))
    c.check("按水果分组准确率", pf["Apple"]["accuracy"] == 1.0 and pf["Banana"]["accuracy"] == 0.5,
            str(pf))


def test_maturity(c: Checker) -> None:
    print("\n[4] 成熟度特征与分档")
    try:
        from PIL import Image
    except Exception as exc:
        c.check(f"PIL 可用（{exc}）", False)
        return

    import maturity as MT

    # 用固定子目录而不是 tempfile.mkdtemp：部分受限环境下运行期新建的临时目录不可写
    tmp = os.path.join(os.getcwd(), "_smoke_tmp")
    os.makedirs(tmp, exist_ok=True)
    paths = []
    # 三张合成图：偏青（成熟度低）、正常、暗褐（过熟/褐变，v<0.35 且 s<0.55）
    specs = [((90, 160, 70), "unripe"), ((220, 150, 40), "ripe"), ((70, 48, 32), "over")]
    for color, name in specs:
        img = Image.new("RGB", (256, 256), color)
        p = os.path.join(tmp, f"{name}.jpg")
        img.save(p)
        paths.append(p)

    feats = [MT.extract_features(Image.open(p).convert("RGB")) for p in paths]
    scores = [MT.maturity_score(f) for f in feats]
    c.check("三个分数都落在 [0,1]", all(0.0 <= s <= 1.0 for s in scores), str(np.round(scores, 3)))
    c.check("褐变图 brown_ratio 最高",
            feats[2]["brown_ratio"] >= max(feats[0]["brown_ratio"], feats[1]["brown_ratio"]),
            str([round(f["brown_ratio"], 3) for f in feats]))

    calib = MT.calibrate(paths, max_images=3, out_json=os.path.join(tmp, "calib.json"))
    c.check("标定产出阈值", len(calib["thresholds"]) == 2, str(calib["thresholds"]))
    c.check("标定产出中心", set(calib["centers"]) == set(MT.LABELS), str(calib["centers"]))

    label, probs = MT.classify(0.5)
    c.check("三档概率和为 1", abs(sum(probs.values()) - 1.0) < 1e-9)
    c.check("分类结论在合法集合内", label in MT.LABELS, label)

    c.check("默认 app 中心", MT.DEFAULT_CENTERS["成熟"] == 0.55)
    print(f"     合成图分数: {[round(s, 4) for s in scores]}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="不依赖 torch 的自检")
    p.add_argument("--metadata-dir", default="metadata")
    p.add_argument("--split-prefix", default=None,
                   help="如 D:/大三作业/机器学习/答辩/split_")
    args = p.parse_args(argv)

    c = Checker()
    if os.path.isdir(args.metadata_dir):
        test_metadata_labels(c, args.metadata_dir)
    else:
        print(f"\n[1] 跳过：未找到 metadata 目录 {args.metadata_dir}")

    if args.split_prefix:
        test_split_txt_flip(c, args.split_prefix)
    else:
        print("\n[2] 跳过：未提供 --split-prefix")

    test_metrics(c)
    test_maturity(c)

    print("\n" + "=" * 56)
    if c.failures:
        print(f"❌ {len(c.failures)} 项未通过: {c.failures}")
        return 1
    print("✅ 全部自检通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
