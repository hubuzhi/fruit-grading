"""测试集评估：把“模型部分报告”需要的东西一次性全部算出来。

    python evaluate.py --ckpt runs/resnet18_e15/best.pth ^
        --metadata-dir metadata --data-root "C:\\...\\Fruit_Project" ^
        --out-dir runs/resnet18_e15/eval

产物：
    metrics.json               全部指标（准确率、每类 P/R/F1、macro-F1、AUC、AP、ECE、混淆矩阵）
    classification_report.txt  表格化报告，可直接贴进附录
    predictions.csv            每张图的真值/预测/概率，用于复核与画图
    probs.npy / labels.npy     概率矩阵，方便二次分析
    figs/*.png                 混淆矩阵、ROC、PR、学习曲线、每水果准确率、置信度分布、错误图册
    report_snippets.md         把上面的数字自动填好的“报告段落 + 表格”，避免手抄出错
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

import data as D
import metrics as M
import model as MDL

FIGS = "figs"


def load_checkpoint(ckpt_path: str, backbone: str, device):
    """加载 state_dict；兼容 DataParallel 前缀与 dropout 版分类头。

    兼容策略：先按标准结构 strict 加载，失败再 strict=False 并打印缺失/多余键，
    这样即使 B 早期保存的权重结构略有差异也能跑通评估（评估与训练解耦很重要）。
    """
    import torch

    net, _ = MDL.build_model(backbone, num_classes=len(D.CLASSES), pretrained=False)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"⚠️  权重非严格匹配。缺失键: {missing[:4]}… 多余键: {unexpected[:4]}…")
    net.to(device).eval()
    return net


def predict(net, loader, device) -> tuple:
    """返回 (probs[N,K], labels[N], paths[N], fruits[N])。"""
    import torch

    all_probs, all_labels, all_paths, all_fruits = [], [], [], []
    dataset = loader.dataset
    with torch.no_grad():
        for images, labels in loader:
            logits = net(images.to(device, non_blocking=True))
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.tolist())

    all_probs = np.concatenate(all_probs, axis=0)
    all_paths = [s.path for s in dataset.samples]
    all_fruits = [s.fruit for s in dataset.samples]
    return all_probs, np.asarray(all_labels), all_paths, all_fruits


def evaluate(cfg: argparse.Namespace) -> Dict:
    import pandas as pd

    device = _device(cfg.device)
    out_dir = cfg.out_dir
    fig_dir = os.path.join(out_dir, FIGS)
    os.makedirs(fig_dir, exist_ok=True)

    splits = D.load_splits(
        metadata_dir=cfg.metadata_dir,
        split_txt_prefix=cfg.split_txt_prefix,
        data_root=cfg.data_root,
    )
    split = cfg.split
    df = splits[split]
    if cfg.limit:
        df = df.head(cfg.limit).copy()

    eval_tf = D.build_transforms(cfg.img_size, train=False)
    loader = D.make_loader(df, eval_tf, cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers)

    net = load_checkpoint(cfg.ckpt, cfg.backbone, device)
    probs, y_true, paths, fruits = predict(net, loader, device)
    y_pred = probs.argmax(axis=1)

    cm = M.confusion_matrix(y_true, y_pred, num_classes=len(D.CLASSES))
    summary = M.summarize(cm, D.CLASSES)
    auc = M.multiclass_auc(probs, y_true, num_classes=len(D.CLASSES))
    ece = M.expected_calibration_error(y_true, probs)
    per_fruit = M.per_group_accuracy(fruits, y_true, y_pred)
    confusions = M.top_confusions(cm, D.CLASSES)

    result = {
        "split": split,
        "n_samples": int(len(y_true)),
        "checkpoint": os.path.abspath(cfg.ckpt),
        "label_convention": "0=Bad(劣质) 1=Good(优质)",
        "class_names": D.CLASSES,
        **summary,
        "auc": auc,
        "ece": round(ece, 4),
        "per_fruit": per_fruit,
        "top_confusions": confusions,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    report_txt = M.classification_report_text(summary, D.CLASSES)
    report_txt += f"\n\nMacro AUC (OvR): {auc['macro_auc_ovr']:.4f}"
    report_txt += f"\nMacro AP  (OvR): {auc['macro_ap_ovr']:.4f}"
    report_txt += f"\nECE (保真度)   : {ece:.4f}"
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt + "\n")
    print("\n" + report_txt)

    pd.DataFrame(
        {
            "filepath": paths,
            "fruit": fruits,
            "y_true": y_true,
            "y_true_name": [D.CLASSES[i] for i in y_true],
            "y_pred": y_pred,
            "y_pred_name": [D.CLASSES[i] for i in y_pred],
            "p_bad": probs[:, 0].round(6),
            "p_good": probs[:, 1].round(6),
            "correct": (y_true == y_pred),
        }
    ).to_csv(os.path.join(out_dir, "predictions.csv"), index=False, encoding="utf-8-sig")
    np.save(os.path.join(out_dir, "probs.npy"), probs)
    np.save(os.path.join(out_dir, "labels.npy"), y_true)

    _make_figures(cfg, out_dir, fig_dir, history_csv=None, probs=probs, y_true=y_true,
                  y_pred=y_pred, paths=paths, fruits=fruits, cm=cm, per_fruit=per_fruit)
    _write_snippets(out_dir, result, report_txt)
    print(f"\n✅ 评估完成，产物在 {os.path.abspath(out_dir)}")
    return result


def _device(prefer: str = "auto"):
    import torch

    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_figures(cfg, out_dir, fig_dir, history_csv, probs, y_true, y_pred, paths,
                  fruits, cm, per_fruit) -> None:
    """画图；matplotlib 缺失时只警告不中断（评估数字比图更重要）。"""
    try:
        import plots
    except Exception as exc:  # pragma: no cover
        print(f"⚠️  跳过绘图（{exc}），请先 pip install matplotlib")
        return

    plots.setup_style()
    split_cn = {"train": "训练集", "val": "验证集", "test": "测试集"}.get(cfg.split, cfg.split)
    try:
        plots.plot_confusion_matrix(
            cm, D.CLASSES, os.path.join(fig_dir, "confusion_matrix.png"),
            title=f"{split_cn}混淆矩阵（n={int(cm.sum())}）",
        )
        plots.plot_roc_pr(probs, y_true, D.CLASSES,
                          os.path.join(fig_dir, "roc_curves.png"),
                          os.path.join(fig_dir, "pr_curves.png"))
        plots.plot_per_fruit_accuracy(per_fruit, os.path.join(fig_dir, "per_fruit_accuracy.png"))
        plots.plot_confidence_hist(probs, y_true, os.path.join(fig_dir, "confidence_hist.png"))

        # 错误样本：优先挑“错得最自信”的，最能暴露系统性偏差
        wrong = np.where(y_true != y_pred)[0]
        if len(wrong):
            conf = probs[wrong, y_pred[wrong]]
            order = wrong[np.argsort(-conf)][:12]
            plots.plot_error_gallery(
                [paths[i] for i in order], y_true[order], y_pred[order], probs[order],
                D.CLASSES, os.path.join(fig_dir, "error_gallery.png"),
            )
        if history_csv and os.path.exists(history_csv):
            plots.plot_learning_curves(history_csv, os.path.join(fig_dir, "learning_curves.png"))
    except Exception as exc:
        print(f"⚠️  绘图过程中出现问题: {exc}")


def _write_snippets(out_dir: str, result: Dict, report_txt: str) -> None:
    """生成可直接粘进报告的段落与表格（数字全部来自本次运行，不手抄）。"""
    pc = result["per_class"]
    rows = "\n".join(
        f"| {r['class']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['support']} |"
        for r in pc
    )
    fruit_rows = "\n".join(
        f"| {f} | {v['n']} | {v['accuracy']:.4f} |" for f, v in result["per_fruit"].items()
    )
    confusion_rows = "\n".join(
        f"| {c['true']} → {c['pred']} | {c['count']} | {c['rate_of_true_class']:.2%} |"
        for c in result["top_confusions"]
    ) or "| 无 | 0 | 0 |"

    best_fruit = max(result["per_fruit"].items(), key=lambda kv: kv[1]["accuracy"])
    worst_fruit = min(result["per_fruit"].items(), key=lambda kv: kv[1]["accuracy"])

    md = f"""# 测试集结果（自动生成，可直接粘贴进报告）

> 数据来源：`{result['checkpoint']}`
> 评估集：`{result['split']}`，样本 {result['n_samples']} 张
> 标签口径：{result['label_convention']}

## 1. 总体指标

在测试集（n={result['n_samples']}）上，模型总体准确率为 **{result['accuracy']:.4f}**，
宏平均精确率 {result['macro_precision']:.4f}、宏平均召回率 {result['macro_recall']:.4f}、
宏平均 F1 为 **{result['macro_f1']:.4f}**，加权 F1 为 {result['weighted_f1']:.4f}；
One-vs-Rest 宏平均 AUC 为 {result['auc']['macro_auc_ovr']:.4f}，宏平均 AP 为
{result['auc']['macro_ap_ovr']:.4f}，期望校准误差 ECE 为 {result['ece']:.4f}。

## 2. 分类报告

| 类别 | 精确率 | 召回率 | F1 | 支持数 |
|---|---:|---:|---:|---:|
{rows}
| **宏平均** | {result['macro_precision']:.4f} | {result['macro_recall']:.4f} | {result['macro_f1']:.4f} | {result['support_total']} |

```
{report_txt}
```

## 3. 按水果种类的准确率

| 水果 | 样本数 | 准确率 |
|---|---:|---:|
{fruit_rows}

准确率最高的是 **{best_fruit[0]}（{best_fruit[1]['accuracy']:.4f}）**，
最低的是 **{worst_fruit[0]}（{worst_fruit[1]['accuracy']:.4f}）**。

## 4. 主要混淆对

| 混淆方向 | 次数 | 占该真实类比例 |
|---|---:|---:|
{confusion_rows}

## 5. 需要插入的图

- `figs/confusion_matrix.png`：混淆矩阵（左样本数、右行归一化）
- `figs/roc_curves.png` / `figs/pr_curves.png`：ROC 与 PR 曲线
- `figs/per_fruit_accuracy.png`：各水果准确率
- `figs/confidence_hist.png`：置信度分布（正确 vs 错误，对数纵轴）
- `figs/error_gallery.png`：典型误判样本
"""
    with open(os.path.join(out_dir, "report_snippets.md"), "w", encoding="utf-8") as f:
        f.write(md)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="测试集评估：指标、混淆矩阵、ROC/PR、错误样本")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--metadata-dir", default="metadata")
    p.add_argument("--split-txt-prefix", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="runs/eval")
    return p.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
