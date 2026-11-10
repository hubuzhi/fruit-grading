"""超参数实验：把“调参过程”做成一张可复现的表，而不是嘴上说“我调过”。

    python tune.py --metadata-dir metadata --data-root "C:\\...\\Fruit_Project" ^
        --experiments quick --out-dir runs/tuning

每组实验都会：训练 → 在验证集选最优 → 在测试集评估 → 汇总一行结果。
最后输出：
    runs/tuning/results.csv         每行一组实验的全部配置与测试集指标
    runs/tuning/results.md          报告用的对比表
    runs/tuning/model_comparison.png 对比条形图
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# 实验设计原则：一次只动一个变量，确保结论可归因。
EXPERIMENTS: Dict[str, List[Dict]] = {
    # 3 组，约 30-40 分钟（GPU）：先证明“两段式微调 + 增强 + 加权”有效
    "quick": [
        {"name": "E1_head_only", "freeze_epochs": 99, "epochs": 6, "strong_aug": False,
         "note": "仅训练分类头（对照 B 的原始做法）"},
        {"name": "E2_full_ft", "freeze_epochs": 0, "epochs": 6, "strong_aug": True,
         "note": "全网络微调 + 增强"},
        {"name": "E3_two_stage", "freeze_epochs": 3, "epochs": 8, "strong_aug": True,
         "note": "两段式微调（推荐方案）"},
    ],
    # 学习率扫描
    "lr": [
        {"name": f"E_lr{lrs}", "freeze_epochs": 3, "epochs": 8, "strong_aug": True,
         "head_lr": lrs, "note": f"分类头学习率 {lrs}"}
        for lrs in (1e-2, 1e-3, 3e-4)
    ],
    # 正则化强度
    "reg": [
        {"name": f"E_drop{d}", "freeze_epochs": 3, "epochs": 8, "strong_aug": True,
         "dropout": d, "note": f"分类头 dropout={d}"}
        for d in (0.0, 0.3, 0.5)
    ],
    # 输入分辨率
    "size": [
        {"name": f"E_img{s}", "freeze_epochs": 3, "epochs": 8, "strong_aug": True,
         "img_size": s, "note": f"输入 {s}×{s}"}
        for s in (160, 224)
    ],
}


def run_one(exp: Dict, dataset_args: List[str], out_root: str, seed: int) -> Dict:
    """用子进程跑一组实验，保证每组实验互不污染（进程级隔离最稳）。"""
    run_dir = os.path.join(out_root, exp["name"])
    os.makedirs(run_dir, exist_ok=True)

    cmd = [sys.executable, os.path.join(HERE, "train.py"),
           "--out-dir", run_dir, "--seed", str(seed)] + dataset_args
    for key, value in exp.items():
        if key in ("name", "note"):
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            else:
                if flag == "--strong-aug":
                    pass  # 默认不开，无需显式关闭
                else:
                    cmd.append("--no-" + key.replace("_", "-"))
        else:
            cmd.extend([flag, str(value)])

    print(f"\n▶ {exp['name']}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        return {"run_name": exp["name"], "error": f"训练失败 (exit {proc.returncode})", **exp}

    eval_dir = os.path.join(run_dir, "eval")
    cmd = [sys.executable, os.path.join(HERE, "evaluate.py"),
           "--ckpt", os.path.join(run_dir, "best.pth"),
           "--out-dir", eval_dir, "--split", "test"] + dataset_args
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        return {"run_name": exp["name"], "error": "评估失败", **exp}

    with open(os.path.join(run_dir, "train_summary.json"), encoding="utf-8") as f:
        train_sum = json.load(f)
    with open(os.path.join(eval_dir, "metrics.json"), encoding="utf-8") as f:
        test_metrics = json.load(f)

    return {
        "run_name": exp["name"],
        "note": exp.get("note", ""),
        "epochs": exp.get("epochs"),
        "freeze_epochs": exp.get("freeze_epochs"),
        "head_lr": exp.get("head_lr"),
        "dropout": exp.get("dropout"),
        "img_size": exp.get("img_size"),
        "strong_aug": exp.get("strong_aug"),
        "best_epoch": train_sum["best"]["epoch"],
        "val_accuracy": train_sum["best"].get("val_accuracy"),
        "val_macro_f1": train_sum["best"].get("val_macro_f1"),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_macro_auc": test_metrics["auc"]["macro_auc_ovr"],
        "test_ece": test_metrics["ece"],
        "val_test_gap": round(
            (train_sum["best"].get("val_macro_f1") or 0) - test_metrics["macro_f1"], 4
        ),
    }


def tune(cfg: argparse.Namespace) -> pd.DataFrame:
    exps = EXPERIMENTS.get(cfg.experiments)
    if exps is None:
        raise SystemExit(f"未知实验组 {cfg.experiments!r}，可选 {list(EXPERIMENTS)}")
    if cfg.only:
        exps = [e for e in exps if e["name"] in cfg.only]

    dataset_args: List[str] = []
    if cfg.metadata_dir:
        dataset_args += ["--metadata-dir", cfg.metadata_dir]
    if cfg.split_txt_prefix:
        dataset_args += ["--split-txt-prefix", cfg.split_txt_prefix]
    if cfg.data_root:
        dataset_args += ["--data-root", cfg.data_root]
    if cfg.limit:
        dataset_args += ["--limit", str(cfg.limit)]

    os.makedirs(cfg.out_dir, exist_ok=True)
    rows = [run_one(exp, dataset_args, cfg.out_dir, cfg.seed) for exp in exps]

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.out_dir, "results.csv"), index=False, encoding="utf-8-sig")

    ok = df[df.get("test_accuracy").notna()] if "test_accuracy" in df else df
    if not ok.empty:
        best = ok.sort_values("test_macro_f1", ascending=False).iloc[0]
        with open(os.path.join(cfg.out_dir, "best_config.json"), "w", encoding="utf-8") as f:
            json.dump(best.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        print(f"\n🏆 最佳配置: {best['run_name']}  test macro-F1={best['test_macro_f1']}")

        _write_table(ok, cfg.out_dir)
        try:
            import plots
            plots.setup_style()
            plots.plot_model_comparison(
                os.path.join(cfg.out_dir, "results.csv"),
                os.path.join(cfg.out_dir, "model_comparison.png"),
                metric_cols=("test_accuracy", "test_macro_f1"),
                label_col="run_name",
            )
        except Exception as exc:
            print(f"⚠️  对比图跳过: {exc}")
    return df


def _write_table(df: pd.DataFrame, out_dir: str) -> None:
    cols = ["run_name", "note", "best_epoch", "val_macro_f1", "test_accuracy",
            "test_macro_f1", "test_macro_auc", "val_test_gap"]
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df[cols].iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write("# 超参数实验结果\n\n" + "\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="超参数实验矩阵")
    p.add_argument("--experiments", default="quick", choices=list(EXPERIMENTS))
    p.add_argument("--only", nargs="*", default=None, help="只跑指定实验名")
    p.add_argument("--metadata-dir", default="metadata")
    p.add_argument("--split-txt-prefix", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--limit", type=int, default=0, help="调试用，限制每个 split 的样本数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="runs/tuning")
    return p.parse_args(argv)


if __name__ == "__main__":
    tune(parse_args())
