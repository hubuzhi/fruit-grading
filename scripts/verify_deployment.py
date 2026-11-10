"""上线前体检：确认 checkpoint 的标签口径与 app.py 是否一致。

为什么必须做这一步
------------------
本项目存在两套相反的标签口径：

    split_*.txt      : 0=Good  1=Bad
    metadata/*.csv   : 0=Bad   1=Good
    train_quality.py : 0=Bad   1=Good
    app.py           : index0=劣质(Bad)  index1=优质(Good)

如果训练时用了 `split_*.txt` 的顺序、却把 `metadata` 的 `label` 当标签，模型的
“优质”和“劣质”就会整体互换。这种错误在训练日志里完全看不出来（loss 一样降、
准确率一样高），只有上线后才会发现“好果被判劣质”。本脚本就是用来提前抓它的。

用法：
    python verify_deployment.py --ckpt quality_model.pth ^
        --metadata-dir metadata --data-root "C:\\...\\Fruit_Project" --n 200

判定逻辑（不依赖人工看图）：
    1. 在测试集上按“0=Bad, 1=Good”算准确率 A_norm；
    2. 把模型输出交换后再算准确率 A_swap；
    3. A_norm 明显高于 A_swap → 口径正确；反之 → checkpoint 的标签是反的，
       必须在 app.py 的 QUALITY_CLASSES 或加载处做交换。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

import data as D
import model as MDL


def verify(ckpt: str, metadata_dir: Optional[str], split_txt_prefix: Optional[str],
           data_root: Optional[str], split: str = "test", n: int = 300,
           backbone: str = "resnet18", img_size: int = 224,
           device_pref: str = "auto", out_json: Optional[str] = None) -> Dict:
    import torch

    device = torch.device(device_pref if device_pref != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    splits = D.load_splits(metadata_dir=metadata_dir, split_txt_prefix=split_txt_prefix,
                           data_root=data_root)
    df = splits[split]
    # 均衡取样：一半 Good 一半 Bad，避免“全判多数类”也能得到高准确率
    good = df[df["label"] == 1].head(n // 2)
    bad = df[df["label"] == 0].head(n // 2)
    import pandas as pd

    sample = pd.concat([good, bad]).reset_index(drop=True)

    net, _ = MDL.build_model(backbone, num_classes=len(D.CLASSES), pretrained=False)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    net.to(device).eval()

    loader = D.make_loader(sample, D.build_transforms(img_size, train=False),
                           batch_size=32, shuffle=False, num_workers=0)

    preds, probs_good, labels = [], [], []
    with torch.no_grad():
        for images, y in loader:
            logits = net(images.to(device))
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            preds.extend(probs.argmax(1).tolist())
            probs_good.extend(probs[:, 1].tolist())
            labels.extend(y.tolist())

    preds = np.asarray(preds)
    labels = np.asarray(labels)
    probs_good = np.asarray(probs_good)

    acc_norm = float((preds == labels).mean())
    acc_swap = float(((1 - preds) == labels).mean())

    # 更强的证据：优质图的平均 P(优质) 是否高于劣质图
    mg = probs_good[labels == 1].mean() if (labels == 1).any() else float("nan")
    mb = probs_good[labels == 0].mean() if (labels == 0).any() else float("nan")
    margin = float(mg - mb)

    verdict = "口径正确（0=Bad, 1=Good）" if acc_norm >= acc_swap else "口径相反（0=Good, 1=Bad）"
    ok = (acc_norm >= acc_swap) and (margin > 0)

    result = {
        "checkpoint": os.path.abspath(ckpt),
        "n_samples": int(len(labels)),
        "load_missing_keys": len(missing),
        "load_unexpected_keys": len(unexpected),
        "accuracy_as_0Bad_1Good": round(acc_norm, 4),
        "accuracy_after_swap": round(acc_swap, 4),
        "mean_p_good_on_good_images": round(float(mg), 4),
        "mean_p_good_on_bad_images": round(float(mb), 4),
        "p_good_margin": round(margin, 4),
        "verdict": verdict,
        "pass": bool(ok),
        "action": (
            "无需修改" if ok else
            "在 app.py 中把 QUALITY_CLASSES 改为 ['优质','劣质']，或加载权重后交换输出通道"
        ),
    }

    print("\n=== 上线前标签口径体检 ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    if not ok:
        print("\n❌ 未通过：模型输出的正类含义与 app.py 不一致，直接部署会把好坏果判反。")
    else:
        print("\n✅ 通过：模型输出可直接被 app.py 使用。")

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="检查 checkpoint 的标签口径是否与 app.py 一致")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--metadata-dir", default="metadata")
    p.add_argument("--split-txt-prefix", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-json", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = vars(parse_args())
    # CLI 用 --device 更直观，但函数参数叫 device_pref；这里做一次显式映射，
    # 避免 "unexpected keyword argument 'device'" 这类入口错误。
    args["device_pref"] = args.pop("device")
    verify(**args)
