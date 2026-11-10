# 果蔬品质数据集说明

生成时间：2026-11-03T19:23:07

## 1. 数据范围

- 原始图片总数：**19,526**
- 用于品质分类：**18,452** 张（Good + Bad）
- 排除的 `Mixed Qualit_Fruits`：**1,074** 张
- Good：**11,664** 张（63.2%）
- Bad：**6,788** 张（36.8%）

`Mixed Qualit_Fruits` 是多果混拍图，不适合作为单果品质分类的一个类别，因此未纳入训练。

## 2. 官方数据划分

| Split | 图片数 | 占比 |
|---|---:|---:|
| train | 12,911 | 70.0% |
| val | 2,763 | 15.0% |
| test | 2,778 | 15.1% |

**直接使用官方划分，不要重新随机划分。**  
`split_*.txt` 中的原始路径指向 `D:\fruit_project\data`，生成脚本已自动映射到本地数据目录。

## 3. 类别分布

| 水果 | Good | Bad | 合计 |
|---|---:|---:|---:|
| Apple | 1,149 | 1,141 | 2,290 |
| Banana | 1,113 | 1,087 | 2,200 |
| Guava | 1,152 | 1,129 | 2,281 |
| Lime | 1,094 | 1,085 | 2,179 |
| Orange | 1,216 | 1,159 | 2,375 |
| Pomegranate | 5,940 | 1,187 | 7,127 |

注意：`Pomegranate_Good` 有 5,940 张，占全部数据的
32.2%，存在明显类别不均衡。  
建议训练时使用 `WeightedRandomSampler` 或加权 `CrossEntropyLoss`，并报告每个水果的准确率、召回率和混淆矩阵。

## 4. 文件说明

| 文件 | 用途 |
|---|---|
| `fruit_dataset.csv` | 全量元数据 |
| `train.csv` / `val.csv` / `test.csv` | 按官方划分拆好的文件 |
| `per_class_counts.csv` | 每个 split 中水果/品质类别的数量 |
| `image_audit.csv` | 图片尺寸、MD5 审计信息 |
| `bad_images.csv` | 坏图列表（应为空） |
| `duplicate_images.csv` | 完全重复图片列表（应为空） |
| `class_weights.json` | 二分类加权损失建议值 |
| `summary.json` | 机器可读的统计汇总 |
| `data_summary.txt` | 纯文本统计汇总 |
| `class_distribution.png` | 各水果 Good/Bad 数量 |
| `split_distribution.png` | 官方训练/验证/测试集分布 |
| `quality_balance.png` | Good/Bad 总体比例 |
| `sample_grid.png` | 每个水果的 Good/Bad 代表性样本 |
| `image_size_distribution.png` | 图片尺寸分布 |

## 5. CSV 字段

| 字段 | 含义 |
|---|---|
| `file_path` | 本机绝对路径 |
| `relative_path` | 相对 `Fruit_Project/` 的路径，推荐训练时使用 |
| `fruit` | Apple / Banana / Guava / Lime / Orange / Pomegranate |
| `quality` | Good / Bad |
| `label` | Good=1，Bad=0 |
| `split` | train / val / test |
| `width`, `height` | 图片尺寸 |

训练代码建议：

```python
DATA_ROOT = Path(r"C:\Users\WZY\Desktop\Fruit_Project\Fruit_Project")
img_path = DATA_ROOT / row["relative_path"]
```

这样即使项目目录移动，也不需要改 CSV。

## 6. 数据审计结果

- 坏图：**0**
- 完全重复图片组：**0**
- 跨 split 重复：未发现。
- 图片尺寸种类：7 种，最常见为
  256x256（12,508 张）, 192x256（3,968 张）, 256x192（1,391 张）

## 7. 重要限制：没有成熟度标签

这份数据集只有 **Good / Bad 品质标签**，没有“未熟 / 成熟 / 过熟”的真实成熟度标签。

因此：

1. 不要使用旧 `process_data.py` 中基于 HSV 均值的伪成熟度标签；
2. 不要用“绿色=未熟、红色=成熟”这种规则生成标签；
3. 如果课程必须做成熟度，需要另外找带成熟度标注的数据，或制定标准后人工标注一部分图片；
4. 品质分类可以先独立完成，作为项目的主任务。

## 8. 训练建议

- 输入尺寸：`224 × 224`；
- 训练增强：`RandomResizedCrop`、`RandomHorizontalFlip`、`ColorJitter`、`RandomRotation(10)`；
- 验证/测试：`Resize(256)` + `CenterCrop(224)`；
- 模型：ResNet18 / MobileNetV3 / EfficientNet-B0 迁移学习；
- 损失：加权 `CrossEntropyLoss`，权重见 `class_weights.json`；
- 评价：总体准确率 + 每类召回率 + 宏平均 F1 + 混淆矩阵；
- 因为水果种类也会影响颜色，建议额外报告“每个水果单独的品质准确率”，避免模型只记住水果颜色。

## 9. 复现

在项目根目录运行：

```powershell
python build_metadata.py
```

脚本会重新读取官方 split，并覆盖生成 `metadata/` 下的 CSV、统计和图表。
