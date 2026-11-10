"""模型构建：ResNet18 迁移学习 + 两段式微调。

与 `app.py` 的推理契约必须完全一致（否则加载权重会报 key 不匹配）：
    models.resnet18(weights=None); model.fc = nn.Linear(512, 2)
    torch.save(model.state_dict(), "quality_model.pth")

改进点（每一条都对应 B 那版 toy 脚本的一个具体问题）：
1. 不只训分类头 → 支持“冻结→解冻最后两个 stage”两段式微调；
2. 冻结阶段 BN 必须 eval，否则 running_mean/var 被小批量统计量污染；
3. 分类头加 dropout 可选；
4. 保存 state_dict 仅此一项，保证 app.py 能直接 load_state_dict。
"""

from __future__ import annotations

from typing import Dict, List, Optional

BACKBONES = ("resnet18", "resnet34", "resnet50", "mobilenet_v3_small", "efficientnet_b0")

# torchvision 的权重枚举类名与函数名并不规则（ResNet18_Weights 但 MobileNet_V3_Small_Weights），
# 必须显式映射；靠字符串拼接在 mobilenet_v3_small 上会直接 KeyError。
WEIGHT_ENUMS = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
    "efficientnet_b0": "EfficientNet_B0_Weights",
}


def build_model(
    backbone: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.0,
):
    """构建分类模型，返回 (model, feature_dim)。

    `weights` 用新 API（torchvision≥0.13 的枚举）；旧版本回退到 `pretrained=True`，
    避免不同 torchvision 版本在答辩现场直接报错。
    """
    import torch.nn as nn
    from torchvision import models

    if backbone not in BACKBONES:
        raise ValueError(f"未知 backbone {backbone!r}，可选 {BACKBONES}")

    weights = None
    if pretrained:
        enum_name = WEIGHT_ENUMS.get(backbone)
        weights = getattr(getattr(models, enum_name), "DEFAULT") if enum_name else None

    try:
        model = getattr(models, backbone)(weights=weights)
    except TypeError:
        model = getattr(models, backbone)(pretrained=pretrained)

    in_features = _replace_classifier(model, backbone, num_classes, dropout)
    return model, in_features


def _replace_classifier(model, backbone: str, num_classes: int, dropout: float) -> int:
    """替换分类头，返回原特征维度。"""
    import torch.nn as nn

    if backbone.startswith("resnet"):
        in_features = model.fc.in_features
        head: nn.Module = nn.Linear(in_features, num_classes)
        if dropout > 0:
            head = nn.Sequential(nn.Dropout(dropout), head)
        model.fc = head
        return in_features

    if backbone == "mobilenet_v3_small":
        in_features = model.classifier[-1].in_features
        layers: List[nn.Module] = list(model.classifier[:-1])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features, num_classes))
        model.classifier = nn.Sequential(*layers)
        return in_features

    if backbone == "efficientnet_b0":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return in_features

    raise ValueError(f"不支持的 backbone: {backbone}")


def set_frozen(model, freeze: bool) -> None:
    """冻结/解冻除分类头以外的全部参数。"""
    head_params = _head_parameter_ids(model)
    for name, param in model.named_parameters():
        if id(param) in head_params:
            param.requires_grad = True
        else:
            param.requires_grad = not freeze


def unfreeze_last_blocks(model, backbone: str = "resnet18", n_blocks: int = 2) -> List[str]:
    """解冻 ResNet 最后 n 个 stage（layer3/layer4），返回被解冻的参数名前缀。"""
    if not backbone.startswith("resnet"):
        set_frozen(model, freeze=False)
        return ["all"]

    names = []
    targets = [f"layer{i}" for i in range(4, 4 - n_blocks, -1)]
    for name, param in model.named_parameters():
        if any(name.startswith(t) for t in targets):
            param.requires_grad = True
            names.append(name)
    return sorted({n.split(".")[0] for n in names})


def _head_parameter_ids(model) -> set:
    """定位分类头参数（用于只训头 / 判别式学习率）。"""
    head = getattr(model, "fc", None)
    if head is not None and hasattr(head, "parameters"):
        return {id(p) for p in head.parameters()}
    for attr in ("classifier", "heads"):
        head = getattr(model, attr, None)
        if head is not None:
            return {id(p) for p in head.parameters()}
    return {id(p) for p in model.parameters()}


def count_parameters(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def build_optimizer(
    model,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    backbone: str = "resnet18",
):
    """分类头用大学习率、主干用小学习率（判别式微调），避免把预训练特征冲掉。"""
    import torch

    head_ids = _head_parameter_ids(model)
    head_params, base_params = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (head_params if id(param) in head_ids else base_params).append(param)

    groups = [{"params": head_params, "lr": head_lr, "name": "head"}]
    if base_params:
        groups.append({"params": base_params, "lr": backbone_lr, "name": "backbone"})

    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def bce_style_loss(use_class_weight: Optional[object], label_smoothing: float = 0.05):
    """加权 + 标签平滑的交叉熵。

    标签平滑 0.05 的作用：单果图像里“很确定是优质”的样本并不绝对，
    轻微平滑可以抑制模型过度自信，降低 ECE、提升泛化（这正是过拟合处理的一环）。
    """
    import torch.nn as nn

    return nn.CrossEntropyLoss(weight=use_class_weight, label_smoothing=label_smoothing)
