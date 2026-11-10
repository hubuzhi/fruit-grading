# -*- coding: utf-8 -*-
# 任务4/5：果蔬分级可视化系统 (Streamlit) - 水果种类+成熟度+品质 + 水果报告
# 用法: D:\anaconda3\Scripts\streamlit.exe run D:\fruit_project\scripts\app.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from fruit_reports import get_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

FRUIT_CLASSES = ["苹果", "香蕉", "番石榴", "青柠", "橙子", "石榴"]
RIPE_CLASSES = ["未成熟", "成熟", "过熟/腐败"]
QUAL_CLASSES = ["A级优等", "B级合格", "C级残次"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ---------- 隐藏 Streamlit 右上角无用UI(部署/停等工具栏) ----------
_HIDE_UI = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}
[data-testid="stHeader"] {display: none;}
[data-testid="stStatusWidget"] {display: none;}
.stDeployButton {display: none !important;}
[data-testid="stActionButton"] {visibility: hidden;}
#stActionButton {display:none;}
</style>
"""
st.markdown(_HIDE_UI, unsafe_allow_html=True)

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


@st.cache_resource
def load_models():
    def _build(num):
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num)
        return m

    fruit = _build(6)
    fruit.load_state_dict(torch.load(os.path.join(MODEL_DIR, "fruit_mobilenet_weighted.pth"), map_location="cpu"))
    fruit.eval()
    ripe = _build(3)
    ripe.load_state_dict(torch.load(os.path.join(MODEL_DIR, "ripeness_mobilenet_full.pth"), map_location="cpu"))
    ripe.eval()
    qual = _build(3)
    qual.load_state_dict(torch.load(os.path.join(MODEL_DIR, "quality_mobilenet_full.pth"), map_location="cpu"))
    qual.eval()
    return fruit, ripe, qual


def predict(model, img, classes):
    t = tf(img).unsqueeze(0)
    with torch.no_grad():
        prob = torch.softmax(model(t), 1)[0]
    idx = int(prob.argmax().item())
    return classes[idx], float(prob[idx]), prob.tolist()


st.set_page_config(page_title="果蔬品质与成熟度分级系统", layout="wide")
st.title("🍎 果蔬品质与成熟度分级系统")
st.caption("上传图片 → 智能识别水果种类、成熟度、品质，并输出该水果分级报告与分拣方案")

fruit_model, ripe_model, qual_model = load_models()

upload = st.file_uploader("上传果蔬图片", type=["jpg", "jpeg", "png"])

if upload is not None:
    img = Image.open(upload).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="原始图片", use_container_width=True)

    fruit_cls, fruit_conf, fruit_probs = predict(fruit_model, img, FRUIT_CLASSES)
    ripe_cls, ripe_conf, ripe_probs = predict(ripe_model, img, RIPE_CLASSES)
    qual_cls, qual_conf, qual_probs = predict(qual_model, img, QUAL_CLASSES)

    with col2:
        st.subheader("🔍 识别结果")
        st.success(f"**水果种类**：{fruit_cls}  （置信度 {fruit_conf:.1%}）")
        st.write(f"**成熟度**：{ripe_cls}  （置信度 {ripe_conf:.1%}）")
        st.write(f"**品质等级**：{qual_cls}  （置信度 {qual_conf:.1%}）")

    st.subheader("各类别概率分布")
    st.caption("柱条越长 = 系统越认为是该类别；柱上数字 = 该类别的概率（百分比）")

    def _prob_chart(classes, probs, title):
        import matplotlib.font_manager as fm
        try:
            font = fm.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
        except Exception:
            font = None
        fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=120)
        ax.barh(classes, probs, color="#2e8b57")
        ax.set_xlim(0, 1.05)
        ax.set_title(title, fontproperties=font, fontsize=11)
        ax.tick_params(labelsize=8)
        if font:
            for lbl in ax.get_yticklabels():
                lbl.set_fontproperties(font)
        for i, p in enumerate(probs):
            ax.text(p + 0.01, i, f"{p*100:.1f}%", va="center", fontsize=9,
                    fontproperties=font)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return fig

    c1, c2, c3 = st.columns(3)
    with c1:
        st.pyplot(_prob_chart(FRUIT_CLASSES, fruit_probs, "水果种类概率"))
    with c2:
        st.pyplot(_prob_chart(RIPE_CLASSES, ripe_probs, "成熟度概率"))
    with c3:
        st.pyplot(_prob_chart(QUAL_CLASSES, qual_probs, "品质概率"))

    # ---------- 识别出水果后，显示该水果的报告与分拣方案 ----------
    report = get_report(fruit_cls)
    if report:
        st.divider()
        st.header(f"📋 {fruit_cls} 品质报告与分拣方案")

        st.markdown("#### 基本档案")
        st.write(f"**学名**：{report['学名']}")
        st.write(f"**产地**：{report['产地']}")
        st.write(f"**上市季节**：{report['季节']}")
        st.write(f"**主要营养**：{report['营养']}")
        st.write(f"**特性**：{report['特性']}")

        st.markdown("#### 成熟度标准")
        for k, v in report["成熟度标准"].items():
            st.write(f"- **{k}**：{v}")

        st.markdown("#### 品质标准")
        for k, v in report["品质标准"].items():
            st.write(f"- **{k}**：{v}")

        st.markdown("#### 分拣流向（按识别结果）")
        for k, v in report["分拣流向"].items():
            st.write(f"- **{k}** → {v}")

        st.markdown("#### 分拣流程优化方案")
        for item in report["分拣优化"]:
            st.write(f"- {item}")

        st.info(f"当前图片识别为 **{fruit_cls} · {ripe_cls} · {qual_cls}**，请按上方分拣流向安排去向。")
else:
    st.write("请上传一张果蔬图片开始识别，系统将自动输出该水果的报告与分拣方案。")
