# -*- coding: utf-8 -*-
import os, sys, base64
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import torch
import torch.nn as nn
from torchvision import models

MODEL_DIR = r"D:\fruit_project\models"

def build_model(num_classes):
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

# 1. 导出ONNX（opset 13 兼容浏览器）
configs = [
    ("fruit_mobilenet_weighted.pth", "fruit", 6),
    ("ripeness_mobilenet_full.pth", "ripeness", 3),
    ("quality_mobilenet_full.pth", "quality", 3),
]

b64_data = {}
for pth_name, key, num_cls in configs:
    model = build_model(num_cls)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, pth_name), map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = os.path.join(MODEL_DIR, key + ".onnx")
    torch.onnx.export(model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=13, dynamo=False, do_constant_folding=True)
    with open(onnx_path, "rb") as f:
        b64_data[key] = base64.b64encode(f.read()).decode("ascii")
    os.remove(onnx_path)
    print(f"{key}: {len(b64_data[key])/1024/1024:.1f} MB base64")

# 2. 生成HTML
html = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>果蔬品质与成熟度分级系统</title>
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:"Microsoft YaHei","PingFang SC",sans-serif; background:#f5f7fa; color:#1a1a2e; }
.container { max-width:1000px; margin:0 auto; padding:24px; }
h1 { font-size:28px; margin-bottom:4px; }
.subtitle { color:#888; font-size:14px; margin-bottom:20px; }
.upload-zone { position:relative; border:2px dashed #bbb; border-radius:12px; padding:40px; text-align:center; background:#fff; cursor:pointer; transition:all .2s; margin-bottom:20px; }
.upload-zone:hover, .upload-zone.dragover { border-color:#2e8b57; background:#e8f5e9; }
.upload-zone p { color:#666; margin-top:8px; }
.upload-zone .icon { font-size:48px; }
.upload-zone input[type=file] { position:absolute; inset:0; width:100%; height:100%; opacity:0; cursor:pointer; }
.result-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:20px; }
@media(max-width:700px){ .result-grid { grid-template-columns:1fr; } }
.card { background:#fff; border-radius:12px; padding:20px; box-shadow:0 2px 8px rgba(0,0,0,.06); }
.card h3 { font-size:18px; margin-bottom:12px; }
.result-img { width:100%; border-radius:8px; margin-bottom:12px; }
.badge { display:inline-block; padding:6px 14px; border-radius:20px; font-size:14px; margin:4px 0; }
.badge-green { background:#e8f5e9; color:#2e7d32; }
.badge-blue { background:#e3f2fd; color:#1565c0; }
.badge-orange { background:#fff3e0; color:#e65100; }
.prob-row { display:flex; align-items:center; margin:6px 0; font-size:13px; }
.prob-label { width:80px; text-align:right; padding-right:8px; color:#555; }
.prob-bar-bg { flex:1; height:18px; background:#f0f0f0; border-radius:4px; overflow:hidden; }
.prob-bar { height:100%; background:linear-gradient(90deg,#43a047,#2e8b57); border-radius:4px; }
.prob-val { width:50px; padding-left:6px; font-size:12px; }
.report-section { background:#fff; border-radius:12px; padding:24px; box-shadow:0 2px 8px rgba(0,0,0,.06); margin-bottom:20px; }
.report-section h2 { font-size:20px; margin-bottom:12px; }
.report-section h4 { font-size:15px; margin:14px 0 6px; color:#2e8b57; border-left:4px solid #2e8b57; padding-left:8px; }
.info-grid { display:grid; grid-template-columns:90px 1fr; gap:4px 12px; font-size:14px; }
.info-label { color:#888; }
ul { padding-left:20px; font-size:14px; }
ul li { margin:4px 0; }
.flow-table { width:100%; border-collapse:collapse; font-size:14px; margin-top:6px; }
.flow-table td { padding:6px 10px; border-bottom:1px solid #eee; }
.flow-table td:first-child { color:#666; width:40%; }
.loading { text-align:center; padding:20px; color:#888; }
.spinner { display:inline-block; width:24px; height:24px; border:3px solid #ddd; border-top-color:#2e8b57; border-radius:50%; animation:spin 1s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
</style>
</head>
<body>
<div class="container">
<h1>🍎 果蔬品质与成熟度分级系统</h1>
<p class="subtitle">上传图片 → 智能识别水果种类、成熟度、品质（本地推理，隐私安全）</p>
<div class="upload-zone" id="dropZone">
  <div class="icon">📁</div>
  <p><b>点击 / 拖放 / 直接 Ctrl+V 粘贴图片到这里</b></p>
  <p>JPG / PNG · 首次加载模型约需10秒</p>
  <input type="file" id="fileInput" accept="image/*">
</div>
<div id="resultArea" style="display:none;">
  <div class="loading" id="loading"><div class="spinner"></div><p>正在加载模型并识别...</p></div>
  <div id="results" style="display:none;">
    <div class="result-grid">
      <div class="card"><img class="result-img" id="resultImg"><div id="badges"></div></div>
      <div class="card"><h3>各类别概率分布</h3><p style="font-size:12px;color:#999;margin-bottom:8px;">柱条越长 = 越认为是该类别</p><div id="probChart"></div></div>
    </div>
    <div class="report-section" id="reportArea2"></div>
  </div>
</div>
</div>
<script>
const FRUIT_CLASSES = ["苹果","香蕉","番石榴","青柠","橙子","石榴"];
const RIPE_CLASSES = ["未成熟","成熟","过熟/腐败"];
const QUAL_CLASSES = ["A级优等","B级合格","C级残次"];
const MEAN = [0.485,0.456,0.406];
const STD = [0.229,0.224,0.225];
const MODEL_B64 = { fruit:"__FRUIT_B64__", ripeness:"__RIPENESS_B64__", quality:"__QUALITY_B64__" };
const REPORTS = {
"苹果":{sci:"Malus domestica（苹果）",origin:"主产山东、陕西、河南、甘肃",season:"秋冬季集中上市，全年可储",nutrition:"果胶、膳食纤维、维生素C、多种有机酸",feature:"耐储存、怕压伤、水分流失会变皱",ripeStd:{"未成熟":"果皮青绿、硬脆酸涩","成熟":"果皮红/黄透亮、脆甜多汁","过熟/腐败":"果肉发绵、表皮皱缩或腐烂"},qualStd:{"A级优等":"果形圆整、表皮光洁无伤疤、无虫眼","B级合格":"轻微压痕/划伤、不影响食用","C级残次":"明显磕碰、腐烂、虫蛀"},flow:{"成熟+A级":"商超精品架","成熟/B级":"加工（果汁/果醋/果干）","过熟+C级":"剔除/榨汁降级","未成熟":"冷库后熟储存"},optimize:["防磕碰：分拣托盘加缓冲垫","甜度分级：可结合糖度传感器","储存：冷藏0-4℃","外观检测：重点识别虫眼、碰伤"]},
"香蕉":{sci:"Musa nana（香蕉）",origin:"广东、广西、海南、云南",season:"全年供应，春夏旺季",nutrition:"碳水化合物(约22%)、钾、维生素B6、膳食纤维",feature:"后熟型水果，怕冷怕压怕碰",ripeStd:{"未成熟":"表皮青绿、硬挺","成熟":"表皮亮黄、略带斑点","过熟/腐败":"表皮大量黑斑、果肉软烂发黑"},qualStd:{"A级优等":"表皮黄亮、无碰伤、无黑斑、果形完整","B级合格":"少量黑斑/轻微碰压、不影响食用","C级残次":"明显黑斑、软烂、擦伤严重"},flow:{"成熟+A级":"商超精品架","成熟/B级":"加工（香蕉片/果酱/烘焙）","过熟+C级":"剔除/做饲料","未成熟":"转催熟库"},optimize:["防碰压：输送带加软垫缓冲","催熟分流：青香蕉先进催熟库","颜色阈值：按香蕉'青→黄→黑'色阶分级","温度控制：分拣车间14-20℃","斑点检测：重点识别黑斑"]},
"番石榴":{sci:"Psidium guajava（番石榴/芭乐）",origin:"福建、广东、海南、广西",season:"热带水果，多季上市",nutrition:"维生素C含量极高、膳食纤维、叶酸",feature:"表皮脆弱易碰伤、后熟型",ripeStd:{"未成熟":"果皮青绿、果肉硬脆","成熟":"果皮黄绿、果肉软糯香甜","过熟/腐败":"果皮发黑、果肉软烂"},qualStd:{"A级优等":"果形完整、表皮无破损","B级合格":"轻微擦伤","C级残次":"明显破损、软烂"},flow:{"成熟+A级":"商超精品架","成熟/B级":"加工（果汁/果酱）","过熟+C级":"剔除","未成熟":"催熟后销售"},optimize:["防破损：需海绵托盘分拣","成熟判断：以果皮颜色+硬度双重标准","快速周转：尽快进入冷链"]},
"青柠":{sci:"Citrus limon（柠檬）",origin:"四川安岳、广东、云南",season:"四季挂果，全年供应",nutrition:"维生素C、柠檬酸、类黄酮",feature:"耐储存、怕冻伤",ripeStd:{"未成熟":"果皮深绿、果肉极酸","成熟":"果皮黄/浅绿、酸香浓郁","过熟/腐败":"果皮干皱、发霉"},qualStd:{"A级优等":"果皮光滑、色泽均匀","B级合格":"轻微麻点/划伤","C级残次":"干皱、霉变"},flow:{"成熟+A级":"商超/鲜销","成熟/B级":"加工（榨汁/柠檬干）","过熟+C级":"剔除/榨汁降级","未成熟":"继续生长/催色"},optimize:["防冻伤：分拣车间不低于8℃","酸度分级：榨汁用果降低外观门槛","霉斑检测：重点识别表皮霉变"]},
"橙子":{sci:"Citrus sinensis（甜橙）",origin:"江西赣南、湖南、湖北、四川",season:"秋冬上市，耐储",nutrition:"维生素C、类胡萝卜素、钾、膳食纤维",feature:"果皮厚耐储运、需防霉",ripeStd:{"未成熟":"果皮青绿、酸涩","成熟":"果皮橙黄、汁多香甜","过熟/腐败":"果皮干皱、发霉"},qualStd:{"A级优等":"果皮橙亮、果形圆、无伤疤虫眼","B级合格":"轻微划伤/麻点","C级残次":"霉变、严重碰伤"},flow:{"成熟+A级":"商超精品架","成熟/B级":"加工（橙汁/橙皮）","过熟+C级":"剔除/榨汁","未成熟":"入库催熟/储存"},optimize:["防霉：分拣后涂保鲜蜡+控温控湿","糖酸比分级：结合光谱测糖度","外观检测：重点识别霉斑、虫眼"]},
"石榴":{sci:"Punica granatum（石榴）",origin:"陕西临潼、云南、四川、山东",season:"秋季集中上市",nutrition:"多酚类抗氧化物质、维生素C、钾",feature:"果皮坚硬耐储、怕裂口",ripeStd:{"未成熟":"果皮青绿、籽粒白硬","成熟":"果皮红/黄、籽粒饱满红亮","过熟/腐败":"果皮干裂、籽粒发黑"},qualStd:{"A级优等":"果皮红亮、无裂口、无霉斑","B级合格":"轻微裂口/擦伤","C级残次":"严重裂口、霉烂"},flow:{"成熟+A级":"商超精品架（中秋礼盒）","成熟/B级":"加工（石榴汁/籽）","过熟+C级":"剔除/榨汁","未成熟":"后熟储存"},optimize:["防裂口：防挤压防跌落","裂口检测：重点识别","礼盒优先：A级中秋礼盒价值高","储存：低温5℃+控湿"]}
};
let sessions = null;
function b64ToBuffer(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}
async function loadModels() {
  if (sessions) return;
  const [f, r, q] = await Promise.all([
    ort.InferenceSession.create(b64ToBuffer(MODEL_B64.fruit)),
    ort.InferenceSession.create(b64ToBuffer(MODEL_B64.ripeness)),
    ort.InferenceSession.create(b64ToBuffer(MODEL_B64.quality))
  ]);
  sessions = { fruit: f, ripeness: r, quality: q };
}
function preprocess(img) {
  const canvas = document.createElement("canvas");
  canvas.width = 224; canvas.height = 224;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, 224, 224);
  const data = ctx.getImageData(0, 0, 224, 224).data;
  const input = new Float32Array(3 * 224 * 224);
  for (let i = 0; i < 224 * 224; i++) {
    for (let c = 0; c < 3; c++) {
      input[c * 224 * 224 + i] = (data[i*4+c]/255.0 - MEAN[c]) / STD[c];
    }
  }
  return new ort.Tensor("float32", input, [1,3,224,224]);
}
function softmax(arr) {
  const mx = Math.max(...arr);
  const e = arr.map(x => Math.exp(x - mx));
  const s = e.reduce((a,b)=>a+b,0);
  return e.map(x=>x/s);
}
async function predict(sess, img) {
  const t = preprocess(img);
  const out = await sess.run({ input: t });
  return softmax(Array.from(out.output.data));
}
function makeChart(title, classes, probs) {
  let h = `<div style="margin-bottom:10px;"><b style="font-size:13px;">${title}</b></div>`;
  classes.forEach((c,i) => {
    const pct = (probs[i]*100).toFixed(1);
    h += `<div class="prob-row"><span class="prob-label">${c}</span><div class="prob-bar-bg"><div class="prob-bar" style="width:${pct}%"></div></div><span class="prob-val">${pct}%</span></div>`;
  });
  return h;
}
function renderReport(fruitName, ripeCls, qualCls) {
  const r = REPORTS[fruitName];
  if (!r) return "";
  let h = `<h2>📋 ${fruitName} 品质报告与分拣方案</h2>`;
  h += `<h4>基本档案</h4><div class="info-grid">`;
  h += `<span class="info-label">学名</span><span>${r.sci}</span>`;
  h += `<span class="info-label">产地</span><span>${r.origin}</span>`;
  h += `<span class="info-label">季节</span><span>${r.season}</span>`;
  h += `<span class="info-label">营养</span><span>${r.nutrition}</span>`;
  h += `<span class="info-label">特性</span><span>${r.feature}</span></div>`;
  h += `<h4>成熟度标准</h4><ul>`;
  for (const [k,v] of Object.entries(r.ripeStd)) h += `<li><b>${k}</b>：${v}</li>`;
  h += `</ul><h4>品质标准</h4><ul>`;
  for (const [k,v] of Object.entries(r.qualStd)) h += `<li><b>${k}</b>：${v}</li>`;
  h += `</ul><h4>分拣流向</h4><table class="flow-table">`;
  for (const [k,v] of Object.entries(r.flow)) h += `<tr><td>${k}</td><td>${v}</td></tr>`;
  h += `</table><h4>分拣流程优化方案</h4><ul>`;
  for (const o of r.optimize) h += `<li>${o}</li>`;
  h += `</ul><p style="margin-top:10px;padding:10px;background:#e3f2fd;border-radius:6px;font-size:13px;">当前图片识别为 <b>${fruitName} · ${ripeCls} · ${qualCls}</b></p>`;
  return h;
}
async function handleFile(file) {
  document.getElementById("resultArea").style.display = "block";
  document.getElementById("loading").style.display = "block";
  document.getElementById("results").style.display = "none";
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = async () => {
    document.getElementById("resultImg").src = url;
    try {
      document.getElementById("loading").innerHTML = '<div class="spinner"></div><p>正在加载模型（约10秒）...</p>';
      await loadModels();
      const fp = await predict(sessions.fruit, img);
      const rp = await predict(sessions.ripeness, img);
      const qp = await predict(sessions.quality, img);
      const fi = fp.indexOf(Math.max(...fp));
      const ri = rp.indexOf(Math.max(...rp));
      const qi = qp.indexOf(Math.max(...qp));
      const fn = FRUIT_CLASSES[fi], rn = RIPE_CLASSES[ri], qn = QUAL_CLASSES[qi];
      document.getElementById("badges").innerHTML =
        `<div><span class="badge badge-green">水果：${fn}（${(fp[fi]*100).toFixed(1)}%）</span></div>` +
        `<div><span class="badge badge-blue">成熟度：${rn}（${(rp[ri]*100).toFixed(1)}%）</span></div>` +
        `<div><span class="badge badge-orange">品质：${qn}（${(qp[qi]*100).toFixed(1)}%）</span></div>`;
      document.getElementById("probChart").innerHTML =
        makeChart("水果种类", FRUIT_CLASSES, fp) +
        makeChart("成熟度", RIPE_CLASSES, rp) +
        makeChart("品质", QUAL_CLASSES, qp);
      document.getElementById("reportArea2").innerHTML = renderReport(fn, rn, qn);
      document.getElementById("loading").style.display = "none";
      document.getElementById("results").style.display = "block";
    } catch(e) {
      document.getElementById("loading").innerHTML = "<p style='color:red'>出错：" + e.message + "</p>";
    }
  };
  img.src = url;
}
const dz = document.getElementById("dropZone");
const fi = document.getElementById("fileInput");
fi.onchange = e => { if(e.target.files[0]) handleFile(e.target.files[0]); };
dz.ondragover = e => { e.preventDefault(); dz.classList.add("dragover"); };
dz.ondragleave = () => dz.classList.remove("dragover");
dz.ondrop = e => { e.preventDefault(); dz.classList.remove("dragover"); if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); };
document.addEventListener("paste", e => {
  for (const item of (e.clipboardData || {}).items || []) {
    if (item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) { e.preventDefault(); handleFile(file); break; }
    }
  }
});
</script>
</body>
</html>
"""

html = html.replace("__FRUIT_B64__", b64_data["fruit"])
html = html.replace("__RIPENESS_B64__", b64_data["ripeness"])
html = html.replace("__QUALITY_B64__", b64_data["quality"])

out = r"D:\fruit_project\果蔬识别系统.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"生成: {out}  ({os.path.getsize(out)/1024/1024:.1f} MB)")
