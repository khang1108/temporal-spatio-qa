#!/usr/bin/env python3
"""Script to fix empty visualizations and enrich data in notebooks/002_eval_spatialmqa_kaggle.ipynb.

Root causes of "trắng bóc" (empty plots):
1. In `llava_1.5_lora_test_predictions.jsonl`, `item.get("id")` is `None` because `test.jsonl` has no "id" field.
2. `s.get("id", idx)` returned `None` because the key "id" existed in `s`. Thus `id_map` had key `None` only!
3. `showcase_ids = [5, 20, 22, 26]` could never find samples in `id_map`, causing `if s:` to be skipped completely.
4. If local image paths differed on Kaggle, images failed to load. We add multi-path checking + HF CDN direct fallback.
5. In cells 18 and 20, `axis` and `perspective_key` were missing from raw JSONL outputs, causing empty matrices.
"""

import json
import os

NOTEBOOK_PATH = "notebooks/002_eval_spatialmqa_kaggle.ipynb"

# 1. New Cell 16 Source (Load & Enrich Samples)
CELL_16_SOURCE = """# Phân tích hiệu năng theo số lượng Options và so sánh với Random Baseline
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Hàm nạp và làm giàu thuộc tính (axis, perspective, is_correct, id) cho mẫu
def load_and_enrich_samples():
    pred_file = "/kaggle/working/llava_1.5_lora_test_predictions.jsonl"
    fallback_json = "visualizer/src/data/predictions_data.json"
    raw_list = []
    
    if os.path.exists(pred_file):
        with open(pred_file, "r", encoding="utf-8") as f:
            raw_list = [json.loads(line) for line in f]
    elif os.path.exists(fallback_json):
        with open(fallback_json, "r", encoding="utf-8") as f:
            raw_list = json.load(f).get("items", [])
    elif os.path.exists("data/spatial_mqa/test.jsonl"):
        with open("data/spatial_mqa/test.jsonl", "r", encoding="utf-8") as f:
            raw_list = [json.loads(line) for line in f]

    axis_map = {
        "left of": "A_x (Horizontal)", "right of": "A_x (Horizontal)",
        "in front of": "A_y (Depth)", "behind": "A_y (Depth)",
        "on/above": "A_z (Vertical)", "below": "A_z (Vertical)"
    }
    
    enriched = []
    for idx, s in enumerate(raw_list):
        item = dict(s)
        gt = item.get("answer", "").strip().lower()
        pred = item.get("output", "").strip().lower()
        q = item.get("question", "")

        # 1. ID chuẩn (1-indexed)
        if item.get("id") is None:
            item["id"] = idx + 1

        # 2. Trục tọa độ 3D
        if not item.get("axis"):
            item["axis"] = axis_map.get(gt, "Other")

        # 3. Phân loại hệ quy chiếu
        if not item.get("perspective_key"):
            ql = q.lower()
            if "from your perspective" in ql or "from the perspective" in ql:
                item["perspective_key"] = "Q3_ThirdPerson"
            elif "if you" in ql:
                item["perspective_key"] = "Q2_FirstPerson"
            else:
                item["perspective_key"] = "Q1_OutOfImage"

        # 4. Cờ đánh giá đúng/sai
        if "is_correct" not in item:
            item["is_correct"] = (gt == pred) if (gt and pred) else False

        enriched.append(item)
    return enriched

samples = load_and_enrich_samples()

if samples:
    opt_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for s in samples:
        n = len(s.get("options", []))
        opt_stats[n]["total"] += 1
        if s.get("is_correct"):
            opt_stats[n]["correct"] += 1

    table_data = []
    for n in sorted(opt_stats.keys()):
        tot = opt_stats[n]["total"]
        corr = opt_stats[n]["correct"]
        acc = corr / tot * 100 if tot else 0
        rnd = 100.0 / n if n else 0
        delta = acc - rnd
        table_data.append([f"{n} Options", tot, corr, f"{acc:.2f}%", f"{rnd:.2f}%", f"{delta:+.2f}%"])

    df_opts = pd.DataFrame(table_data, columns=["Nhóm", "Số Mẫu", "Số Đúng", "LLaVA Accuracy", "Random Chance", "Delta (+/-)"])
    display(df_opts)

    plt.figure(figsize=(8, 4.5))
    x_labels = [f"{n} Options" for n in sorted(opt_stats.keys())]
    llava_vals = [opt_stats[n]["correct"] / opt_stats[n]["total"] * 100 for n in sorted(opt_stats.keys())]
    rand_vals = [100.0 / n for n in sorted(opt_stats.keys())]

    x = range(len(x_labels))
    width = 0.35
    plt.bar([i - width/2 for i in x], llava_vals, width=width, label="LLaVA-1.5 (LoRA)", color="#3b82f6")
    plt.bar([i + width/2 for i in x], rand_vals, width=width, label="Random Chance Baseline", color="#ef4444", alpha=0.7)

    plt.ylabel("Độ chính xác (%)")
    plt.title("So sánh LLaVA-1.5 vs Random Guessing theo số lượng Options")
    plt.xticks(x, x_labels)
    plt.ylim(0, 60)
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()
else:
    print("Chưa nạp được dữ liệu kết quả. Hãy chạy đánh giá ở mục 5 trước!")
"""

# 2. New Cell 18 Source (Confusion Matrix)
CELL_18_SOURCE = """# Lập và trực quan hóa Ma trận nhầm lẫn cho Ax (Horizontal) và Ay (Depth)
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Đảm bảo nạp dữ liệu kết quả nếu chạy cell độc lập
if 'samples' not in globals() or not samples or not samples[0].get("axis"):
    if 'load_and_enrich_samples' in globals():
        samples = load_and_enrich_samples()
    else:
        pred_file = "/kaggle/working/llava_1.5_lora_test_predictions.jsonl"
        fallback_json = "visualizer/src/data/predictions_data.json"
        raw_list = []
        if os.path.exists(pred_file):
            with open(pred_file, "r", encoding="utf-8") as f:
                raw_list = [json.loads(line) for line in f]
        elif os.path.exists(fallback_json):
            with open(fallback_json, "r", encoding="utf-8") as f:
                raw_list = json.load(f).get("items", [])
        
        axis_map = {
            "left of": "A_x (Horizontal)", "right of": "A_x (Horizontal)",
            "in front of": "A_y (Depth)", "behind": "A_y (Depth)",
            "on/above": "A_z (Vertical)", "below": "A_z (Vertical)"
        }
        samples = []
        for idx, s in enumerate(raw_list):
            item = dict(s)
            gt = item.get("answer", "").strip().lower()
            pred = item.get("output", "").strip().lower()
            q = item.get("question", "")
            item["id"] = idx + 1
            item["axis"] = axis_map.get(gt, "Other")
            ql = q.lower()
            if "from your perspective" in ql or "from the perspective" in ql:
                item["perspective_key"] = "Q3_ThirdPerson"
            elif "if you" in ql:
                item["perspective_key"] = "Q2_FirstPerson"
            else:
                item["perspective_key"] = "Q1_OutOfImage"
            item["is_correct"] = (gt == pred)
            samples.append(item)

if samples:
    # 1. Ma trận Ax trong Q2 (Horizontal: left/right)
    ax_q2_samples = [s for s in samples if s.get("axis", "").startswith("A_x") and s.get("perspective_key") == "Q2_FirstPerson"]
    ax_pairs = Counter((s["answer"].strip().lower(), s["output"].strip().lower()) for s in ax_q2_samples)

    labels_ax = ["left of", "right of", "in front of", "behind"]
    mat_ax = [[ax_pairs.get((gt, pred), 0) for pred in labels_ax] for gt in ["left of", "right of"]]

    df_mat_ax = pd.DataFrame(mat_ax, index=["GT: left of", "GT: right of"], columns=labels_ax)
    print("=== Ma Trận Nhầm Lẫn: Trục Ngang Ax trong Góc Nhìn Nhập Vai Q2 ===")
    display(df_mat_ax)

    # 2. Ma trận Ay trong Q2 (Depth: in front of / behind)
    ay_q2_samples = [s for s in samples if s.get("axis", "").startswith("A_y") and s.get("perspective_key") == "Q2_FirstPerson"]
    ay_pairs = Counter((s["answer"].strip().lower(), s["output"].strip().lower()) for s in ay_q2_samples)

    mat_ay = [[ay_pairs.get((gt, pred), 0) for pred in labels_ax] for gt in ["in front of", "behind"]]
    df_mat_ay = pd.DataFrame(mat_ay, index=["GT: in front of", "GT: behind"], columns=labels_ax)
    print("\\n=== Ma Trận Nhầm Lẫn: Trục Chiều Sâu Ay trong Góc Nhìn Nhập Vai Q2 ===")
    display(df_mat_ay)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    im1 = ax1.imshow(df_mat_ax.values, cmap="Blues")
    ax1.set_xticks(range(len(labels_ax)))
    ax1.set_yticks(range(2))
    ax1.set_xticklabels(labels_ax)
    ax1.set_yticklabels(df_mat_ax.index)
    ax1.set_title("Ax in Q2 (Đảo Chiều Trái/Phải 180°)")
    for i in range(2):
        for j in range(len(labels_ax)):
            val = df_mat_ax.values[i, j]
            ax1.text(j, i, str(val), ha="center", va="center", color="black" if val < 35 else "white", fontweight="bold")

    im2 = ax2.imshow(df_mat_ay.values, cmap="Reds")
    ax2.set_xticks(range(len(labels_ax)))
    ax2.set_yticks(range(2))
    ax2.set_xticklabels(labels_ax)
    ax2.set_yticklabels(df_mat_ay.index)
    ax2.set_title("Ay in Q2 (Sụp Đổ Chiều Sâu Z sang Ngang X)")
    for i in range(2):
        for j in range(len(labels_ax)):
            val = df_mat_ay.values[i, j]
            ax2.text(j, i, str(val), ha="center", va="center", color="black" if val < 30 else "white", fontweight="bold")

    plt.tight_layout()
    plt.show()
"""

# 3. New Cell 20 Source (Taxonomy)
CELL_20_SOURCE = """# Phân loại tự động toàn bộ ca lỗi theo Taxonomy và vẽ biểu đồ phân phối
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Đảm bảo nạp dữ liệu kết quả nếu chạy cell độc lập
if 'samples' not in globals() or not samples or not samples[0].get("axis"):
    if 'load_and_enrich_samples' in globals():
        samples = load_and_enrich_samples()
    else:
        pred_file = "/kaggle/working/llava_1.5_lora_test_predictions.jsonl"
        fallback_json = "visualizer/src/data/predictions_data.json"
        raw_list = []
        if os.path.exists(pred_file):
            with open(pred_file, "r", encoding="utf-8") as f:
                raw_list = [json.loads(line) for line in f]
        elif os.path.exists(fallback_json):
            with open(fallback_json, "r", encoding="utf-8") as f:
                raw_list = json.load(f).get("items", [])
        
        axis_map = {
            "left of": "A_x (Horizontal)", "right of": "A_x (Horizontal)",
            "in front of": "A_y (Depth)", "behind": "A_y (Depth)",
            "on/above": "A_z (Vertical)", "below": "A_z (Vertical)"
        }
        samples = []
        for idx, s in enumerate(raw_list):
            item = dict(s)
            gt = item.get("answer", "").strip().lower()
            pred = item.get("output", "").strip().lower()
            q = item.get("question", "")
            item["id"] = idx + 1
            item["axis"] = axis_map.get(gt, "Other")
            ql = q.lower()
            if "from your perspective" in ql or "from the perspective" in ql:
                item["perspective_key"] = "Q3_ThirdPerson"
            elif "if you" in ql:
                item["perspective_key"] = "Q2_FirstPerson"
            else:
                item["perspective_key"] = "Q1_OutOfImage"
            item["is_correct"] = (gt == pred)
            samples.append(item)

def classify_failure(item):
    axis = item.get("axis", "")
    persp = item.get("perspective_key", "")
    gt = item.get("answer", "").strip().lower()
    pred = item.get("output", "").strip().lower()

    if persp == "Q2_FirstPerson" and axis.startswith("A_x") and ((gt == "right of" and pred == "left of") or (gt == "left of" and pred == "right of")):
        return "Camera Fallback (180° Inversion)"
    elif axis.startswith("A_y") and pred in ["left of", "right of"]:
        return "Depth-to-2D Degeneracy (Ay -> Ax)"
    elif axis.startswith("A_y") and ((gt == "in front of" and pred == "behind") or (gt == "behind" and pred == "in front of")):
        return "Depth Occlusion Inversion"
    elif persp == "Q2_FirstPerson" and ((axis.startswith("A_x") and pred in ["in front of", "behind"]) or (axis.startswith("A_y") and pred in ["left of", "right of"])):
        return "Egocentric Axis Leakage (Q2)"
    elif axis.startswith("A_z") and ((gt == "on/above" and pred == "below") or (gt == "below" and pred == "on/above")):
        return "Vertical Layout / OCR Inversion (Az)"
    elif persp == "Q1_OutOfImage" and ((gt == "left of" and pred == "right of") or (gt == "right of" and pred == "left of")):
        return "Camera-Frame Inversion (Q1)"
    else:
        return "Perspective Reference Ambiguity"

if samples:
    fails = [s for s in samples if not s.get("is_correct")]
    tax_counts = Counter(classify_failure(f) for f in fails)

    df_tax = pd.DataFrame([
        [k, v, f"{v / len(fails) * 100:.1f}%"] for k, v in tax_counts.most_common()
    ], columns=["Cơ Chế Thất Bại (Failure Mechanism)", "Số Ca Lỗi", "Tỷ Lệ %"])

    print(f"Tổng số ca thất bại: {len(fails)} / {len(samples)} ({len(fails)/len(samples)*100:.1f}%)")
    display(df_tax)

    plt.figure(figsize=(10, 5))
    cats = [x[0] for x in tax_counts.most_common()[::-1]]
    vals = [x[1] for x in tax_counts.most_common()[::-1]]

    bars = plt.barh(cats, vals, color="#6366f1")
    plt.xlabel("Số lượng ca lỗi (mẫu)")
    plt.title("Phân phối 6 Cơ Chế Thất Bại Cốt Lõi của LLaVA-1.5 trên SpatialMQA")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 3, bar.get_y() + bar.get_height()/2, f"{w} ({w/len(fails)*100:.1f}%)", va="center", fontsize=10, fontweight="bold")
    plt.xlim(0, max(vals) + 30)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()
"""

# 4. New Cell 22 Source (Top Canonical Failure Cases Showcase)
CELL_22_SOURCE = """# Trực quan hóa 4 ca kinh điển kèm hình ảnh và phân tích chẩn đoán
import os
import json
import io
import textwrap
import urllib.request
import matplotlib.pyplot as plt
from PIL import Image

# 1. Đảm bảo nạp dữ liệu kết quả nếu chạy cell độc lập
if 'samples' not in globals() or not samples:
    if 'load_and_enrich_samples' in globals():
        samples = load_and_enrich_samples()
    else:
        pred_file = "/kaggle/working/llava_1.5_lora_test_predictions.jsonl"
        fallback_json = "visualizer/src/data/predictions_data.json"
        raw_list = []
        if os.path.exists(pred_file):
            with open(pred_file, "r", encoding="utf-8") as f:
                raw_list = [json.loads(line) for line in f]
        elif os.path.exists(fallback_json):
            with open(fallback_json, "r", encoding="utf-8") as f:
                raw_list = json.load(f).get("items", [])
        elif os.path.exists("data/spatial_mqa/test.jsonl"):
            with open("data/spatial_mqa/test.jsonl", "r", encoding="utf-8") as f:
                raw_list = [json.loads(line) for line in f]
        
        samples = []
        for idx, s in enumerate(raw_list):
            item = dict(s)
            item["id"] = idx + 1
            samples.append(item)

# 2. Định nghĩa 4 ca thất bại kinh điển
canonical_cases = [
    {
        "id": 5, "img": "000000034288.jpg",
        "note": "Suy biến Z -> X: Ấm nước nằm trước lò vi sóng trong 3D, góc chụp lệch trái -> LLaVA đoán left of."
    },
    {
        "id": 20, "img": "000000131634.jpg",
        "note": "Đảo chiều 180°: Xe đối diện camera. Phải tài xế = Trái camera -> LLaVA sập bẫy camera đoán left of."
    },
    {
        "id": 22, "img": "000000158946.jpg",
        "note": "Ego-to-2D Shift: Người đeo kính nhìn thẳng. Cửa sổ bên phải người nhưng ở bên trái ảnh -> LLaVA đoán left of."
    },
    {
        "id": 26, "img": "000000029745.jpg",
        "note": "Rò rỉ trục: Điện thoại trước mặt người quàng khăn nhưng lệch pixel sang trái -> LLaVA đoán left of."
    }
]

# 3. Hàm tải ảnh đa tầng (Local paths + CDN direct fallback)
def load_showcase_img(img_name):
    # Kiểm tra các đường dẫn cục bộ khả dĩ trên Kaggle
    candidates = [
        os.path.join("data/spatial_mqa/images", img_name),
        os.path.join("/kaggle/working/temporal-spatio-qa/data/spatial_mqa/images", img_name),
        os.path.join("/kaggle/working/data/spatial_mqa/images", img_name),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return Image.open(p)
            except Exception:
                pass
    
    # Fallback trực tiếp từ CDN nếu chưa tải ảnh về disk
    try:
        url = f"https://huggingface.co/datasets/liuziyan/SpatialMQA/resolve/main/images/{img_name}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            return Image.open(io.BytesIO(resp.read()))
    except Exception:
        return None

# 4. Vẽ trực quan hóa 2x2
fig, axes = plt.subplots(2, 2, figsize=(15, 14))
axes = axes.flatten()

for idx, case in enumerate(canonical_cases):
    ax = axes[idx]
    target_img = case["img"]
    cid = case["id"]

    # Tìm sample theo image hoặc ID
    match = None
    for s in samples:
        if s.get("image") == target_img or s.get("id") == cid:
            match = s
            break

    # Hiển thị ảnh
    img_obj = load_showcase_img(target_img)
    if img_obj:
        ax.imshow(img_obj)
    else:
        ax.text(0.5, 0.5, f"[Không tải được ảnh: {target_img}]", ha="center", va="center", color="red")
    ax.axis("off")

    gt = match.get("answer", "") if match else ""
    pred = match.get("output", "") if match else ""
    q = match.get("question", "") if match else ""
    if len(q) > 60:
        q = q[:60] + "..."

    note_wrapped = "\\n".join(textwrap.wrap(case["note"], width=65))
    title_text = (
        f"Mẫu #{cid} ({target_img})\\n"
        f"Q: {q}\\n"
        f"GT: {gt}  |  LLaVA: {pred} (SAI)\\n"
        f"Cơ chế: {note_wrapped}"
    )
    ax.set_title(title_text, fontsize=9.5, fontweight="bold", color="#991b1b", pad=8)

plt.tight_layout(h_pad=4.0, w_pad=2.0)
plt.show()
"""

def split_lines(src_text):
    lines = src_text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []

def main():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Update cells
    nb["cells"][16]["source"] = split_lines(CELL_16_SOURCE.strip())
    nb["cells"][18]["source"] = split_lines(CELL_18_SOURCE.strip())
    nb["cells"][20]["source"] = split_lines(CELL_20_SOURCE.strip())
    nb["cells"][22]["source"] = split_lines(CELL_22_SOURCE.strip())

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print("[SUCCESS] Updated cells 16, 18, 20, 22 in notebook 002!")

if __name__ == "__main__":
    main()
