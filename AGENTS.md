# AGENTS.md — Điều Lệ Nghiên Cứu & Kiến Trúc Cốt Lõi
## Dự án: Trí Tuệ Không Gian 3D trong MLLM — Phân Tích Lỗi Nhập Vai & Giải Pháp Prefix Tuning
### Benchmark Mỏ Neo: SpatialMQA (ACL 2025 Long) | Mô hình Nền tảng: LLaVA-1.5 & SpaceLLaVA

---

## 🎯 1. Tuyên Bố Sứ Mệnh & Bài Toán Nghiên Cứu (Mission Statement)

Dự án này tập trung giải quyết bài toán cốt lõi: **Nâng cao năng lực suy luận không gian 3D (3D Spatial Reasoning) và khả năng chuyển đổi hệ quy chiếu ngôi thứ nhất (Egocentric Frame of Reference Shift - FRS) trong các Mô hình Ngôn ngữ - Thị giác Lớn (MLLM)**.

Các MLLM tiêu chuẩn hiện nay (LLaVA-1.5, GPT-4V, Gemini) đều mắc phải **"điểm nghẽn trí tuệ không gian"** mang tính cấu trúc: bộ mã hóa thị giác (CLIP ViT) ép phẳng thế giới vật lý 3D thành lưới pixel 2D, khiến mô hình hoàn toàn "mù" trước hình học 3D, chiều sâu khoảng cách và sự biến đổi góc nhìn của người quan sát.

### Benchmark Mỏ Neo: SpatialMQA (ACL 2025)
* **Quy mô tập dữ liệu:** 5,392 câu hỏi trên 5,063 ảnh MS-COCO tự nhiên, không có bounding box sẵn.
* **Các trục tọa độ đánh giá:** Trục ngang ($A_x$: left/right), Trục chiều sâu ($A_y$: in front of/behind), Trục đứng ($A_z$: on/above/below).
* **Các dạng góc nhìn (Perspectives):** 
  - **Q1 (Ngoài ảnh / Camera Viewpoint):** Góc nhìn tĩnh từ camera nhìn vào khung cảnh (452 câu test / 2,183 câu tổng).
  - **Q2 (Ngôi thứ nhất / Nhập vai trực tiếp - Ego-centric):** *"Nếu bạn là chú hươu cao cổ trong ảnh, mặt trời ở đâu so với bạn?"* (590 câu test / 3,028 câu tổng).
  - **Q3 (Ngôi thứ ba trong ảnh / In-image Observer):** Người quan sát ở vị trí $X$ nhận định quan hệ giữa $A$ và $B$ (34 câu test / 181 câu tổng).
* **Trần hiệu năng của Baselines:** 
  - **LLaVA-1.5-7B (LoRA):** Độ chính xác toàn cục **46.56%** (Trục chiều sâu $A_y = 29.64\%$, Câu hỏi nhập vai $\text{Q2} = 40.99\%$).
  - **SpaceLLaVA (LoRA):** Độ chính xác toàn cục **48.14%** (Trục đứng $A_z$ sụp đổ xuống $31.41\%$).

---

## 🏆 2. Lộ Trình Công Bố Khoa Học (Publication Strategy)

Chiến lược nghiên cứu được chia làm 2 giai đoạn kế thừa chặt chẽ:

### 🚩 Giai đoạn 1 (Ngắn hạn — Mục tiêu Hội nghị Rank B: ACIIDS):
* **Trọng tâm:** 
  1. Xây dựng **Hệ thống phân loại lỗi toàn diện (Comprehensive Failure Taxonomy)** và chẩn đoán định lượng chính xác các kịch bản MLLM bị thất bại khi trả lời câu hỏi nhập vai (Q2/Q3).
  2. Bóc tách hiện tượng "sập bẫy camera" (Camera-Default Bias), mâu thuẫn giữa hướng đầu và thân (Head-Torso Disagreement), và sự suy giảm của mô hình ở ranh giới góc nhìn.
  3. Đề xuất giải pháp kiến trúc thử nghiệm ban đầu dựa trên **Prefix Tuning / Embodiment Tokens** can thiệp trực tiếp vào mô hình để cải thiện câu hỏi nhập vai so với baseline LoRA chuẩn.

### 🚀 Giai đoạn 2 (Dài hạn — Mở rộng cho Hội nghị Rank A / A* như ACL, CVPR, NeurIPS, EMNLP, AAAI):
* **Trọng tâm:**
  1. Hoàn thiện giải pháp **Geometric Prefix Tuning & Coordinate Transformation Modules** tác động sâu vào tầng tham số và biểu diễn Token (K, V cache attention layers / Cross-modal Projector).
  2. Triển khai khuôn khổ thực nghiệm can thiệp đa tầng (Factorial Interventions) bóc tách triệt để: Perception (Grounding + Heading) vs. Reasoning (Coordinate Rotation).
  3. Đạt bước nhảy vọt hiệu năng vượt trần 56–60% trên SpatialMQA và chuyển giao sang các benchmark 3D VLM khác (VSR, ViewSpatial, SpatialSense).

---

## 🔬 3. Hai Trụ Cột Nghiên Cứu Cốt Lõi (Core Research Pillars)

```
                            ┌── TRỤ CỘT 1: Chẩn đoán & Phân loại Lỗi Nhập vai (Failure Taxonomy)
                            │   ├── Frame Conflict (Xung đột góc nhìn Camera vs. Ego)
                            │   ├── Head vs. Torso Disagreement (Mâu thuẫn hướng đầu và thân)
                            │   ├── Relation Boundary Sensitivity (Độ nhạy ranh giới góc quay)
HƯỚNG NGHIÊN CỨU TRỌNG TÂM ──┤   └── Entity Mis-grounding (Nhầm lẫn định vị trong cảnh phức tạp)
                            │
                            └── TRỤ CỘT 2: Tác Động Kiến Trúc bằng Prefix Tuning (Architectural Intervention)
                                ├── Parameter-Level Intervention (Tác động trực tiếp vào Weights / Soft Prompts)
                                ├── Embodiment & Rotation Prefix Tokens (Token hóa hệ quy chiếu hình học)
                                └── Cross-Attention Modulation (Điều hướng sự chú ý của LLM theo trục quan sát)
```

### 🔍 Trụ Cột 1: Phân Tích & Bóc Tách Các Trường Hợp Thất Bại (Failure Mode Diagnostics)
Thay vì coi MLLM là hộp đen chấm điểm trắc nghiệm, ta mổ xẻ chuỗi suy luận thành 4 kịch bản thất bại:
1. **Xung đột Hệ quy chiếu (Frame Conflict):** Góc nhìn Camera và góc nhìn của nhân vật cho ra kết quả trái ngược (ví dụ: nhìn từ ngoài là *phải*, nhưng với nhân vật là *trái*). Chứng minh tỷ lệ mô hình "lười biếng" copy đáp án từ camera.
2. **Bất đồng Đầu vs Thân (Head-Torso Disagreement):** Nhân vật có hướng mặt khác hướng thân (ví dụ: người đi xe đạp quay đầu sang bên). Khảo sát xem mô hình bối rối hay ưu tiên hệ quy chiếu nào.
3. **Ranh giới quan hệ mấp mé (Relation Boundary Sensitivity):** Vật thể nằm ở góc lệch ~80° - 100° giữa các quan hệ. Đánh giá độ bền vững trước sai số ước lượng góc nhìn.
4. **Sai lệch định vị đối tượng (Entity Mis-grounding):** Mô hình gán nhầm nhân vật được chỉ định trong câu hỏi khi khung cảnh có nhiều đối tượng tương tự.

### ⚙️ Trụ Cột 2: Giải Pháp Prefix Tuning & Tác Động Kiến Trúc (Prefix Tuning & Architectural Intervention)
Không dừng lại ở việc tinh chỉnh câu lệnh (Prompt Engineering) bề mặt, ta can thiệp trực tiếp vào tầng tham số và biểu diễn token:
1. **Geometric Prefix Tokens (Soft Prompts dạng hình học):**
   - Thay vì ép LLM tự tưởng tượng phép quay, ta biến đổi thông tin hình học (vị trí chủ thể, hướng nhìn $\vec{h}$, ma trận xoay $R_{\text{cam}\rightarrow\text{obj}}$) thành chuỗi **Prefix Embedding Tokens** có thể học được.
2. **Can thiệp vào Tầng Transformer (Key-Value Attention Intervention):**
   - Chèn các Prefix Vectors vào các tầng Self-Attention của LLM backbone hoặc tầng Cross-Modal Projector của LLaVA.
   - Điều này buộc không gian biểu diễn ngôn ngữ của LLM phải bị "uốn cong" theo đúng hệ quy chiếu của nhân vật trước khi sinh ra câu trả lời.
3. **Huấn luyện PEFT / Prefix Tuning Tối ưu:**
   - Cố định phần lớn Vision Encoder và LLM base, chỉ cập nhật Prefix Parameters kết hợp LoRA trên 1x GPU A100 (40GB) đảm bảo tính nhẹ, khả thi và hội tụ nhanh.

---

## 🏛️ 4. Mục Tiêu Chỉ Số Hiệu Năng (Target Goals)

| Chỉ số Đánh Giá | LLaVA-1.5 Baseline | SpaceLLaVA Baseline | **Mục tiêu Giai đoạn 1 (ACIIDS)** | **Mục tiêu Giai đoạn 2 (Rank A/A\*)** |
|---|---|---|---|---|
| **Độ chính xác Tổng (Overall)** | 46.56% | 48.14% | **$\ge$ 51.5% (+3 to 5%)** | **$\ge$ 56.0% - 60.0%** |
| **Q1 (Góc nhìn Camera)** | 53.14% | 54.87% | **$\ge$ 56.0%** | **$\ge$ 62.0%** |
| **Q2 (Nhập vai Ego-centric)** | **40.99%** | **42.37%** | **$\ge$ 50.0% (+8 to 10%)** | **$\ge$ 65.0% (+23% boost)** |
| **Q3 (Góc nhìn thứ 3 trong ảnh)** | 64.71% | 58.82% | **$\ge$ 65.0%** | **$\ge$ 70.0%** |
| **Trục Ngang ($A_x$)** | 55.71% | 56.00% | **$\ge$ 60.0%** | **$\ge$ 65.0%** |
| **Trục Chiều sâu ($A_y$)** | **29.64%** | 51.85% | **$\ge$ 45.0%** | **$\ge$ 60.0% (+30% boost)** |
| **Trục Đứng ($A_z$)** | 48.13% | 31.41% | **$\ge$ 45.0%** | **$\ge$ 55.0%** |

---

## 📂 5. Cấu Trúc Mã Nguồn & Tài Nguyên Dự Án

```
Spatial-Temporal-KG/
├── AGENTS.md                               # Điều lệ nghiên cứu, kiến trúc lõi & quy định agent
├── docs/
│   ├── TRAINING_GUIDE_A100.md             # Hướng dẫn huấn luyện A100 (ZeRO-2, LoRA/Prefix, BF16)
│   ├── visualizations/                    # Visualizations học thuật và portal phân tích
│   │   ├── index.html                     # Portal nghiên cứu hợp nhất (Unified Dashboard)
│   │   ├── survey.html                    # Khảo sát 27 bài báo, phản biện & failure taxonomy
│   │   ├── serve.sh                       # Script chạy local server
│   │   └── serve_survey.py                # Python HTTP Server
│   └── survey/
│       ├── SURVEY_MASTER.md               # Bản khảo sát 25 bài báo phân loại theo 4 trường phái
│       ├── survey_papers.xlsx             # Bảng theo dõi tiến độ đọc và ghi chú giới hạn
│       ├── survey_papers.csv              # Dữ liệu xuất từ Google Sheet
│       └── papers_pdf/                    # Thư viện PDF gốc các bài báo quan trọng
├── data/
│   └── spatial_mqa/
│       ├── train.jsonl (3,780 câu)        # Tập huấn luyện thô
│       ├── dev.jsonl (536 câu)            # Tập kiểm định thô
│       ├── test.jsonl (1,076 câu)         # Tập đánh giá chính thức (Test benchmark)
│       ├── invisible.jsonl (811 câu)      # Tập thử thách góc khuất bổ sung
│       ├── train_3780.json                # Dữ liệu định dạng hội thoại cho LLaVA
│       ├── dev_536.json                   # Dữ liệu dev cho LLaVA
│       └── images/ (5,063 ảnh)            # Toàn bộ 800MB ảnh MS-COCO từ Hugging Face CDN
├── scripts/
│   ├── download_images.py                 # Tool tải nhanh ảnh từ HF CDN
│   ├── train_mem.py                       # Điểm vào huấn luyện tương thích SDPA
│   ├── train_llava_lora.sh                # Script huấn luyện LoRA cho LLaVA-1.5 (ZeRO-2)
│   ├── train_spacellava_lora.sh           # Script huấn luyện LoRA cho SpaceLLaVA
│   ├── run_eval.py                        # Script chạy suy luận & tính toán toàn bộ chỉ số
│   └── zero2.json                         # Cấu hình DeepSpeed ZeRO-2
├── src/
│   ├── data/format_llava.py               # Chuyển đổi định dạng JSONL sang LLaVA Conversation
│   └── evaluation/metrics.py              # Bộ đánh giá Macro-F1, Precision, Recall, Per-axis, Q1/Q2/Q3
└── experiments/
    ├── checkpoints/                       # Trọng số adapter / prefix sau huấn luyện
    └── predictions/                       # Kết quả dự đoán chi tiết phục vụ phân tích lỗi
```

---

## 🤖 6. Quy Tắc Hành Vi & Chỉ Thị Vận Hành Cho Agent

Bất kỳ AI Agent nào làm việc trong không gian mã nguồn này bắt buộc phải tuân thủ nghiêm ngặt các điều lệ sau:

1. **Bảo Vệ Phạm Vi Nghiên Cứu (Scope Protection):**
   - **Tuyệt đối KHÔNG** đưa lại các mã nguồn, tập dữ liệu hoặc thư viện Knowledge Graph (KGQA / STCQA) cũ đã bị loại bỏ.
   - **Tuyệt đối KHÔNG** thêm các mô hình baseline không liên quan (BLIP, IDEFICS, mPLUG, các API thương mại đóng). Dự án chỉ tập trung vào **LLaVA-1.5-7B** và **SpaceLLaVA**.
2. **Tôn Trọng Trọng Tâm Nghiên Cứu:**
   - Mọi cải tiến kỹ thuật phải hướng trực tiếp vào giải quyết **lỗi chuyển đổi góc nhìn FRS (Q2)** và **lỗi trục chiều sâu $A_y$**.
   - Mọi giải pháp phải gắn với **phân tích lỗi (Failure Analysis)** hoặc **tác động kiến trúc (Prefix/Token/Parameter-level Intervention)**, không chấp nhận prompt engineering đơn thuần thiếu cơ sở hình học.
3. **Tương Thích Môi Trường & Phần Cứng:**
   - Toàn bộ mã nguồn phải chạy ổn định trên **1x NVIDIA A100 (40GB)** dưới Linux với **PyTorch 2.5.1 + CUDA 12.1**.
   - Luôn ưu tiên dùng cơ chế bản địa PyTorch SDPA (Scaled Dot-Product Attention) để tránh lỗi biên dịch của `flash-attn`.
   - Luôn thiết lập `TRITON_CACHE_DIR=/tmp/triton_${USER}` để tránh tình trạng treo I/O trên ổ mạng NFS.
4. **Bảo Vệ Tính Toàn Vẹn Của Dữ Liệu:**
   - Bộ ảnh chính thức gồm **chính xác 5,063 ảnh (~800 MB)** trong thư mục `data/spatial_mqa/images/`.
   - Tuyệt đối không tải lại tệp `test2017.zip` 6.2GB từ máy chủ COCO bị bóp băng thông.
5. **Đồng Bộ Git & Chuẩn Mực Commit:**
   - Luôn kiểm tra `git status` trước và sau khi chỉnh sửa. Tạo các commit ngữ nghĩa chuẩn (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
   - Đảm bảo đồng bộ với nhánh từ xa `origin/main`.
