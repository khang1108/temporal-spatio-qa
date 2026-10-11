# AGENTS.md — Điều Lệ Nghiên Cứu & Kiến Trúc Cốt Lõi (Anchor Charter)
## Dự án: Phân Tích Cơ Chế Thất Bại Tầng Token/Tham Số & Giải Pháp Prefix Tuning Cho Trí Tuệ Không Gian 3D Nhập Vai (FRS) trong MLLM
### Benchmark Mỏ Neo: SpatialMQA (ACL 2025 Long) | Mô hình Nền tảng: LLaVA-1.5-7B & SpaceLLaVA

---

## 🎯 1. Tuyên Bố Sứ Mệnh & Bài Toán Nghiên Cứu (Mission Statement)

Dự án này tập trung giải quyết bài toán cốt lõi: **Nâng cao năng lực suy luận không gian 3D (3D Spatial Reasoning) và khả năng chuyển đổi hệ quy chiếu ngôi thứ nhất (Egocentric Frame of Reference Shift - FRS) trong các Mô hình Ngôn ngữ - Thị giác Lớn (MLLM)**.

Các MLLM thịnh hành (LLaVA-1.5, GPT-4V, Gemini) đều gặp **"điểm nghẽn trí tuệ không gian"** mang tính cấu trúc: bộ mã hóa thị giác (CLIP ViT) ép phẳng thế giới vật lý 3D thành lưới pixel 2D, khiến mô hình hoàn toàn "mù" trước hình học 3D, chiều sâu khoảng cách và sự biến đổi góc nhìn của người quan sát.

### 🌟 Triết Lý Nghiên Cứu: Định Hình Lại Cách Tiếp Cận Khoa Học Cho ACIIDS
Dự án **KHÔNG** biến quy trình nghiên cứu ($E_0 \rightarrow E_1 \rightarrow E_2 \rightarrow RQ_2$) thành một pipeline cố định mang tính giáo điều, giả định trước như thể chúng ta đã biết lỗi chắc chắn nằm ở attention heads và chỉ cần tìm đúng vị trí để sửa.

**Thực tế nghiên cứu khách quan:**
Chúng ta **chưa biết trước điều đó**. Hiện tại, chúng ta mới chỉ có một hiện tượng thực nghiệm cần giải thích: **LLaVA-1.5-7B + LoRA vẫn trả lời sai nghiêm trọng nhiều câu hỏi SpatialMQA yêu cầu nhập vai ($Q_2$) và suy luận vị trí tương đối**. 
Chúng ta có một số giả thuyết xuất phát từ các nghiên cứu trước (prior research), nhưng **chưa biết giả thuyết nào thực sự đúng trên mô hình của mình**.

Do đó, toàn bộ nghiên cứu được vận hành theo **Vòng lặp Khoa học Hướng Giả thuyết (Hypothesis-Driven Scientific Inquiry Loop)**:

```
                  ┌──────────────────────────────────────────────┐
                  │          1. OBSERVATION (Quan sát)           │
                  │  Mô hình thất bại ở đâu? Patterns lỗi gì?    │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │           2. HYPOTHESIS (Giả thuyết)         │
                  │ Dựa vào prior research, cơ chế nào giải thích?│
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │            3. PROBING (Thiết kế đo)          │
                  │ Nếu giả thuyết đúng, kỳ vọng quan sát được gì?│
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │         4. INTERVENTION (Kiểm tra nhân quả)  │
                  │ Can thiệp thành phần nghi vấn có đổi hành vi?│
                  └──────────────────────┬───────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
       [CÓ BẰNG CHỨNG HỖ TRỢ]                      [KHÔNG CÓ BẰNG CHỨNG]
       Thu hẹp cơ chế giải thích                   Kiểm tra lại phép đo
       Kiểm thử nhân quả mạnh hơn                  Bác bỏ hoặc sửa đổi giả thuyết
       Đánh giá khả năng steering                  Chuyển sang khảo sát cơ chế khác
```

Một vòng lặp nghiên cứu, không phải chuỗi bước mà mọi giả thuyết đều phải đi qua theo cùng thứ tự.

> [!IMPORTANT]
> **Điểm cốt lõi: Tính Khả Bác Bỏ (Falsifiability First)**  
> Chúng ta **bắt buộc phải quyết định trước điều gì có thể bác bỏ giả thuyết**, chứ không chỉ thiết kế thí nghiệm để tìm bằng chứng ủng hộ nó. Một thí nghiệm chỉ có giá trị khoa học khi nó có khả năng chứng minh giả thuyết là sai.

---

### 🎯 1.1. Trở Lại Hai Câu Hỏi Nghiên Cứu (Research Questions): Chúng Ta Thực Sự Chưa Biết Điều Gì?

#### 🔹 $\mathbf{RQ_1}$ — Cơ chế Thất bại (Mechanistic Failure Breakdown):
*Trong quá trình suy luận không gian, sự tương tác giữa Text Tokens và Visual Tokens bị đứt gãy tại những layers/heads nào? Liệu attention leakage và camera-default representation collapse có giải thích được thất bại khi nhập vai hay không?*

Câu hỏi này được bóc tách rành mạch thành **hai tầng nhận thức**:
1. **Tầng Hiện Tượng (Phenomenological Level):**
   - Mô hình có thực sự bị ảnh hưởng bởi góc nhìn mặc định của camera không?
   - Các lỗi có thực sự liên quan đến việc chú ý sai đối tượng (wrong object attention) hay không?
2. **Tầng Cơ Chế (Mechanistic Level):**
   - Nếu hiện tượng tồn tại, nó gắn với:
     * Trọng số chú ý (Attention Weights $\mathbf{A}$)?
     * Đầu ra của từng head (Head Outputs $\mathbf{h}_{l,h}$)?
     * Không gian biểu diễn ẩn (Hidden Representations / Residual Stream Trajectory)?
     * Hay một cơ chế phân tán (Distributed Mechanism) lan tỏa qua nhiều layers?
   - **Quy tắc bắt buộc:** Chúng ta **không được xem tầng cơ chế là kết luận có sẵn**.

#### 🔹 $\mathbf{RQ_2}$ — Can Thiệp Để Cải Thiện (Conditioned Frame-Shift Intervention):
*Nghiên cứu cơ chế Soft Coordinate Steering Prefix Tokens đưa vào Key-Value Cache ($K, V$), điều kiện hóa theo bounding box và vector hướng nhìn $\vec{h}_{\text{agent}}$ của nhân vật.*

Ba điều **chưa biết bắt buộc phải kiểm chứng bằng Ablation Study** (tuyệt đối không kết luận từ các paper khác):
1. **Vị trí can thiệp:** Đưa prefix vào $K, V$ ở đâu và ở layer nào có ảnh hưởng đáng kể?
2. **Bản chất tác động:** Tác động ấy có thực sự giúp mô hình sử dụng hệ quy chiếu nhân vật (Agent Reference Frame) hay chỉ đơn thuần làm thay đổi phân phối từ ngữ đầu ra (Output Distribution)?
3. **Thành phần tạo ra cải thiện:** Nếu có cải thiện độ chính xác, thành phần nào thực sự tạo ra cải thiện:
   - Bản thân các tokens prefix?
   - Điều kiện không gian (Spatial Condition: bounding box + hướng nhìn)?
   - Chiến lược chọn tầng can thiệp (Layer Selection)?
   - Hay chỉ đơn thuần do việc thêm tham số có thể huấn luyện (Trainable Parameters)?

---

### 📚 1.2. Dùng Prior Research Để Hình Thành Giả Thuyết, Không Dùng Để Tiên Đoán Kết Quả
Ba công trình/trường phái tiêu biểu chúng ta đã khảo sát cung cấp **ba manh mối khác nhau**:
1. **Manh mối 1 — Biểu diễn Tuyến tính & Phân tích Cơ chế VLMs (Engels et al., 2026 / Linear Mech VLMs):**  
   - *Gợi ý giả thuyết:* Khái niệm không gian có thể được mã hóa thành các hướng tuyến tính (Linear Representations) trong không gian ẩn, hoặc do các attention heads chuyên biệt phụ trách.
   - *Điều kiện bác bỏ:* Nếu linear probes không tìm thấy phân tách có ý nghĩa hoặc việc can thiệp vào các heads nghi vấn không làm thay đổi phân phối dự đoán $\rightarrow$ Bác bỏ giả thuyết head chuyên môn hóa, chuyển sang khảo sát cơ chế phân tán đa tầng.
2. **Manh mối 2 — Điểm Mù Chiều Sâu từ Bộ Mã Hóa Thị Giác (SpatialVLM, Perception Encoder, SpatioLM):**  
   - *Gợi ý giả thuyết:* Đứt gãy có thể xuất phát từ chính việc bộ mã hóa CLIP ViT 2D làm mất chiều sâu tương đối ngay từ tầng trích xuất visual tokens, khiến LLM thiếu thông tin hình học trước khi suy luận.
   - *Điều kiện bác bỏ:* Nếu ngay cả khi cung cấp rõ ràng thông tin depth/bbox 3D mà mô hình vẫn thất bại trong việc xoay hệ quy chiếu $\rightarrow$ Điểm nghẽn nằm ở khâu frame transformation của LLM chứ không đơn thuần do thiếu depth tokens.
3. **Manh mối 3 — Khả năng & Giới hạn của Prefix Tuning (Li & Liang 2021, MoReS 2025, ICLR 2024 Limitations):**  
   - *Gợi ý giả thuyết:* Bơm learnable prefix vào $K, V$ cache có thể hoạt động như một toán tử xoay hệ quy chiếu mềm để "lái" biểu diễn.
   - *Điều kiện bác bỏ:* ICLR 2024 chứng minh prefix tuning có thể gặp giới hạn về sức chứa biểu diễn (capacity limitation) và chỉ làm trôi phân phối ngôn ngữ. Nếu prefix không vượt qua được LoRA baseline trên các ca ego-centric đối ngẫu $\rightarrow$ Bác bỏ prefix tự do, bắt buộc chuyển sang prefix có gating điều kiện hình học rõ ràng.

---

## 📊 2. Dữ Liệu & Quy Định Benchmark Mỏ Neo (SpatialMQA - ACL 2025)

Dự án sử dụng **chính thức và duy nhất** benchmark học thuật đã công bố tại **ACL 2025 Long: SpatialMQA**.

### A. Quy Mô Tập Dữ Liệu:
* **Tập Huấn Luyện (`train.jsonl` / `train_3780.json`):** 3,780 mẫu câu hỏi - đáp.
* **Tập Kiểm Định (`dev.jsonl` / `dev_536.json`):** 536 mẫu.
* **Tập Đánh Giá Chính Thức (`test.jsonl`):** 1,076 mẫu test chuẩn của ACL 2025.
* **Tập Thử Thách Góc Khuất (`invisible.jsonl`):** 811 mẫu đo độ ảo giác khi vật thể bị che khuất.
* **Kho Ảnh Gốc (`data/spatial_mqa/images/`):** Đúng **5,063 ảnh MS-COCO** tự nhiên (~800MB tải từ Hugging Face CDN).

### B. Quy Tắc Toàn Vẹn Dữ Liệu & Gán Nhãn (Data & Annotation Policy):
* **ĐÃ ĐỦ 100%, KHÔNG CẦN TẠO THÊM DATASET:** Toàn bộ dữ liệu train/dev/test đã được chuẩn hóa bởi tác giả SpatialMQA. Tuyệt đối không tự ý thêm các câu hỏi bên ngoài để đảm bảo tính so sánh công bằng (fair comparison) với các baseline công bố quốc tế.
* **KHÔNG ANNOTATE BẰNG TAY:** Toàn bộ nhãn câu hỏi, options, đáp án, trục tọa độ ($A_x, A_y, A_z$), và góc nhìn (Q1, Q2, Q3) đã có sẵn ground-truth chính xác.
* **Diagnostic Probe Set:** Để visualize cơ chế và mổ xẻ token, trích xuất một tập con thăm dò gồm ~30–50 trường hợp xung đột góc nhìn tiêu biểu từ tập test có sẵn.
* **Trích xuất đặc trưng tự động:** Nếu cần thông tin hướng nhìn $\vec{h}$ hoặc tọa độ bounding box hỗ trợ Prefix, dùng các mô hình zero-shot có sẵn (ZoeDepth, Grounding DINO, 3D Pose), tuyệt đối không gán nhãn thủ công.

---

## 🔬 3. Hai Trụ Cột Nghiên Cứu Cốt Lõi (Core Research Pillars)

```
                            ┌── TRỤ CỘT 1: Chẩn đoán Cơ chế Thất bại Tầng Token & Parameter
                            │   ├── Cross-Attention Probing (Đo rò rỉ attention từ Camera sang Ego)
                            │   ├── Hidden State Geometry (Chứng minh sụp đổ không gian biểu diễn)
HƯỚNG NGHIÊN CỨU TRỌNG TÂM ──┤   ├── Layer-wise Bottleneck Localization (Xác định tầng Transformer bị nghẽn)
                            │   └── Visualizations (Attention Heatmaps, t-SNE Trái/Phải, Trajectory)
                            │
                            └── TRỤ CỘT 2: Can thiệp Kiến trúc bằng Prefix Tuning (Principled Intervention)
                                ├── Embodiment / Soft Coordinate Steering Prefix Tokens
                                ├── Key-Value Cache Injection tại các tầng Transformer bị nghẽn
                                └── Before vs. After Validation (Attention phục hồi & Điểm số bứt phá)
```

### 🔍 Trụ Cột 1: Phân Tích & Trực Quan Hóa Cơ Chế Thất Bại (Mechanistic Failure Analysis)
Ta mổ xẻ nội tại mô hình LLaVA-1.5 và SpaceLLaVA qua 4 kịch bản thất bại cốt lõi:
1. **Sập bẫy Camera (Camera-Default Representation Collapse):**
   - Khi góc nhìn Camera và nhân vật cho ra kết quả trái ngược (ví dụ: Camera thấy *phải*, nhân vật thấy *trái*).
   - *Cơ chế:* Chứng minh Attention của LLM bỏ qua token của nhân vật và bám chặt vào visual tokens toàn cảnh; Hidden states tại token đáp án bị cụm chặt vào đáp án của Camera thay vì phân tách theo nhân vật.
2. **Mâu thuẫn Hướng Đầu vs Thân (Head-Torso Disagreement):**
   - Nhân vật có mặt quay một hướng, thân hướng một hướng khác.
   - *Cơ chế:* Xác định xem Attention Heads ưu tiên thông tin vùng mặt hay vùng thân của đối tượng.
3. **Điểm Mù Trục Chiều Sâu ($A_y$ Depth Blindness):**
   - LLaVA-1.5 chỉ đạt 29.64% trên trục chiều sâu (in front of / behind).
   - *Cơ chế:* Bộ mã hóa CLIP ViT 2D làm mất chiều sâu tương đối, khiến các token chiều sâu bị nhiễu loạn trong không gian biểu diễn.
4. **Ranh Giới Góc Nhìn Nhạy Cảm (Perspective Boundary Drift):**
   - Vật thể nằm ở góc chuyển tiếp (~80° - 100°). Đo độ bất định (entropy) của các attention heads khi đối mặt với ranh giới góc nhìn.

### ⚙️ Trụ Cột 2: Giải Pháp Prefix Tuning Có Cơ Sở Lý Thuyết (Principled Prefix Tuning)
Từ phát hiện ở Trụ Cột 1 (xác định chính xác tầng $L$ và cơ chế bị nghẽn), ta thiết kế module can thiệp:
1. **Embodiment / Coordinate Steering Prefix Tokens:**
   - Chuỗi $M$ vector prefix có thể học được (learnable soft tokens), đóng vai trò như toán tử xoay hệ quy chiếu mềm (Soft Frame-Shift Operator).
2. **Can thiệp vào Tầng Chú Ý (Key-Value Attention Intervention):**
   - Bơm Prefix trực tiếp vào $K, V$ cache của các tầng Self-Attention trong LLM backbone (hoặc tầng Cross-Modal Projector).
   - Dòng thông tin khi đi qua các tầng này bị "uốn cong" theo trục quan sát của nhân vật, ép mô hình thoát khỏi quán tính camera.
3. **Bằng Chứng Phục Hồi (Before vs. After):**
   - Trực quan hóa sự thay đổi: Attention tập trung đúng vào nhân vật; Hidden states tách biệt rõ ràng giữa các quan hệ không gian đối ngẫu.
   - Hiệu quả tham số: Cố định toàn bộ LLM và ViT, chỉ cập nhật Prefix (kết hợp LoRA nhẹ) trên 1x GPU A100 (40GB).

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

## 📂 5. Bản Đồ Tài Nguyên & Mã Nguồn Cốt Lõi

```
Spatial-Temporal-KG/
├── AGENTS.md                               # ĐIỀU LỆ NEO: Tôn chỉ nghiên cứu & kiến trúc cốt lõi
├── docs/
│   ├── TRAINING_GUIDE_A100.md             # Hướng dẫn huấn luyện A100 (ZeRO-2, LoRA/Prefix, BF16)
│   ├── visualizations/                    # Visualizations học thuật và portal phân tích
│   │   ├── index.html                     # Portal nghiên cứu hợp nhất (Unified Dashboard)
│   │   ├── survey.html                    # Khảo sát 27 bài báo, phản biện & failure taxonomy
│   │   ├── serve.sh                       # Script chạy local server
│   │   └── serve_survey.py                # Python HTTP Server
│   └── survey/
│       ├── SURVEY_MASTER.md               # Khảo sát 25 bài báo phân loại theo 4 trường phái
│       └── survey_papers.xlsx             # Bảng theo dõi phân tích giới hạn các bài báo
├── data/
│   └── spatial_mqa/
│       ├── train.jsonl (3,780 câu)        # Dữ liệu huấn luyện gốc
│       ├── dev.jsonl (536 câu)            # Dữ liệu kiểm định gốc
│       ├── test.jsonl (1,076 câu)         # Benchmark đánh giá chính thức (ACL 2025)
│       ├── invisible.jsonl (811 câu)      # Tập thử thách che khuất bổ sung
│       ├── train_3780.json                # Định dạng chuẩn hội thoại cho LLaVA
│       ├── dev_536.json                   # Định dạng chuẩn hội thoại dev
│       └── images/ (5,063 ảnh)            # Toàn bộ 800MB ảnh MS-COCO từ HF CDN
├── scripts/
│   ├── download_images.py                 # Tool tải nhanh ảnh từ HF CDN
│   ├── train_mem.py                       # Điểm vào huấn luyện tương thích SDPA
│   ├── train_llava_lora.sh                # Huấn luyện LoRA baseline cho LLaVA-1.5
│   ├── train_spacellava_lora.sh           # Huấn luyện LoRA baseline cho SpaceLLaVA
│   ├── run_eval.py                        # Suy luận và tính toán chỉ số (Overall, Q1-Q3, Ax-Az)
│   └── zero2.json                         # Cấu hình DeepSpeed ZeRO-2
├── src/
│   ├── data/format_llava.py               # Converter định dạng JSONL -> LLaVA format
│   └── evaluation/metrics.py              # Thư viện đo lường Macro-F1, Acc per axis, Perspective
└── experiments/
    ├── checkpoints/                       # Trọng số adapter / prefix sau huấn luyện
    └── predictions/                       # Kết quả dự đoán và probe traces phục vụ visualize
```

---

## 🤖 6. Điều Lệ Vận Hành Bắt Buộc Cho AI Agents (Agent Guardrails)

Mọi AI Agent làm việc trong dự án này bắt buộc tuân thủ các nguyên tắc bất di bất dịch sau:

1. **Bảo Vệ Tính Toàn Vẹn Của AGENTS.md (No Progress Clutter):**
   - Tệp này là **Điều lệ neo kiến trúc và mục tiêu dự án (Anchor Charter)**.
   - **TUYỆT ĐỐI KHÔNG** biến tệp này thành sổ nhật ký tiến độ hàng ngày (Progress Report), to-do list tạm bợ hay bảng chấm công. Các kế hoạch tạm thời phải lưu trong các artifact riêng hoặc tài liệu tài nguyên khác.
2. **Bảo Vệ Phạm Vi Nghiên Cứu (Strict Scope Protection):**
   - **TUYỆT ĐỐI KHÔNG** đưa lại mã nguồn, tập dữ liệu hoặc thư viện Knowledge Graph (KGQA / STCQA) cũ.
   - Tập trung vào **LLaVA-1.5-7B** và **SpaceLLaVA** trên benchmark **SpatialMQA**.
3. **Tôn Trọng Phương Pháp Luận Cơ Chế (Mechanistic Focus):**
   - Mọi đề xuất cải tiến phải gắn liền với **tầng Token, Parameter, và Attention Visualization**. Không chấp nhận các giải pháp chỉ là Prompt Engineering bề mặt thiếu cơ sở hình học.
   - Không được đề xuất ngồi gán nhãn lại dữ liệu thủ công bằng tay.
4. **Chuẩn Mực Môi Trường & Phần Cứng:**
   - Hoạt động tối ưu trên **1x NVIDIA A100 (40GB)** dưới Linux với **PyTorch 2.5.1 + CUDA 12.1 + DeepSpeed ZeRO-2**.
   - Dùng cơ chế native SDPA (Scaled Dot-Product Attention) của PyTorch để tránh lỗi biên dịch của `flash-attn`.
   - Thiết lập `TRITON_CACHE_DIR=/tmp/triton_${USER}` để chống nghẽn I/O trên ổ mạng NFS.
5. **Kỷ Luật Git:**
   - Kiểm tra `git status` trước và sau khi can thiệp mã nguồn.
   - Commit với thông điệp rõ ràng theo chuẩn: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.
6. **Tư Duy Đồng Đội Phản Biện & Đề Xuất Giải Pháp (Constructive Teammate Critique & Solution-Oriented Collaboration):**
   - **Tuyệt đối không tâng bốc suông (No Flattery):** Không dùng lời khen đãi bôi, sáo rỗng hoặc vội vã đồng ý mù quáng.
   - **Phản biện mang tính xây dựng của một người đồng đội (Constructive Teammate Mindset):** Không phủ định triệt tiêu hay bác bỏ cực đoan ý tưởng của người dùng. Hãy coi ý tưởng của người dùng là một trực giác nghiên cứu (research intuition) quý giá: phân tích điểm nghẽn kỹ thuật, chỉ ra các rủi ro lý thuyết dựa trên các công trình đã khảo sát, nhưng **bắt buộc phải cùng tìm kiếm giải pháp (proactive solutions), đề xuất các ngả rẽ khả thi hoặc cách biến ý tưởng đó thành hiện thực trong khuôn khổ tài nguyên cho phép**.
   - **Lập luận và gợi mở bằng bằng chứng (Evidence-Based Suggestions):** Mọi lời khuyên hoặc hướng đi mới phải được neo vào các cơ chế, phát hiện thực nghiệm và phương pháp luận từ các paper đã khảo sát (SpatialMQA, Perception Encoder, SpatioLM, Kang et al., Prefix-Tuning, v.v.).
7. **Kỷ Luật Khoa Học Hướng Giả Thuyết & Chống Thiên Kiến Xác Nhận (Falsifiability Discipline & Anti-Confirmation Bias):**
   - Tuyệt đối không giả định trước cơ chế thất bại hay kết quả can thiệp khi chưa có số liệu thực nghiệm trên LLaVA-1.5 / SpaceLLaVA.
   - Mọi thí nghiệm probing hay intervention phải xác định trước: *Kết quả thực nghiệm nào sẽ bác bỏ giả thuyết này?*
   - Nếu thực nghiệm cho thấy giả thuyết sai, phải dũng cảm bác bỏ và cập nhật giả thuyết mới, tuyệt đối không cố gượng ép số liệu để bảo vệ định kiến ban đầu.
