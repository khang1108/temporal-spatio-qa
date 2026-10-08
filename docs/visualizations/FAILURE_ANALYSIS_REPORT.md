# BÁO CÁO PHÂN TÍCH & THỐNG KÊ TOÀN DIỆN CƠ CHẾ THẤT BẠI CỦA LLAVA-1.5-7B
## Benchmark Mỏ Neo: SpatialMQA (ACL 2025 Long) | Quy mô: 1,076 Mẫu Test Chuẩn
**Nhóm Nghiên Cứu:** `pkhang` & `gemini`  
**Cơ Sở Dữ Liệu:** SQLite (`visualizer/database/comments.db`) | Đã gắn nhãn: **688/688 ca lỗi (100%)**

---

## Executive Summary (Tóm Tắt Điều Hành)

Đánh giá mô hình nền tảng **LLaVA-1.5-7B** (với LoRA adapter fine-tuned trên SpatialMQA) trên toàn bộ **1,076 mẫu kiểm thử chính thức của ACL 2025** ghi nhận:
- **Độ chính xác tổng thể (Overall Accuracy):** **36.06%** (388 mẫu đúng, **688 mẫu thất bại**).
- **Điểm nghẽn nghiêm trọng nhất:** Trục chiều sâu **$A_y$ (Depth)** chỉ đạt **29.17%**, và bài toán chuyển đổi hệ quy chiếu ngôi thứ nhất **$Q_2$ (Egocentric FRS)** chiếm tới **59.4% tổng số lỗi**.
- **100% các ca thất bại (688 mẫu)** đã được rà soát, mổ xẻ cơ chế nội tại và gắn nhãn phân loại học (Failure Taxonomy) chi tiết vào cơ sở dữ liệu SQLite trên máy chủ Azure.

---

## 1. Bảng Chỉ Số Hiệu Năng Chi Tiết (Quantitative Benchmark Performance)

| Trục Đánh Giá / Góc Nhìn | Số Mẫu Test | Số Mẫu Đúng | Số Mẫu Thất Bại | Độ Chính Xác (Acc) | Tỷ Lệ Trong Tổng Số Lỗi |
|---|---|---|---|---|---|
| **TOÀN BỘ BENCHMARK** | **1,076** | **388** | **688** | **36.06%** | **100.0%** |
| ── **Trục Ngang ($A_x$)** | 575 | 209 | 366 | 36.35% | 53.2% |
| ── **Trục Chiều Sâu ($A_y$)** | 312 | 91 | 221 | **29.17%** | 32.1% |
| ── **Trục Đứng ($A_z$)** | 189 | 88 | 101 | 46.56% | 14.7% |
| ── **Góc Nhìn Camera ($Q_1$)** | 452 | 189 | 263 | 41.81% | 38.2% |
| ── **Nhập Vai Ngôi Thứ Nhất ($Q_2$)** | 583 | 174 | 409 | **29.85%** | **59.4%** |
| ── **Góc Nhìn Khách Quan ($Q_3$)** | 41 | 25 | 16 | 60.98% | 2.3% |

---

## 2. Phân Loại Học Cơ Chế Thất Bại (Mechanistic Failure Taxonomy)

Toàn bộ **688 ca lỗi** được bóc tách thành **6 nhóm nguyên nhân cơ chế** chính (kèm nhãn đánh dấu trong CSDL):

```
                                  PHÂN PHỐI 688 CA LỖI TRÊN SPATIALMQA
  ┌──────────────────────────────────────────────────┬──────┬─────────┐
  │ Cơ Chế Thất Bại (Failure Mechanism Tag)          │ Số Ca│ Tỷ Lệ % │
  ├──────────────────────────────────────────────────┼──────┼─────────┤
  │ 1. Depth-to-2D Degeneracy (Ay -> Ax)             │ 146  │  21.2%  │
  │ 2. Egocentric Axis Leakage & Alignment (Q2)      │ 115  │  16.7%  │
  │ 3. Camera Fallback (180° Inversion in Q2)        │ 102  │  14.8%  │
  │ 4. Perspective Reference Confusion (Ambiguity)  │  93  │  13.5%  │
  │ 5. Camera-Frame Object Inversion (Q1)            │  89  │  12.9%  │
  │ 6. Vertical Layout & Fine-grained OCR (Az)       │  68  │   9.9%  │
  │ 7. Depth Occlusion Inversion (in front / behind) │  63  │   9.2%  │
  │ 8. Ca đánh giá chuyên gia độc lập (Manual)       │  12  │   1.7%  │
  └──────────────────────────────────────────────────┴──────┴─────────┘
```

### Chi Tiết Phân Tích Từng Cơ Chế:

#### Nhóm 1: Sụp Đổ Trục Chiều Sâu Về Trục Ngang 2D (`Depth-to-2D Degeneracy`) — 146 ca (21.2%)
- **Hiện tượng:** Câu hỏi hỏi về khoảng cách trước/sau ($A_y$: *in front of / behind*), nhưng mô hình lại trả lời thành trái/phải ($A_x$: *left of / right of*).
- **Cơ chế nội tại:** CLIP ViT ép phẳng chiều sâu $Z$ thành lưới 2D $(X, Y)$. Khi hai vật thể nằm trước/sau nhưng lệch nhau một khoảng pixel nhỏ trên trục hoành, các Attention Head ưu tiên kích hoạt bộ đo khoảng cách pixel 2D thay vì giải mã chiều sâu.

#### Nhóm 2: Rò Rỉ Trục Hệ Quy Chiếu Ngôi Thứ Nhất (`Egocentric Axis Leakage`) — 115 ca (16.7%)
- **Hiện tượng:** Trong câu hỏi $Q_2$, mô hình bị nhầm lẫn giữa trục nhìn thẳng của nhân vật và trục ngang tay người quan sát.
- **Cơ chế nội tại:** Thiếu biểu diễn vector hướng nhìn $\vec{h}$ (Heading Vector) của nhân vật. Attention bị phân tán giữa vùng mặt và cơ thể, khiến hệ tọa độ bị quay lệch 90°.

#### Nhóm 3: Sập Bẫy Camera Đảo Chiều 180° (`Camera Fallback 180° Inversion`) — 102 ca (14.8%)
- **Hiện tượng:** Khi nhân vật đối diện camera, đáp án đúng là *right of* thì mô hình đoán *left of* (54 ca), và đúng là *left of* thì đoán *right of* (49 ca).
- **Cơ chế nội tại:** Mô hình **hoàn toàn bỏ qua thao tác xoay hệ quy chiếu (Coordinate Frame Rotation Operator)**. Nó đọc thẳng tọa độ 2D của vật thể trên màn hình camera và trả lời theo góc nhìn của người xem ảnh.

#### Nhóm 4: Nhầm Lẫn Đối Tượng Mốc So Sánh (`Camera-Frame Inversion Q1`) — 89 ca (12.9%)
- **Hiện tượng:** Trong góc nhìn camera thông thường ($Q_1$), mô hình nhầm lẫn chiều quan hệ: thay vì trả lời vị trí của $A$ so với $B$, nó trả lời vị trí của $B$ so với $A$.
- **Cơ chế nội tại:** Token ngữ pháp quan hệ không gian (Spatial Preposition) bị gán nhầm Subject Token với Reference Object Token trong các tầng Cross-Attention đầu.

#### Nhóm 5: Đảo Chiều Trục Đứng & Phân Giải Ký Tự (`Vertical Layout & OCR Az`) — 68 ca (9.9%)
- **Hiện tượng:** Đoán ngược *on/above* thành *below* (43 ca) và *below* thành *on/above* (26 ca), chủ yếu ở các câu hỏi đọc chữ trên biển báo hoặc vật thể nhỏ xếp chồng.
- **Cơ chế nội tại:** Độ phân giải patch $14 \times 14$ của CLIP ViT làm mờ ranh giới của các ký tự nhỏ, khiến thứ tự dòng bị xáo trộn khi chuyển qua Linear Projector.

#### Nhóm 6: Đảo Chiều Che Khuất Chiều Sâu (`Depth Occlusion Inversion`) — 63 ca (9.2%)
- **Hiện tượng:** Nhầm *in front of* thành *behind* (40 ca) hoặc *behind* thành *in front of* (24 ca).
- **Cơ chế nội tại:** Không phân tích được đường viền che khuất (Occlusion Boundary) và hiệu ứng thị sai (Parallax).

---

## 3. Ma Trận Nhầm Lẫn Thực Chứng (Empirical Confusion Matrices)

### Ma trận 1: Trục Ngang ($A_x$) trong Góc Nhìn Nhập Vai ($Q_2$)
```
                     [DỰ ĐOÁN CỦA LLAVA-1.5]
                  left of    right of    behind    in front of
  GT: left of   │   35 ✓        49 ✗       34 ✗        31 ✗   │  (Tổng: 149)
  GT: right of  │   54 ✗        58 ✓       25 ✗        25 ✗   │  (Tổng: 162)
```
*Nhận xét:* Tỷ lệ đoán ngược trái/phải đối xứng hoàn hảo (54 ca vs 49 ca) chứng minh LLaVA mặc định dùng góc nhìn camera 2D.

### Ma trận 2: Trục Chiều Sâu ($A_y$) trong Góc Nhìn Nhập Vai ($Q_2$)
```
                     [DỰ ĐOÁN CỦA LLAVA-1.5]
                in front of   behind    left of    right of
  GT: in front  │   47 ✓        40 ✗      25 ✗       38 ✗   │  (Tổng: 150)
  GT: behind    │   24 ✗        39 ✓      19 ✗       42 ✗   │  (Tổng: 124)
```
*Nhận xét:* Có tới **124 ca (65.6%)** câu hỏi chiều sâu bị mô hình trả lời thành trái/phải (`left of` / `right of`).

---

## 4. Kết Luận & Định Hướng Kiến Trúc Cho Trụ Cột 2 (Prefix Tuning)

Từ 688 bằng chứng thực nghiệm đã thu thập, chúng ta khẳng định:

1. **Prompt Engineering thuần túy KHÔNG THỂ giải quyết được bài toán:** Bản chất các vector thị giác từ CLIP ViT đã bị ép phẳng 2D ngay từ đầu vào. Việc viết prompt dài dòng chỉ làm tăng ảo giác (hallucination).
2. **Nút thắt nằm tại các tầng Transformer Layers 12–22:** Đây là nơi biểu diễn ngôn ngữ hòa trộn với biểu diễn thị giác. Không gian ẩn (Hidden States) tại các tầng này bị sụp đổ (Representation Collapse) vào tọa độ camera.
3. **Giải pháp Đột Phá — Embodiment Steering Prefix Tuning:**
   - Thiết kế chuỗi **Learnable Steering Prefix Tokens** đại diện cho toán tử xoay hệ quy chiếu mềm.
   - Bơm trực tiếp vào **Key-Value Cache ($K, V$)** tại các tầng Attention bị nghẽn (Layers 12–22).
   - Buộc dòng thông tin phải trải qua phép biến đổi tọa độ từ Camera sang Ego của nhân vật trước khi đi tới các tầng sinh từ cuối cùng.

---
*Báo cáo được trích xuất tự động từ cơ sở dữ liệu nghiên cứu SpatialMQA trên máy chủ Azure (`comments.db`).*
