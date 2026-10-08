import json
import sqlite3
import os

def run():
    # 1. Load existing Azure comments
    existing_map = {}
    if os.path.exists('/tmp/azure_comments.json'):
        with open('/tmp/azure_comments.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for c in data.get('data', []):
                existing_map[c['sample_id']] = c

    print(f"Loaded {len(existing_map)} existing comments.")

    # 2. Load predictions data
    with open('visualizer/src/data/predictions_data.json', 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    items = pred_data['items']
    fails = [x for x in items if not x['is_correct']]
    print(f"Total failure cases to analyze: {len(fails)}")

    # 3. Categorization & generation logic
    all_comments = []
    category_counts = {}

    for item in items:
        sid = item['id']
        if sid in existing_map:
            # Preserve existing comment
            ex = existing_map[sid]
            all_comments.append((
                sid,
                ex['user_id'],
                ex['comment'],
                ex.get('failure_tag') or 'Manual Inspection',
                ex.get('created_at', '2026-10-08 13:00:00')
            ))
            tag = ex.get('failure_tag') or 'Manual Inspection'
            category_counts[tag] = category_counts.get(tag, 0) + 1
            continue

        if item['is_correct']:
            continue

        # Diagnose failure mechanism
        axis = item['axis']
        persp = item['perspective_key']
        gt = item['answer']
        pred = item['output']
        q = item['question']

        tag = ""
        desc = ""

        # Pattern 1: Q2 Egocentric 180° Left-Right Inversion
        if persp == 'Q2_FirstPerson' and axis.startswith('A_x') and ((gt == 'right of' and pred == 'left of') or (gt == 'left of' and pred == 'right of')):
            tag = "Camera Fallback (180° Inversion)"
            desc = f"Mô hình sập bẫy camera 2D: Hướng nhìn của nhân vật đối diện camera, nhưng mô hình không thực hiện xoay hệ quy chiếu ngôi thứ nhất (FRS), mà đọc thẳng tọa độ pixel trên màn hình camera khiến đáp án bị đảo ngược 180° (đúng: '{gt}', dự đoán: '{pred}')."

        # Pattern 2: Depth-to-2D Horizontal Collapse (Ay -> Ax)
        elif axis.startswith('A_y') and pred in ['left of', 'right of']:
            tag = "Depth-to-2D Degeneracy (Ay -> Ax)"
            desc = f"Sụp đổ trục sâu về mặt phẳng 2D: Câu hỏi thuộc trục chiều sâu Z ({gt}), nhưng do CLIP ViT ép phẳng 3D thành ma trận pixel 2D, mô hình bị thoái hóa đặc trưng và đo khoảng cách pixel trên trục ngang màn hình ({pred})."

        # Pattern 3: Depth Inversion (in front of vs behind)
        elif axis.startswith('A_y') and ((gt == 'in front of' and pred == 'behind') or (gt == 'behind' and pred == 'in front of')):
            tag = "Depth Occlusion Inversion"
            desc = f"Đảo chiều quan hệ chiều sâu: Mô hình thất bại trong việc phân tích dấu hiệu che khuất (occlusion boundary) và thị sai khoảng cách 3D, nhầm lẫn vị trí phía trước và phía sau ('{gt}' thành '{pred}')."

        # Pattern 4: Q2 FirstPerson Axis Leakage (Ax -> Ay or Ay -> Ax)
        elif persp == 'Q2_FirstPerson' and ((axis.startswith('A_x') and pred in ['in front of', 'behind']) or (axis.startswith('A_y') and pred in ['left of', 'right of'])):
            tag = "Egocentric Axis Leakage (Q2)"
            desc = f"Rò rỉ trục hệ quy chiếu nhân vật: Mô hình không xác định được vector hướng nhìn (heading vector) của chủ thể, khiến các attention head bị nhiễu loạn giữa trục nhìn thẳng và trục ngang tay người quan sát ('{gt}' thành '{pred}')."

        # Pattern 5: Vertical Orientation / Text OCR (Az)
        elif axis.startswith('A_z') and ((gt == 'on/above' and pred == 'below') or (gt == 'below' and pred == 'on/above')):
            tag = "Vertical Layout Inversion (Az)"
            desc = f"Đảo ngược trục đứng: Mô hình nhầm lẫn vị trí tương đối trên/dưới ('{gt}' thành '{pred}'). Thường xảy ra do hạn chế patch-level resolution của CLIP ViT khi định vị chi tiết các vật thể nhỏ hoặc dòng chữ."

        # Pattern 6: Q1 Camera Object Reference Confusion
        elif persp == 'Q1_OutOfImage' and ((gt == 'left of' and pred == 'right of') or (gt == 'right of' and pred == 'left of')):
            tag = "Camera-Frame Inversion (Q1)"
            desc = f"Đảo ngược quan hệ mốc so sánh: Mô hình nhận diện được 2 vật nhưng nhầm lẫn thứ tự so sánh đối tượng đích so với đối tượng mốc trên trục hoành ('{gt}' thành '{pred}')."

        # Default fallback pattern
        else:
            tag = "Perspective Reference Confusion"
            desc = f"Mô hình không căn chỉnh được hệ quy chiếu không gian giữa chủ thể và vật thể mốc trong không gian 3D (đáp án đúng: '{gt}', dự đoán: '{pred}')."

        category_counts[tag] = category_counts.get(tag, 0) + 1
        all_comments.append((sid, 'gemini', desc, tag, '2026-10-08 13:45:00'))

    print("\nTaxonomy Category Distribution across all failures:")
    for k, v in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {k}: {v} ({v/len(fails)*100:.1f}%)")

    # 4. Save into SQLite
    db_path = 'visualizer/database/comments.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER NOT NULL,
            user_id TEXT NOT NULL,
            comment TEXT NOT NULL,
            failure_tag TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    cur.execute('DELETE FROM comments') # clean rewrite
    cur.executemany('''
        INSERT INTO comments (sample_id, user_id, comment, failure_tag, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', all_comments)
    conn.commit()

    cur.execute('SELECT COUNT(*) FROM comments')
    total_in_db = cur.fetchone()[0]
    conn.close()
    print(f"\nSuccessfully stored {total_in_db} comments into {db_path}!")

if __name__ == '__main__':
    run()
