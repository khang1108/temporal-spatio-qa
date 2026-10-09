import json
import os
import re

def clean_and_add_section13():
    nb_path = "notebooks/002_eval_spatialmqa_kaggle.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Clean all emojis and icons from all existing cells
    # Comprehensive emoji/symbol regex
    emoji_pattern = re.compile(
        r'[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]|[\ufe00-\ufe0f]|[\u200d]'
    )

    cleaned_cells = []
    for c in nb["cells"][:24]:  # Keep cells up to section 12
        new_source = []
        for line in c["source"]:
            cleaned = emoji_pattern.sub("", line)
            cleaned = cleaned.replace("✓ ", "").replace("✓", "")
            cleaned = cleaned.replace("🚀 ", "").replace("🚀", "")
            cleaned = cleaned.replace("⚙️ ", "").replace("⚙️", "")
            cleaned = cleaned.replace("❓ ", "").replace("❓", "")
            cleaned = cleaned.replace("🔬 ", "").replace("🔬", "")
            cleaned = cleaned.replace("🏛️ ", "").replace("🏛️", "")
            cleaned = cleaned.replace("️ ", "")
            # Clean possible duplicate spaces in headers
            if cleaned.startswith("#"):
                cleaned = re.sub(r'#\s+', '# ', cleaned)
            new_source.append(cleaned)
        c["source"] = new_source
        cleaned_cells.append(c)

    # 2. Add clean Section 13: Commit & Push to GitHub
    sec13_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## **13. Commit & Push Kết Quả Đánh Giá Lên GitHub**\n",
            "\n",
            "Sau khi quá trình đánh giá hoàn tất, toàn bộ tệp kết quả dự đoán (`.jsonl`) và bảng chỉ số (`_metrics.json`) được lưu trữ vào thư mục chuẩn `experiments/predictions/` và đẩy lên nhánh `main` của repository GitHub.\n",
            "\n",
            "### Hướng dẫn thiết lập GitHub Token trên Kaggle:\n",
            "- **Cách 1 (Khuyến nghị):** Vào menu **Add-ons** (thanh trên cùng của Kaggle) $\\rightarrow$ **Secrets** $\\rightarrow$ Bấm **Add Secret** với Label là `GITHUB_TOKEN` và dán GitHub Personal Access Token của bạn.\n",
            "- **Cách 2:** Nếu chưa cấu hình Secrets, cell bên dưới sẽ tự động hiển thị ô nhập token bảo mật (sử dụng `getpass` ẩn ký tự).\n"
        ]
    }

    sec13_code_lines = [
        "# Copy kết quả vào experiments/predictions và push lên GitHub\n",
        "import os\n",
        "import shutil\n",
        "import getpass\n",
        "\n",
        "REPO_DIR = \"/kaggle/working/temporal-spatio-qa\"\n",
        "if not os.path.exists(REPO_DIR):\n",
        "    REPO_DIR = os.getcwd()\n",
        "\n",
        "# 1. Tạo thư mục lưu trữ kết quả chuẩn theo AGENTS.md\n",
        "target_dir = os.path.join(REPO_DIR, \"experiments/predictions\")\n",
        "os.makedirs(target_dir, exist_ok=True)\n",
        "\n",
        "# 2. Copy kết quả từ /kaggle/working/ vào repository\n",
        "src_jsonl = \"/kaggle/working/llava_1.5_lora_test_predictions.jsonl\"\n",
        "src_metrics = \"/kaggle/working/llava_1.5_lora_test_predictions_metrics.json\"\n",
        "\n",
        "if os.path.exists(src_jsonl):\n",
        "    shutil.copy(src_jsonl, os.path.join(target_dir, \"llava_1.5_lora_test_predictions.jsonl\"))\n",
        "    print(f\"Đã copy {src_jsonl} -> {target_dir}\")\n",
        "\n",
        "if os.path.exists(src_metrics):\n",
        "    shutil.copy(src_metrics, os.path.join(target_dir, \"llava_1.5_lora_test_predictions_metrics.json\"))\n",
        "    print(f\"Đã copy {src_metrics} -> {target_dir}\")\n",
        "\n",
        "# 3. Cấu hình định danh Git\n",
        "!git config --global user.name \"Nguyen Phuc Khang\"\n",
        "!git config --global user.email \"nguyenphuc.khang110806@gmail.com\"\n",
        "\n",
        "# 4. Lấy GitHub Token từ Kaggle Secrets hoặc nhập trực tiếp\n",
        "github_token = None\n",
        "try:\n",
        "    from kaggle_secrets import UserSecretsClient\n",
        "    user_secrets = UserSecretsClient()\n",
        "    github_token = user_secrets.get_secret(\"GITHUB_TOKEN\")\n",
        "    print(\"Đã tải GitHub Token thành công từ Kaggle Secrets!\")\n",
        "except Exception:\n",
        "    pass\n",
        "\n",
        "if not github_token:\n",
        "    github_token = getpass.getpass(\"Nhập GitHub Personal Access Token (ghp_...): \")\n",
        "\n",
        "# 5. Commit và Push lên GitHub\n",
        "if github_token:\n",
        "    %cd {REPO_DIR}\n",
        "    !git add experiments/predictions/\n",
        "    !git commit -m \"eval: add LLaVA-1.5 test predictions and metrics from Kaggle run\"\n",
        "    !git push https://{github_token}@github.com/khang1108/temporal-spatio-qa.git main\n",
        "    print(\"Đã push kết quả đánh giá lên GitHub repository thành công!\")\n",
        "else:\n",
        "    print(\"Chưa có token. Không thể push lên GitHub.\")\n"
    ]

    sec13_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": sec13_code_lines
    }

    cleaned_cells.extend([sec13_md, sec13_code])
    nb["cells"] = cleaned_cells

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Successfully cleaned notebook and added Section 13! Total cells: {len(nb['cells'])}")

if __name__ == "__main__":
    clean_and_add_section13()
