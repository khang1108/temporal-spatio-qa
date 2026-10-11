#!/usr/bin/env bash
# ==============================================================================
# Script biên dịch LaTeX tự động cho paper/main.tex
# Hỗ trợ cả latexmk và pdflatex, tích hợp cơ chế tự động dọn dẹp (auto-clean)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MAIN_TEX="main.tex"
MAIN_BASE="main"
PDF_OUT="${MAIN_BASE}.pdf"

# Danh sách phần mở rộng tệp trung gian cần dọn dẹp
TEMP_EXTS=("aux" "log" "bbl" "blg" "out" "fls" "fdb_latexmk" "synctex.gz" "toc" "lof" "lot")

clean_temp() {
    echo "[Clean] Đang dọn dẹp các tệp trung gian tạm thời..."
    for ext in "${TEMP_EXTS[@]}"; do
        rm -f "${MAIN_BASE}.${ext}"
    done
    rm -f "${MAIN_BASE}.run.xml" "${MAIN_BASE}-blx.bib"
    echo "[Clean] Hoàn tất dọn dẹp tệp tạm thời."
}

clean_all() {
    clean_temp
    echo "[Clean] Xóa tệp PDF xuất ra: ${PDF_OUT}..."
    rm -f "${PDF_OUT}"
    echo "[Clean] Đã xóa toàn bộ sản phẩm biên dịch."
}

compile_latexmk() {
    echo "[Compile] Đang biên dịch bằng latexmk..."
    latexmk -pdf -interaction=nonstopmode -halt-on-error "${MAIN_TEX}"
}

compile_pdflatex_fallback() {
    echo "[Compile] latexmk không khả dụng. Biên dịch bằng chuỗi pdflatex + bibtex..."
    pdflatex -interaction=nonstopmode -halt-on-error "${MAIN_TEX}"
    bibtex "${MAIN_BASE}" || true
    pdflatex -interaction=nonstopmode -halt-on-error "${MAIN_TEX}"
    pdflatex -interaction=nonstopmode -halt-on-error "${MAIN_TEX}"
}

build_pdf() {
    if command -v latexmk >/dev/null 2>&1; then
        compile_latexmk
    elif command -v pdflatex >/dev/null 2>&1; then
        compile_pdflatex_fallback
    else
        echo "[Lỗi] Không tìm thấy latexmk hoặc pdflatex trên hệ thống!" >&2
        exit 1
    fi

    if [ -f "${PDF_OUT}" ]; then
        echo "=========================================================="
        echo "[Thành công] Đã tạo PDF: ${SCRIPT_DIR}/${PDF_OUT}"
        echo "Kích thước: $(du -h "${PDF_OUT}" | cut -f1)"
        echo "=========================================================="
    else
        echo "[Lỗi] Biên dịch thất bại, không tìm thấy ${PDF_OUT}!" >&2
        exit 1
    fi
}

# Xử lý tham số dòng lệnh
MODE="${1:-build}"

case "${MODE}" in
    clean)
        clean_temp
        ;;
    clean-all)
        clean_all
        ;;
    --auto-clean|autoclean)
        build_pdf
        clean_temp
        echo "[Info] Đã giữ lại tệp PDF sạch: ${PDF_OUT}"
        ;;
    build|*)
        build_pdf
        # Nếu người dùng truyền cờ phụ --clean-temp
        if [[ "${2:-}" == "--clean-temp" || "${2:-}" == "-c" ]]; then
            clean_temp
        fi
        ;;
esac
