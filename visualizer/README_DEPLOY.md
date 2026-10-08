# HƯỚNG DẪN TRIỂN KHAI AZURE & KẾT NỐI CLOUDFLARE TUNNEL (npl.iamphuckhang.dev)

Ứng dụng: **SpatialMQA Failure Visualizer & Multi-User Annotation Portal**  
Kiến trúc: **React + Node.js (Express) + Native SQLite (`node:sqlite`) chạy trên Port 3000**

---

## 🟢 THÔNG TIN MÁY CHỦ AZURE ĐÃ KHỞI TẠO THÀNH CÔNG

- **Tên VM:** `vm-spatial-mqa`
- **Resource Group:** `rg-dres-server`
- **Vị trí:** `eastasia` (Hong Kong / East Asia)
- **Kích thước VM:** `Standard_B2s_v2` (2 vCPUs, 4GB RAM)
- **Địa chỉ IP Public:** `20.205.104.181`
- **Cổng ứng dụng:** `3000` (`http://20.205.104.181:3000`)
- **Tài khoản SSH:** `azureuser` (Đã tích hợp sẵn SSH key của bạn)
  ```bash
  ssh azureuser@20.205.104.181
  ```
- **Trạng thái dịch vụ:** PM2 daemon `spatial-visualizer` đang chạy online, tự khởi động cùng hệ điều hành.
- **cloudflared:** Đã cài sẵn phiên bản `2026.10.0` trên máy ảo.

---

## 🚀 1. Chuẩn Bị Môi Trường Trên Azure VM (Ubuntu Linux)

Mở SSH vào máy chủ Azure và thực hiện các lệnh sau:

### Bước 1.1: Cài đặt Node.js v22 (LTS) & PM2
Hệ thống sử dụng tính năng `node:sqlite` tích hợp sẵn trong Node 22+, hoàn toàn không cần cài đặt thêm C++ build tools (như `node-gyp`, `make`, `gcc`).

```bash
# Cập nhật hệ thống
sudo apt update && sudo apt upgrade -y

# Cài đặt Node.js 22.x
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs git

# Kiểm tra phiên bản (yêu cầu Node >= 22.0.0)
node -v
npm -v

# Cài đặt PM2 để quản lý tiến trình chạy nền
sudo npm install -g pm2
```

---

## 📂 2. Cài Đặt Ứng Dụng Trên Azure VM

### Bước 2.1: Kéo mã nguồn về VM
```bash
# Di chuyển đến thư mục làm việc (ví dụ /var/www hoặc home user)
cd ~
git clone https://github.com/phuckhangne/Spatial-Temporal-KG.git
cd Spatial-Temporal-KG/visualizer
```
*(Nếu đã có sẵn repo trên máy chủ, chỉ cần chạy `git pull origin main`)*

### Bước 2.2: Cài đặt thư viện & Build Frontend
```bash
# Cài đặt dependencies
npm install

# Build bản production tĩnh ra thư mục dist/
npm run build
```

### Bước 2.3: Khởi chạy Backend Server với PM2
Server sẽ lắng nghe trên cổng `3000` (`http://localhost:3000`), đồng thời tự động khởi tạo database SQLite tại `database/comments.db`.

```bash
# Khởi động dịch vụ
pm2 start server.js --name "spatial-visualizer"

# Lưu cấu hình PM2 để tự động chạy lại khi VM reboot
pm2 save
sudo env PATH=$PATH:/usr/bin pm2 startup systemd -u $USER --hp $HOME
```

Kiểm tra trạng thái server:
```bash
pm2 status
curl -s http://localhost:3000/api/comments
```
*(Nếu trả về `{"success":true,"count":0,"comments":[]}` là server đã hoạt động hoàn hảo)*

---

## 🌐 3. Kết Nối Cloudflare Tunnel Đến Domain `npl.iamphuckhang.dev`

Vì sử dụng Cloudflare Tunnel (`cloudflared`), **bạn KHÔNG CẦN mở cổng 3000 trên Azure Network Security Group (NSG)**. Tunnel sẽ tạo kết nối mã hóa an toàn outbound từ VM về Cloudflare Edge.

Có 2 cách thực hiện:

### CÁCH 1: Cấu hình qua Cloudflare Zero Trust Dashboard (Khuyến nghị - Nhanh nhất)
1. Đăng nhập [Cloudflare One / Zero Trust Dashboard](https://one.dash.cloudflare.com/).
2. Vào **Networks** $\rightarrow$ **Tunnels** $\rightarrow$ Chọn **Create a Tunnel** (hoặc chọn Tunnel đã có).
3. Đặt tên (ví dụ: `azure-spatialmqa`).
4. Tại tab **Install connector**, chọn **Debian / Ubuntu 64-bit**, copy dòng lệnh cài đặt có chứa Token và chạy trên Azure VM:
   ```bash
   curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && sudo dpkg -i cloudflared.deb && sudo cloudflared service install <YOUR_TOKEN>
   ```
5. Chuyển sang tab **Public Hostnames**:
   - **Subdomain:** `npl`
   - **Domain:** `iamphuckhang.dev`
   - **Type:** `HTTP`
   - **URL:** `localhost:3000`
6. Bấm **Save tunnel**. Trong vòng vài giây, truy cập `https://npl.iamphuckhang.dev` sẽ hiển thị ngay giao diện!

---

### CÁCH 2: Cấu hình thủ công bằng CLI trên Azure VM
Nếu bạn thích dùng dòng lệnh:

```bash
# 1. Tải và cài đặt cloudflared
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

# 2. Đăng nhập Cloudflare
cloudflared tunnel login
# (Trình duyệt sẽ hiển thị link xác thực với domain iamphuckhang.dev)

# 3. Tạo tunnel
cloudflared tunnel create npl-tunnel
# Lưu lại mã Tunnel-UUID được sinh ra

# 4. Trỏ DNS domain về tunnel
cloudflared tunnel route dns npl-tunnel npl.iamphuckhang.dev

# 5. Tạo file cấu hình ~/.cloudflared/config.yml
cat << 'EOF' > ~/.cloudflared/config.yml
tunnel: <TUNNEL_UUID>
credentials-file: /home/phuckhang/.cloudflared/<TUNNEL_UUID>.json

ingress:
  - hostname: npl.iamphuckhang.dev
    service: http://localhost:3000
  - service: http_status:404
EOF

# 6. Cài đặt và kích hoạt systemd service
sudo cloudflared --config ~/.cloudflared/config.yml service install
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
sudo systemctl status cloudflared
```

---

## 💾 4. Quản Lý Dữ Liệu SQLite & Backup Nhận Xét

- **Vị trí file CSDL:** `visualizer/database/comments.db`
- **Tải toàn bộ nhận xét về máy cá nhân:**
  ```bash
  curl -o comments_backup.json https://npl.iamphuckhang.dev/api/export
  ```
- **Copy file `.db` về máy:**
  ```bash
  scp user@azure-vm:~/Spatial-Temporal-KG/visualizer/database/comments.db ./comments.db
  ```
- **Cấu trúc bảng nhận xét:**
  - `sample_id`: ID câu hỏi (từ 0 đến 1075)
  - `user_id`: Tên/ID người kiểm tra (ví dụ: `khang`, `researcher_a`)
  - `comment`: Nội dung phân tích lỗi
  - `failure_tag`: Phân loại lỗi (`Camera Fallback`, `Depth Blindness`, `Perspective Drift`, `Visual Occlusion`)
  - `created_at`: Thời gian nhận xét (UTC)
