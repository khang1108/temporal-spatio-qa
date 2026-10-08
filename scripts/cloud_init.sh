#!/bin/bash
set -e
exec > >(tee -i /var/log/cloud-init-deploy.log) 2>&1

export DEBIAN_FRONTEND=noninteractive

echo "=== [1/6] Updating system packages ==="
apt-get update -y
apt-get install -y curl wget git

echo "=== [2/6] Installing Node.js 22 LTS & PM2 ==="
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
npm install -g pm2

echo "=== [3/6] Installing cloudflared ==="
curl -L --output /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i /tmp/cloudflared.deb || apt-get install -f -y
rm -f /tmp/cloudflared.deb

echo "=== [4/6] Cloning SpatialMQA Repository ==="
mkdir -p /opt/app
git clone https://github.com/khang1108/temporal-spatio-qa.git /opt/app/temporal-spatio-qa
chown -R azureuser:azureuser /opt/app

echo "=== [5/6] Building Visualizer Frontend & Setting up Node Server ==="
cd /opt/app/temporal-spatio-qa/visualizer
sudo -u azureuser npm install
sudo -u azureuser npm run build

echo "=== [6/6] Starting Server with PM2 on Port 3000 ==="
sudo -u azureuser pm2 start server.js --name "spatial-visualizer"
sudo -u azureuser pm2 save

# Setup PM2 startup script
env PATH=$PATH:/usr/bin /usr/lib/node_modules/pm2/bin/pm2 startup systemd -u azureuser --hp /home/azureuser

echo "=== Deployment Completed Successfully at $(date) ===" > /var/log/deploy_finished.log
