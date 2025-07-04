#!/bin/bash
# amazon_setup.sh - Cài đặt môi trường MCCVA trên Amazon Cloud Ubuntu
set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; }
if [[ $EUID -eq 0 ]]; then error "Không chạy script này bằng root!"; exit 1; fi
CURRENT_USER=$(whoami)
log "Cài đặt MCCVA cho user: $CURRENT_USER"
log "Cập nhật hệ thống..."; sudo apt update && sudo apt upgrade -y
log "Cài đặt package cần thiết..."; sudo apt install -y python3 python3-pip python3-venv nginx curl wget git unzip software-properties-common
log "Cài đặt OpenResty..."; wget -qO - https://openresty.org/package/pubkey.gpg | sudo apt-key add -; sudo add-apt-repository -y "deb https://openresty.org/package/ubuntu $(lsb_release -sc) openresty"; sudo apt update; sudo apt install -y openresty
log "Tạo thư mục ứng dụng..."; sudo mkdir -p /opt/mccva; sudo chown $CURRENT_USER:$CURRENT_USER /opt/mccva; cd /opt/mccva
if [ -d "/home/$CURRENT_USER/mccva" ]; then log "Copy project từ home..."; cp -r /home/$CURRENT_USER/mccva/* /opt/mccva/; else log "Clone project từ GitHub..."; git clone https://github.com/nhiwentwest/mccva.git temp_mccva; cp -r temp_mccva/* /opt/mccva/; rm -rf temp_mccva; fi
log "Tạo virtualenv Python..."; if [ -d "venv" ]; then rm -rf venv; fi; python3 -m venv venv; source venv/bin/activate; pip install --upgrade pip
log "Cài đặt Python packages..."; pip install numpy==1.26.4 scipy==1.12.0 scikit-learn==1.4.0 pandas==2.2.0 joblib==1.3.2 Flask==3.0.0 Werkzeug==3.0.1 gunicorn==21.2.0 requests==2.31.0 matplotlib==3.8.2 seaborn==0.13.0
python -c "import numpy, scipy, sklearn, pandas, joblib, flask; print('✅ All packages installed successfully')"
log "Tạo thư mục log và set quyền..."; sudo mkdir -p /var/log/mccva /var/log/nginx /var/www/html /var/run/nginx /var/run/openresty; sudo chown -R $CURRENT_USER:$CURRENT_USER /var/log/mccva /var/log/nginx /var/run/nginx /var/run/openresty; sudo chmod -R 755 /var/log/mccva /var/log/nginx /var/run/nginx /var/run/openresty
log "Tạo systemd service..."
sudo tee /etc/systemd/system/mccva-ml.service > /dev/null <<EOF
[Unit]
Description=MCCVA ML Service
After=network.target
[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=/opt/mccva
Environment=PATH=/opt/mccva/venv/bin
ExecStart=/opt/mccva/venv/bin/python ml_service.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/mccva/mccva-ml.log
StandardError=append:/var/log/mccva/mccva-ml.log
[Install]
WantedBy=multi-user.target
EOF
sudo tee /etc/systemd/system/mccva-mock-servers.service > /dev/null <<EOF
[Unit]
Description=MCCVA Mock Servers
After=network.target
[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=/opt/mccva
Environment=PATH=/opt/mccva/venv/bin
ExecStart=/opt/mccva/venv/bin/python mock_servers.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/mccva/mock-servers.log
StandardError=append:/var/log/mccva/mock-servers.log
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
log "Cài đặt nginx.conf và Lua..."; if [ -f "/etc/nginx/nginx.conf" ]; then sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup; fi; sudo cp nginx.conf /etc/nginx/nginx.conf; sudo mkdir -p /usr/local/openresty/nginx/conf/lua; sudo cp lua/*.lua /usr/local/openresty/nginx/conf/lua/

# Đảm bảo dòng pid trong nginx.conf trùng với systemd
NGINX_CONF="/usr/local/openresty/nginx/conf/nginx.conf"
PIDFILE="/usr/local/openresty/nginx/logs/nginx.pid"

# Sửa dòng pid trong nginx.conf nếu chưa đúng
if grep -q "^pid " "$NGINX_CONF"; then
    sudo sed -i "s|^pid .*;|pid $PIDFILE;|" "$NGINX_CONF"
else
    # Nếu chưa có dòng pid, thêm vào sau worker_processes
    sudo sed -i "/worker_processes/a pid $PIDFILE;" "$NGINX_CONF"
fi

# Tạo thư mục logs và set quyền đúng
sudo mkdir -p /usr/local/openresty/nginx/logs
sudo chown -R ubuntu:ubuntu /usr/local/openresty/nginx/logs
sudo chmod -R 755 /usr/local/openresty/nginx/logs

# Cài đặt Docker nếu chưa có
if ! command -v docker &> /dev/null; then
  log "Cài đặt Docker..."; 
  curl -fsSL https://get.docker.com -o get-docker.sh; 
  sh get-docker.sh; 
  sudo usermod -aG docker $CURRENT_USER; 
  rm get-docker.sh
  echo "\n\033[1;33m👉 Đã cài Docker và thêm user vào group docker.\033[0m"
  echo "\033[1;33m👉 Bạn cần logout/login lại hoặc chạy 'newgrp docker' rồi chạy lại script này để tiếp tục.\033[0m"
  exit 0
else
  log "Docker đã được cài đặt."
fi

# Tạo Dockerfile cho ml_service.py nếu chưa có
if [ ! -f Dockerfile ]; then
  log "Tạo Dockerfile cho ML Service..."
  cat <<EOF > Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "ml_service.py"]
EOF
else
  log "Đã có Dockerfile."
fi

# Build Docker image cho ML Service
log "Build Docker image cho ML Service..."
docker build -t mccva-ml-service .

log "✅ Cài đặt hoàn tất! Chạy run.sh để khởi động hệ thống." 