#!/bin/bash
# run.sh - Khởi động và kiểm thử hệ thống MCCVA
set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"; }
CURRENT_USER=$(whoami)

# Đọc config
if [ -f ./mccva_env.conf ]; then
  source ./mccva_env.conf
else
  log "Không tìm thấy file config mccva_env.conf, dùng giá trị mặc định."
  NGINX_USER=ubuntu
  LOGS_DIR=/usr/local/openresty/nginx/logs
  MOCK_SERVER_COUNT=8
fi

# Đảm bảo dòng pid trong nginx.conf trùng với systemd
NGINX_CONF="/usr/local/openresty/nginx/conf/nginx.conf"
PIDFILE="/usr/local/openresty/nginx/logs/nginx.pid"

if grep -q "^pid " "$NGINX_CONF"; then
    sudo sed -i "s|^pid .*;|pid $PIDFILE;|" "$NGINX_CONF"
else
    sudo sed -i "/worker_processes/a pid $PIDFILE;" "$NGINX_CONF"
fi

sudo mkdir -p /usr/local/openresty/nginx/logs
sudo chown -R $NGINX_USER:$NGINX_USER /usr/local/openresty/nginx/logs
sudo chmod -R 755 /usr/local/openresty/nginx/logs

log "Reload systemd..."; sudo systemctl daemon-reload

# Stop systemd ML service nếu đang chạy
if systemctl is-active --quiet mccva-ml; then
  log "Stop systemd ML Service để dùng Docker..."; sudo systemctl stop mccva-ml
fi

# Stop container cũ nếu có
if docker ps -a --format '{{.Names}}' | grep -q '^mccva-ml$'; then
  log "Xóa container ML Service cũ..."; docker rm -f mccva-ml
fi

# Run ML Service bằng Docker
log "Chạy ML Service bằng Docker..."
docker run -d --name mccva-ml -p 5000:5000 --restart unless-stopped mccva-ml-service

log "Start Mock Servers..."; sudo systemctl start mccva-mock-servers

# Function để start OpenResty với retry logic
start_openresty() {
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        log "Attempt $((retry_count + 1))/$max_retries: Starting OpenResty..."
        
        # Stop OpenResty nếu đang chạy
        if systemctl is-active --quiet openresty; then
            log "Stopping existing OpenResty service..."
            sudo systemctl stop openresty
            sleep 2
        fi
        
        # Kill any process using port 80
        if sudo lsof -i :80 >/dev/null 2>&1; then
            warn "Port 80 is in use. Killing processes..."
            sudo fuser -k 80/tcp || true
            sleep 3
        fi
        
        # Remove old PID file if exists
        if [ -f "$PIDFILE" ]; then
            sudo rm -f "$PIDFILE"
        fi
        
        # Start OpenResty
        if sudo systemctl start openresty; then
            log "✅ OpenResty started successfully!"
            return 0
        else
            retry_count=$((retry_count + 1))
            error "Failed to start OpenResty (attempt $retry_count/$max_retries)"
            
            # Show detailed error
            sudo systemctl status openresty --no-pager || true
            sudo journalctl -xeu openresty.service --no-pager | tail -10 || true
            
            if [ $retry_count -lt $max_retries ]; then
                warn "Retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done
    
    error "Failed to start OpenResty after $max_retries attempts"
    return 1
}

# Start OpenResty với retry logic
if start_openresty; then
    log "✅ OpenResty started successfully"
else
    error "❌ Failed to start OpenResty. Please check logs manually:"
    error "   sudo systemctl status openresty"
    error "   sudo journalctl -xeu openresty.service"
    exit 1
fi

sleep 5

# Kiểm tra trạng thái services
log "Kiểm tra trạng thái service..."
sudo systemctl status mccva-mock-servers --no-pager
sudo systemctl status openresty --no-pager

# Test health endpoints
log "Test health endpoint..."
if curl -s http://localhost/health >/dev/null; then
    log "✅ OpenResty health check passed"
else
    error "❌ OpenResty health check failed"
fi

if curl -s http://localhost:5000/health >/dev/null; then
    log "✅ ML Service health check passed"
else
    error "❌ ML Service health check failed"
fi

# Test mock servers
for port in $(seq 8081 $((8080+MOCK_SERVER_COUNT))); do
    if curl -s http://localhost:$port/health >/dev/null; then
        log "✅ Mock server $port health check passed"
    else
        error "❌ Mock server $port health check failed"
    fi
done

# Test AI routing
log "Test AI routing..."
if curl -s -X POST http://localhost/mccva/route -H "Content-Type: application/json" -d '{"cpu_cores": 4, "memory": 8, "storage": 100, "network_bandwidth": 1000, "priority": 3}' >/dev/null; then
    log "✅ AI routing test passed"
else
    error "❌ AI routing test failed"
fi

log "✅ Hệ thống đã sẵn sàng!" 