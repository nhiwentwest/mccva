#!/bin/bash
# fix_permissions.sh - Fix permissions cho MCCVA system

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"; }

log "🔧 Fixing MCCVA permissions..."

# Kiểm tra thư mục tồn tại
if [ ! -d "/opt/mccva" ]; then
    error "MCCVA directory not found at /opt/mccva"
    log "Please clone the repository first:"
    log "   cd /opt && sudo git clone https://github.com/nhiwentwest/mccva.git"
    exit 1
fi

# Di chuyển vào thư mục
cd /opt/mccva

# Fix ownership
log "👤 Setting ownership to ubuntu user..."
sudo chown -R ubuntu:ubuntu /opt/mccva

# Fix permissions cho scripts
log "🔐 Setting execute permissions..."
chmod +x *.sh 2>/dev/null || warn "No .sh files in root directory"
chmod +x cloud_scripts/*.sh 2>/dev/null || warn "No cloud_scripts directory"

# Fix permissions cho Python files
log "🐍 Setting Python file permissions..."
chmod +x *.py 2>/dev/null || warn "No .py files in root directory"

# Tạo config file nếu chưa có
if [ ! -f "mccva_env.conf" ]; then
    log "📝 Creating default config file..."
    cat > mccva_env.conf << 'EOF'
NGINX_USER=ubuntu
LOGS_DIR=/usr/local/openresty/nginx/logs
MOCK_SERVER_COUNT=8
EOF
    log "✅ Created mccva_env.conf"
fi

# Kiểm tra Git status
log "🔍 Checking Git status..."
if [ -d ".git" ]; then
    git status
    log "✅ Git repository is valid"
else
    warn "⚠️ Not a Git repository"
fi

# Kiểm tra files quan trọng
log "📋 Checking important files..."
important_files=("run.sh" "ml_service.py" "nginx.conf" "lua/mccva_routing.lua")

for file in "${important_files[@]}"; do
    if [ -f "$file" ]; then
        log "✅ Found: $file"
    else
        warn "⚠️ Missing: $file"
    fi
done

# Kiểm tra permissions
log "🔍 Checking permissions..."
ls -la *.sh 2>/dev/null || warn "No .sh files found"
ls -la cloud_scripts/*.sh 2>/dev/null || warn "No cloud_scripts found"

log "✅ Permissions fixed successfully!"
log "🎯 You can now run: ./run.sh" 