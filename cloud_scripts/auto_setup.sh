#!/bin/bash
# auto_setup.sh - Tự động setup MCCVA từ GitHub

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"; }

log "🚀 Starting automatic MCCVA setup..."

# Kiểm tra xem có MCCVA cũ không
if [ -d "/opt/mccva" ]; then
    warn "⚠️ Existing MCCVA installation found"
    read -p "Remove existing installation? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "🗑️ Removing old installation..."
        sudo systemctl stop openresty 2>/dev/null || true
        sudo systemctl stop mccva-mock-servers 2>/dev/null || true
        docker stop mccva-ml 2>/dev/null || true
        docker rm mccva-ml 2>/dev/null || true
        sudo rm -rf /opt/mccva
    else
        log "❌ Setup cancelled"
        exit 1
    fi
fi

# Di chuyển vào /opt
cd /opt

# Clone repository
log "📥 Cloning MCCVA repository..."
sudo git clone https://github.com/nhiwentwest/mccva.git

# Fix permissions ngay lập tức
log "🔧 Fixing permissions..."
sudo chown -R ubuntu:ubuntu mccva
cd mccva

# Set execute permissions
chmod +x *.sh 2>/dev/null || warn "No .sh files in root"
chmod +x cloud_scripts/*.sh 2>/dev/null || warn "No cloud_scripts"

# Tạo config file
if [ ! -f "mccva_env.conf" ]; then
    log "📝 Creating config file..."
    cat > mccva_env.conf << 'EOF'
NGINX_USER=ubuntu
LOGS_DIR=/usr/local/openresty/nginx/logs
MOCK_SERVER_COUNT=8
EOF
fi

# Kiểm tra repository
log "🔍 Checking repository..."
git status
git log --oneline -3

# Kiểm tra dependencies
log "🔍 Checking dependencies..."

# Check Docker
if ! command -v docker &> /dev/null; then
    error "❌ Docker not installed"
    log "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    log "✅ Docker installed"
fi

# Check Python3
if ! command -v python3 &> /dev/null; then
    error "❌ Python3 not installed"
    log "Installing Python3..."
    sudo apt update
    sudo apt install -y python3 python3-pip
    log "✅ Python3 installed"
fi

# Check OpenResty
if ! command -v openresty &> /dev/null; then
    warn "⚠️ OpenResty not installed"
    log "OpenResty will be installed during setup"
fi

log "✅ Dependencies check passed"

# Build Docker image nếu có Dockerfile
if [ -f "Dockerfile" ]; then
    log "🐳 Building Docker image..."
    docker build -t mccva-ml-service .
    log "✅ Docker image built"
else
    warn "⚠️ No Dockerfile found"
fi

# Start services
log "🚀 Starting MCCVA system..."
./run.sh

# Kiểm tra services
log "🔍 Checking services..."
sleep 5

if systemctl is-active --quiet openresty; then
    log "✅ OpenResty is running"
else
    error "❌ OpenResty failed to start"
fi

if systemctl is-active --quiet mccva-mock-servers; then
    log "✅ Mock servers are running"
else
    error "❌ Mock servers failed to start"
fi

if docker ps --filter name=mccva-ml --format "{{.Status}}" | grep -q "Up"; then
    log "✅ ML container is running"
else
    error "❌ ML container failed to start"
fi

# Health checks
log "🏥 Running health checks..."
sleep 3

if curl -s http://localhost/health >/dev/null; then
    log "✅ OpenResty health check passed"
else
    warn "⚠️ OpenResty health check failed"
fi

if curl -s http://localhost:5000/health >/dev/null; then
    log "✅ ML Service health check passed"
else
    warn "⚠️ ML Service health check failed"
fi

log "✅ Automatic setup completed successfully!"
log "🎯 MCCVA system is ready for use"

echo ""
log "📋 Quick commands:"
log "   cd /opt/mccva"
log "   ./run.sh                    # Start system"
log "   python3 test_ai_routing_simple.py  # Test AI routing"
log "   ./cloud_scripts/update_mccva.sh    # Update from GitHub" 