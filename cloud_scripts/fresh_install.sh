#!/bin/bash
# fresh_install.sh - Fresh install MCCVA từ GitHub

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"; }

log "🚀 Starting fresh MCCVA installation from GitHub..."

# Kiểm tra xem có MCCVA cũ không
if [ -d "/opt/mccva" ]; then
    warn "⚠️ Existing MCCVA installation found at /opt/mccva"
    read -p "Do you want to backup before removing? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "📦 Creating backup..."
        ./cloud_scripts/backup_mccva.sh
    fi
    
    log "🗑️ Removing old installation..."
    sudo systemctl stop openresty 2>/dev/null || true
    sudo systemctl stop mccva-mock-servers 2>/dev/null || true
    docker stop mccva-ml 2>/dev/null || true
    docker rm mccva-ml 2>/dev/null || true
    sudo rm -rf /opt/mccva
fi

# Di chuyển vào /opt
cd /opt

# Clone repository
log "📥 Cloning MCCVA repository..."
sudo git clone https://github.com/nhiwentwest/mccva.git

# Set permissions
log "🔧 Setting permissions..."
sudo chown -R ubuntu:ubuntu mccva
cd mccva
chmod +x *.sh
chmod +x cloud_scripts/*.sh

# Kiểm tra repository
log "🔍 Checking repository..."
git status
git log --oneline -3

# Tạo config file mặc định nếu chưa có
if [ ! -f "mccva_env.conf" ]; then
    log "📝 Creating default config..."
    cat > mccva_env.conf << 'EOF'
NGINX_USER=ubuntu
LOGS_DIR=/usr/local/openresty/nginx/logs
MOCK_SERVER_COUNT=8
EOF
fi

# Kiểm tra dependencies
log "🔍 Checking dependencies..."

# Check Docker
if ! command -v docker &> /dev/null; then
    error "❌ Docker not installed"
    log "Please install Docker first:"
    log "   curl -fsSL https://get.docker.com -o get-docker.sh"
    log "   sudo sh get-docker.sh"
    exit 1
fi

# Check Python3
if ! command -v python3 &> /dev/null; then
    error "❌ Python3 not installed"
    log "Please install Python3 first:"
    log "   sudo apt update && sudo apt install python3 python3-pip"
    exit 1
fi

# Check OpenResty
if ! command -v openresty &> /dev/null; then
    warn "⚠️ OpenResty not installed"
    log "OpenResty will be installed during setup"
fi

log "✅ Dependencies check passed"

# Build Docker image
log "🐳 Building Docker image..."
if [ -f "Dockerfile" ]; then
    docker build -t mccva-ml-service .
    log "✅ Docker image built successfully"
else
    warn "⚠️ No Dockerfile found, skipping Docker build"
fi

# Start services
log "🚀 Starting MCCVA system..."
./run.sh

# Kiểm tra services
log "🔍 Checking services status..."
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

# Test AI routing
log "🧪 Testing AI routing..."
if python3 test_ai_routing_simple.py; then
    log "✅ AI routing test passed"
else
    warn "⚠️ AI routing test had issues"
fi

log "✅ Fresh installation completed successfully!"
log "🎯 MCCVA system is ready for use"

echo ""
log "📋 Quick commands:"
log "   cd /opt/mccva"
log "   ./run.sh                    # Start system"
log "   python3 test_ai_routing_simple.py  # Test AI routing"
log "   ./cloud_scripts/update_mccva.sh    # Update from GitHub"
log "   ./cloud_scripts/backup_mccva.sh    # Create backup" 