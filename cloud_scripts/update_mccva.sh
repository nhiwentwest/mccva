#!/bin/bash
# update_mccva.sh - Update MCCVA từ GitHub

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"; }

log "🔄 Starting MCCVA update from GitHub..."

# Kiểm tra repository tồn tại
if [ ! -d "/opt/mccva/.git" ]; then
    error "MCCVA repository not found at /opt/mccva"
    log "Please clone the repository first:"
    log "   cd /opt && sudo git clone https://github.com/nhiwentwest/mccva.git"
    exit 1
fi

# Backup trước khi update
log "📦 Creating backup..."
./cloud_scripts/backup_mccva.sh

# Di chuyển vào thư mục project
cd /opt/mccva

# Kiểm tra remote
log "🔍 Checking remote repository..."
if ! git remote -v | grep -q "origin"; then
    error "No remote origin found"
    exit 1
fi

# Fetch updates
log "📥 Fetching updates from GitHub..."
git fetch origin

# Kiểm tra có updates không
LOCAL_COMMIT=$(git rev-parse HEAD)
REMOTE_COMMIT=$(git rev-parse origin/main)

if [ "$LOCAL_COMMIT" = "$REMOTE_COMMIT" ]; then
    log "✅ Already up to date with GitHub"
    log "Local commit: $LOCAL_COMMIT"
    exit 0
fi

log "🔄 Updates available:"
log "Local:  $LOCAL_COMMIT"
log "Remote: $REMOTE_COMMIT"

# Dừng services
log "🛑 Stopping services..."
sudo systemctl stop openresty 2>/dev/null || warn "OpenResty not running"
sudo systemctl stop mccva-mock-servers 2>/dev/null || warn "Mock servers not running"
docker stop mccva-ml 2>/dev/null || warn "ML container not running"

# Backup config files
log "💾 Backing up config files..."
cp mccva_env.conf ~/mccva_env.conf.backup 2>/dev/null || warn "No mccva_env.conf to backup"
cp nginx.conf ~/nginx.conf.backup 2>/dev/null || warn "No nginx.conf to backup"

# Reset về trạng thái GitHub
log "🔄 Resetting to GitHub state..."
git reset --hard origin/main
git clean -fd

# Restore config files
log "🔄 Restoring config files..."
if [ -f ~/mccva_env.conf.backup ]; then
    cp ~/mccva_env.conf.backup mccva_env.conf
    log "✅ Restored mccva_env.conf"
fi

if [ -f ~/nginx.conf.backup ]; then
    cp ~/nginx.conf.backup nginx.conf
    log "✅ Restored nginx.conf"
fi

# Set permissions
log "🔧 Setting permissions..."
sudo chown -R ubuntu:ubuntu /opt/mccva
chmod +x /opt/mccva/*.sh

# Start services
log "🚀 Starting services..."
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

# Show update summary
log "📊 Update Summary:"
git log --oneline "$LOCAL_COMMIT..HEAD"

log "✅ Update completed successfully!"
log "🎯 System is ready for use" 