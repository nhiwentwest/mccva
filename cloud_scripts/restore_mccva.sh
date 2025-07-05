#!/bin/bash
# restore_mccva.sh - Restore MCCVA system từ backup

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"; }

# Kiểm tra backup directory
if [ $# -eq 0 ]; then
    error "Usage: $0 <backup_directory>"
    echo "Example: $0 ~/mccva_backup_20241201_143022"
    exit 1
fi

BACKUP_DIR="$1"

if [ ! -d "$BACKUP_DIR" ]; then
    error "Backup directory not found: $BACKUP_DIR"
    exit 1
fi

log "🔄 Starting MCCVA restore from: $BACKUP_DIR"

# Kiểm tra backup info
if [ -f "$BACKUP_DIR/backup_info.txt" ]; then
    log "📄 Backup information:"
    cat "$BACKUP_DIR/backup_info.txt"
    echo ""
fi

# Dừng services hiện tại
log "🛑 Stopping current services..."
sudo systemctl stop openresty 2>/dev/null || warn "OpenResty not running"
sudo systemctl stop mccva-mock-servers 2>/dev/null || warn "Mock servers not running"
docker stop mccva-ml 2>/dev/null || warn "ML container not running"

# Restore config files
if [ -f "$BACKUP_DIR/mccva_env.conf" ]; then
    cp "$BACKUP_DIR/mccva_env.conf" /opt/mccva/
    log "✅ Restored mccva_env.conf"
fi

if [ -f "$BACKUP_DIR/nginx.conf" ]; then
    cp "$BACKUP_DIR/nginx.conf" /opt/mccva/
    log "✅ Restored nginx.conf"
fi

# Restore systemd service file
if [ -f "$BACKUP_DIR/mccva-mock-servers.service" ]; then
    sudo cp "$BACKUP_DIR/mccva-mock-servers.service" /etc/systemd/system/
    sudo systemctl daemon-reload
    log "✅ Restored systemd service file"
fi

# Restore logs nếu có
if [ -d "$BACKUP_DIR/mccva" ]; then
    sudo cp -r "$BACKUP_DIR/mccva" /var/log/
    log "✅ Restored logs directory"
fi

# Set permissions
log "🔧 Setting permissions..."
sudo chown -R ubuntu:ubuntu /opt/mccva
chmod +x /opt/mccva/*.sh

# Start services
log "🚀 Starting services..."
cd /opt/mccva
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

log "✅ Restore completed successfully!"
log "🎯 System is ready for use" 