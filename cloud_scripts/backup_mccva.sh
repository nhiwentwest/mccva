#!/bin/bash
# backup_mccva.sh - Backup MCCVA system trước khi update

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"; }

BACKUP_DIR="$HOME/mccva_backup_$(date +%Y%m%d_%H%M%S)"

log "🔄 Starting MCCVA backup..."

# Tạo thư mục backup
mkdir -p "$BACKUP_DIR"
log "📁 Backup directory: $BACKUP_DIR"

# Backup config files
if [ -f "/opt/mccva/mccva_env.conf" ]; then
    cp /opt/mccva/mccva_env.conf "$BACKUP_DIR/"
    log "✅ Backup mccva_env.conf"
fi

if [ -f "/opt/mccva/nginx.conf" ]; then
    cp /opt/mccva/nginx.conf "$BACKUP_DIR/"
    log "✅ Backup nginx.conf"
fi

# Backup logs nếu có
if [ -d "/var/log/mccva" ]; then
    cp -r /var/log/mccva "$BACKUP_DIR/"
    log "✅ Backup logs directory"
fi

# Backup Docker image info
if docker images | grep -q mccva-ml-service; then
    docker images mccva-ml-service > "$BACKUP_DIR/docker_image_info.txt"
    log "✅ Backup Docker image info"
fi

# Stop services để backup sạch
log "🛑 Stopping services..."
sudo systemctl stop openresty 2>/dev/null || warn "OpenResty not running"
sudo systemctl stop mccva-mock-servers 2>/dev/null || warn "Mock servers not running"
docker stop mccva-ml 2>/dev/null || warn "ML container not running"

# Backup systemd service files
if [ -f "/etc/systemd/system/mccva-mock-servers.service" ]; then
    cp /etc/systemd/system/mccva-mock-servers.service "$BACKUP_DIR/"
    log "✅ Backup systemd service file"
fi

# Backup current Git status
if [ -d "/opt/mccva/.git" ]; then
    cd /opt/mccva
    git log --oneline -10 > "$BACKUP_DIR/git_history.txt"
    git status > "$BACKUP_DIR/git_status.txt"
    log "✅ Backup Git status"
fi

# Tạo file info
cat > "$BACKUP_DIR/backup_info.txt" << EOF
MCCVA Backup Information
========================
Date: $(date)
Backup Directory: $BACKUP_DIR
System: $(uname -a)
User: $(whoami)
Git Commit: $(cd /opt/mccva && git rev-parse HEAD 2>/dev/null || echo "N/A")

Services Status:
- OpenResty: $(systemctl is-active openresty 2>/dev/null || echo "inactive")
- Mock Servers: $(systemctl is-active mccva-mock-servers 2>/dev/null || echo "inactive")
- ML Container: $(docker ps --filter name=mccva-ml --format "{{.Status}}" 2>/dev/null || echo "not running")

Files Backed Up:
$(ls -la "$BACKUP_DIR")
EOF

log "✅ Backup completed successfully!"
log "📁 Backup location: $BACKUP_DIR"
log "📄 Backup info: $BACKUP_DIR/backup_info.txt"

echo ""
log "🎯 To restore from this backup:"
log "   ./cloud_scripts/restore_mccva.sh $BACKUP_DIR" 