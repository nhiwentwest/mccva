#!/bin/bash
# run.sh - Khởi động và kiểm thử hệ thống MCCVA
set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; }
CURRENT_USER=$(whoami)
log "Reload systemd..."; sudo systemctl daemon-reload
log "Start ML Service..."; sudo systemctl start mccva-ml
log "Start Mock Servers..."; sudo systemctl start mccva-mock-servers
log "Start OpenResty..."; sudo systemctl start openresty || true
sleep 5
log "Kiểm tra trạng thái service..."; sudo systemctl status mccva-ml --no-pager; sudo systemctl status mccva-mock-servers --no-pager; sudo systemctl status openresty --no-pager
log "Test health endpoint..."; curl -s http://localhost/health || error "OpenResty health failed"; curl -s http://localhost:5000/health || error "ML Service health failed"
for port in 8081 8082 8083 8084 8085 8086 8087 8088; do curl -s http://localhost:$port/health || error "Mock server $port health failed"; done
log "Test AI routing..."; curl -s -X POST http://localhost/mccva/route -H "Content-Type: application/json" -d '{"cpu_cores": 4, "memory": 8, "storage": 100, "network_bandwidth": 1000, "priority": 3}' || error "AI routing test failed"
log "✅ Hệ thống đã sẵn sàng!" 