#!/bin/bash

# Create Mock Servers Service Script
# Chạy script này để tạo systemd service cho mock servers

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo "================================"
    echo "$1"
    echo "================================"
}

print_header "Create Mock Servers Service"

USER=$(whoami)
PROJECT_DIR="/opt/mccva"

print_status "Creating systemd service for mock servers..."

# Create systemd service file
sudo tee /etc/systemd/system/mccva-mock-servers.service > /dev/null <<EOF
[Unit]
Description=MCCVA Mock Servers
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
Environment=PYTHONPATH=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/python mock_servers.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

print_success "✅ Mock servers service file created"

# Reload systemd and enable service
print_status "Enabling and starting mock servers service..."

sudo systemctl daemon-reload
sudo systemctl enable mccva-mock-servers
sudo systemctl start mccva-mock-servers

print_success "✅ Mock servers service started"

# Check service status
print_status "Checking service status..."
sleep 5

if systemctl is-active --quiet mccva-mock-servers; then
    print_success "✅ Mock servers service is running"
else
    print_error "❌ Mock servers service failed to start"
    sudo systemctl status mccva-mock-servers --no-pager
    sudo journalctl -u mccva-mock-servers --no-pager -n 10
fi

# Test mock servers
print_status "Testing mock servers..."
for port in 8081 8082 8083 8084 8085 8086 8087 8088; do
    if curl -s http://localhost:$port/health > /dev/null; then
        print_success "✅ Mock server port $port is responding"
    else
        print_warning "⚠️ Mock server port $port test failed"
    fi
done

print_header "Service Created Successfully"
print_success "✅ Mock servers service has been created and started!"
print_status "Management commands:"
print_status "  • Status: sudo systemctl status mccva-mock-servers"
print_status "  • Logs: sudo journalctl -u mccva-mock-servers -f"
print_status "  • Restart: sudo systemctl restart mccva-mock-servers"
print_status "  • Stop: sudo systemctl stop mccva-mock-servers" 