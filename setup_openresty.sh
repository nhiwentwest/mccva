#!/bin/bash

# setup_openresty.sh - Setup OpenResty cho MCCVA deployment
# Khắc phục vấn đề PID file conflicts và permissions

set -e

echo "🚀 Setting up OpenResty for MCCVA deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# 1. Create necessary directories with correct permissions
print_status "Creating OpenResty directories..."
sudo mkdir -p /usr/local/openresty/nginx/logs
sudo mkdir -p /usr/local/openresty/nginx/conf/lua
sudo mkdir -p /var/log/nginx
sudo mkdir -p /var/www/html

# 2. Set correct ownership
print_status "Setting correct permissions..."
sudo chown -R $USER:$USER /usr/local/openresty/nginx/logs
sudo chown -R $USER:$USER /usr/local/openresty/nginx/conf
sudo chmod 755 /usr/local/openresty/nginx/logs
sudo chmod 755 /usr/local/openresty/nginx/conf

# 3. Remove any existing PID files
print_status "Cleaning up old PID files..."
sudo rm -f /var/run/nginx.pid
sudo rm -f /usr/local/openresty/nginx/logs/nginx.pid

# 4. Kill any existing nginx/openresty processes
print_status "Stopping existing processes..."
sudo pkill -f nginx || true
sudo pkill -f openresty || true

# 5. Backup existing config if exists
if [ -f /etc/openresty/nginx.conf ]; then
    print_warning "Backing up existing config..."
    sudo cp /etc/openresty/nginx.conf /etc/openresty/nginx.conf.backup.$(date +%Y%m%d_%H%M%S)
fi

# 6. Copy new config
print_status "Installing new nginx config..."
sudo cp nginx.conf /etc/openresty/nginx.conf

# 7. Test configuration
print_status "Testing nginx configuration..."
if sudo openresty -t; then
    print_status "✅ Configuration test passed"
else
    print_error "❌ Configuration test failed"
    exit 1
fi

# 8. Create systemd override if needed
print_status "Creating systemd service override..."
sudo mkdir -p /etc/systemd/system/openresty.service.d
sudo tee /etc/systemd/system/openresty.service.d/override.conf > /dev/null << EOF
[Service]
# Fix PID file path
PIDFile=/usr/local/openresty/nginx/logs/nginx.pid

# Increase timeout for ML model loading
TimeoutStartSec=120
TimeoutStopSec=60

# Restart policy
Restart=always
RestartSec=10

# Environment
Environment=MCCVA_ENV=production
EOF

# 9. Reload systemd and restart service
print_status "Reloading systemd and starting OpenResty..."
sudo systemctl daemon-reload
sudo systemctl enable openresty
sudo systemctl restart openresty

# 10. Wait for service to start
sleep 3

# 11. Check service status
print_status "Checking service status..."
if sudo systemctl is-active --quiet openresty; then
    print_status "✅ OpenResty is running successfully"
    
    # Test basic endpoints
    print_status "Testing basic endpoints..."
    if curl -s http://localhost/health > /dev/null; then
        print_status "✅ Health endpoint responding"
    else
        print_warning "⚠️  Health endpoint not responding (normal if ML service not started)"
    fi
    
else
    print_error "❌ OpenResty failed to start"
    print_error "Check logs: sudo journalctl -xeu openresty.service"
    exit 1
fi

# 12. Display service information
print_status "Service Information:"
echo "  - Status: $(sudo systemctl is-active openresty)"
echo "  - Config: /etc/openresty/nginx.conf"
echo "  - PID file: /usr/local/openresty/nginx/logs/nginx.pid"
echo "  - Logs: /var/log/nginx/"
echo "  - Service logs: sudo journalctl -u openresty.service"

print_status "🎉 OpenResty setup completed successfully!"
print_status "Next steps:"
echo "  1. Start ML service: cd /opt/mccva && python3 ml_service.py"
echo "  2. Test Meta-Learning: curl -X POST http://localhost/api/meta_learning -H 'Content-Type: application/json' -d '{\"features\": [85.5, 70.2, 0.8, 0.6, 0.3, 1, 0, 0, 1, 2, 1, 0.75, 0.82]}'"
echo "  3. Monitor logs: sudo tail -f /var/log/nginx/access.log" 