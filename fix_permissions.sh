#!/bin/bash

# Fix Permissions Script for MCCVA
# Chạy script này để fix permission issues trên server hiện tại

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

print_header "Fix MCCVA Permissions"

# Get current user
USER=$(whoami)
PROJECT_DIR="/opt/mccva"

print_status "Fixing permissions for user: $USER"
print_status "Project directory: $PROJECT_DIR"

# Step 1: Create log directories
print_header "Step 1: Create Log Directories"
print_status "Creating log directories..."

sudo mkdir -p /var/log/mccva
sudo mkdir -p /var/log/nginx
sudo mkdir -p /usr/local/openresty/nginx/logs

print_success "✅ Log directories created"

# Step 2: Set permissions
print_header "Step 2: Set Permissions"
print_status "Setting permissions..."

sudo chown -R $USER:$USER /var/log/mccva
sudo chmod -R 755 /var/log/mccva
sudo chown -R $USER:$USER /usr/local/openresty/nginx/logs
sudo chmod -R 755 /usr/local/openresty/nginx/logs

print_success "✅ Permissions set"

# Step 3: Create log files
print_header "Step 3: Create Log Files"
print_status "Creating log files..."

sudo touch /var/log/mccva/mccva-ml.log
sudo touch /var/log/mccva/mccva-mock-servers.log
sudo chown $USER:$USER /var/log/mccva/mccva-ml.log
sudo chown $USER:$USER /var/log/mccva/mccva-mock-servers.log
sudo chmod 644 /var/log/mccva/mccva-ml.log
sudo chmod 644 /var/log/mccva/mccva-mock-servers.log

print_success "✅ Log files created"

# Step 4: Fix project directory permissions
print_header "Step 4: Fix Project Directory Permissions"
print_status "Fixing project directory permissions..."

if [ -d "$PROJECT_DIR" ]; then
    sudo chown -R $USER:$USER $PROJECT_DIR
    sudo chmod -R 755 $PROJECT_DIR
    print_success "✅ Project directory permissions fixed"
else
    print_warning "⚠️ Project directory not found: $PROJECT_DIR"
fi

# Step 5: Restart services
print_header "Step 5: Restart Services"
print_status "Restarting services..."

sudo systemctl daemon-reload
sudo systemctl restart mccva-ml
sudo systemctl restart mccva-mock-servers
sudo systemctl restart openresty

print_success "✅ Services restarted"

# Step 6: Check service status
print_header "Step 6: Check Service Status"
print_status "Checking service status..."

sleep 5

if systemctl is-active --quiet mccva-ml; then
    print_success "✅ MCCVA ML Service is running"
else
    print_error "❌ MCCVA ML Service failed to start"
    sudo systemctl status mccva-ml --no-pager
    sudo journalctl -u mccva-ml --no-pager -n 10
fi

if systemctl is-active --quiet mccva-mock-servers; then
    print_success "✅ MCCVA Mock Servers are running"
else
    print_error "❌ MCCVA Mock Servers failed to start"
    sudo systemctl status mccva-mock-servers --no-pager
    sudo journalctl -u mccva-mock-servers --no-pager -n 10
fi

if systemctl is-active --quiet openresty; then
    print_success "✅ OpenResty is running"
else
    print_error "❌ OpenResty failed to start"
    sudo systemctl status openresty --no-pager
fi

# Step 7: Test endpoints
print_header "Step 7: Test Endpoints"
print_status "Testing endpoints..."

if curl -s http://localhost/health | grep -q "mccva-openresty-gateway"; then
    print_success "✅ Health endpoint is working"
else
    print_warning "⚠️ Health endpoint test failed"
fi

if curl -s http://localhost:5000/health | grep -q "healthy"; then
    print_success "✅ ML Service is responding"
else
    print_warning "⚠️ ML Service test failed"
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

print_header "Fix Complete"
print_success "✅ Permissions have been fixed!"
print_status "You can now use the management script:"
print_status "  ~/mccva_manage.sh status"
print_status "  ~/mccva_manage.sh test"
print_status "  ~/mccva_manage.sh demo" 