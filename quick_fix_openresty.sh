#!/bin/bash

# Quick Fix for OpenResty PID File Issue
# Chạy script này để fix nhanh OpenResty startup issue

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

print_header "Quick Fix OpenResty PID File Issue"

USER=$(whoami)

# Step 1: Create necessary directories
print_header "Step 1: Create Directories"
print_status "Creating necessary directories..."

sudo mkdir -p /var/log/nginx
sudo mkdir -p /usr/local/openresty/nginx/logs
sudo mkdir -p /var/www/html
sudo mkdir -p /var/run/nginx

print_success "✅ Directories created"

# Step 2: Set permissions
print_header "Step 2: Set Permissions"
print_status "Setting permissions..."

sudo chown -R $USER:$USER /var/log/nginx
sudo chmod -R 755 /var/log/nginx
sudo chown -R $USER:$USER /usr/local/openresty/nginx/logs
sudo chmod -R 755 /usr/local/openresty/nginx/logs
sudo chown $USER:$USER /var/run/nginx
sudo chmod 755 /var/run/nginx

print_success "✅ Permissions set"

# Step 3: Kill existing processes
print_header "Step 3: Kill Existing Processes"
print_status "Killing existing nginx processes..."

sudo pkill -f nginx || true
sudo pkill -f openresty || true

sleep 2

# Force kill if still running
if pgrep -f nginx > /dev/null; then
    print_warning "⚠️ Force killing remaining nginx processes..."
    sudo pkill -9 -f nginx || true
    sleep 1
fi

print_success "✅ Processes killed"

# Step 4: Test nginx configuration
print_header "Step 4: Test Configuration"
print_status "Testing nginx configuration..."

if sudo /usr/local/openresty/nginx/sbin/nginx -t; then
    print_success "✅ Configuration is valid"
else
    print_error "❌ Configuration has errors"
    exit 1
fi

# Step 5: Start OpenResty
print_header "Step 5: Start OpenResty"
print_status "Starting OpenResty..."

sudo systemctl start openresty

sleep 5

# Check if it's running
if systemctl is-active --quiet openresty; then
    print_success "✅ OpenResty started successfully"
else
    print_error "❌ OpenResty failed to start"
    sudo systemctl status openresty --no-pager
    exit 1
fi

# Step 6: Test endpoints
print_header "Step 6: Test Endpoints"
print_status "Testing endpoints..."

# Test health endpoint
response=$(curl -s http://localhost/health 2>/dev/null || echo "ERROR")
if echo "$response" | grep -q "mccva-openresty-gateway"; then
    print_success "✅ Health endpoint working"
    echo "Response: $response"
else
    print_warning "⚠️ Health endpoint test failed"
    echo "Response: $response"
fi

# Test root endpoint
response=$(curl -s http://localhost/ 2>/dev/null || echo "ERROR")
if echo "$response" | grep -q "MCCVA OpenResty Gateway"; then
    print_success "✅ Root endpoint working"
else
    print_warning "⚠️ Root endpoint test failed"
    echo "Response: $response"
fi

print_header "Quick Fix Complete"
print_success "✅ OpenResty PID file issue has been resolved!"
print_status "OpenResty is now running successfully."
print_status "Test commands:"
print_status "  curl http://localhost/health"
print_status "  curl http://localhost/"
print_status "  ~/mccva_manage.sh status" 