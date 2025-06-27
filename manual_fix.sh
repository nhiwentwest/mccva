#!/bin/bash

# Manual Fix for OpenResty
# Chạy từng bước một cách manual

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

print_header "Manual Fix for OpenResty"

USER=$(whoami)

# Step 1: Stop services manually
print_header "Step 1: Stop Services Manually"
print_status "Stopping services one by one..."

echo "Stopping OpenResty..."
sudo systemctl stop openresty || echo "OpenResty already stopped"

echo "Stopping ML service..."
sudo systemctl stop mccva-ml || echo "ML service already stopped"

echo "Stopping mock servers..."
sudo systemctl stop mccva-mock-servers || echo "Mock servers already stopped"

print_success "✅ Services stopped"

# Step 2: Kill processes manually
print_header "Step 2: Kill Processes Manually"
print_status "Killing processes one by one..."

echo "Killing nginx processes..."
sudo pkill -f nginx || echo "No nginx processes found"

echo "Killing openresty processes..."
sudo pkill -f openresty || echo "No openresty processes found"

echo "Waiting 3 seconds..."
sleep 3

echo "Force killing remaining processes..."
sudo pkill -9 -f nginx || echo "No nginx processes to force kill"
sudo pkill -9 -f openresty || echo "No openresty processes to force kill"

print_success "✅ Processes killed"

# Step 3: Check what's running
print_header "Step 3: Check What's Running"
print_status "Checking current processes..."

echo "Nginx processes:"
ps aux | grep nginx | grep -v grep || echo "No nginx processes"

echo ""
echo "OpenResty processes:"
ps aux | grep openresty | grep -v grep || echo "No openresty processes"

echo ""
echo "Python processes:"
ps aux | grep python | grep -v grep || echo "No python processes"

# Step 4: Fix permissions
print_header "Step 4: Fix Permissions"
print_status "Fixing permissions..."

echo "Creating directories..."
sudo mkdir -p /var/log/nginx
sudo mkdir -p /usr/local/openresty/nginx/logs
sudo mkdir -p /var/www/html
sudo mkdir -p /var/run/nginx

echo "Setting permissions..."
sudo chown -R $USER:$USER /var/log/nginx
sudo chmod -R 755 /var/log/nginx
sudo chown -R $USER:$USER /usr/local/openresty/nginx/logs
sudo chmod -R 755 /usr/local/openresty/nginx/logs
sudo chown $USER:$USER /var/run/nginx
sudo chmod 755 /var/run/nginx

print_success "✅ Permissions fixed"

# Step 5: Test nginx manually
print_header "Step 5: Test Nginx Manually"
print_status "Testing nginx configuration..."

if sudo /usr/local/openresty/nginx/sbin/nginx -t; then
    print_success "✅ Configuration is valid"
else
    print_error "❌ Configuration has errors"
    exit 1
fi

# Step 6: Start nginx manually
print_header "Step 6: Start Nginx Manually"
print_status "Starting nginx manually..."

echo "Starting nginx..."
sudo /usr/local/openresty/nginx/sbin/nginx

echo "Waiting 3 seconds..."
sleep 3

echo "Checking if nginx is running..."
if pgrep -f nginx > /dev/null; then
    print_success "✅ Nginx started manually"
else
    print_error "❌ Nginx failed to start manually"
    exit 1
fi

# Step 7: Test endpoints
print_header "Step 7: Test Endpoints"
print_status "Testing endpoints..."

echo "Testing health endpoint..."
response=$(curl -s http://localhost/health 2>/dev/null || echo "ERROR")
if echo "$response" | grep -q "mccva-openresty-gateway"; then
    print_success "✅ Health endpoint working"
    echo "Response: $response"
else
    print_warning "⚠️ Health endpoint test failed"
    echo "Response: $response"
fi

echo "Testing root endpoint..."
response=$(curl -s http://localhost/ 2>/dev/null || echo "ERROR")
if echo "$response" | grep -q "MCCVA OpenResty Gateway"; then
    print_success "✅ Root endpoint working"
else
    print_warning "⚠️ Root endpoint test failed"
    echo "Response: $response"
fi

# Step 8: Start other services
print_header "Step 8: Start Other Services"
print_status "Starting other services..."

echo "Starting ML service..."
sudo systemctl start mccva-ml
sleep 5

if systemctl is-active --quiet mccva-ml; then
    print_success "✅ ML service started"
else
    print_warning "⚠️ ML service failed to start"
fi

echo "Starting mock servers..."
sudo systemctl start mccva-mock-servers
sleep 5

if systemctl is-active --quiet mccva-mock-servers; then
    print_success "✅ Mock servers started"
else
    print_warning "⚠️ Mock servers failed to start"
fi

# Step 9: Test all services
print_header "Step 9: Test All Services"
print_status "Testing all services..."

echo "Testing ML service..."
response=$(curl -s http://localhost:5000/health 2>/dev/null || echo "ERROR")
if echo "$response" | grep -q "healthy"; then
    print_success "✅ ML service working"
else
    print_warning "⚠️ ML service test failed"
    echo "Response: $response"
fi

echo "Testing mock servers..."
for port in 8081 8082 8083 8084 8085 8086 8087 8088; do
    if curl -s http://localhost:$port/health > /dev/null; then
        print_success "✅ Mock server port $port working"
    else
        print_warning "⚠️ Mock server port $port test failed"
    fi
done

print_header "Manual Fix Complete"
print_success "✅ Manual fix completed!"
print_status "Current status:"
print_status "  • Nginx: $(pgrep -f nginx > /dev/null && echo "running" || echo "stopped")"
print_status "  • ML Service: $(systemctl is-active mccva-ml)"
print_status "  • Mock Servers: $(systemctl is-active mccva-mock-servers)"

print_status "Test commands:"
print_status "  curl http://localhost/health"
print_status "  curl http://localhost/"
print_status "  ~/mccva_manage.sh status" 