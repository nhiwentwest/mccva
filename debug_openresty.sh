#!/bin/bash

# Debug OpenResty Timeout Issues
# Chạy script này để debug và fix OpenResty timeout

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

print_header "Debug OpenResty Timeout Issues"

# Step 1: Check detailed logs
print_header "Step 1: Check Detailed Logs"
print_status "Checking OpenResty logs..."

sudo journalctl -u openresty --no-pager -n 30

# Step 2: Check nginx error logs
print_header "Step 2: Check Nginx Error Logs"
print_status "Checking nginx error logs..."

if [ -f "/var/log/nginx/error.log" ]; then
    print_status "Nginx error log:"
    sudo tail -20 /var/log/nginx/error.log
else
    print_warning "⚠️ Nginx error log not found"
fi

# Step 3: Check system resources
print_header "Step 3: Check System Resources"
print_status "Checking system resources..."

echo "Memory usage:"
free -h

echo ""
echo "Disk usage:"
df -h

echo ""
echo "Process count:"
ps aux | wc -l

# Step 4: Check port conflicts
print_header "Step 4: Check Port Conflicts"
print_status "Checking port 80..."

if netstat -tlnp | grep :80; then
    print_warning "⚠️ Port 80 is in use"
    netstat -tlnp | grep :80
else
    print_success "✅ Port 80 is free"
fi

# Step 5: Check nginx configuration
print_header "Step 5: Check Nginx Configuration"
print_status "Testing nginx configuration..."

if sudo /usr/local/openresty/nginx/sbin/nginx -t; then
    print_success "✅ Configuration is valid"
else
    print_error "❌ Configuration has errors"
fi

# Step 6: Try manual start
print_header "Step 6: Try Manual Start"
print_status "Trying to start nginx manually..."

# Kill any existing processes
sudo pkill -f nginx || true
sudo pkill -f openresty || true
sleep 2

# Try manual start
print_status "Starting nginx manually..."
sudo /usr/local/openresty/nginx/sbin/nginx

sleep 3

# Check if it's running
if pgrep -f nginx > /dev/null; then
    print_success "✅ Nginx started manually"
    
    # Test endpoints
    print_status "Testing endpoints..."
    
    response=$(curl -s http://localhost/health 2>/dev/null || echo "ERROR")
    if echo "$response" | grep -q "mccva-openresty-gateway"; then
        print_success "✅ Health endpoint working"
        echo "Response: $response"
    else
        print_warning "⚠️ Health endpoint test failed"
        echo "Response: $response"
    fi
    
    # Stop manual nginx
    sudo /usr/local/openresty/nginx/sbin/nginx -s stop
    sleep 2
    
else
    print_error "❌ Manual start failed"
fi

# Step 7: Create ultra-minimal config
print_header "Step 7: Create Ultra-Minimal Config"
print_status "Creating ultra-minimal nginx configuration..."

sudo tee /usr/local/openresty/nginx/conf/nginx.conf > /dev/null <<EOF
worker_processes 1;
error_log /var/log/nginx/error.log;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /usr/local/openresty/nginx/conf/mime.types;
    default_type application/octet-stream;
    
    access_log /var/log/nginx/access.log;
    
    sendfile on;
    keepalive_timeout 65;
    
    server {
        listen 80;
        server_name localhost;
        
        location /health {
            content_by_lua_block {
                ngx.header.content_type = "application/json";
                ngx.say('{"status": "healthy", "service": "mccva-openresty-gateway"}');
            }
        }
        
        location / {
            return 200 "MCCVA OpenResty Gateway is running";
        }
    }
}
EOF

print_success "✅ Ultra-minimal configuration created"

# Step 8: Test ultra-minimal config
print_header "Step 8: Test Ultra-Minimal Config"
print_status "Testing ultra-minimal configuration..."

if sudo /usr/local/openresty/nginx/sbin/nginx -t; then
    print_success "✅ Ultra-minimal configuration is valid"
else
    print_error "❌ Ultra-minimal configuration has errors"
    exit 1
fi

# Step 9: Start with ultra-minimal config
print_header "Step 9: Start with Ultra-Minimal Config"
print_status "Starting OpenResty with ultra-minimal config..."

sudo systemctl start openresty

sleep 10

# Check if it's running
if systemctl is-active --quiet openresty; then
    print_success "✅ OpenResty started successfully with ultra-minimal config"
    
    # Test endpoints
    print_status "Testing endpoints..."
    
    response=$(curl -s http://localhost/health 2>/dev/null || echo "ERROR")
    if echo "$response" | grep -q "mccva-openresty-gateway"; then
        print_success "✅ Health endpoint working"
        echo "Response: $response"
    else
        print_warning "⚠️ Health endpoint test failed"
        echo "Response: $response"
    fi
    
    response=$(curl -s http://localhost/ 2>/dev/null || echo "ERROR")
    if echo "$response" | grep -q "MCCVA OpenResty Gateway"; then
        print_success "✅ Root endpoint working"
    else
        print_warning "⚠️ Root endpoint test failed"
        echo "Response: $response"
    fi
    
else
    print_error "❌ OpenResty still failed to start"
    sudo systemctl status openresty --no-pager
    sudo journalctl -u openresty --no-pager -n 20
fi

print_header "Debug Complete"
print_success "✅ Debug process completed!"
print_status "If OpenResty is running, you can now gradually add back features."
print_status "Test commands:"
print_status "  curl http://localhost/health"
print_status "  curl http://localhost/"
print_status "  ~/mccva_manage.sh status" 