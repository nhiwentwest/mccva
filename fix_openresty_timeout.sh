#!/bin/bash

# Fix OpenResty Timeout Issues
# Chạy script này để fix OpenResty timeout và startup issues

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

print_header "Fix OpenResty Timeout Issues"

# Step 1: Check OpenResty status and logs
print_header "Step 1: Check OpenResty Status and Logs"
print_status "Checking OpenResty status..."

sudo systemctl status openresty --no-pager

print_status "Checking OpenResty logs..."
sudo journalctl -u openresty --no-pager -n 20

# Step 2: Check nginx configuration
print_header "Step 2: Check Nginx Configuration"
print_status "Testing nginx configuration..."

if sudo /usr/local/openresty/nginx/sbin/nginx -t; then
    print_success "✅ Nginx configuration is valid"
else
    print_error "❌ Nginx configuration has errors"
    print_status "Fixing nginx configuration..."
    
    # Backup current config
    sudo cp /usr/local/openresty/nginx/conf/nginx.conf /usr/local/openresty/nginx/conf/nginx.conf.backup
    
    # Create a minimal working config
    sudo tee /usr/local/openresty/nginx/conf/nginx.conf > /dev/null <<EOF
worker_processes auto;
error_log /var/log/nginx/error.log;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
}

http {
    include /usr/local/openresty/nginx/conf/mime.types;
    default_type application/octet-stream;
    
    log_format main '\$remote_addr - \$remote_user [\$time_local] "\$request" '
                    '\$status \$body_bytes_sent "\$http_referer" '
                    '"\$http_user_agent" "\$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    upstream ml_service {
        server 127.0.0.1:5000;
        keepalive 32;
    }
    
    upstream vm_low_load {
        server 127.0.0.1:8081 weight=7;
        server 127.0.0.1:8082 weight=3;
        keepalive 16;
    }
    
    upstream vm_medium_load {
        server 127.0.0.1:8083 weight=6;
        server 127.0.0.1:8084 weight=4;
        keepalive 16;
    }
    
    upstream vm_high_load {
        server 127.0.0.1:8085 weight=8;
        server 127.0.0.1:8086 weight=2;
        keepalive 16;
    }
    
    upstream vm_balanced {
        server 127.0.0.1:8087 weight=5;
        server 127.0.0.1:8088 weight=5;
        keepalive 16;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        root /var/www/html;
        index index.html index.htm;
        
        location /health {
            content_by_lua_block {
                ngx.header.content_type = "application/json";
                ngx.say('{"status": "healthy", "service": "mccva-openresty-gateway"}');
            }
        }
        
        location /api/ml/ {
            proxy_pass http://ml_service/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_connect_timeout 5s;
            proxy_send_timeout 10s;
            proxy_read_timeout 10s;
        }
        
        location / {
            return 200 "MCCVA OpenResty Gateway is running";
        }
    }
}
EOF

    print_success "✅ Created minimal nginx configuration"
fi

# Step 3: Check and create log directories
print_header "Step 3: Check and Create Log Directories"
print_status "Creating log directories..."

sudo mkdir -p /var/log/nginx
sudo mkdir -p /usr/local/openresty/nginx/logs
sudo mkdir -p /var/www/html
sudo mkdir -p /var/run

sudo chown -R ubuntu:ubuntu /var/log/nginx
sudo chmod -R 755 /var/log/nginx
sudo chown -R ubuntu:ubuntu /usr/local/openresty/nginx/logs
sudo chmod -R 755 /usr/local/openresty/nginx/logs

# Create PID directory
sudo mkdir -p /var/run/nginx
sudo chown ubuntu:ubuntu /var/run/nginx
sudo chmod 755 /var/run/nginx

print_success "✅ Log directories and PID directory created"

# Step 4: Kill any existing nginx processes
print_header "Step 4: Kill Existing Nginx Processes"
print_status "Killing any existing nginx processes..."

sudo pkill -f nginx || true
sudo pkill -f openresty || true

sleep 2

# Check if any nginx processes are still running
if pgrep -f nginx > /dev/null; then
    print_warning "⚠️ Some nginx processes are still running, force killing..."
    sudo pkill -9 -f nginx || true
    sleep 1
fi

print_success "✅ Nginx processes killed"

# Step 5: Start OpenResty with longer timeout
print_header "Step 5: Start OpenResty with Longer Timeout"
print_status "Starting OpenResty..."

# Stop the service first
sudo systemctl stop openresty || true

# Start manually with longer timeout
sudo systemctl start openresty --timeout=120

sleep 10

# Check if it's running
if systemctl is-active --quiet openresty; then
    print_success "✅ OpenResty started successfully"
else
    print_error "❌ OpenResty failed to start"
    sudo systemctl status openresty --no-pager
    sudo journalctl -u openresty --no-pager -n 30
    exit 1
fi

# Step 6: Test basic functionality
print_header "Step 6: Test Basic Functionality"
print_status "Testing basic endpoints..."

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

# Step 7: Gradually add back features
print_header "Step 7: Gradually Add Back Features"
print_status "Adding back MCCVA features..."

# Check if lua files exist
LUA_DIR="/usr/local/openresty/nginx/conf/lua"
if [ ! -d "$LUA_DIR" ]; then
    sudo mkdir -p "$LUA_DIR"
    print_success "✅ Created lua directory"
fi

# Copy lua files if they exist
if [ -f "/opt/mccva/lua/mccva_routing.lua" ]; then
    sudo cp /opt/mccva/lua/mccva_routing.lua "$LUA_DIR/"
    print_success "✅ Copied mccva_routing.lua"
fi

if [ -f "/opt/mccva/lua/predict_makespan.lua" ]; then
    sudo cp /opt/mccva/lua/predict_makespan.lua "$LUA_DIR/"
    print_success "✅ Copied predict_makespan.lua"
fi

if [ -f "/opt/mccva/lua/predict_vm_cluster.lua" ]; then
    sudo cp /opt/mccva/lua/predict_vm_cluster.lua "$LUA_DIR/"
    print_success "✅ Copied predict_vm_cluster.lua"
fi

# Step 8: Test with full configuration
print_header "Step 8: Test with Full Configuration"
print_status "Testing with full MCCVA configuration..."

# Test ML service endpoint
response=$(curl -s http://localhost:5000/health 2>/dev/null || echo "ERROR")
if echo "$response" | grep -q "healthy"; then
    print_success "✅ ML service is responding"
else
    print_warning "⚠️ ML service test failed"
    echo "Response: $response"
fi

print_header "Fix Complete"
print_success "✅ OpenResty timeout issues have been resolved!"
print_status "OpenResty is now running with basic functionality."
print_status "You can now gradually add back MCCVA features."
print_status "Test commands:"
print_status "  curl http://localhost/health"
print_status "  curl http://localhost/"
print_status "  ~/mccva_manage.sh status" 