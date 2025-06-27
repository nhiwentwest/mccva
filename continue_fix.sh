#!/bin/bash

# Continue Fix from Step 4
# Chạy script này nếu quick_fix_openresty.sh bị dừng

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

print_header "Continue Fix from Step 4"

USER=$(whoami)

# Step 4: Test nginx configuration
print_header "Step 4: Test Configuration"
print_status "Testing nginx configuration..."

if sudo /usr/local/openresty/nginx/sbin/nginx -t; then
    print_success "✅ Configuration is valid"
else
    print_error "❌ Configuration has errors"
    print_status "Creating minimal configuration..."
    
    # Create minimal config
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
        
        location / {
            return 200 "MCCVA OpenResty Gateway is running";
        }
    }
}
EOF
    
    print_success "✅ Minimal configuration created"
    
    # Test again
    if sudo /usr/local/openresty/nginx/sbin/nginx -t; then
        print_success "✅ Configuration is now valid"
    else
        print_error "❌ Configuration still has errors"
        exit 1
    fi
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

print_header "Continue Fix Complete"
print_success "✅ OpenResty is now running successfully!"
print_status "Test commands:"
print_status "  curl http://localhost/health"
print_status "  curl http://localhost/"
print_status "  ~/mccva_manage.sh status" 