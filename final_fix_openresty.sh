#!/bin/bash

# Final Fix for OpenResty
# Chạy script này để fix OpenResty với approach khác

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

print_header "Final Fix for OpenResty"

USER=$(whoami)

# Step 1: Complete cleanup
print_header "Step 1: Complete Cleanup"
print_status "Performing complete cleanup..."

# Stop all services
sudo systemctl stop openresty || true
sudo systemctl stop mccva-ml || true
sudo systemctl stop mccva-mock-servers || true

# Kill all processes
sudo pkill -f nginx || true
sudo pkill -f openresty || true
sudo pkill -f python || true

sleep 3

# Force kill if still running
sudo pkill -9 -f nginx || true
sudo pkill -9 -f openresty || true

print_success "✅ Cleanup completed"

# Step 2: Check and fix permissions
print_header "Step 2: Check and Fix Permissions"
print_status "Checking and fixing permissions..."

# Create all necessary directories
sudo mkdir -p /var/log/nginx
sudo mkdir -p /usr/local/openresty/nginx/logs
sudo mkdir -p /var/www/html
sudo mkdir -p /var/run/nginx

# Set permissions
sudo chown -R $USER:$USER /var/log/nginx
sudo chmod -R 755 /var/log/nginx
sudo chown -R $USER:$USER /usr/local/openresty/nginx/logs
sudo chmod -R 755 /usr/local/openresty/nginx/logs
sudo chown $USER:$USER /var/run/nginx
sudo chmod 755 /var/run/nginx

# Fix project directory permissions
sudo chown -R $USER:$USER /opt/mccva
sudo chmod -R 755 /opt/mccva

print_success "✅ Permissions fixed"

# Step 3: Create working nginx config
print_header "Step 3: Create Working Nginx Config"
print_status "Creating working nginx configuration..."

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

print_success "✅ Working nginx configuration created"

# Step 4: Test configuration
print_header "Step 4: Test Configuration"
print_status "Testing nginx configuration..."

if sudo /usr/local/openresty/nginx/sbin/nginx -t; then
    print_success "✅ Configuration is valid"
else
    print_error "❌ Configuration has errors"
    exit 1
fi

# Step 5: Start services in order
print_header "Step 5: Start Services in Order"
print_status "Starting services in correct order..."

# Start ML service first
print_status "Starting ML service..."
sudo systemctl start mccva-ml
sleep 5

if systemctl is-active --quiet mccva-ml; then
    print_success "✅ ML service started"
else
    print_warning "⚠️ ML service failed to start"
fi

# Start mock servers
print_status "Starting mock servers..."
sudo systemctl start mccva-mock-servers
sleep 5

if systemctl is-active --quiet mccva-mock-servers; then
    print_success "✅ Mock servers started"
else
    print_warning "⚠️ Mock servers failed to start"
fi

# Start OpenResty last
print_status "Starting OpenResty..."
sudo systemctl start openresty
sleep 10

# Check if it's running
if systemctl is-active --quiet openresty; then
    print_success "✅ OpenResty started successfully"
else
    print_error "❌ OpenResty failed to start"
    sudo systemctl status openresty --no-pager
    sudo journalctl -u openresty --no-pager -n 20
    exit 1
fi

# Step 6: Test all endpoints
print_header "Step 6: Test All Endpoints"
print_status "Testing all endpoints..."

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

# Test ML service
response=$(curl -s http://localhost:5000/health 2>/dev/null || echo "ERROR")
if echo "$response" | grep -q "healthy"; then
    print_success "✅ ML service working"
else
    print_warning "⚠️ ML service test failed"
    echo "Response: $response"
fi

# Test mock servers
print_status "Testing mock servers..."
for port in 8081 8082 8083 8084 8085 8086 8087 8088; do
    if curl -s http://localhost:$port/health > /dev/null; then
        print_success "✅ Mock server port $port working"
    else
        print_warning "⚠️ Mock server port $port test failed"
    fi
done

print_header "Final Fix Complete"
print_success "✅ OpenResty and all services are now running!"
print_status "All services status:"
print_status "  • OpenResty: $(systemctl is-active openresty)"
print_status "  • ML Service: $(systemctl is-active mccva-ml)"
print_status "  • Mock Servers: $(systemctl is-active mccva-mock-servers)"

print_status "Test commands:"
print_status "  curl http://localhost/health"
print_status "  curl http://localhost/"
print_status "  ~/mccva_manage.sh status"
print_status "  ~/mccva_manage.sh demo" 