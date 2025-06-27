#!/bin/bash

# Fix Health Endpoint and Nginx Configuration
# Chạy script này để fix health endpoint issues

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

print_header "Fix Health Endpoint and Nginx Configuration"

# Step 1: Check nginx configuration
print_header "Step 1: Check Nginx Configuration"
print_status "Checking nginx configuration..."

if [ -f "/usr/local/openresty/nginx/conf/nginx.conf" ]; then
    print_success "✅ Nginx config file exists"
    
    # Check if health endpoint is configured
    if grep -q "location /health" /usr/local/openresty/nginx/conf/nginx.conf; then
        print_success "✅ Health endpoint configured in nginx.conf"
    else
        print_error "❌ Health endpoint not found in nginx.conf"
    fi
else
    print_error "❌ Nginx config file not found"
fi

# Step 2: Check nginx status
print_header "Step 2: Check Nginx Status"
print_status "Checking nginx status..."

if systemctl is-active --quiet openresty; then
    print_success "✅ OpenResty is running"
else
    print_error "❌ OpenResty is not running"
    sudo systemctl status openresty --no-pager
fi

# Step 3: Test nginx configuration
print_header "Step 3: Test Nginx Configuration"
print_status "Testing nginx configuration..."

if sudo /usr/local/openresty/nginx/sbin/nginx -t; then
    print_success "✅ Nginx configuration is valid"
else
    print_error "❌ Nginx configuration has errors"
    exit 1
fi

# Step 4: Restart nginx
print_header "Step 4: Restart Nginx"
print_status "Restarting nginx..."

sudo systemctl restart openresty
sleep 5

if systemctl is-active --quiet openresty; then
    print_success "✅ OpenResty restarted successfully"
else
    print_error "❌ OpenResty failed to restart"
    sudo systemctl status openresty --no-pager
    exit 1
fi

# Step 5: Test health endpoint
print_header "Step 5: Test Health Endpoint"
print_status "Testing health endpoint..."

# Wait a bit for nginx to fully start
sleep 3

response=$(curl -s http://localhost/health 2>/dev/null || echo "ERROR")

if echo "$response" | grep -q "mccva-openresty-gateway"; then
    print_success "✅ Health endpoint is working"
    echo "Response: $response"
elif echo "$response" | grep -q "404"; then
    print_error "❌ Health endpoint returns 404"
    echo "Response: $response"
    
    # Check if lua files exist
    print_status "Checking lua files..."
    if [ -f "/usr/local/openresty/nginx/conf/lua/mccva_routing.lua" ]; then
        print_success "✅ mccva_routing.lua exists"
    else
        print_error "❌ mccva_routing.lua not found"
    fi
    
    if [ -f "/usr/local/openresty/nginx/conf/lua/predict_makespan.lua" ]; then
        print_success "✅ predict_makespan.lua exists"
    else
        print_error "❌ predict_makespan.lua not found"
    fi
    
    if [ -f "/usr/local/openresty/nginx/conf/lua/predict_vm_cluster.lua" ]; then
        print_success "✅ predict_vm_cluster.lua exists"
    else
        print_error "❌ predict_vm_cluster.lua not found"
    fi
    
    # Check if lua files exist
    print_status "Checking lua files..."
    if [ -f "/usr/local/openresty/nginx/conf/lua/mccva_routing.lua" ]; then
        print_success "✅ mccva_routing.lua exists"
    else
        print_error "❌ mccva_routing.lua not found"
    fi
    
    if [ -f "/usr/local/openresty/nginx/conf/lua/predict_makespan.lua" ]; then
        print_success "✅ predict_makespan.lua exists"
    else
        print_error "❌ predict_makespan.lua not found"
    fi
    
    if [ -f "/usr/local/openresty/nginx/conf/lua/predict_vm_cluster.lua" ]; then
        print_success "✅ predict_vm_cluster.lua exists"
    else
        print_error "❌ predict_vm_cluster.lua not found"
    fi
else
    print_warning "⚠️ Health endpoint test failed"
    echo "Response: $response"
fi

# Step 6: Test ML service endpoints
print_header "Step 6: Test ML Service Endpoints"
print_status "Testing ML service endpoints..."

# Test SVM prediction
response=$(curl -s -X POST http://localhost/predict/makespan \
    -H "Content-Type: application/json" \
    -d '{"features": [4, 8, 100, 1000, 3]}' 2>/dev/null || echo "ERROR")

if echo "$response" | grep -q "prediction"; then
    print_success "✅ SVM prediction endpoint working"
else
    print_warning "⚠️ SVM prediction endpoint test failed"
    echo "Response: $response"
fi

# Test K-Means prediction
response=$(curl -s -X POST http://localhost/predict/vm_cluster \
    -H "Content-Type: application/json" \
    -d '{"vm_features": [0.5, 0.5, 0.5]}' 2>/dev/null || echo "ERROR")

if echo "$response" | grep -q "cluster"; then
    print_success "✅ K-Means prediction endpoint working"
else
    print_warning "⚠️ K-Means prediction endpoint test failed"
    echo "Response: $response"
fi

# Test MCCVA routing
response=$(curl -s -X POST http://localhost/mccva/route \
    -H "Content-Type: application/json" \
    -d '{"features": [4, 8, 100, 1000, 3], "vm_features": [0.5, 0.5, 0.5]}' 2>/dev/null || echo "ERROR")

if echo "$response" | grep -q "target_vm"; then
    print_success "✅ MCCVA routing endpoint working"
else
    print_warning "⚠️ MCCVA routing endpoint test failed"
    echo "Response: $response"
fi

# Step 7: Check lua files
print_header "Step 7: Check Lua Files"
print_status "Checking lua files in nginx conf directory..."

LUA_DIR="/usr/local/openresty/nginx/conf/lua"
if [ -d "$LUA_DIR" ]; then
    print_success "✅ Lua directory exists: $LUA_DIR"
    ls -la "$LUA_DIR"
else
    print_error "❌ Lua directory not found: $LUA_DIR"
    
    # Create lua directory and copy files
    print_status "Creating lua directory and copying files..."
    sudo mkdir -p "$LUA_DIR"
    
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
    
    # Restart nginx again
    sudo systemctl restart openresty
    sleep 3
fi

print_header "Fix Complete"
print_success "✅ Health endpoint and nginx configuration have been checked!"
print_status "You can now test the endpoints:"
print_status "  curl http://localhost/health"
print_status "  ~/mccva_manage.sh demo"
print_status "  ~/mccva_manage.sh test" 