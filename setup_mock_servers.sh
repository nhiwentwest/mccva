#!/bin/bash

# Setup Mock Servers for MCCVA
# Chạy script này để setup mock servers service trên server hiện tại

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

print_header "Setup Mock Servers for MCCVA"

USER=$(whoami)
PROJECT_DIR="/opt/mccva"

print_status "Setting up mock servers service..."

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    print_error "Project directory not found: $PROJECT_DIR"
    exit 1
fi

# Check if mock_servers.py exists
if [ ! -f "$PROJECT_DIR/mock_servers.py" ]; then
    print_error "mock_servers.py not found in $PROJECT_DIR"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$PROJECT_DIR/venv" ]; then
    print_error "Virtual environment not found: $PROJECT_DIR/venv"
    exit 1
fi

print_success "✅ Prerequisites check passed"

# Step 1: Create systemd service
print_header "Step 1: Create Systemd Service"
print_status "Creating systemd service for mock servers..."

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

# Step 2: Enable and start service
print_header "Step 2: Enable and Start Service"
print_status "Enabling and starting mock servers service..."

sudo systemctl daemon-reload
sudo systemctl enable mccva-mock-servers
sudo systemctl start mccva-mock-servers

print_success "✅ Mock servers service started"

# Step 3: Check service status
print_header "Step 3: Check Service Status"
print_status "Checking service status..."
sleep 10

if systemctl is-active --quiet mccva-mock-servers; then
    print_success "✅ Mock servers service is running"
else
    print_error "❌ Mock servers service failed to start"
    sudo systemctl status mccva-mock-servers --no-pager
    sudo journalctl -u mccva-mock-servers --no-pager -n 20
    exit 1
fi

# Step 4: Test mock servers
print_header "Step 4: Test Mock Servers"
print_status "Testing mock servers..."

MOCK_PORTS=(8081 8082 8083 8084 8085 8086 8087 8088)
MOCK_NAMES=("VM-Low-1" "VM-Low-2" "VM-Medium-1" "VM-Medium-2" "VM-High-1" "VM-High-2" "VM-Balanced-1" "VM-Balanced-2")

for i in "${!MOCK_PORTS[@]}"; do
    port=${MOCK_PORTS[$i]}
    name=${MOCK_NAMES[$i]}
    
    print_status "Testing $name (port $port)..."
    
    # Wait a bit for server to start
    sleep 2
    
    response=$(curl -s http://localhost:$port/health 2>/dev/null || echo "ERROR")
    
    if echo "$response" | grep -q "$name"; then
        print_success "✅ $name (port $port) is responding"
    else
        print_warning "⚠️ $name (port $port) test failed"
        echo "Response: $response"
    fi
done

# Step 5: Test processing endpoints
print_header "Step 5: Test Processing Endpoints"
print_status "Testing processing endpoints..."

for port in "${MOCK_PORTS[@]}"; do
    response=$(curl -s -X POST http://localhost:$port/process \
        -H "Content-Type: application/json" \
        -d '{"test": "data"}' 2>/dev/null || echo "ERROR")
    
    if echo "$response" | grep -q "processed"; then
        print_success "✅ Port $port processing working"
    else
        print_warning "⚠️ Port $port processing test failed"
    fi
done

# Step 6: Update management script
print_header "Step 6: Update Management Script"
print_status "Updating management script..."

if [ -f "/home/$USER/mccva_manage.sh" ]; then
    print_success "✅ Management script exists"
else
    print_warning "⚠️ Management script not found"
fi

print_header "Setup Complete"
print_success "✅ Mock servers have been set up successfully!"
print_status "Services running:"
print_status "  • MCCVA ML Service: $(systemctl is-active mccva-ml)"
print_status "  • Mock Servers: $(systemctl is-active mccva-mock-servers)"
print_status "  • OpenResty: $(systemctl is-active openresty)"

print_status "Management commands:"
print_status "  • Status: ~/mccva_manage.sh status"
print_status "  • Test: ~/mccva_manage.sh test"
print_status "  • Demo: ~/mccva_manage.sh demo"
print_status "  • Mock logs: sudo journalctl -u mccva-mock-servers -f"

print_status "Mock servers are now ready for MCCVA testing!" 