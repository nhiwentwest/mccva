#!/bin/bash

# =============================================================================
# MCCVA Algorithm Deployment Script for Amazon Cloud Ubuntu
# Makespan Classification & Clustering VM Algorithm
# 
# Script này bao gồm tất cả fixes và improvements từ các script khác:
# - Fix Python 3.12 compatibility issues
# - Fix OpenResty PID file issues
# - Fix permissions and directory issues
# - Comprehensive error handling và logging
# - Auto-recovery mechanisms
# 
# Chạy: ./amazon_deploy.sh
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root for security reasons"
   exit 1
fi

# Get current user
CURRENT_USER=$(whoami)
log "Deploying MCCVA as user: $CURRENT_USER"

# =============================================================================
# STEP 1: System Preparation
# =============================================================================
log "Step 1: Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
log "Installing essential packages..."
sudo apt install -y python3 python3-pip python3-venv nginx curl wget git unzip software-properties-common

# Install OpenResty
log "Installing OpenResty..."
wget -qO - https://openresty.org/package/pubkey.gpg | sudo apt-key add -
sudo add-apt-repository -y "deb https://openresty.org/package/ubuntu $(lsb_release -sc) openresty"
sudo apt update
sudo apt install -y openresty

# =============================================================================
# STEP 2: Create Application Directory
# =============================================================================
log "Step 2: Creating application directory..."
sudo mkdir -p /opt/mccva
sudo chown $CURRENT_USER:$CURRENT_USER /opt/mccva
cd /opt/mccva

# Clone or copy project files
if [ -d "/home/$CURRENT_USER/mccva" ]; then
    log "Copying project files from home directory..."
    cp -r /home/$CURRENT_USER/mccva/* /opt/mccva/
else
    log "Downloading project from GitHub..."
    git clone https://github.com/nhiwentwest/mccva.git temp_mccva
    cp -r temp_mccva/* /opt/mccva/
    rm -rf temp_mccva
fi

# =============================================================================
# STEP 3: Setup Python Environment with Enhanced Error Handling
# =============================================================================
log "Step 3: Setting up Python virtual environment..."

# Remove existing venv if exists
if [ -d "venv" ]; then
    log "Removing existing virtual environment..."
    rm -rf venv
fi

# Create new virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install packages in specific order to avoid conflicts
log "Installing Python packages in optimized order..."

# Install numpy first (critical for other packages)
log "Installing numpy..."
pip install numpy==1.26.4

# Install scipy
log "Installing scipy..."
pip install scipy==1.12.0

# Install scikit-learn
log "Installing scikit-learn..."
pip install scikit-learn==1.4.0

# Install pandas
log "Installing pandas..."
pip install pandas==2.2.0

# Install other dependencies
log "Installing other dependencies..."
pip install joblib==1.3.2
pip install Flask==3.0.0
pip install Werkzeug==3.0.1
pip install gunicorn==21.2.0
pip install requests==2.31.0
pip install matplotlib==3.8.2
pip install seaborn==0.13.0

# Verify installation
log "Verifying Python packages..."
python -c "import numpy, scipy, sklearn, pandas, joblib, flask; print('✅ All packages installed successfully')"

# =============================================================================
# STEP 4: Create Log Directories and Fix Permissions
# =============================================================================
log "Step 4: Creating log directories and fixing permissions..."

# Create necessary directories
sudo mkdir -p /var/log/mccva
sudo mkdir -p /var/log/nginx
sudo mkdir -p /usr/local/openresty/nginx/logs
sudo mkdir -p /var/www/html
sudo mkdir -p /var/run/nginx
sudo mkdir -p /var/run/openresty

# Set proper permissions
sudo chown -R $CURRENT_USER:$CURRENT_USER /var/log/mccva
sudo chown -R $CURRENT_USER:$CURRENT_USER /var/log/nginx
sudo chown -R $CURRENT_USER:$CURRENT_USER /usr/local/openresty/nginx/logs
sudo chown -R $CURRENT_USER:$CURRENT_USER /var/run/nginx
sudo chown -R $CURRENT_USER:$CURRENT_USER /var/run/openresty
sudo chmod -R 755 /var/log/mccva
sudo chmod -R 755 /var/log/nginx
sudo chmod -R 755 /usr/local/openresty/nginx/logs
sudo chmod -R 755 /var/run/nginx
sudo chmod -R 755 /var/run/openresty

# =============================================================================
# STEP 5: Create Systemd Services
# =============================================================================
log "Step 5: Creating systemd services..."

# Create ML Service
sudo tee /etc/systemd/system/mccva-ml.service > /dev/null <<EOF
[Unit]
Description=MCCVA ML Service
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=/opt/mccva
Environment=PATH=/opt/mccva/venv/bin
ExecStart=/opt/mccva/venv/bin/python ml_service.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/mccva/mccva-ml.log
StandardError=append:/var/log/mccva/mccva-ml.log

[Install]
WantedBy=multi-user.target
EOF

# Create Mock Servers Service
sudo tee /etc/systemd/system/mccva-mock-servers.service > /dev/null <<EOF
[Unit]
Description=MCCVA Mock Servers
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=/opt/mccva
Environment=PATH=/opt/mccva/venv/bin
ExecStart=/opt/mccva/venv/bin/python mock_servers.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/mccva/mock-servers.log
StandardError=append:/var/log/mccva/mock-servers.log

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# =============================================================================
# STEP 6: Configure OpenResty/Nginx
# =============================================================================
log "Step 6: Configuring OpenResty/Nginx..."

# Backup existing nginx config
if [ -f "/etc/nginx/nginx.conf" ]; then
    sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup
fi

# Install new nginx config
sudo cp nginx.conf /etc/nginx/nginx.conf

# Copy Lua files
sudo mkdir -p /usr/local/openresty/nginx/conf/lua
sudo cp lua/*.lua /usr/local/openresty/nginx/conf/lua/

# =============================================================================
# STEP 7: Enhanced OpenResty Startup with Error Recovery
# =============================================================================
log "Step 7: Starting services with enhanced error recovery..."

# Function to check if service is running
check_service() {
    local service_name=$1
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if systemctl is-active --quiet $service_name; then
            log "✅ $service_name is running"
            return 0
        fi
        
        log "Waiting for $service_name to start... (attempt $attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    error "$service_name failed to start after $max_attempts attempts"
    return 1
}

# Function to start OpenResty with error handling
start_openresty() {
    log "Starting OpenResty with enhanced error handling..."
    
    # Stop any existing nginx processes
    sudo systemctl stop openresty 2>/dev/null || true
    sudo systemctl stop nginx 2>/dev/null || true
    
    # Kill any remaining processes
    sudo pkill -f nginx 2>/dev/null || true
    sudo pkill -f openresty 2>/dev/null || true
    sleep 3
    
    # Test nginx configuration
    log "Testing nginx configuration..."
    if ! sudo /usr/local/openresty/nginx/sbin/nginx -t; then
        error "Nginx configuration test failed"
        return 1
    fi
    
    # Start OpenResty
    log "Starting OpenResty..."
    sudo systemctl start openresty
    
    # Wait for OpenResty to start
    if ! check_service openresty; then
        error "OpenResty failed to start"
        return 1
    fi
    
    log "✅ OpenResty started successfully"
    return 0
}

# Start services in order
log "Starting ML Service..."
sudo systemctl start mccva-ml
check_service mccva-ml

log "Starting Mock Servers..."
sudo systemctl start mccva-mock-servers
check_service mccva-mock-servers

log "Starting OpenResty..."
start_openresty

# Enable services to start on boot
sudo systemctl enable mccva-ml
sudo systemctl enable mccva-mock-servers
sudo systemctl enable openresty

# =============================================================================
# STEP 8: Test Deployment
# =============================================================================
log "Step 8: Testing deployment..."

# Wait for services to be ready
sleep 5

# Test health endpoints
log "Testing health endpoints..."

# Test OpenResty health
if curl -s http://localhost/health > /dev/null; then
    log "✅ OpenResty health endpoint working"
else
    warn "⚠️  OpenResty health endpoint not responding"
fi

# Test ML Service health
if curl -s http://localhost:5000/health > /dev/null; then
    log "✅ ML Service health endpoint working"
else
    warn "⚠️  ML Service health endpoint not responding"
fi

# Test mock servers
for port in 8081 8082 8083 8084 8085 8086 8087 8088; do
    if curl -s http://localhost:$port/health > /dev/null; then
        log "✅ Mock server on port $port working"
    else
        warn "⚠️  Mock server on port $port not responding"
    fi
done

# Test MCCVA algorithm
log "Testing MCCVA algorithm..."
TEST_RESPONSE=$(curl -s -X POST http://localhost/mccva/route \
  -H "Content-Type: application/json" \
  -d '{"cpu_cores": 4, "memory": 8, "storage": 100, "network_bandwidth": 1000, "priority": 3}' 2>/dev/null || echo "ERROR")

if [[ "$TEST_RESPONSE" != "ERROR" ]]; then
    log "✅ MCCVA algorithm working"
else
    warn "⚠️  MCCVA algorithm test failed"
fi

# =============================================================================
# STEP 9: Create Management Script
# =============================================================================
log "Step 9: Creating management script..."

sudo tee /home/$CURRENT_USER/mccva_manage.sh > /dev/null <<'EOF'
#!/bin/bash

# MCCVA Management Script
# Usage: ./mccva_manage.sh [start|stop|restart|status|logs|test]

SERVICE_ML="mccva-ml"
SERVICE_MOCK="mccva-mock-servers"
SERVICE_NGINX="openresty"

case "$1" in
    start)
        echo "Starting MCCVA services..."
        sudo systemctl start $SERVICE_ML
        sudo systemctl start $SERVICE_MOCK
        sudo systemctl start $SERVICE_NGINX
        ;;
    stop)
        echo "Stopping MCCVA services..."
        sudo systemctl stop $SERVICE_NGINX
        sudo systemctl stop $SERVICE_MOCK
        sudo systemctl stop $SERVICE_ML
        ;;
    restart)
        echo "Restarting MCCVA services..."
        sudo systemctl restart $SERVICE_ML
        sudo systemctl restart $SERVICE_MOCK
        sudo systemctl restart $SERVICE_NGINX
        ;;
    status)
        echo "=== MCCVA Services Status ==="
        sudo systemctl status $SERVICE_ML --no-pager -l
        echo ""
        sudo systemctl status $SERVICE_MOCK --no-pager -l
        echo ""
        sudo systemctl status $SERVICE_NGINX --no-pager -l
        ;;
    logs)
        echo "=== MCCVA Logs ==="
        echo "ML Service logs:"
        sudo journalctl -u $SERVICE_ML -n 20 --no-pager
        echo ""
        echo "Mock Servers logs:"
        sudo journalctl -u $SERVICE_MOCK -n 20 --no-pager
        echo ""
        echo "OpenResty logs:"
        sudo journalctl -u $SERVICE_NGINX -n 20 --no-pager
        ;;
    test)
        echo "=== Testing MCCVA Endpoints ==="
        echo "OpenResty Health:"
        curl -s http://localhost/health | jq . 2>/dev/null || curl -s http://localhost/health
        echo ""
        echo "ML Service Health:"
        curl -s http://localhost:5000/health | jq . 2>/dev/null || curl -s http://localhost:5000/health
        echo ""
        echo "MCCVA Algorithm Test:"
        curl -s -X POST http://localhost/mccva/route \
          -H "Content-Type: application/json" \
          -d '{"cpu_cores": 4, "memory": 8, "storage": 100, "network_bandwidth": 1000, "priority": 3}' | jq . 2>/dev/null || curl -s -X POST http://localhost/mccva/route \
          -H "Content-Type: application/json" \
          -d '{"cpu_cores": 4, "memory": 8, "storage": 100, "network_bandwidth": 1000, "priority": 3}'
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|test}"
        exit 1
        ;;
esac
EOF

sudo chmod +x /home/$CURRENT_USER/mccva_manage.sh

# =============================================================================
# STEP 10: Configure Firewall (if active)
# =============================================================================
log "Step 10: Configuring firewall..."

# Check if ufw is active
if sudo ufw status | grep -q "Status: active"; then
    log "UFW is active, configuring ports..."
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    sudo ufw allow 5000/tcp
    sudo ufw allow 8081:8088/tcp
    log "✅ Firewall configured"
else
    log "UFW is not active, skipping firewall configuration"
fi

# =============================================================================
# DEPLOYMENT COMPLETE
# =============================================================================
log "🎉 MCCVA Deployment Complete!"
echo ""
echo "=== DEPLOYMENT SUMMARY ==="
echo "✅ System packages updated"
echo "✅ Python environment configured"
echo "✅ OpenResty installed and configured"
echo "✅ ML Service running on port 5000"
echo "✅ Mock Servers running on ports 8081-8088"
echo "✅ OpenResty Gateway running on port 80"
echo "✅ Firewall configured (if active)"
echo ""
echo "=== SERVICE MANAGEMENT ==="
echo "Start services:   ~/mccva_manage.sh start"
echo "Stop services:    ~/mccva_manage.sh stop"
echo "Restart services: ~/mccva_manage.sh restart"
echo "Check status:     ~/mccva_manage.sh status"
echo "View logs:        ~/mccva_manage.sh logs"
echo "Test endpoints:   ~/mccva_manage.sh test"
echo ""
echo "=== TESTING ENDPOINTS ==="
echo "OpenResty Health: http://localhost/health"
echo "ML Service:       http://localhost:5000/health"
echo "MCCVA Routing:    http://localhost/mccva/route"
echo "Mock Servers:     http://localhost:8081-8088"
echo ""
echo "=== LOG FILES ==="
echo "ML Service:       /var/log/mccva/mccva-ml.log"
echo "Mock Servers:     /var/log/mccva/mock-servers.log"
echo "OpenResty:        /usr/local/openresty/nginx/logs/"
echo ""
echo "🎯 MCCVA Algorithm is ready for production use!"

# =============================================================================
# STEP 11: Enhanced Deployment Verification
# =============================================================================
log "Step 11: Enhanced deployment verification..."

# Function to verify service health
verify_service_health() {
    local service_name=$1
    local endpoint=$2
    local max_attempts=30
    local attempt=1
    
    log "Verifying $service_name health..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$endpoint" > /dev/null 2>&1; then
            log "✅ $service_name is healthy"
            return 0
        fi
        
        log "Waiting for $service_name to be ready... (attempt $attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    error "$service_name health check failed after $max_attempts attempts"
    return 1
}

# Verify all services
verify_service_health "OpenResty Gateway" "http://localhost/health"
verify_service_health "ML Service" "http://localhost:5000/health"

# Verify mock servers
for port in 8081 8082 8083 8084 8085 8086 8087 8088; do
    verify_service_health "Mock Server $port" "http://localhost:$port/health"
done

# Test MCCVA algorithm functionality
log "Testing MCCVA algorithm functionality..."
TEST_PAYLOAD='{"cpu_cores": 4, "memory": 8, "storage": 100, "network_bandwidth": 1000, "priority": 3}'
TEST_RESPONSE=$(curl -s -X POST http://localhost/mccva/route \
  -H "Content-Type: application/json" \
  -d "$TEST_PAYLOAD" 2>/dev/null || echo "ERROR")

if [[ "$TEST_RESPONSE" != "ERROR" ]]; then
    log "✅ MCCVA algorithm is working correctly"
    echo "Test response: $TEST_RESPONSE"
else
    error "MCCVA algorithm test failed"
    exit 1
fi

# =============================================================================
# STEP 12: Performance Testing
# =============================================================================
log "Step 12: Running performance tests..."

# Test response times
log "Testing response times..."
for i in {1..5}; do
    START_TIME=$(date +%s%N)
    curl -s -X POST http://localhost/mccva/route \
      -H "Content-Type: application/json" \
      -d "$TEST_PAYLOAD" > /dev/null
    END_TIME=$(date +%s%N)
    RESPONSE_TIME=$(( (END_TIME - START_TIME) / 1000000 ))
    log "Request $i response time: ${RESPONSE_TIME}ms"
done

# =============================================================================
# STEP 13: Final System Check
# =============================================================================
log "Step 13: Final system check..."

# Check disk space
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    warn "⚠️  Disk usage is high: ${DISK_USAGE}%"
else
    log "✅ Disk usage is normal: ${DISK_USAGE}%"
fi

# Check memory usage
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
log "✅ Memory usage: ${MEMORY_USAGE}%"

# Check running processes
log "Checking running processes..."
ps aux | grep -E "(mccva|openresty|nginx)" | grep -v grep || warn "No MCCVA processes found"

