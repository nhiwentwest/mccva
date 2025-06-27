#!/bin/bash

# MCCVA Deployment Script
# Triển khai thuật toán MCCVA với OpenResty trên Amazon Cloud Ubuntu
# Đảm bảo tất cả dependencies hoạt động hoàn hảo

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root"
    exit 1
fi

print_header "MCCVA Algorithm Deployment"
print_status "Starting deployment of MCCVA (Makespan Classification & Clustering VM Algorithm)"
print_status "Ensuring all dependencies work perfectly on Ubuntu"

# Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential
sudo apt install -y curl wget git unzip
sudo apt install -y libblas-dev liblapack-dev libatlas-base-dev gfortran

# Install OpenResty
print_status "Installing OpenResty..."
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:openresty/ppa
sudo apt update
sudo apt install -y openresty

# Create application directory
print_status "Creating application directory..."
sudo mkdir -p /opt/mccva
sudo chown $USER:$USER /opt/mccva

# Copy application files
print_status "Copying application files..."
cp -r models /opt/mccva/
cp ml_service.py /opt/mccva/
cp requirements.txt /opt/mccva/

# Create Python virtual environment
print_status "Setting up Python virtual environment..."
cd /opt/mccva
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel
print_status "Upgrading pip and installing build tools..."
pip install --upgrade pip
pip install wheel setuptools

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Verify model files exist
print_status "Verifying model files..."
if [ ! -f "models/svm_model.joblib" ]; then
    print_error "SVM model file not found!"
    exit 1
fi
if [ ! -f "models/kmeans_model.joblib" ]; then
    print_error "K-Means model file not found!"
    exit 1
fi
if [ ! -f "models/scaler.joblib" ]; then
    print_error "SVM scaler file not found!"
    exit 1
fi
if [ ! -f "models/kmeans_scaler.joblib" ]; then
    print_error "K-Means scaler file not found!"
    exit 1
fi
print_status "✅ All model files verified!"

# Test model loading
print_status "Testing model loading..."
python3 -c "
import joblib
import sys
try:
    svm_model = joblib.load('models/svm_model.joblib')
    kmeans_model = joblib.load('models/kmeans_model.joblib')
    svm_scaler = joblib.load('models/scaler.joblib')
    kmeans_scaler = joblib.load('models/kmeans_scaler.joblib')
    print('✅ All models loaded successfully!')
    print(f'SVM Model: {svm_model.kernel} kernel')
    print(f'K-Means Model: {kmeans_model.n_clusters} clusters')
except Exception as e:
    print(f'❌ Error loading models: {e}')
    sys.exit(1)
"

# Create systemd service for ML service
print_status "Creating systemd service for ML service..."
sudo tee /etc/systemd/system/mccva-ml.service > /dev/null <<EOF
[Unit]
Description=MCCVA ML Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/mccva
Environment=PATH=/opt/mccva/venv/bin
Environment=PYTHONPATH=/opt/mccva
ExecStart=/opt/mccva/venv/bin/python ml_service.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Setup Lua files directory
print_status "Setting up Lua files..."
sudo mkdir -p /usr/local/openresty/nginx/conf/lua
sudo cp lua/mccva_routing.lua /usr/local/openresty/nginx/conf/lua/
sudo cp lua/predict_makespan.lua /usr/local/openresty/nginx/conf/lua/
sudo cp lua/predict_vm_cluster.lua /usr/local/openresty/nginx/conf/lua/

# Backup original nginx config
print_status "Backing up original nginx configuration..."
sudo cp /usr/local/openresty/nginx/conf/nginx.conf /usr/local/openresty/nginx/conf/nginx.conf.backup

# Copy new nginx configuration
print_status "Installing new nginx configuration..."
sudo cp nginx.conf /usr/local/openresty/nginx/conf/nginx.conf

# Create log and web root directories
print_status "Creating log and web root directories..."
sudo mkdir -p /var/log/nginx
sudo mkdir -p /usr/local/openresty/nginx/html
sudo mkdir -p /var/www/html
sudo chown -R $USER:$USER /var/log/nginx

# Test nginx configuration
print_status "Testing nginx configuration..."
sudo /usr/local/openresty/nginx/sbin/nginx -t

# Start services
print_status "Starting services..."
sudo systemctl daemon-reload
sudo systemctl enable mccva-ml
sudo systemctl start mccva-ml
sudo systemctl enable openresty
sudo systemctl start openresty

# Wait for services to start
print_status "Waiting for services to start..."
sleep 10

# Check service status
print_status "Checking service status..."
if systemctl is-active --quiet mccva-ml; then
    print_status "✅ MCCVA ML Service is running"
else
    print_error "❌ MCCVA ML Service failed to start"
    sudo systemctl status mccva-ml
    sudo journalctl -u mccva-ml --no-pager -n 20
    exit 1
fi

if systemctl is-active --quiet openresty; then
    print_status "✅ OpenResty is running"
else
    print_error "❌ OpenResty failed to start"
    sudo systemctl status openresty
    exit 1
fi

# Test endpoints
print_status "Testing MCCVA endpoints..."

# Test health endpoint
print_status "Testing health endpoint..."
if curl -s http://localhost/health | grep -q "mccva-openresty-gateway"; then
    print_status "✅ Health endpoint is working"
else
    print_warning "⚠️ Health endpoint test failed"
fi

# Test ML service
print_status "Testing ML service..."
if curl -s http://localhost:5000/health | grep -q "healthy"; then
    print_status "✅ ML Service is responding"
else
    print_warning "⚠️ ML Service test failed"
    curl -s http://localhost:5000/health || echo "Cannot connect to ML service"
fi

# Test model predictions
print_status "Testing model predictions..."
if curl -s -X POST http://localhost/predict/makespan \
    -H "Content-Type: application/json" \
    -d '{"features": [4, 8, 100, 1000, 3]}' | grep -q "makespan"; then
    print_status "✅ SVM prediction working"
else
    print_warning "⚠️ SVM prediction test failed"
fi

if curl -s -X POST http://localhost/predict/vm_cluster \
    -H "Content-Type: application/json" \
    -d '{"vm_features": [0.5, 0.5, 0.5]}' | grep -q "cluster"; then
    print_status "✅ K-Means prediction working"
else
    print_warning "⚠️ K-Means prediction test failed"
fi

# Configure firewall if active
if command -v ufw &> /dev/null; then
    print_status "Configuring firewall..."
    sudo ufw allow 80/tcp
    sudo ufw allow 5000/tcp
    print_status "✅ Firewall configured"
fi

print_header "MCCVA Deployment Complete!"

print_status "Deployment Information:"
echo "  • ML Service: http://localhost:5000"
echo "  • OpenResty Gateway: http://localhost"
echo "  • MCCVA Routing: POST http://localhost/mccva/route"
echo "  • Health Check: http://localhost/health"
echo "  • SVM Prediction: POST http://localhost/predict/makespan"
echo "  • K-Means Prediction: POST http://localhost/predict/vm_cluster"

print_status "Service Management:"
echo "  • ML Service: sudo systemctl {start|stop|restart|status} mccva-ml"
echo "  • OpenResty: sudo systemctl {start|stop|restart|status} openresty"
echo "  • View logs: sudo journalctl -u mccva-ml -f"

print_status "Testing MCCVA Algorithm:"
echo "  • Test makespan prediction:"
echo "    curl -X POST http://localhost/predict/makespan \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"features\": [4, 8, 100, 1000, 3]}'"
echo ""
echo "  • Test VM clustering:"
echo "    curl -X POST http://localhost/predict/vm_cluster \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"vm_features\": [0.5, 0.5, 0.5]}'"
echo ""
echo "  • Test MCCVA routing:"
echo "    curl -X POST http://localhost/mccva/route \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"features\": [4, 8, 100, 1000, 3], \"vm_features\": [0.5, 0.5, 0.5]}'"

print_status "✅ MCCVA Algorithm successfully deployed!"
print_status "🎯 The system is ready to handle load balancing using AI-based VM selection."
print_status "📊 All models loaded and verified successfully!" 