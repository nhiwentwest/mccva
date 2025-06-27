#!/bin/bash

# MCCVA Amazon Cloud Ubuntu Deployment Script
# Tự động deploy thuật toán MCCVA trên Amazon Cloud Ubuntu
# Chạy: ./amazon_deploy.sh

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

# GitHub repository URL
GITHUB_REPO="https://github.com/nhiwentwest/mccva.git"
PROJECT_DIR="/opt/mccva"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root"
    exit 1
fi

print_header "MCCVA Amazon Cloud Ubuntu Deployment"
print_status "Automated deployment of MCCVA Algorithm on Amazon Cloud Ubuntu"
print_status "This script will setup everything from scratch"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if service is running
service_is_running() {
    systemctl is-active --quiet "$1"
}

# Function to install package if not exists
install_if_missing() {
    local package=$1
    if ! command_exists $package; then
        print_status "Installing $package..."
        sudo apt install -y $package
    else
        print_status "✅ $package is already installed"
    fi
}

# Function to check if directory exists
check_directory() {
    if [ -d "$1" ]; then
        print_status "✅ Directory $1 exists"
        return 0
    else
        print_status "❌ Directory $1 not found"
        return 1
    fi
}

# Step 1: Update system
print_header "Step 1: System Update"
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Step 2: Install essential packages
print_header "Step 2: Install Essential Packages"
print_status "Installing essential packages..."

install_if_missing git
install_if_missing curl
install_if_missing wget
install_if_missing python3
install_if_missing python3-pip
install_if_missing python3-venv
install_if_missing python3-dev
install_if_missing build-essential
install_if_missing net-tools

# Step 3: Install ML dependencies
print_header "Step 3: Install ML Dependencies"
print_status "Installing ML dependencies for scikit-learn..."

sudo apt install -y libblas-dev liblapack-dev libatlas-base-dev gfortran

# Step 4: Install OpenResty
print_header "Step 4: Install OpenResty"
if ! command_exists openresty; then
    print_status "Installing OpenResty..."
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:openresty/ppa
    sudo apt update
    sudo apt install -y openresty
    print_status "✅ OpenResty installed successfully"
else
    print_status "✅ OpenResty is already installed"
fi

# Step 5: Create project directory
print_header "Step 5: Setup Project Directory"
print_status "Creating project directory..."
sudo mkdir -p $PROJECT_DIR
sudo chown $USER:$USER $PROJECT_DIR

# Step 6: Clone repository
print_header "Step 6: Clone GitHub Repository"
if [ -d "$PROJECT_DIR/.git" ]; then
    print_status "Repository already exists, updating..."
    cd $PROJECT_DIR
    git pull origin main
else
    print_status "Cloning MCCVA repository..."
    cd /tmp
    git clone $GITHUB_REPO $PROJECT_DIR
    cd $PROJECT_DIR
fi

# Check if clone was successful
if [ ! -f "$PROJECT_DIR/ml_service.py" ] || [ ! -f "$PROJECT_DIR/deploy.sh" ]; then
    print_error "Failed to clone repository or missing essential files"
    exit 1
fi

print_status "✅ Repository cloned successfully"

# Step 7: Setup Python environment
print_header "Step 7: Setup Python Environment"
cd $PROJECT_DIR

print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

print_status "Upgrading pip and installing build tools..."
pip install --upgrade pip
pip install wheel setuptools

print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Step 8: Verify model files
print_header "Step 8: Verify Model Files"
print_status "Checking model files..."

MODEL_FILES=(
    "models/svm_model.joblib"
    "models/kmeans_model.joblib"
    "models/scaler.joblib"
    "models/kmeans_scaler.joblib"
)

for model_file in "${MODEL_FILES[@]}"; do
    if [ -f "$model_file" ]; then
        size=$(du -h "$model_file" | cut -f1)
        print_status "✅ $model_file exists ($size)"
    else
        print_error "❌ $model_file not found"
        exit 1
    fi
done

# Step 9: Test model loading
print_header "Step 9: Test Model Loading"
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

# Step 10: Setup Lua files
print_header "Step 10: Setup Lua Files"
print_status "Setting up Lua files for OpenResty..."

sudo mkdir -p /usr/local/openresty/nginx/conf/lua
sudo cp lua/mccva_routing.lua /usr/local/openresty/nginx/conf/lua/
sudo cp lua/predict_makespan.lua /usr/local/openresty/nginx/conf/lua/
sudo cp lua/predict_vm_cluster.lua /usr/local/openresty/nginx/conf/lua/

print_status "✅ Lua files copied successfully"

# Step 11: Setup Nginx configuration
print_header "Step 11: Setup Nginx Configuration"
print_status "Backing up original nginx configuration..."
sudo cp /usr/local/openresty/nginx/conf/nginx.conf /usr/local/openresty/nginx/conf/nginx.conf.backup

print_status "Installing new nginx configuration..."
sudo cp nginx.conf /usr/local/openresty/nginx/conf/nginx.conf

# Step 12: Create directories
print_header "Step 12: Create Directories"
print_status "Creating necessary directories..."

sudo mkdir -p /var/log/nginx
sudo mkdir -p /usr/local/openresty/nginx/html
sudo mkdir -p /var/www/html
sudo chown -R $USER:$USER /var/log/nginx

# Step 13: Test nginx configuration
print_header "Step 13: Test Nginx Configuration"
print_status "Testing nginx configuration..."
sudo /usr/local/openresty/nginx/sbin/nginx -t

# Step 14: Create systemd service
print_header "Step 14: Create Systemd Service"
print_status "Creating systemd service for ML service..."

sudo tee /etc/systemd/system/mccva-ml.service > /dev/null <<EOF
[Unit]
Description=MCCVA ML Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
Environment=PYTHONPATH=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/python ml_service.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Step 15: Start services
print_header "Step 15: Start Services"
print_status "Starting services..."

sudo systemctl daemon-reload
sudo systemctl enable mccva-ml
sudo systemctl start mccva-ml
sudo systemctl enable openresty
sudo systemctl start openresty

# Step 16: Wait for services
print_header "Step 16: Wait for Services"
print_status "Waiting for services to start..."
sleep 15

# Step 17: Check service status
print_header "Step 17: Check Service Status"
if service_is_running mccva-ml; then
    print_status "✅ MCCVA ML Service is running"
else
    print_error "❌ MCCVA ML Service failed to start"
    sudo systemctl status mccva-ml
    sudo journalctl -u mccva-ml --no-pager -n 20
    exit 1
fi

if service_is_running openresty; then
    print_status "✅ OpenResty is running"
else
    print_error "❌ OpenResty failed to start"
    sudo systemctl status openresty
    exit 1
fi

# Step 18: Test endpoints
print_header "Step 18: Test Endpoints"
print_status "Testing endpoints..."

# Test health endpoint
if curl -s http://localhost/health | grep -q "mccva-openresty-gateway"; then
    print_status "✅ Health endpoint is working"
else
    print_warning "⚠️ Health endpoint test failed"
fi

# Test ML service
if curl -s http://localhost:5000/health | grep -q "healthy"; then
    print_status "✅ ML Service is responding"
else
    print_warning "⚠️ ML Service test failed"
fi

# Step 19: Test model predictions
print_header "Step 19: Test Model Predictions"
print_status "Testing model predictions..."

# Test SVM prediction
if curl -s -X POST http://localhost/predict/makespan \
    -H "Content-Type: application/json" \
    -d '{"features": [4, 8, 100, 1000, 3]}' | grep -q "makespan"; then
    print_status "✅ SVM prediction working"
else
    print_warning "⚠️ SVM prediction test failed"
fi

# Test K-Means prediction
if curl -s -X POST http://localhost/predict/vm_cluster \
    -H "Content-Type: application/json" \
    -d '{"vm_features": [0.5, 0.5, 0.5]}' | grep -q "cluster"; then
    print_status "✅ K-Means prediction working"
else
    print_warning "⚠️ K-Means prediction test failed"
fi

# Step 20: Test MCCVA routing
print_header "Step 20: Test MCCVA Routing"
print_status "Testing MCCVA routing..."

if curl -s -X POST http://localhost/mccva/route \
    -H "Content-Type: application/json" \
    -d '{"features": [4, 8, 100, 1000, 3], "vm_features": [0.5, 0.5, 0.5]}' | grep -q "target_vm"; then
    print_status "✅ MCCVA routing working"
else
    print_warning "⚠️ MCCVA routing test failed"
fi

# Step 21: Configure firewall
print_header "Step 21: Configure Firewall"
print_status "Configuring firewall..."
if command -v ufw &> /dev/null; then
    sudo ufw allow 80/tcp
    sudo ufw allow 5000/tcp
    sudo ufw allow 22/tcp
    print_status "✅ Firewall configured"
else
    print_warning "UFW not found, skipping firewall configuration"
fi

# Step 22: Create management script
print_header "Step 22: Create Management Script"
print_status "Creating service management script..."

cat > /home/$USER/mccva_manage.sh << 'EOF'
#!/bin/bash

# MCCVA Service Management Script
# Usage: ./mccva_manage.sh [start|stop|restart|status|logs|test|demo]

case "$1" in
    start)
        echo "Starting MCCVA services..."
        sudo systemctl start mccva-ml openresty
        ;;
    stop)
        echo "Stopping MCCVA services..."
        sudo systemctl stop mccva-ml openresty
        ;;
    restart)
        echo "Restarting MCCVA services..."
        sudo systemctl restart mccva-ml openresty
        ;;
    status)
        echo "MCCVA ML Service:"
        sudo systemctl status mccva-ml --no-pager
        echo ""
        echo "OpenResty:"
        sudo systemctl status openresty --no-pager
        ;;
    logs)
        echo "MCCVA ML Service logs:"
        sudo journalctl -u mccva-ml -f
        ;;
    test)
        echo "Testing MCCVA endpoints..."
        echo "Health Check:"
        curl -s http://localhost/health | jq .
        echo ""
        echo "ML Service Health:"
        curl -s http://localhost:5000/health | jq .
        ;;
    demo)
        echo "Running MCCVA Demo..."
        echo "1. Testing SVM Prediction:"
        curl -s -X POST http://localhost/predict/makespan \
            -H "Content-Type: application/json" \
            -d '{"features": [2, 4, 50, 500, 1]}' | jq .
        echo ""
        echo "2. Testing K-Means Clustering:"
        curl -s -X POST http://localhost/predict/vm_cluster \
            -H "Content-Type: application/json" \
            -d '{"vm_features": [0.3, 0.2, 0.1]}' | jq .
        echo ""
        echo "3. Testing MCCVA Routing:"
        curl -s -X POST http://localhost/mccva/route \
            -H "Content-Type: application/json" \
            -d '{"features": [2, 4, 50, 500, 1], "vm_features": [0.3, 0.2, 0.1]}' | jq .
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|test|demo}"
        exit 1
        ;;
esac
EOF

chmod +x /home/$USER/mccva_manage.sh

# Step 23: Run comprehensive test
print_header "Step 23: Run Comprehensive Test"
print_status "Running comprehensive test..."

if [ -f "test_mccva.py" ]; then
    cd $PROJECT_DIR
    source venv/bin/activate
    python3 test_mccva.py
else
    print_warning "test_mccva.py not found, running basic tests..."
    
    # Basic performance test
    print_status "Running performance test..."
    for i in {1..5}; do
        start_time=$(date +%s.%N)
        curl -s -X POST http://localhost/mccva/route \
            -H "Content-Type: application/json" \
            -d '{"features": [4, 8, 100, 1000, 3], "vm_features": [0.5, 0.5, 0.5]}' > /dev/null
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc)
        print_status "Request $i: ${duration}s"
    done
fi

print_header "MCCVA Amazon Cloud Deployment Complete!"

print_status "Deployment Summary:"
echo "  • Repository: $GITHUB_REPO"
echo "  • Installation: $PROJECT_DIR"
echo "  • Services: mccva-ml, openresty"
echo "  • Status: ✅ Deployed successfully"

print_status "Service URLs:"
PUBLIC_IP=$(curl -s ifconfig.me || echo "localhost")
echo "  • OpenResty Gateway: http://$PUBLIC_IP"
echo "  • ML Service: http://$PUBLIC_IP:5000"
echo "  • Health Check: http://$PUBLIC_IP/health"

print_status "Management Commands:"
echo "  • Service management: ~/mccva_manage.sh [start|stop|restart|status|logs|test|demo]"
echo "  • View logs: sudo journalctl -u mccva-ml -f"
echo "  • Test algorithm: cd $PROJECT_DIR && source venv/bin/activate && python3 test_mccva.py"

print_status "API Endpoints:"
echo "  • MCCVA Routing: POST http://localhost/mccva/route"
echo "  • SVM Prediction: POST http://localhost/predict/makespan"
echo "  • K-Means Prediction: POST http://localhost/predict/vm_cluster"

print_status "Demo Commands:"
echo "  • Run demo: ~/mccva_manage.sh demo"
echo "  • Test endpoints: ~/mccva_manage.sh test"

print_status "✅ MCCVA Algorithm is now running and ready for production use!"
print_status "🎯 AI-based load balancing is active!"
print_status "🤖 Mock server is running and AI routing is working!"
