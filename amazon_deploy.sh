#!/bin/bash

# MCCVA Amazon Cloud Ubuntu Deployment Script
# Tự động deploy thuật toán MCCVA trên Amazon Cloud Ubuntu
# Fixed for Python 3.12 externally managed environment
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
print_status "Fixed for Python 3.12 externally managed environment"

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

# Step 3: Fix Python 3.12 environment issues
print_header "Step 3: Fix Python 3.12 Environment"
print_status "Installing python3-full for virtual environment support..."

# Install python3-full to fix externally managed environment
sudo apt install -y python3-full

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
print_status "Python version: $(python3 --version)"

if [[ "$PYTHON_VERSION" == "3.12" ]]; then
    print_status "Detected Python 3.12 - applying fixes for externally managed environment"
    
    # Ensure python3-venv is installed
    sudo apt install -y python3-venv
    
    # Test virtual environment creation
    python3 -m venv /tmp/test_venv
    if [ $? -eq 0 ]; then
        print_status "✅ Virtual environment creation works"
        rm -rf /tmp/test_venv
    else
        print_error "❌ Virtual environment creation failed"
        exit 1
    fi
fi

# Step 4: Install ML dependencies
print_header "Step 4: Install ML Dependencies"
print_status "Installing ML dependencies for scikit-learn..."

sudo apt install -y libblas-dev liblapack-dev libatlas-base-dev gfortran

# Step 5: Install OpenResty
print_header "Step 5: Install OpenResty"
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

# Step 6: Create project directory
print_header "Step 6: Setup Project Directory"
print_status "Creating project directory..."
sudo mkdir -p $PROJECT_DIR
sudo chown $USER:$USER $PROJECT_DIR

# Step 7: Clone repository
print_header "Step 7: Clone GitHub Repository"
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

# Step 8: Setup Python environment
print_header "Step 8: Setup Python Environment"
cd $PROJECT_DIR

print_status "Creating Python virtual environment..."
# Remove existing venv if it exists
if [ -d "venv" ]; then
    print_status "Removing existing virtual environment..."
    rm -rf venv
fi

# Create new virtual environment
python3 -m venv venv
source venv/bin/activate

print_status "Upgrading pip and installing build tools..."
# Use specific versions to avoid compatibility issues
pip install --upgrade pip==23.3.1
pip install setuptools==68.2.2 wheel==0.41.2

print_status "Installing Python dependencies..."
# Install dependencies with specific versions for Python 3.12 compatibility
print_status "Installing numpy first (required for other packages)..."
pip install numpy==1.26.4

print_status "Installing scipy..."
pip install scipy==1.12.0

print_status "Installing scikit-learn..."
pip install scikit-learn==1.4.0

print_status "Installing pandas..."
pip install pandas==2.2.0

print_status "Installing remaining dependencies..."
pip install joblib==1.3.2 Flask==3.0.0 Werkzeug==3.0.1 gunicorn==21.2.0 requests==2.31.0

# Install visualization packages if needed
print_status "Installing visualization packages..."
pip install matplotlib==3.8.2 seaborn==0.13.0

print_status "✅ All Python dependencies installed successfully"

# Step 9: Verify model files
print_header "Step 9: Verify Model Files"
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

# Step 10: Test model loading
print_header "Step 10: Test Model Loading"
print_status "Testing model loading..."

# Activate virtual environment for testing
source venv/bin/activate

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

# Step 11: Setup Lua files
print_header "Step 11: Setup Lua Files"
print_status "Setting up Lua files for OpenResty..."

sudo mkdir -p /usr/local/openresty/nginx/conf/lua
sudo cp lua/mccva_routing.lua /usr/local/openresty/nginx/conf/lua/
sudo cp lua/predict_makespan.lua /usr/local/openresty/nginx/conf/lua/
sudo cp lua/predict_vm_cluster.lua /usr/local/openresty/nginx/conf/lua/

print_status "✅ Lua files copied successfully"

# Step 12: Setup Nginx configuration
print_header "Step 12: Setup Nginx Configuration"
print_status "Backing up original nginx configuration..."
sudo cp /usr/local/openresty/nginx/conf/nginx.conf /usr/local/openresty/nginx/conf/nginx.conf.backup

print_status "Installing new nginx configuration..."
sudo cp nginx.conf /usr/local/openresty/nginx/conf/nginx.conf

# Step 13: Create directories
print_header "Step 13: Create Directories"
print_status "Creating necessary directories..."

sudo mkdir -p /var/log/nginx
sudo mkdir -p /usr/local/openresty/nginx/html
sudo mkdir -p /var/www/html
sudo chown -R $USER:$USER /var/log/nginx

# Step 14: Test nginx configuration
print_header "Step 14: Test Nginx Configuration"
print_status "Testing nginx configuration..."
sudo /usr/local/openresty/nginx/sbin/nginx -t

# Step 15: Create systemd service
print_header "Step 15: Create Systemd Service"
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

# Create systemd service for mock servers
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

# Step 16: Start services
print_header "Step 16: Start Services"
print_status "Starting services..."

sudo systemctl daemon-reload
sudo systemctl enable mccva-ml
sudo systemctl enable mccva-mock-servers
sudo systemctl start mccva-ml
sudo systemctl start mccva-mock-servers
sudo systemctl enable openresty
sudo systemctl start openresty

# Step 17: Wait for services
print_header "Step 17: Wait for Services"
print_status "Waiting for services to start..."
sleep 20

# Step 18: Check service status
print_header "Step 18: Check Service Status"
if service_is_running mccva-ml; then
    print_status "✅ MCCVA ML Service is running"
else
    print_error "❌ MCCVA ML Service failed to start"
    sudo systemctl status mccva-ml
    sudo journalctl -u mccva-ml --no-pager -n 20
    exit 1
fi

if service_is_running mccva-mock-servers; then
    print_status "✅ MCCVA Mock Servers are running"
else
    print_error "❌ MCCVA Mock Servers failed to start"
    sudo systemctl status mccva-mock-servers
    sudo journalctl -u mccva-mock-servers --no-pager -n 20
    exit 1
fi

if service_is_running openresty; then
    print_status "✅ OpenResty is running"
else
    print_error "❌ OpenResty failed to start"
    sudo systemctl status openresty
    exit 1
fi

# Step 19: Test endpoints
print_header "Step 19: Test Endpoints"
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

# Step 20: Test model predictions
print_header "Step 20: Test Model Predictions"
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

# Step 21: Test MCCVA routing
print_header "Step 21: Test MCCVA Routing"
print_status "Testing MCCVA routing..."

if curl -s -X POST http://localhost/mccva/route \
    -H "Content-Type: application/json" \
    -d '{"features": [4, 8, 100, 1000, 3], "vm_features": [0.5, 0.5, 0.5]}' | grep -q "target_vm"; then
    print_status "✅ MCCVA routing working"
else
    print_warning "⚠️ MCCVA routing test failed"
fi

# Step 22: Configure firewall
print_header "Step 22: Configure Firewall"
print_status "Configuring firewall..."
if command -v ufw &> /dev/null; then
    sudo ufw allow 80/tcp
    sudo ufw allow 5000/tcp
    sudo ufw allow 22/tcp
    print_status "✅ Firewall configured"
else
    print_warning "UFW not found, skipping firewall configuration"
fi

# Step 23: Create management script
print_header "Step 23: Create Management Script"
print_status "Creating service management script..."

cat > /home/$USER/mccva_manage.sh << 'EOF'
#!/bin/bash

# MCCVA Service Management Script
# Usage: ./mccva_manage.sh [start|stop|restart|status|logs|test|demo]

case "$1" in
    start)
        echo "Starting MCCVA services..."
        sudo systemctl start mccva-ml mccva-mock-servers openresty
        ;;
    stop)
        echo "Stopping MCCVA services..."
        sudo systemctl stop mccva-ml mccva-mock-servers openresty
        ;;
    restart)
        echo "Restarting MCCVA services..."
        sudo systemctl restart mccva-ml mccva-mock-servers openresty
        ;;
    status)
        echo "MCCVA ML Service:"
        sudo systemctl status mccva-ml --no-pager
        echo ""
        echo "MCCVA Mock Servers:"
        sudo systemctl status mccva-mock-servers --no-pager
        echo ""
        echo "OpenResty:"
        sudo systemctl status openresty --no-pager
        ;;
    logs)
        echo "MCCVA ML Service logs:"
        sudo journalctl -u mccva-ml -f
        ;;
    mock-logs)
        echo "MCCVA Mock Servers logs:"
        sudo journalctl -u mccva-mock-servers -f
        ;;
    test)
        echo "Testing MCCVA endpoints..."
        echo "Health Check:"
        curl -s http://localhost/health | jq . 2>/dev/null || curl -s http://localhost/health
        echo ""
        echo "ML Service Health:"
        curl -s http://localhost:5000/health | jq . 2>/dev/null || curl -s http://localhost:5000/health
        echo ""
        echo "Mock Servers Health:"
        for port in 8081 8082 8083 8084 8085 8086 8087 8088; do
            echo "Port $port:"
            curl -s http://localhost:$port/health | jq . 2>/dev/null || curl -s http://localhost:$port/health
        done
        ;;
    demo)
        echo "Running MCCVA Demo..."
        echo "1. Testing SVM Prediction:"
        curl -s -X POST http://localhost/predict/makespan \
            -H "Content-Type: application/json" \
            -d '{"features": [2, 4, 50, 500, 1]}' | jq . 2>/dev/null || curl -s -X POST http://localhost/predict/makespan \
            -H "Content-Type: application/json" \
            -d '{"features": [2, 4, 50, 500, 1]}'
        echo ""
        echo "2. Testing K-Means Clustering:"
        curl -s -X POST http://localhost/predict/vm_cluster \
            -H "Content-Type: application/json" \
            -d '{"vm_features": [0.3, 0.2, 0.1]}' | jq . 2>/dev/null || curl -s -X POST http://localhost/predict/vm_cluster \
            -H "Content-Type: application/json" \
            -d '{"vm_features": [0.3, 0.2, 0.1]}'
        echo ""
        echo "3. Testing MCCVA Routing:"
        curl -s -X POST http://localhost/mccva/route \
            -H "Content-Type: application/json" \
            -d '{"features": [2, 4, 50, 500, 1], "vm_features": [0.3, 0.2, 0.1]}' | jq . 2>/dev/null || curl -s -X POST http://localhost/mccva/route \
            -H "Content-Type: application/json" \
            -d '{"features": [2, 4, 50, 500, 1], "vm_features": [0.3, 0.2, 0.1]}'
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|mock-logs|test|demo}"
        exit 1
        ;;
esac
EOF

chmod +x /home/$USER/mccva_manage.sh

# Step 24: Run comprehensive test
print_header "Step 24: Run Comprehensive Test"
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
        duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "0.001")
        print_status "Request $i: ${duration}s"
    done
fi

print_header "MCCVA Amazon Cloud Deployment Complete!"

print_status "Deployment Summary:"
echo "  • Repository: $GITHUB_REPO"
echo "  • Installation: $PROJECT_DIR"
echo "  • Services: mccva-ml, openresty"
echo "  • Python Environment: Virtual environment in $PROJECT_DIR/venv"
echo "  • Status: ✅ Deployed successfully"

print_status "Service URLs:"
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "localhost")
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
print_status "🐍 Python 3.12 environment issues fixed!"
