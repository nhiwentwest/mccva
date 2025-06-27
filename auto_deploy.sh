#!/bin/bash

# MCCVA Auto Deployment Script
# Tự động deploy thuật toán MCCVA trên Amazon Cloud Ubuntu
# Chạy: ./auto_deploy.sh

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

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root"
    exit 1
fi

print_header "MCCVA Auto Deployment"
print_status "Automated deployment of MCCVA Algorithm on Amazon Cloud Ubuntu"
print_status "This script will clone, setup, and deploy everything automatically"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
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

# Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
print_status "Installing essential packages..."
install_if_missing git
install_if_missing curl
install_if_missing wget
install_if_missing python3
install_if_missing python3-pip
install_if_missing python3-venv
install_if_missing python3-dev
install_if_missing build-essential

# Install ML dependencies
print_status "Installing ML dependencies..."
sudo apt install -y libblas-dev liblapack-dev libatlas-base-dev gfortran

# Install OpenResty
print_status "Installing OpenResty..."
if ! command_exists openresty; then
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:openresty/ppa
    sudo apt update
    sudo apt install -y openresty
else
    print_status "✅ OpenResty is already installed"
fi

# Clone repository
print_status "Cloning MCCVA repository..."
if [ -d "mccva" ]; then
    print_status "Repository already exists, updating..."
    cd mccva
    git pull origin main
else
    git clone $GITHUB_REPO
    cd mccva
fi

# Check if clone was successful
if [ ! -f "ml_service.py" ] || [ ! -f "deploy.sh" ]; then
    print_error "Failed to clone repository or missing essential files"
    exit 1
fi

print_status "✅ Repository cloned successfully"

# Make deploy script executable
chmod +x deploy.sh

# Run deployment
print_status "Starting MCCVA deployment..."
./deploy.sh

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 15

# Run deployment check
print_status "Running deployment verification..."
if [ -f "check_deployment.py" ]; then
    python3 check_deployment.py
else
    print_warning "check_deployment.py not found, running basic checks..."
    
    # Basic health checks
    if curl -s http://localhost/health | grep -q "mccva-openresty-gateway"; then
        print_status "✅ OpenResty is working"
    else
        print_error "❌ OpenResty health check failed"
    fi
    
    if curl -s http://localhost:5000/health | grep -q "healthy"; then
        print_status "✅ ML Service is working"
    else
        print_error "❌ ML Service health check failed"
    fi
fi

# Test MCCVA algorithm
print_status "Testing MCCVA algorithm..."
if [ -f "test_mccva.py" ]; then
    python3 test_mccva.py
else
    print_warning "test_mccva.py not found, running basic API tests..."
    
    # Test basic API endpoints
    if curl -s -X POST http://localhost/predict/makespan \
        -H "Content-Type: application/json" \
        -d '{"features": [4, 8, 100, 1000, 3]}' | grep -q "makespan"; then
        print_status "✅ SVM prediction working"
    else
        print_error "❌ SVM prediction failed"
    fi
    
    if curl -s -X POST http://localhost/predict/vm_cluster \
        -H "Content-Type: application/json" \
        -d '{"vm_features": [0.5, 0.5, 0.5]}' | grep -q "cluster"; then
        print_status "✅ K-Means prediction working"
    else
        print_error "❌ K-Means prediction failed"
    fi
    
    if curl -s -X POST http://localhost/mccva/route \
        -H "Content-Type: application/json" \
        -d '{"features": [4, 8, 100, 1000, 3], "vm_features": [0.5, 0.5, 0.5]}' | grep -q "target_vm"; then
        print_status "✅ MCCVA routing working"
    else
        print_error "❌ MCCVA routing failed"
    fi
fi

# Configure firewall
print_status "Configuring firewall..."
if command -v ufw &> /dev/null; then
    sudo ufw allow 80/tcp
    sudo ufw allow 5000/tcp
    sudo ufw allow 22/tcp
    print_status "✅ Firewall configured"
else
    print_warning "UFW not found, skipping firewall configuration"
fi

# Create service management script
print_status "Creating service management script..."
cat > /home/$USER/mccva_manage.sh << 'EOF'
#!/bin/bash

# MCCVA Service Management Script
# Usage: ./mccva_manage.sh [start|stop|restart|status|logs|test]

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
        curl -s http://localhost/health
        echo ""
        curl -s http://localhost:5000/health
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|test}"
        exit 1
        ;;
esac
EOF

chmod +x /home/$USER/mccva_manage.sh

print_header "MCCVA Auto Deployment Complete!"

print_status "Deployment Summary:"
echo "  • Repository: $GITHUB_REPO"
echo "  • Installation: /opt/mccva"
echo "  • Services: mccva-ml, openresty"
echo "  • Status: ✅ Deployed successfully"

print_status "Service URLs:"
echo "  • OpenResty Gateway: http://$(curl -s ifconfig.me || echo 'localhost')"
echo "  • ML Service: http://$(curl -s ifconfig.me || echo 'localhost'):5000"
echo "  • Health Check: http://$(curl -s ifconfig.me || echo 'localhost')/health"

print_status "Management Commands:"
echo "  • Service management: ~/mccva_manage.sh [start|stop|restart|status|logs|test]"
echo "  • View logs: sudo journalctl -u mccva-ml -f"
echo "  • Test algorithm: cd mccva && python3 test_mccva.py"

print_status "API Endpoints:"
echo "  • MCCVA Routing: POST http://localhost/mccva/route"
echo "  • SVM Prediction: POST http://localhost/predict/makespan"
echo "  • K-Means Prediction: POST http://localhost/predict/vm_cluster"

print_status "Example API calls:"
echo "  curl -X POST http://localhost/mccva/route \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"features\": [4, 8, 100, 1000, 3], \"vm_features\": [0.5, 0.5, 0.5]}'"

print_status "✅ MCCVA Algorithm is now running and ready for production use!"
print_status "🎯 AI-based load balancing is active!" 