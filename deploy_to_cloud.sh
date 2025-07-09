#!/bin/bash

# 🚀 MCCVA ML Service - Cloud Deployment Script
# Run this script on your EC2/Cloud server

set -e  # Exit on any error

echo "🚀 MCCVA ML Service - Cloud Deployment Started"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "Please don't run this script as root. Use a regular user with sudo privileges."
fi

# 1. Update system
log "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# 2. Install Docker if not installed
if ! command -v docker &> /dev/null; then
    log "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    warn "Docker installed! You need to logout and login again, or run 'newgrp docker'"
    echo "After that, run this script again."
    exit 0
else
    log "🐳 Docker is already installed"
fi

# 3. Install Docker Compose if not installed
if ! command -v docker-compose &> /dev/null; then
    log "🐙 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
else
    log "🐙 Docker Compose is already installed"
fi

# 4. Clone/Update repository
REPO_DIR="mccva"
if [ -d "$REPO_DIR" ]; then
    log "📁 Updating existing repository..."
    cd $REPO_DIR
    git pull origin main
else
    log "📥 Cloning repository..."
    git clone https://github.com/nhiwentwest/mccva.git
    cd $REPO_DIR
fi

# 5. Check if models directory exists
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    error "❌ Models directory is missing or empty! Please upload your trained models to the models/ directory."
fi

log "✅ Found $(ls models/ | wc -l) model files in models/ directory"

# 6. Create logs directory
log "📝 Creating logs directory..."
mkdir -p logs
chmod 755 logs

# 7. Stop existing containers if running
log "🛑 Stopping existing containers..."
docker-compose down 2>/dev/null || true

# 8. Build and start the service
log "🔨 Building and starting ML Service..."
docker-compose up --build -d

# 9. Wait for service to be ready
log "⏳ Waiting for service to start..."
sleep 30

# 10. Check service health
log "🏥 Checking service health..."
for i in {1..10}; do
    if curl -f http://localhost:5000/health > /dev/null 2>&1; then
        log "✅ Service is healthy and running!"
        break
    else
        if [ $i -eq 10 ]; then
            error "❌ Service failed to start properly"
        fi
        echo "Attempt $i/10 - waiting..."
        sleep 10
    fi
done

# 11. Show service status
log "📊 Service Status:"
docker-compose ps

# 12. Show useful commands
echo ""
echo -e "${BLUE}🎉 DEPLOYMENT SUCCESSFUL!${NC}"
echo "=================================================="
echo -e "${YELLOW}📋 USEFUL COMMANDS:${NC}"
echo ""
echo -e "${GREEN}# Check service status:${NC}"
echo "docker-compose ps"
echo ""
echo -e "${GREEN}# View logs:${NC}"
echo "docker-compose logs -f mccva-ml-service"
echo ""
echo -e "${GREEN}# Stop service:${NC}"
echo "docker-compose down"
echo ""
echo -e "${GREEN}# Restart service:${NC}"
echo "docker-compose restart"
echo ""
echo -e "${GREEN}# Test API endpoints:${NC}"
echo "curl http://localhost:5000/health"
echo "curl http://localhost:5000/admin/metrics"
echo ""
echo -e "${YELLOW}🌐 Your ML Service is running on:${NC}"
echo "- Health Check: http://$(curl -s ifconfig.me):5000/health"
echo "- API Endpoint: http://$(curl -s ifconfig.me):5000/predict"
echo "- Admin Metrics: http://$(curl -s ifconfig.me):5000/admin/metrics"
echo ""
echo -e "${YELLOW}🔧 Admin Endpoints:${NC}"
echo "- /admin/metrics - Performance metrics"
echo "- /admin/debug - Debug information"
echo "- /admin/cache/clear - Clear prediction cache"
echo "- /admin/models/reload - Reload models"
echo "- /predict/compare - Compare predictions"
echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}" 