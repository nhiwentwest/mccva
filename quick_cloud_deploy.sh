#!/bin/bash
# Quick Cloud Deployment for Perfect SVM API
# Run: wget -O - https://raw.githubusercontent.com/nhiwentwest/mccva/main/quick_cloud_deploy.sh | bash

echo "🚀 Deploying Perfect SVM API to Cloud..."
echo "========================================"

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip git docker.io docker-compose

# Clone repository
echo "📥 Cloning repository..."
if [ -d "mccva" ]; then
    cd mccva
    git pull origin main
else
    git clone https://github.com/nhiwentwest/mccva.git
    cd mccva
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install flask scikit-learn joblib pandas numpy

# Check if models exist, if not train them
echo "🧠 Checking ML models..."
if [ ! -f "models/svm_model.joblib" ]; then
    echo "📊 Training perfect accuracy model..."
    python3 perfect_accuracy_train_svm.py
fi

# Start Flask API
echo "🌐 Starting Flask API..."
# Kill any existing Flask processes
sudo pkill -f "python3 app.py" 2>/dev/null || true

# Start API in background
nohup python3 app.py > api.log 2>&1 &

# Wait for API to start
sleep 5

# Test API
echo "🧪 Testing API..."
API_RESPONSE=$(curl -s http://localhost:8080/health)
if [[ $API_RESPONSE == *"healthy"* ]]; then
    echo "✅ API is running successfully!"
    echo "🌍 API accessible at: http://$(curl -s ifconfig.me):8080"
else
    echo "❌ API failed to start. Check logs:"
    tail -10 api.log
    exit 1
fi

# Test prediction
echo "🔮 Testing prediction endpoint..."
PREDICTION=$(curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"cpu_cores": 8, "memory_mb": 16384, "jobs_1min": 12, "jobs_5min": 8, "network_receive": 1500, "network_transmit": 1200, "cpu_speed": 3.0}')

if [[ $PREDICTION == *"large"* ]]; then
    echo "✅ Prediction test passed!"
else
    echo "❌ Prediction test failed:"
    echo "$PREDICTION"
fi

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "======================="
echo "🌐 API URL: http://$(curl -s ifconfig.me):8080"
echo "📊 Health Check: http://$(curl -s ifconfig.me):8080/health"
echo "🔮 Prediction: http://$(curl -s ifconfig.me):8080/predict"
echo ""
echo "📋 Quick Test Commands:"
echo "curl http://$(curl -s ifconfig.me):8080/health"
echo ""
echo 'curl -X POST http://$(curl -s ifconfig.me):8080/predict \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"cpu_cores": 4, "memory_mb": 8192, "jobs_1min": 8, "jobs_5min": 6, "network_receive": 1000, "network_transmit": 800, "cpu_speed": 2.8}'"'"
echo ""
echo "📝 Logs: tail -f api.log"
echo "🛑 Stop API: sudo pkill -f 'python3 app.py'" 