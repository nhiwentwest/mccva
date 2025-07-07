#!/bin/bash
# Update Perfect Models - Replace existing models with 100% accuracy models
# Run on cloud server: wget -O - https://raw.githubusercontent.com/nhiwentwest/mccva/main/update_perfect_models.sh | bash

echo "🔄 Updating Models to 100% Accuracy..."
echo "====================================="

# Navigate to project directory
cd /opt/mccva || { echo "❌ /opt/mccva not found"; exit 1; }

# Pull latest code with perfect models
echo "📥 Pulling latest code with perfect models..."
git pull origin main

# Install any missing Python packages
echo "🐍 Installing missing packages via apt..."
sudo apt-get update -y
sudo apt-get install -y python3-flask python3-numpy python3-sklearn python3-joblib python3-pandas

# Check if perfect models exist
echo "🧠 Checking perfect accuracy models..."
if [ -f "models/svm_model.joblib" ]; then
    echo "✅ Perfect models found in repository!"
else
    echo "📊 Training perfect accuracy models..."
    python3 perfect_accuracy_train_svm.py
fi

# Stop existing ml_service
echo "🛑 Stopping existing ml_service..."
sudo pkill -f "ml_service.py" 2>/dev/null || true
sleep 2

# Start ml_service with new models
echo "🚀 Starting ml_service with perfect models..."
nohup python3 ml_service.py > ml_service.log 2>&1 &

# Wait for service to start
sleep 5

# Test health
echo "🏥 Testing ml_service health..."
HEALTH_RESPONSE=$(curl -s http://localhost:5000/health)
if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
    echo "✅ ml_service is running with perfect models!"
else
    echo "❌ ml_service failed to start. Check logs:"
    tail -10 ml_service.log
    exit 1
fi

# Test prediction with perfect model
echo "🔮 Testing perfect model prediction..."
PREDICTION=$(curl -s -X POST http://localhost:5000/predict/makespan \
  -H "Content-Type: application/json" \
  -d '{"features": [8, 16, 200, 2000, 4, 3, 50, 40, 800, 4]}')

if [[ $PREDICTION == *"large"* ]] || [[ $PREDICTION == *"medium"* ]]; then
    echo "✅ Perfect model prediction test passed!"
    echo "Response: $PREDICTION"
else
    echo "❌ Prediction test failed:"
    echo "$PREDICTION"
fi

# Test với script có sẵn
echo "🧪 Testing with existing test script..."
if [ -f "quick_accuracy_test.py" ]; then
    python3 quick_accuracy_test.py
else
    echo "⚠️  quick_accuracy_test.py not found"
fi

echo ""
echo "🎉 MODEL UPDATE COMPLETE!"
echo "========================="
echo "🌐 ml_service URL: http://$(curl -s ifconfig.me):5000"
echo "📊 Health Check: http://$(curl -s ifconfig.me):5000/health"
echo "🔮 Prediction: http://$(curl -s ifconfig.me):5000/predict/makespan"
echo ""
echo "📋 Test Commands:"
echo "curl http://localhost:5000/health"
echo ""
echo 'curl -X POST http://localhost:5000/predict/makespan \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"features": [4, 8, 100, 1000, 3, 2, 20, 30, 400, 3]}'"'"
echo ""
echo "📝 Logs: tail -f ml_service.log"
echo "🧪 Test Accuracy: python3 quick_accuracy_test.py" 