#!/bin/bash
# Fix AI Accuracy Issue - Restart ML Service with New Model
# Run this script on your AWS EC2 instance

echo "🚀 Fixing MCCVA AI Accuracy Issue"
echo "================================="

# Check current directory
echo "📍 Current directory: $(pwd)"
if [ ! -f "ml_service.py" ]; then
    echo "❌ Not in /opt/mccva directory. Changing..."
    cd /opt/mccva
fi

echo "✅ Working directory: $(pwd)"

# Check if models exist
echo ""
echo "🔍 Checking for new trained models..."
if [ -d "models" ] && [ -f "models/svm_model.joblib" ]; then
    echo "✅ New models found!"
    ls -la models/
else
    echo "❌ Models not found. Need to retrain first."
    echo "Run: python3 retrain_svm_fixed.py"
    exit 1
fi

# Check current ML service
echo ""
echo "🔍 Checking current ML service..."
ML_PID=$(ps aux | grep '[m]l_service.py' | awk '{print $2}')
if [ -n "$ML_PID" ]; then
    echo "⚠️  ML service running with PID: $ML_PID (using old model)"
    echo "🔄 Stopping old ML service..."
    sudo pkill -f ml_service.py
    sleep 2
    
    # Verify it's stopped
    if ps aux | grep '[m]l_service.py' > /dev/null; then
        echo "❌ Failed to stop ML service. Trying force kill..."
        sudo pkill -9 -f ml_service.py
        sleep 1
    fi
    echo "✅ Old ML service stopped"
else
    echo "ℹ️  No ML service currently running"
fi

# Start new ML service with retrained model
echo ""
echo "🚀 Starting ML service with new model..."
nohup python3 ml_service.py > ml_service.log 2>&1 &
sleep 3

# Verify new service is running
NEW_PID=$(ps aux | grep '[m]l_service.py' | awk '{print $2}')
if [ -n "$NEW_PID" ]; then
    echo "✅ New ML service started with PID: $NEW_PID"
else
    echo "❌ Failed to start ML service. Check logs:"
    tail -20 ml_service.log
    exit 1
fi

# Test the ML service endpoints
echo ""
echo "🧪 Testing ML service endpoints..."

# Test basic health
curl -s http://localhost:5000/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ ML service health check passed"
else
    echo "❌ ML service health check failed"
fi

# Test SVM prediction
echo "🧪 Testing SVM prediction (should be 'medium')..."
PREDICTION=$(curl -s -X POST http://localhost:5000/predict/makespan \
  -H "Content-Type: application/json" \
  -d '{"cpu_cores": 4, "memory": 8, "storage": 100, "network_bandwidth": 1000, "priority": 3}' | grep -o '"[^"]*"' | head -1 | tr -d '"')

echo "📊 SVM Prediction Result: $PREDICTION"

if [ "$PREDICTION" = "medium" ]; then
    echo "✅ SVM model is working correctly!"
else
    echo "❌ SVM model still predicting incorrectly: $PREDICTION"
fi

# Run full AI routing test
echo ""
echo "🎯 Running full AI routing test..."
python3 test_ai_routing.py

echo ""
echo "🏁 Fix completed! Check the AI accuracy above."
echo "Expected: AI accuracy should now be 80%+ instead of 33.3%"
echo ""
echo "📊 Service Status:"
ps aux | grep -E '(ml_service|nginx|openresty)' | grep -v grep 