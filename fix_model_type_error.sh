#!/bin/bash

echo "🔧 Fixing model type error in ml_service.py..."

cd /opt/mccva

# Stop any running ml_service
echo "⏹️  Stopping ml_service..."
pkill -f "python.*ml_service"
sleep 2

# Backup current ml_service.py
echo "💾 Creating backup..."
cp ml_service.py ml_service.py.backup.$(date +%s)

# Fix the model info logging to handle both SVM and RandomForest
echo "🔧 Fixing model info logging..."
sed -i 's/logger.info(f"SVM Model: {svm_model.kernel} kernel, {sum(svm_model.n_support_)} support vectors")/if hasattr(svm_model, "kernel"):\n        logger.info(f"SVM Model: {svm_model.kernel} kernel, {sum(svm_model.n_support_)} support vectors")\n    else:\n        logger.info(f"Model Type: {type(svm_model).__name__}, Features: {getattr(svm_model, \"n_features_in_\", \"unknown\")}")/' ml_service.py

# Also check if we need to retrain with correct SVM model
echo "🧠 Checking what model type is saved..."
python3 -c "
import joblib
try:
    model = joblib.load('models/svm_model.joblib')
    print(f'Current model type: {type(model).__name__}')
    if 'RandomForest' in str(type(model)):
        print('❌ Need to retrain with SVM model!')
    else:
        print('✅ Model type is correct')
except Exception as e:
    print(f'Error loading model: {e}')
"

# If RandomForest detected, retrain with SVM
echo "🔄 Retraining with SVM model..."
python3 perfect_accuracy_train_svm.py

echo "🚀 Starting ml_service with fixed model handling..."
nohup python3 ml_service.py > ml_service.log 2>&1 &
sleep 5

# Check if service started successfully
if pgrep -f "python.*ml_service" > /dev/null; then
    echo "✅ ml_service started successfully!"
    
    # Test health
    echo "🏥 Testing service health..."
    response=$(curl -s http://localhost:5000/health)
    echo "Health response: $response"
    
    # Test prediction
    echo "🧪 Testing prediction..."
    curl -s -X POST http://localhost:5000/predict/makespan \
         -H "Content-Type: application/json" \
         -d '{"features": [4, 8, 100, 1000, 3, 2, 50, 20, 500, 3]}' | python3 -m json.tool
    
    echo "🎯 Running accuracy test..."
    python3 quick_accuracy_test.py
else
    echo "❌ ml_service failed to start. Check logs:"
    tail -20 ml_service.log
fi 