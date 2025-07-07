#!/bin/bash
# Fix SVM prediction decoding in ml_service.py
# Run: wget -O - https://raw.githubusercontent.com/nhiwentwest/mccva/main/fix_svm_prediction_decode.sh | bash

echo "🔧 Fixing SVM prediction decoding..."
echo "==================================="

cd /opt/mccva || { echo "❌ /opt/mccva not found"; exit 1; }

# Stop ml_service
echo "🛑 Stopping ml_service..."
sudo pkill -f "ml_service.py" 2>/dev/null || true
sleep 2

# Create backup
echo "💾 Creating backup..."
cp ml_service.py ml_service.py.backup

# Fix the load_models function to include label_encoder
echo "📝 Adding label_encoder loading..."
sed -i '/global svm_model, kmeans_model, svm_scaler, kmeans_scaler/c\
    global svm_model, kmeans_model, svm_scaler, kmeans_scaler, svm_label_encoder' ml_service.py

sed -i '/svm_scaler = joblib.load("models\/scaler.joblib")/a\
        svm_label_encoder = joblib.load("models/label_encoder.joblib")' ml_service.py

# Fix the prediction decoding
echo "🔄 Fixing prediction decoding..."
sed -i '/prediction = svm_model.predict(features_scaled)\[0\]/c\
        prediction_encoded = svm_model.predict(features_scaled)[0]\
        prediction = svm_label_encoder.inverse_transform([prediction_encoded])[0]' ml_service.py

# Initialize the global variable
sed -i '/svm_scaler = None/a\
svm_label_encoder = None' ml_service.py

# Show the changes
echo "✅ Changes made:"
echo "=== Load models function ==="
grep -A5 -B5 "svm_label_encoder" ml_service.py
echo ""
echo "=== Prediction function ==="
grep -A3 -B3 "prediction_encoded" ml_service.py

# Restart ml_service
echo "🚀 Restarting ml_service with fixed decoding..."
nohup python3 ml_service.py > ml_service_fixed.log 2>&1 &
sleep 5

# Test health
echo "🏥 Testing ml_service health..."
HEALTH_RESPONSE=$(curl -s http://localhost:5000/health 2>/dev/null)
if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
    echo "✅ ml_service is healthy!"
else
    echo "❌ ml_service failed to start. Check logs:"
    tail -10 ml_service_fixed.log
    exit 1
fi

# Test prediction with proper decoding
echo "🧪 Testing SVM prediction decoding..."
PREDICTION=$(curl -s -X POST http://localhost:5000/predict/makespan \
  -H "Content-Type: application/json" \
  -d '{"features": [4, 8, 100, 1000, 3, 2, 50, 30, 400, 3]}')

echo "Response: $PREDICTION"

if [[ $PREDICTION == *"small"* ]] || [[ $PREDICTION == *"medium"* ]] || [[ $PREDICTION == *"large"* ]]; then
    echo "✅ Prediction decoding fixed! Running accuracy test..."
    if [ -f "quick_accuracy_test.py" ]; then
        python3 quick_accuracy_test.py
    else
        echo "⚠️  quick_accuracy_test.py not found"
    fi
else
    echo "❌ Still returning encoded numbers:"
    echo "$PREDICTION"
    echo ""
    echo "Check if label_encoder.joblib exists:"
    ls -la models/label_encoder.joblib
fi

echo ""
echo "🎉 DECODING FIX COMPLETE!"
echo "========================="
echo "📝 Backup: ml_service.py.backup"
echo "📊 Logs: tail -f ml_service_fixed.log" 