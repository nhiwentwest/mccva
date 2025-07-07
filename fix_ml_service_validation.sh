#!/bin/bash
# Fix ml_service validation ranges for perfect model
# Run: wget -O - https://raw.githubusercontent.com/nhiwentwest/mccva/main/fix_ml_service_validation.sh | bash

echo "🔧 Fixing ml_service validation ranges..."
echo "========================================"

cd /opt/mccva || { echo "❌ /opt/mccva not found"; exit 1; }

# Stop ml_service
echo "🛑 Stopping ml_service..."
sudo pkill -f "ml_service.py" 2>/dev/null || true
sleep 2

# Update validation ranges in ml_service.py
echo "📝 Updating validation ranges..."
sed -i 's/Data size must be between 1-5/Data size must be between 1-1000/g' ml_service.py
sed -i 's/(1 <= features\[6\] <= 5)/(1 <= features[6] <= 1000)/g' ml_service.py

# Show the changes
echo "✅ Updated validation ranges:"
grep -n "Data size must be between" ml_service.py
grep -n "features\[6\]" ml_service.py

# Restart ml_service
echo "🚀 Restarting ml_service..."
nohup python3 ml_service.py > ml_service.log 2>&1 &
sleep 3

# Test the fix
echo "🧪 Testing fixed validation..."
PREDICTION=$(curl -s -X POST http://localhost:5000/predict/makespan \
  -H "Content-Type: application/json" \
  -d '{"features": [4, 8, 100, 1000, 3, 2, 50, 30, 400, 3]}')

echo "Response: $PREDICTION"

if [[ $PREDICTION == *"error"* ]]; then
    echo "❌ Still has validation error"
    echo "Check ml_service.log:"
    tail -5 ml_service.log
else
    echo "✅ Validation fixed! Testing accuracy..."
    python3 quick_accuracy_test.py
fi

echo "🎉 Fix complete!" 