#!/bin/bash
# Force restart ml_service with perfect model validation
# Run: wget -O - https://raw.githubusercontent.com/nhiwentwest/mccva/main/force_restart_ml_service.sh | bash

echo "🔥 Force restarting ml_service..."
echo "================================="

cd /opt/mccva || { echo "❌ /opt/mccva not found"; exit 1; }

# Kill ALL Python processes using port 5000
echo "💀 Killing ALL ml_service processes..."
sudo pkill -9 -f "ml_service.py" 2>/dev/null || true
sudo pkill -9 -f "python3.*ml_service" 2>/dev/null || true
sudo pkill -9 -f "flask.*5000" 2>/dev/null || true

# Also kill any process using port 5000
echo "🔫 Killing processes on port 5000..."
sudo lsof -ti:5000 | xargs sudo kill -9 2>/dev/null || true

# Wait for processes to die
sleep 5

# Check if port is free
echo "🔍 Checking port 5000..."
if netstat -tuln | grep -q ":5000 "; then
    echo "⚠️  Port 5000 still busy. Trying harder..."
    sudo fuser -k 5000/tcp 2>/dev/null || true
    sleep 2
fi

# Update validation if not already done
echo "📝 Ensuring validation ranges are correct..."
sed -i 's/Data size must be between 1-5/Data size must be between 1-1000/g' ml_service.py
sed -i 's/(1 <= features\[6\] <= 5)/(1 <= features[6] <= 1000)/g' ml_service.py

# Show current validation
echo "✅ Current validation ranges:"
grep -A2 -B2 "features\[6\]" ml_service.py

# Start fresh ml_service
echo "🚀 Starting fresh ml_service..."
nohup python3 ml_service.py > ml_service_new.log 2>&1 &

# Wait for startup
sleep 8

# Check if it started
echo "🏥 Checking ml_service health..."
HEALTH_RESPONSE=$(curl -s http://localhost:5000/health 2>/dev/null)
if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
    echo "✅ ml_service is healthy!"
else
    echo "❌ ml_service not responding. Check logs:"
    echo "=== ml_service_new.log ==="
    tail -10 ml_service_new.log
    echo "==========================="
    exit 1
fi

# Test with problematic data_size
echo "🧪 Testing with data_size=50..."
PREDICTION=$(curl -s -X POST http://localhost:5000/predict/makespan \
  -H "Content-Type: application/json" \
  -d '{"features": [4, 8, 100, 1000, 3, 2, 50, 30, 400, 3]}')

echo "Response: $PREDICTION"

if [[ $PREDICTION == *"error"* ]]; then
    echo "❌ Still has validation error!"
    echo "Check the actual error:"
    echo "$PREDICTION"
    echo ""
    echo "File content check:"
    grep -n "Data size must be between" ml_service.py
else
    echo "✅ Validation fixed! Running full accuracy test..."
    if [ -f "quick_accuracy_test.py" ]; then
        python3 quick_accuracy_test.py
    else
        echo "⚠️  quick_accuracy_test.py not found"
    fi
fi

echo ""
echo "🎉 FORCE RESTART COMPLETE!"
echo "=========================="
echo "📊 Health: curl http://localhost:5000/health"
echo "📝 Logs: tail -f ml_service_new.log" 