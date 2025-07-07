#!/bin/bash
echo "🚀 CLOUD TEST COMMANDS - FIXED MODEL"
echo "======================================"

# Navigate to project directory
cd /opt/mccva

# Pull latest changes
echo "📥 Pulling latest changes..."
git pull origin main

# Stop existing service
echo "🛑 Stopping ML service..."
sudo pkill -f ml_service.py

# Start service
echo "🚀 Starting ML service..."
nohup python3 ml_service.py > ml_service.log 2>&1 &

# Wait for service to fully start (5 seconds instead of 3)
echo "⏳ Waiting 5 seconds for service to start..."
sleep 5

# Check service status
echo "🔍 Checking service status..."
ps aux | grep ml_service.py

# Test model accuracy
echo "🧪 Testing model accuracy..."
python3 quick_accuracy_test.py

echo "✅ Test completed!" 