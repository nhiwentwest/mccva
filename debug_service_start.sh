#!/bin/bash

echo "🔍 Debugging ml_service startup issue..."

cd /opt/mccva

# Kill all processes using port 5000
echo "🔧 Killing processes on port 5000..."
sudo lsof -ti:5000 | xargs -r sudo kill -9
sleep 2

# Check what's in the log
echo "📋 Recent ml_service logs:"
tail -30 ml_service.log

echo ""
echo "🧪 Testing model loading directly..."
python3 -c "
import sys
sys.path.append('/opt/mccva')
try:
    import joblib
    from ml_service import load_models
    print('✅ Attempting to load models...')
    load_models()
    print('✅ Models loaded successfully!')
except Exception as e:
    print(f'❌ Error loading models: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "🚀 Starting service manually with verbose output..."
python3 ml_service.py 