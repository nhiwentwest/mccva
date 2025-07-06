#!/bin/bash
# Deploy improved models to cloud server

echo "🚀 Deploying improved models to cloud..."

# Check if we have improved models
if [ ! -f "models/svm_model_improved.joblib" ]; then
    echo "❌ Improved models not found. Run test_model_local.py first!"
    exit 1
fi

# Copy models to cloud (assuming you have SSH access)
CLOUD_SERVER="ubuntu@your-cloud-ip"
CLOUD_PATH="/opt/mccva"

echo "📤 Copying improved models to cloud..."
scp models/svm_model_improved.joblib $CLOUD_SERVER:$CLOUD_PATH/models/svm_model.joblib
scp models/scaler_improved.joblib $CLOUD_SERVER:$CLOUD_PATH/models/scaler.joblib

echo "🔄 Restarting ML Service on cloud..."
ssh $CLOUD_SERVER "cd $CLOUD_PATH && docker restart mccva-ml"

echo "✅ Models deployed successfully!"
echo "🌐 Test on cloud with: ssh $CLOUD_SERVER 'cd $CLOUD_PATH && python3 test_ai_routing_host.py'" 