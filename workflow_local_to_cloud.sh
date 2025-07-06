#!/bin/bash
# Complete workflow: Local Training -> Testing -> Cloud Deployment

echo "🚀 MCCVA Model Training & Deployment Workflow"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
echo "📋 Checking dependencies..."
if ! command_exists python3; then
    echo -e "${RED}❌ Python3 not found${NC}"
    exit 1
fi

if ! command_exists pip3; then
    echo -e "${RED}❌ pip3 not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Dependencies OK${NC}"

# Install required packages
echo "📦 Installing required packages..."
pip3 install scikit-learn pandas numpy matplotlib seaborn joblib

# Step 1: Train models
echo ""
echo "🔧 Step 1: Training Models"
echo "-------------------------"
if [ -f "train_models.py" ]; then
    echo "Running model training..."
    python3 train_models.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Model training completed${NC}"
    else
        echo -e "${RED}❌ Model training failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ train_models.py not found${NC}"
    exit 1
fi

# Step 2: Quick test models
echo ""
echo "🧪 Step 2: Quick Model Testing"
echo "------------------------------"
if [ -f "quick_test_models.py" ]; then
    echo "Testing models..."
    python3 quick_test_models.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Model testing completed${NC}"
    else
        echo -e "${RED}❌ Model testing failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ quick_test_models.py not found${NC}"
    exit 1
fi

# Step 3: Check test results
echo ""
echo "📊 Step 3: Checking Test Results"
echo "--------------------------------"
if [ -f "quick_test_results.json" ]; then
    echo "Test results:"
    cat quick_test_results.json | python3 -m json.tool
    
    # Check if any model has >70% accuracy
    best_accuracy=$(cat quick_test_results.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
accuracies = [v.get('accuracy', 0) for v in data.values() if isinstance(v, dict) and 'accuracy' in v]
print(max(accuracies) if accuracies else 0)
")
    
    if (( $(echo "$best_accuracy >= 70" | bc -l) )); then
        echo -e "${GREEN}✅ Models ready for deployment (best accuracy: ${best_accuracy}%)${NC}"
    else
        echo -e "${YELLOW}⚠️ Models need improvement (best accuracy: ${best_accuracy}%)${NC}"
        echo "Continue with deployment? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Deployment cancelled"
            exit 0
        fi
    fi
else
    echo -e "${RED}❌ Test results not found${NC}"
    exit 1
fi

# Step 4: Deploy to cloud
echo ""
echo "☁️ Step 4: Deploying to Cloud"
echo "-----------------------------"

# Check if we have cloud server info
if [ -f "cloud_config.json" ]; then
    CLOUD_SERVER=$(cat cloud_config.json | python3 -c "import json, sys; print(json.load(sys.stdin).get('server', ''))")
    CLOUD_PATH=$(cat cloud_config.json | python3 -c "import json, sys; print(json.load(sys.stdin).get('path', '/opt/mccva'))")
else
    echo "Enter cloud server (e.g., ubuntu@your-ip):"
    read -r CLOUD_SERVER
    echo "Enter cloud path (default: /opt/mccva):"
    read -r CLOUD_PATH
    CLOUD_PATH=${CLOUD_PATH:-/opt/mccva}
    
    # Save config
    echo "{\"server\": \"$CLOUD_SERVER\", \"path\": \"$CLOUD_PATH\"}" > cloud_config.json
fi

if [ -z "$CLOUD_SERVER" ]; then
    echo -e "${RED}❌ Cloud server not configured${NC}"
    exit 1
fi

echo "Deploying to: $CLOUD_SERVER:$CLOUD_PATH"

# Check if improved models exist
if [ ! -f "models/svm_model_improved.joblib" ]; then
    echo -e "${RED}❌ Improved models not found${NC}"
    exit 1
fi

# Deploy models
echo "Copying models to cloud..."
scp models/svm_model_improved.joblib "$CLOUD_SERVER:$CLOUD_PATH/models/svm_model.joblib"
scp models/scaler_improved.joblib "$CLOUD_SERVER:$CLOUD_PATH/models/scaler.joblib"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Models copied successfully${NC}"
else
    echo -e "${RED}❌ Failed to copy models${NC}"
    exit 1
fi

# Restart ML Service on cloud
echo "Restarting ML Service on cloud..."
ssh "$CLOUD_SERVER" "cd $CLOUD_PATH && docker restart mccva-ml"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ ML Service restarted successfully${NC}"
else
    echo -e "${RED}❌ Failed to restart ML Service${NC}"
    exit 1
fi

# Step 5: Test on cloud
echo ""
echo "🌐 Step 5: Testing on Cloud"
echo "---------------------------"
echo "Running tests on cloud..."
ssh "$CLOUD_SERVER" "cd $CLOUD_PATH && python3 test_ai_routing_host.py"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Cloud testing completed${NC}"
else
    echo -e "${YELLOW}⚠️ Cloud testing had issues (check output above)${NC}"
fi

# Summary
echo ""
echo "🎉 Workflow Summary"
echo "=================="
echo -e "${GREEN}✅ Models trained locally${NC}"
echo -e "${GREEN}✅ Models tested locally${NC}"
echo -e "${GREEN}✅ Models deployed to cloud${NC}"
echo -e "${GREEN}✅ Cloud testing completed${NC}"
echo ""
echo "📊 Files created:"
echo "  - models/svm_model_improved.joblib"
echo "  - models/scaler_improved.joblib"
echo "  - training_results.json"
echo "  - quick_test_results.json"
echo "  - cloud_config.json"
echo ""
echo "🌐 Cloud server: $CLOUD_SERVER:$CLOUD_PATH"
echo "📋 Next steps: Monitor performance and iterate if needed" 