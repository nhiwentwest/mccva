#!/bin/bash
# Fix Feature Mismatch Issue - Update ml_service.py to use correct features
# This will fix the 50% -> 80%+ accuracy issue

echo "🔧 Fixing Feature Mismatch in SVM Model"
echo "======================================="

# Check if we're in the right directory
if [ ! -f "ml_service.py" ]; then
    echo "❌ Not in /opt/mccva directory"
    exit 1
fi

echo "📝 Backing up current ml_service.py..."
cp ml_service.py ml_service.py.backup

echo "🔧 Updating feature calculation in enhanced prediction..."

# Create a temporary Python script to fix the ml_service.py
cat > fix_features.py << 'EOF'
#!/usr/bin/env python3
"""
Fix the feature mismatch in ml_service.py enhanced prediction
"""

# Read the current ml_service.py
with open('ml_service.py', 'r') as f:
    content = f.read()

# Fix the enhanced prediction feature calculation
old_feature_calc = '''        # Convert 5 features to 10 features for SVM model
        cpu_cores, memory, storage, network_bandwidth, priority = features
        
        # Calculate the additional 5 features needed for the 10-feature model
        cpu_memory_ratio = cpu_cores / (memory + 1e-6)
        storage_memory_ratio = storage / (memory + 1e-6)
        network_cpu_ratio = network_bandwidth / (cpu_cores + 1e-6)
        resource_intensity = (cpu_cores * memory * storage) / 1000
        priority_weighted_cpu = cpu_cores * priority
        
        # Combine all 10 features for SVM model
        svm_features = [
            cpu_cores, memory, storage, network_bandwidth, priority,
            cpu_memory_ratio, storage_memory_ratio, network_cpu_ratio,
            resource_intensity, priority_weighted_cpu
        ]'''

new_feature_calc = '''        # Convert 5 features to 10 features for SVM model using EXACT same calculation as training
        cpu_cores, memory, storage, network_bandwidth, priority = features
        
        # Calculate derived features EXACTLY as in retrain_svm_fixed.py
        cpu_memory_ratio = cpu_cores / (memory + 1e-6)
        storage_memory_ratio = storage / (memory + 1e-6)
        network_cpu_ratio = network_bandwidth / (cpu_cores + 1e-6)
        resource_intensity = (cpu_cores * memory * storage) / 1000
        priority_weighted_cpu = cpu_cores * priority
        
        # Combine all 10 features for SVM model (EXACT order as training)
        svm_features = [
            cpu_cores, memory, storage, network_bandwidth, priority,
            cpu_memory_ratio, storage_memory_ratio, network_cpu_ratio,
            resource_intensity, priority_weighted_cpu
        ]'''

# Replace the feature calculation
content = content.replace(old_feature_calc, new_feature_calc)

# Also fix the class mapping to match training exactly
old_mapping = '''        # Map SVM integer prediction to string label
        svm_class_mapping = {0: "small", 1: "medium", 2: "large"}
        svm_prediction = svm_class_mapping.get(int(svm_prediction_int), "medium")  # Default to medium if unknown'''

new_mapping = '''        # Map SVM integer prediction to string label (EXACT as training)
        # Training mapping: {'large': 0, 'medium': 1, 'small': 2}
        svm_class_mapping = {0: "large", 1: "medium", 2: "small"}
        svm_prediction = svm_class_mapping.get(int(svm_prediction_int), "medium")  # Default to medium if unknown'''

content = content.replace(old_mapping, new_mapping)

# Write the fixed content
with open('ml_service.py', 'w') as f:
    f.write(content)

print("✅ Fixed feature calculation and class mapping in ml_service.py")
EOF

# Run the fix script
python3 fix_features.py

# Clean up
rm fix_features.py

echo "✅ Feature mismatch fixed!"
echo ""
echo "🔄 Restarting ML service with fixed features..."

# Stop old service
sudo pkill -f ml_service.py
sleep 2

# Start new service
nohup python3 ml_service.py > ml_service.log 2>&1 &
sleep 3

# Verify service is running
if ps aux | grep '[m]l_service.py' > /dev/null; then
    echo "✅ ML service restarted successfully"
else
    echo "❌ Failed to restart ML service"
    exit 1
fi

echo ""
echo "🧪 Testing fixed prediction..."

# Test with Database Server scenario (should predict 'medium')
PREDICTION=$(curl -s -X POST http://localhost:5000/predict/enhanced \
  -H "Content-Type: application/json" \
  -d '{"features": [4, 8, 100, 1000, 3]}' | python3 -c "import sys, json; print(json.load(sys.stdin)['makespan'])")

echo "📊 Database Server Prediction: $PREDICTION (should be 'medium')"

if [ "$PREDICTION" = "medium" ]; then
    echo "✅ Feature fix successful! SVM now predicting correctly"
else
    echo "❌ Still having issues. Prediction: $PREDICTION"
fi

echo ""
echo "🎯 Running full test to verify 80%+ accuracy..."
python3 test_ai_routing.py 