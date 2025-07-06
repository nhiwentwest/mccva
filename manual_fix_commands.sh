#!/bin/bash
# Manual Fix Commands - Run these directly on cloud server
echo "🔧 Manual Fix for Class Mapping Issue"
echo "====================================="

cat << 'EOF'
# Run these commands directly on your cloud server:

# 1. Check current class mapping in ml_service.py
echo "Current class mapping:"
grep -A3 "svm_class_mapping" ml_service.py

# 2. Create backup
cp ml_service.py ml_service.py.backup_$(date +%s)

# 3. Fix class mapping directly with sed
sed -i 's/svm_class_mapping = {0: "small", 1: "medium", 2: "large"}/svm_class_mapping = {0: "large", 1: "medium", 2: "small"}/' ml_service.py

# 4. Verify the fix
echo "After fix:"
grep -A3 "svm_class_mapping" ml_service.py

# 5. Restart ML service
sudo pkill -f ml_service.py
sleep 2
nohup python3 ml_service.py > ml_service.log 2>&1 &
sleep 3

# 6. Test individual predictions
echo "Testing predictions:"

echo "Web Server (2,4,50,500,1) - should be small:"
curl -s -X POST http://localhost:5000/predict/enhanced \
  -H "Content-Type: application/json" \
  -d '{"features": [2, 4, 50, 500, 1]}' | python3 -c "
import sys, json
result = json.load(sys.stdin)
print(f\"SVM: {result['model_contributions']['svm']['prediction']}\")
print(f\"Final: {result['makespan']}\")
"

echo "Database Server (4,8,100,1000,3) - should be medium:"
curl -s -X POST http://localhost:5000/predict/enhanced \
  -H "Content-Type: application/json" \
  -d '{"features": [4, 8, 100, 1000, 3]}' | python3 -c "
import sys, json
result = json.load(sys.stdin)
print(f\"SVM: {result['model_contributions']['svm']['prediction']}\")
print(f\"Final: {result['makespan']}\")
"

echo "ML Training (12,32,500,5000,5) - should be large:"
curl -s -X POST http://localhost:5000/predict/enhanced \
  -H "Content-Type: application/json" \
  -d '{"features": [12, 32, 500, 5000, 5]}' | python3 -c "
import sys, json
result = json.load(sys.stdin)
print(f\"SVM: {result['model_contributions']['svm']['prediction']}\")
print(f\"Final: {result['makespan']}\")
"

# 7. If still wrong, check label encoder file
echo "Checking label encoder mapping:"
python3 -c "
import joblib
import numpy as np
try:
    le = joblib.load('models/label_encoder.joblib')
    print('Label encoder classes:', le.classes_)
    print('Label mapping:', dict(zip(le.classes_, le.transform(le.classes_))))
except:
    print('No label encoder found')
"

# 8. If label encoder exists, use it instead
echo "Creating fix using label encoder:"
cat > fix_mapping.py << 'PYEOF'
import re

# Read ml_service.py
with open('ml_service.py', 'r') as f:
    content = f.read()

# Find and replace the class mapping section
old_mapping = r'svm_class_mapping = \{.*?\}'
new_mapping = '''# Load label encoder for correct mapping
        try:
            import joblib
            label_encoder = joblib.load('models/label_encoder.joblib')
            svm_prediction = label_encoder.inverse_transform([int(svm_prediction_int)])[0]
        except:
            # Fallback mapping based on training: {large: 0, medium: 1, small: 2}
            svm_class_mapping = {0: "large", 1: "medium", 2: "small"}
            svm_prediction = svm_class_mapping.get(int(svm_prediction_int), "medium")'''

# Replace the mapping
content = re.sub(old_mapping, new_mapping, content, flags=re.DOTALL)

# Also remove the old line that uses svm_class_mapping
content = content.replace(
    'svm_prediction = svm_class_mapping.get(int(svm_prediction_int), "medium")  # Default to medium if unknown',
    '# Mapping handled above'
)

# Write back
with open('ml_service.py', 'w') as f:
    f.write(content)

print("Fixed ml_service.py to use label encoder")
PYEOF

python3 fix_mapping.py
rm fix_mapping.py

# 9. Restart service again
sudo pkill -f ml_service.py
sleep 2
nohup python3 ml_service.py > ml_service.log 2>&1 &
sleep 3

# 10. Final test
echo "Final accuracy test:"
python3 test_ai_routing.py | head -50

EOF 