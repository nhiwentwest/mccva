#!/bin/bash

echo "🔧 Fixing indentation error in ml_service.py..."

cd /opt/mccva

# Stop any running ml_service
echo "⏹️  Stopping ml_service..."
pkill -f "python.*ml_service"
sleep 2

# Backup current ml_service.py
echo "💾 Creating backup..."
cp ml_service.py ml_service.py.backup.$(date +%s)

echo "🔧 Fixing model info logging with proper indentation..."
cat > temp_fix.py << 'EOF'
# Fix the model loading section
import re

# Read the file
with open('ml_service.py', 'r') as f:
    content = f.read()

# Fix the specific indentation issue around line 56-57
# Replace the malformed if statement with proper code
old_pattern = r'if hasattr\(svm_model, "kernel"\):\s*logger\.info\(f"SVM Model: \{svm_model\.kernel\} kernel, \{sum\(svm_model\.n_support_\)\} support vectors"\)\s*else:\s*logger\.info\(f"Model Type: \{type\(svm_model\)\.__name__\}, Features: \{getattr\(svm_model, \\"n_features_in_\", \\"unknown\\"\)\}"\)'

new_code = '''if hasattr(svm_model, "kernel"):
        logger.info(f"SVM Model: {svm_model.kernel} kernel, {sum(svm_model.n_support_)} support vectors")
    else:
        logger.info(f"Model Type: {type(svm_model).__name__}, Features: {getattr(svm_model, 'n_features_in_', 'unknown')}")'''

# Also handle the case where it might be on multiple lines with wrong indentation
lines = content.split('\n')
fixed_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if 'if hasattr(svm_model, "kernel"):' in line:
        # Replace this section with properly formatted code
        fixed_lines.append('    if hasattr(svm_model, "kernel"):')
        fixed_lines.append('        logger.info(f"SVM Model: {svm_model.kernel} kernel, {sum(svm_model.n_support_)} support vectors")')
        fixed_lines.append('    else:')
        fixed_lines.append('        logger.info(f"Model Type: {type(svm_model).__name__}, Features: {getattr(svm_model, \'n_features_in_\', \'unknown\')}")')
        # Skip the malformed lines
        while i < len(lines) and ('logger.info(' in lines[i] or 'else:' in lines[i] or 'Model Type:' in lines[i]):
            i += 1
        i -= 1  # Adjust for the while loop increment
    else:
        fixed_lines.append(line)
    i += 1

# Write back the fixed content
with open('ml_service.py', 'w') as f:
    f.write('\n'.join(fixed_lines))

print("✅ Fixed indentation issues")
EOF

python3 temp_fix.py
rm temp_fix.py

echo "🧪 Testing model loading after fix..."
python3 -c "
import sys
sys.path.append('/opt/mccva')
try:
    from ml_service import load_models
    print('✅ Attempting to load models...')
    load_models()
    print('✅ Models loaded successfully!')
except Exception as e:
    print(f'❌ Error loading models: {e}')
    import traceback
    traceback.print_exc()
"

echo "🚀 Starting ml_service..."
nohup python3 ml_service.py > ml_service.log 2>&1 &
sleep 5

# Check if service started
if pgrep -f "python.*ml_service" > /dev/null; then
    echo "✅ ml_service started successfully!"
    
    # Test health
    echo "🏥 Testing service health..."
    curl -s http://localhost:5000/health
    
    echo ""
    echo "🧪 Testing prediction..."
    curl -s -X POST http://localhost:5000/predict/makespan \
         -H "Content-Type: application/json" \
         -d '{"features": [4, 8, 100, 1000, 3, 2, 50, 20, 500, 3]}' | python3 -m json.tool
    
    echo ""
    echo "🎯 Running accuracy test..."
    python3 quick_accuracy_test.py
else
    echo "❌ ml_service failed to start. Check logs:"
    tail -20 ml_service.log
fi 