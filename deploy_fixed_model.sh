#!/bin/bash
# Deploy Fixed SVM Model - Manual Commands for Cloud Deployment
# Run these commands on your cloud server

echo "🚀 MCCVA Fixed SVM Model Deployment Commands"
echo "============================================="
echo ""
echo "Run these commands on your cloud server:"
echo ""

echo "# 1. Navigate to project directory"
echo "cd /opt/mccva"
echo ""

echo "# 2. Pull latest changes with fixed SVM model"
echo "git pull origin main"
echo ""

echo "# 3. Install Python dependencies (if needed)"
echo "sudo apt-get update"
echo "sudo apt-get install -y python3-flask python3-numpy python3-sklearn python3-joblib python3-pandas"
echo ""

echo "# 4. Retrain model on cloud (creates models/ directory)"
echo "python3 retrain_svm_fixed.py"
echo ""

echo "# 5. Restart ML service to load new model"
echo "sudo pkill -f ml_service.py"
echo "nohup python3 ml_service.py > ml_service.log 2>&1 &"
echo ""

echo "# 6. Test the fixed SVM model"
echo "python3 test_ai_routing.py"
echo ""

echo "# 7. Restart OpenResty to reload any changes"
echo "sudo systemctl restart openresty"
echo ""

echo "# 8. Test the full routing system"
echo 'curl -X POST http://localhost:5000/route/mccva \\'
echo '  -H "Content-Type: application/json" \\'
echo '  -d "{\\"cpu_cores\\": 4, \\"memory\\": 8, \\"storage\\": 100, \\"network_bandwidth\\": 1000, \\"priority\\": 3, \\"data_size\\": 500}"'
echo ""

echo "Expected: The AI should now predict 'medium' correctly!"
echo ""
echo "# 9. Check service status"
echo "ps aux | grep -E '(ml_service|nginx|openresty)'"
echo ""

echo "📊 Expected Results:"
echo "==================="
echo "✅ SVM Model: 100% accuracy on test scenarios"
echo "✅ AI Prediction Accuracy: Should improve from 33% to 80%+"
echo "✅ All 6 test scenarios should predict correctly"
echo ""

echo "🎯 The model will now correctly predict:"
echo "• Web Server (2/4/50/500/1) → small"
echo "• Database Server (4/8/100/1000/3) → medium"  
echo "• ML Training (12/32/500/5000/5) → large"
echo "• Video Rendering (16/64/800/8000/4) → large"
echo "• API Gateway (1/2/20/2000/2) → small"
echo "• File Server (6/12/200/1500/3) → medium" 