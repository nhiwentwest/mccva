# 🚀 EC2 System Restart Guide - After SVM Model Fix

## 📝 Overview
Sau khi fix SVM model từ 5-class → 3-class system, cần restart toàn bộ hệ thống trên EC2 để load model mới.

## 🔄 Complete Restart Sequence

### 1. SSH vào EC2
```bash
# Connect to your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### 2. 🛑 Stop All Services

#### A. Kill ML Service Screen
```bash
# Check current screens
screen -ls

# Kill ML service screen (thường tên là 'ml_service')
screen -S ml_service -X quit

# Or if you know the screen ID
screen -S 12345.ml_service -X quit

# Double check all screens killed
screen -ls
```

#### B. Stop OpenResty/Nginx
```bash
# Stop OpenResty
sudo systemctl stop openresty

# Or if using nginx
sudo systemctl stop nginx

# Check status
sudo systemctl status openresty
```

### 3. 📥 Pull Latest Code
```bash
# Navigate to project directory
cd /opt/mccva

# Pull latest code with SVM fixes
sudo git pull origin main

# Check if new files are there
ls -la models/
```

### 4. 🧠 Retrain SVM Model on EC2
```bash
# Run the fixed SVM training script
cd /opt/mccva
python retrain_balanced_svm.py

# Verify new model files created
ls -la models/svm_*
```

Expected output:
```
✅ SVM Model saved: models/svm_model.joblib
✅ SVM Scaler saved: models/svm_scaler.joblib  
✅ SVM Label Encoder saved: models/svm_label_encoder.joblib
✅ SVM Features saved: models/svm_feature_names.joblib
```

### 5. 🔄 Restart ML Service
```bash
# Create new screen for ML service
screen -S ml_service

# Inside screen, start ML service
cd /opt/mccva
python ml_service.py

# Detach from screen: Ctrl+A then D
```

### 6. 🔄 Restart OpenResty
```bash
# Start OpenResty
sudo systemctl start openresty

# Check status
sudo systemctl status openresty

# Check if port 80 is listening
sudo netstat -tlnp | grep :80
```

### 7. ✅ Verify All Services

#### A. Check ML Service Health
```bash
# Test health endpoint
curl http://localhost:5000/health

# Expected response:
{
  "status": "healthy",
  "models_loaded": {
    "svm_model": true,
    "svm_label_encoder": true,
    "kmeans_model": true,
    "meta_learning_model": true
  }
}
```

#### B. Test SVM Prediction
```bash
# Test single prediction
curl -X POST http://localhost:5000/predict/makespan \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_cores": 8,
    "memory": 32,
    "storage": 500,
    "network_bandwidth": 10000,
    "priority": 1
  }'

# Should now return "large" instead of "small"
```

#### C. Test Full MCCVA Pipeline
```bash
# Test complete pipeline
curl -X POST http://localhost:5000/predict/mccva_complete \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_cores": 8,
    "memory": 32,
    "storage": 500,
    "network_bandwidth": 10000,
    "priority": 1,
    "cpu_usage": 95.8,
    "memory_usage": 88.5,
    "storage_usage": 75.2
  }'
```

### 8. 🎯 Run Demo Test
```bash
# Run demo scenarios to verify fix
cd /opt/mccva
python demo_scenarios.py
```

Expected: Predictions should now be diverse (small/medium/large) instead of all "small"

## 🔍 Troubleshooting

### Screen Issues
```bash
# List all screens
screen -ls

# If screen is "dead", clean it
screen -wipe

# If screen won't die
ps aux | grep python
sudo kill -9 <PID>
```

### OpenResty Issues
```bash
# Check OpenResty config
sudo openresty -t

# Check logs
sudo tail -f /var/log/openresty/error.log
sudo tail -f /var/log/openresty/access.log

# Restart if issues
sudo systemctl restart openresty
```

### Model Loading Issues
```bash
# Check if model files exist
ls -la /opt/mccva/models/

# Check model file permissions
sudo chown -R $USER:$USER /opt/mccva/models/

# Check Python environment
which python
pip list | grep scikit-learn
```

### Port Issues
```bash
# Check what's using ports
sudo netstat -tlnp | grep :5000
sudo netstat -tlnp | grep :80

# Kill process using port
sudo lsof -ti:5000 | xargs sudo kill -9
```

## 📊 Verification Checklist

- [ ] All screens killed and recreated
- [ ] Latest code pulled from GitHub  
- [ ] SVM model retrained with 3-class output
- [ ] ML service running on port 5000
- [ ] OpenResty running on port 80
- [ ] Health endpoint returns "healthy"
- [ ] SVM predictions return diverse classes
- [ ] Demo script shows improved accuracy
- [ ] All 5 scenarios process without errors

## 🎯 Expected Demo Results After Fix

**Before Fix:**
- All predictions: "small" (100% wrong)
- Accuracy: 20% (1/5 correct)

**After Fix:**  
- Diverse predictions: small/medium/large
- Accuracy: 80%+ expected
- Confidence levels: Realistic distributions

## 📞 Support Commands

```bash
# Monitor ML service logs in real-time
screen -r ml_service

# Check system resources
htop
df -h
free -h

# Test individual endpoints
curl http://localhost:5000/health
curl http://localhost:5000/models/info
curl http://localhost:80/health  # Through OpenResty
```

---
**💡 Tips:**
- Always run `screen -ls` to check current sessions
- Use `sudo systemctl status openresty` to check service status  
- Model training takes ~3 minutes on EC2
- Keep terminal open during model training
- Test each endpoint individually before running full demo 