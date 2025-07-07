# 🚀 Cloud Deployment Guide

## Hướng dẫn triển khai model SVM lên cloud và test

### 📋 Tổng quan
Sau khi train model thành công local, bạn cần:
1. Upload project lên cloud server
2. Khởi động ML Service  
3. Test model với script có sẵn

---

## 🔧 Bước 1: Upload lên Cloud

### Option A: Git Clone (Recommended)
```bash
# Trên cloud server
cd /opt/
sudo git clone https://github.com/YOUR_USERNAME/mccva.git
sudo mv mccva /opt/mccva
cd /opt/mccva
```

### Option B: Direct Upload
```bash
# Từ local machine upload lên cloud
scp -r . user@your-cloud-ip:/opt/mccva/
```

---

## 🐍 Bước 2: Setup Environment

```bash
# Trên cloud server
cd /opt/mccva

# Install Python dependencies
sudo apt update
sudo apt install python3-pip python3-venv -y

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 🚀 Bước 3: Khởi động ML Service

### Option A: Development Server với ml_service.py
```bash
# Activate venv
cd /opt/mccva
source venv/bin/activate

# Run ML Service
python3 ml_service.py
```

### Option B: Production với Gunicorn
```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 ml_service:app
```

### Option C: Docker (Recommended for Production)
```bash
# Update Dockerfile để chạy ml_service.py
# Build và chạy với Docker
docker-compose up --build -d

# Hoặc manual Docker
docker build -t mccva-svm .
docker run -d -p 5000:5000 --name mccva-api mccva-svm
```

---

## 🧪 Bước 4: Test Deployed ML Service

### Quick Test từ Local Machine
```bash
# Test từ local về cloud
python3 test_cloud_deployment.py YOUR_CLOUD_IP

# Ví dụ:
python3 test_cloud_deployment.py 192.168.1.100
python3 test_cloud_deployment.py ec2-xx-xx-xx-xx.compute-1.amazonaws.com
```

### Test trực tiếp trên Cloud Server
```bash
# Test local trên cloud server
cd /opt/mccva
python3 test_cloud_deployment.py localhost
```

### Manual API Test với curl

#### Health Check
```bash
curl http://YOUR_CLOUD_IP:5000/health
```

#### Test SVM Prediction
```bash
curl -X POST http://YOUR_CLOUD_IP:5000/predict/makespan \
  -H "Content-Type: application/json" \
  -d '{
    "features": [8, 8, 80, 3500, 4, 4, 450, 35, 800, 4]
  }'
```

#### Test K-Means Clustering
```bash
curl -X POST http://YOUR_CLOUD_IP:5000/predict/vm_cluster \
  -H "Content-Type: application/json" \
  -d '{
    "vm_features": [0.6, 0.7, 0.5]
  }'
```

#### Get Models Info
```bash
curl http://YOUR_CLOUD_IP:5000/models/info
```

---

## 📊 Bước 5: Verify Model Performance

### Expected Test Results
Script `test_cloud_deployment.py` sẽ chạy 5 scenarios:

| Scenario | Expected | Features Description |
|----------|----------|---------------------|
| Light Web Request | `small` | 2 cores, 0.5GB RAM, low network |
| Medium API Call | `medium` | 4 cores, 2GB RAM, medium load |
| Heavy Processing | `large` | 8 cores, 8GB RAM, high network |
| High CPU Only | `large` | 12 cores, 1GB RAM |
| High Memory Only | `large` | 2 cores, 16GB RAM |

### Success Criteria
- ✅ Health check: 200 OK với SVM + K-Means loaded
- ✅ All 5 SVM predictions: Correct class
- ✅ K-Means clustering: Working
- ✅ Average response time: < 1000ms
- ✅ API reliability: 100%

---

## 🔧 Troubleshooting

### Problem: Model files not found
```bash
# Check if models directory exists
ls -la /opt/mccva/models/

# Should see:
# svm_model.joblib, svm_scaler.joblib, svm_label_encoder.joblib
# kmeans_model.joblib, kmeans_scaler.joblib, etc.
```

### Problem: ML Service not starting
```bash
# Check Python version
python3 --version  # Needs 3.8+

# Check dependencies
pip list | grep -E "(flask|joblib|sklearn|numpy|pandas)"

# Test model loading
cd /opt/mccva
python3 -c "import joblib; print(joblib.load('models/svm_model.joblib'))"

# Check port availability
sudo netstat -tlnp | grep :5000
```

### Problem: Permission denied
```bash
# Fix permissions
sudo chown -R $USER:$USER /opt/mccva
chmod +x test_cloud_deployment.py
```

### Problem: Firewall blocking
```bash
# Open port 5000
sudo ufw allow 5000
# hoặc
sudo iptables -A INPUT -p tcp --dport 5000 -j ACCEPT
```

---

## 🌐 Production Setup

### Nginx Reverse Proxy
```nginx
# /etc/nginx/sites-available/mccva
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    location /predict/ {
        proxy_pass http://127.0.0.1:5000/predict/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Systemd Service
```ini
# /etc/systemd/system/mccva.service
[Unit]
Description=MCCVA ML Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/mccva
Environment=PATH=/opt/mccva/venv/bin
ExecStart=/opt/mccva/venv/bin/gunicorn --bind 0.0.0.0:5000 --workers 4 ml_service:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable service
sudo systemctl enable mccva
sudo systemctl start mccva
sudo systemctl status mccva
```

---

## 📱 Integration với OpenResty

### Sample Lua Code
```lua
-- /opt/mccva/lua/ml_predict.lua
local http = require "resty.http"
local cjson = require "cjson"

local function svm_classify_request(features)
    local httpc = http.new()
    
    local request_data = {
        features = features  -- Array of 10 features
    }
    
    local res, err = httpc:request_uri("http://127.0.0.1:5000/predict/makespan", {
        method = "POST",
        body = cjson.encode(request_data),
        headers = {
            ["Content-Type"] = "application/json"
        }
    })
    
    if not res then
        ngx.log(ngx.ERR, "Failed to call SVM API: ", err)
        return "medium"  -- fallback
    end
    
    local result = cjson.decode(res.body)
    return result.makespan or "medium"
end

local function kmeans_cluster_vm(vm_features)
    local httpc = http.new()
    
    local request_data = {
        vm_features = vm_features  -- [cpu_usage, ram_usage, storage_usage]
    }
    
    local res, err = httpc:request_uri("http://127.0.0.1:5000/predict/vm_cluster", {
        method = "POST",
        body = cjson.encode(request_data),
        headers = {
            ["Content-Type"] = "application/json"
        }
    })
    
    if not res then
        ngx.log(ngx.ERR, "Failed to call K-Means API: ", err)
        return 0  -- fallback cluster
    end
    
    local result = cjson.decode(res.body)
    return result.cluster or 0
end

-- Export functions
return {
    svm_classify_request = svm_classify_request,
    kmeans_cluster_vm = kmeans_cluster_vm
}
```

### Example Usage in OpenResty
```lua
-- In your OpenResty config
local ml = require "ml_predict"

-- Example: Classify incoming request
local features = {8, 8, 80, 3500, 4, 4, 450, 35, 800, 4}
local classification = ml.svm_classify_request(features)
ngx.log(ngx.INFO, "Request classified as: " .. classification)

-- Example: Get VM cluster
local vm_usage = {0.6, 0.7, 0.5}  -- cpu, ram, storage usage
local cluster = ml.kmeans_cluster_vm(vm_usage)
ngx.log(ngx.INFO, "VM assigned to cluster: " .. cluster)
```

---

## ✅ Success Checklist

- [ ] Project cloned to `/opt/mccva`
- [ ] Models uploaded to `/opt/mccva/models/` directory
- [ ] ML Service (ml_service.py) running on port 5000
- [ ] Health check returns 200 OK với models loaded
- [ ] Test script passes all 5 SVM scenarios
- [ ] K-Means clustering working
- [ ] Average response time < 1000ms
- [ ] Production setup (systemd/docker) configured
- [ ] Firewall/security configured
- [ ] OpenResty Lua integration tested

**🎉 Khi tất cả checklist xong → ML Service ready for production!**

---

## 🎯 Quick Commands Summary

```bash
# Clone and setup
cd /opt/ && sudo git clone https://github.com/YOUR_USERNAME/mccva.git
sudo mv mccva /opt/mccva && cd /opt/mccva
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Start ML Service
python3 ml_service.py

# Test from another terminal
python3 test_cloud_deployment.py localhost

# Production start
gunicorn --bind 0.0.0.0:5000 --workers 4 ml_service:app
``` 