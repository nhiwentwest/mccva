# 🚀 Cloud Deployment Guide

## Hướng dẫn triển khai model SVM lên cloud và test

### 📋 Tổng quan
Sau khi train model thành công local, bạn cần:
1. Upload project lên cloud server
2. Khởi động Flask API service  
3. Test model với script có sẵn

---

## 🔧 Bước 1: Upload lên Cloud

### Option A: Git Clone (Recommended)
```bash
# Trên cloud server
cd /opt/
sudo git clone https://github.com/YOUR_USERNAME/mccva.git
cd mccva
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

### Option A: Flask Development Server
```bash
# Activate venv
source venv/bin/activate

# Run Flask app
python3 app.py
# hoặc
FLASK_APP=app.py flask run --host=0.0.0.0 --port=5000
```

### Option B: Production với Gunicorn
```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

### Option C: Docker (Recommended for Production)
```bash
# Build và chạy với Docker
docker-compose up --build -d

# Hoặc manual Docker
docker build -t mccva-svm .
docker run -d -p 5000:5000 --name mccva-api mccva-svm
```

---

## 🧪 Bước 4: Test Deployed Model

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
python3 test_cloud_deployment.py localhost
```

### Manual API Test với curl
```bash
# Health check
curl http://YOUR_CLOUD_IP:5000/health

# Test prediction
curl -X POST http://YOUR_CLOUD_IP:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_cores": 8,
    "memory_mb": 8192, 
    "jobs_1min": 45,
    "jobs_5min": 180,
    "network_receive": 2000,
    "network_transmit": 1500,
    "cpu_speed": 3.6
  }'
```

---

## 📊 Bước 5: Verify Model Performance

### Expected Test Results
Script `test_cloud_deployment.py` sẽ chạy 5 scenarios:

| Scenario | Expected | Features |
|----------|----------|----------|
| Light Web Request | `small` | 2 cores, 512MB RAM |
| Medium API Call | `medium` | 4 cores, 2GB RAM |
| Heavy Processing | `large` | 8 cores, 8GB RAM |
| High CPU Only | `large` | 12 cores, 1GB RAM |
| High Memory Only | `large` | 2 cores, 16GB RAM |

### Success Criteria
- ✅ Health check: 200 OK
- ✅ All 5 predictions: Correct class
- ✅ Average response time: < 1000ms
- ✅ API reliability: 100%

---

## 🔧 Troubleshooting

### Problem: Model files not found
```bash
# Check if models directory exists
ls -la models/

# Nếu không có, clone lại hoặc upload models/
```

### Problem: Flask app not starting
```bash
# Check Python version
python3 --version  # Needs 3.8+

# Check dependencies
pip list | grep -E "(flask|joblib|sklearn|numpy|pandas)"

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
    }
}
```

### Systemd Service
```ini
# /etc/systemd/system/mccva.service
[Unit]
Description=MCCVA SVM API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/mccva
Environment=PATH=/opt/mccva/venv/bin
ExecStart=/opt/mccva/venv/bin/gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
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
-- /opt/mccva/lua/svm_predict.lua
local http = require "resty.http"
local cjson = require "cjson"

local function classify_request(cpu_cores, memory_mb, jobs_1min, jobs_5min, network_receive, network_transmit, cpu_speed)
    local httpc = http.new()
    
    local request_data = {
        cpu_cores = cpu_cores,
        memory_mb = memory_mb,
        jobs_1min = jobs_1min,
        jobs_5min = jobs_5min,
        network_receive = network_receive,
        network_transmit = network_transmit,
        cpu_speed = cpu_speed
    }
    
    local res, err = httpc:request_uri("http://127.0.0.1:5000/predict", {
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
    return result.prediction or "medium"
end

-- Export function
return {
    classify_request = classify_request
}
```

---

## ✅ Success Checklist

- [ ] Models uploaded to `models/` directory
- [ ] Flask app running on port 5000
- [ ] Health check returns 200 OK
- [ ] Test script passes all 5 scenarios
- [ ] Average response time < 1000ms
- [ ] Production setup (systemd/docker) configured
- [ ] Firewall/security configured
- [ ] OpenResty integration tested

**🎉 Khi tất cả checklist xong → Model ready for production!** 