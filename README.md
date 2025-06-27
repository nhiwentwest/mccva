# 🤖 MCCVA Algorithm Implementation

**Makespan Classification & Clustering VM Algorithm** - Triển khai thuật toán MCCVA với OpenResty trên Amazon Cloud Ubuntu.

## 🎯 **Mục tiêu**

Triển khai thuật toán MCCVA (Makespan Classification & Clustering VM Algorithm) là giải pháp cân bằng tải sử dụng trí tuệ nhân tạo, gồm ba module chính:

1. **SVM Classification** - Phân loại yêu cầu dựa trên makespan
2. **K-Means Clustering** - Phân cụm các máy ảo (VM) theo tài nguyên
3. **MCCVA Routing** - Phân phối yêu cầu tới VM phù hợp

## 🏗️ **Kiến trúc hệ thống**

```
Client Request → OpenResty → MCCVA Algorithm → Target VM
                      ↓
              [SVM + K-Means Models]
              [AI-based Load Balancing]
```

### **Components:**

- **OpenResty + LuaJIT**: API Gateway và routing logic
- **Flask API**: ML Service với trained models
- **SVM Model**: Makespan classification
- **K-Means Model**: VM clustering
- **MCCVA Algorithm**: Intelligent load balancing

## 🚀 **Triển khai nhanh**

### **1. Chuẩn bị server Amazon Cloud Ubuntu:**

```bash
# Kết nối vào EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Clone hoặc upload project
git clone <repository-url>
cd mccva-project
```

### **2. Chạy deployment script:**

```bash
# Cấp quyền thực thi
chmod +x deploy.sh

# Chạy deployment
./deploy.sh
```

### **3. Kiểm tra deployment:**

```bash
# Comprehensive deployment check
python3 check_deployment.py

# Test MCCVA algorithm
python3 test_mccva.py

# Kiểm tra services
sudo systemctl status mccva-ml
sudo systemctl status openresty
```

## 📊 **API Endpoints**

### **1. MCCVA Routing (Main Endpoint):**
```bash
curl -X POST http://localhost/mccva/route \
  -H "Content-Type: application/json" \
  -d '{
    "features": [4, 8, 100, 1000, 3],
    "vm_features": [0.5, 0.5, 0.5]
  }'
```

**Response:**
```json
{
  "target_vm": "http://127.0.0.1:8083",
  "routing_info": {
    "method": "mccva_svm_primary",
    "algorithm": "SVM Classification",
    "confidence": 2.847
  },
  "mccva_decision": {
    "makespan_prediction": "medium",
    "cluster_prediction": 3,
    "confidence_score": 2.847,
    "algorithm_used": "SVM Classification"
  }
}
```

### **2. SVM Classification:**
```bash
curl -X POST http://localhost/predict/makespan \
  -H "Content-Type: application/json" \
  -d '{"features": [4, 8, 100, 1000, 3]}'
```

### **3. K-Means Clustering:**
```bash
curl -X POST http://localhost/predict/vm_cluster \
  -H "Content-Type: application/json" \
  -d '{"vm_features": [0.5, 0.5, 0.5]}'
```

### **4. Health Check:**
```bash
curl http://localhost/health
```

## 🔧 **Cấu hình**

### **File cấu hình chính:**

- **`nginx.conf`**: OpenResty configuration
- **`lua/mccva_routing.lua`**: MCCVA algorithm implementation
- **`ml_service.py`**: Flask API cho ML models
- **`models/`**: Trained AI models (SVM + K-Means)

### **Service Management:**

```bash
# ML Service
sudo systemctl start|stop|restart|status mccva-ml

# OpenResty
sudo systemctl start|stop|restart|status openresty

# View logs
sudo journalctl -u mccva-ml -f
sudo tail -f /var/log/nginx/access.log
```

## 📈 **MCCVA Algorithm Logic**

### **1. SVM Classification:**
- Input: Task features `[cpu_cores, memory_gb, storage_gb, network_bandwidth, priority]`
- Output: Makespan classification `[small, medium, large]`
- Confidence score for routing decision

### **2. K-Means Clustering:**
- Input: VM usage features `[cpu_usage, ram_usage, storage_usage]`
- Output: VM cluster `[0-5]` based on resource patterns
- 6 clusters: Low, Medium, High, Balanced, CPU-intensive, Storage-intensive

### **3. MCCVA Routing:**
- Priority 1: High confidence SVM routing
- Priority 2: K-Means cluster-based routing
- Priority 3: Ensemble decision (SVM + K-Means)
- Load balancing with weighted distribution

## 🧪 **Testing**

### **Run comprehensive tests:**
```bash
# Full deployment check
python3 check_deployment.py

# MCCVA algorithm test
python3 test_mccva.py
```

### **Test individual components:**
```bash
# Test SVM
curl -X POST http://localhost/predict/makespan \
  -d '{"features": [2, 4, 50, 500, 1]}'

# Test K-Means
curl -X POST http://localhost/predict/vm_cluster \
  -d '{"vm_features": [0.3, 0.2, 0.1]}'

# Test MCCVA
curl -X POST http://localhost/mccva/route \
  -d '{"features": [2, 4, 50, 500, 1], "vm_features": [0.3, 0.2, 0.1]}'
```

## 📁 **Project Structure**

```
mccva-project/
├── lua/
│   ├── mccva_routing.lua      # MCCVA algorithm implementation
│   ├── predict_makespan.lua   # SVM endpoint
│   └── predict_vm_cluster.lua # K-Means endpoint
├── models/
│   ├── svm_model.joblib       # Trained SVM model
│   ├── kmeans_model.joblib    # Trained K-Means model
│   ├── scaler.joblib          # SVM scaler
│   └── kmeans_scaler.joblib   # K-Means scaler
├── ml_service.py              # Flask ML API
├── nginx.conf                 # OpenResty configuration
├── deploy.sh                  # Deployment script
├── check_deployment.py        # Deployment verification
├── test_mccva.py             # Test suite
└── requirements.txt           # Python dependencies
```

## 🎯 **Performance Metrics**

### **Expected Performance:**
- **Routing Time**: < 100ms (excellent), < 500ms (good)
- **SVM Prediction**: < 50ms
- **K-Means Clustering**: < 30ms
- **Load Balancing**: Weighted distribution (70/30, 60/40, 80/20)

### **Monitoring:**
```bash
# Check performance
python3 check_deployment.py

# Monitor logs
sudo tail -f /var/log/nginx/access.log

# Check service status
sudo systemctl status mccva-ml openresty
```

## 🔒 **Security**

### **Firewall Configuration:**
```bash
# Allow necessary ports
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 5000/tcp  # ML Service
sudo ufw allow 22/tcp    # SSH
```

### **Service Isolation:**
- ML Service runs in isolated virtual environment
- OpenResty handles external requests
- Internal communication via localhost

## 📚 **Research Paper Implementation**

Dự án này triển khai thuật toán MCCVA từ research paper với:

- **SVM Classification**: Makespan prediction based on task features
- **K-Means Clustering**: VM resource usage clustering
- **Intelligent Routing**: AI-based load balancing
- **OpenResty Integration**: High-performance API gateway

## 🆘 **Troubleshooting**

### **Deployment Verification:**
```bash
# Comprehensive check
python3 check_deployment.py

# Check specific components
sudo systemctl status mccva-ml openresty
sudo journalctl -u mccva-ml --no-pager -n 20
```

### **Common Issues:**

1. **ML Service not starting:**
   ```bash
   sudo systemctl status mccva-ml
   sudo journalctl -u mccva-ml -f
   cd /opt/mccva && source venv/bin/activate && python ml_service.py
   ```

2. **Model loading issues:**
   ```bash
   # Check model files
   ls -la /opt/mccva/models/
   
   # Test model loading
   cd /opt/mccva
   source venv/bin/activate
   python3 -c "import joblib; joblib.load('models/svm_model.joblib')"
   ```

3. **OpenResty configuration error:**
   ```bash
   sudo /usr/local/openresty/nginx/sbin/nginx -t
   sudo systemctl restart openresty
   ```

4. **Port conflicts:**
   ```bash
   # Check what's using ports
   sudo netstat -tlnp | grep :80
   sudo netstat -tlnp | grep :5000
   ```

### **Log Locations:**
- ML Service: `sudo journalctl -u mccva-ml -f`
- OpenResty: `/var/log/nginx/access.log`, `/var/log/nginx/error.log`
- Application: `/var/log/mccva-ml.log`

### **Quick Fixes:**
```bash
# Restart everything
sudo systemctl restart mccva-ml openresty

# Reinstall dependencies
cd /opt/mccva
source venv/bin/activate
pip install -r requirements.txt

# Check file permissions
sudo chown -R $USER:$USER /opt/mccva
sudo chmod -R 755 /opt/mccva
```

## 📞 **Support**

- **Documentation**: Xem `DEPLOYMENT_README.md` cho hướng dẫn chi tiết
- **Testing**: Chạy `python3 check_deployment.py` để kiểm tra hệ thống
- **Logs**: Kiểm tra logs để debug issues
- **Verification**: `python3 test_mccva.py` để test thuật toán

---

**MCCVA Algorithm** - Intelligent Load Balancing with AI 🚀 