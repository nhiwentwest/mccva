# 🚀 Hướng dẫn triển khai ML Service + OpenResty lên Amazon Cloud

Hướng dẫn chi tiết để triển khai hệ thống tích hợp ML Service (SVM + K-Means) với OpenResty lên Amazon Cloud Ubuntu.

## 📋 Yêu cầu hệ thống

- **OS**: Ubuntu 18.04+ hoặc Amazon Linux 2
- **RAM**: Tối thiểu 2GB (khuyến nghị 4GB+)
- **Storage**: Tối thiểu 10GB
- **Network**: Cổng 80 và 5000 mở
- **Python**: 3.7+
- **OpenResty**: 1.15+

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client/Browser│    │   OpenResty     │    │   ML Service    │
│                 │    │   (Port 80)     │    │   (Port 5000)   │
│                 │◄──►│                 │◄──►│                 │
│                 │    │ • API Gateway   │    │ • Flask API     │
│                 │    │ • Caching       │    │ • SVM Model     │
│                 │    │ • Load Balancing│    │ • K-Means Model │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Triển khai nhanh

### Bước 1: Chuẩn bị server

```bash
# Kết nối vào Amazon EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Cập nhật hệ thống
sudo apt update && sudo apt upgrade -y
```

### Bước 2: Upload files

```bash
# Tạo thư mục project
mkdir ~/ml-openresty-integration
cd ~/ml-openresty-integration

# Upload tất cả files từ máy local
# (Sử dụng scp hoặc git clone)
scp -r /path/to/local/project/* ubuntu@your-ec2-ip:~/ml-openresty-integration/
```

### Bước 3: Chạy script triển khai

```bash
# Cấp quyền thực thi
chmod +x deploy.sh

# Chạy script triển khai
bash deploy.sh
```

### Bước 4: Kiểm tra triển khai

```bash
# Test integration
python3 test_integration.py

# Kiểm tra services
sudo systemctl status ml-service
sudo systemctl status openresty
```

## 📁 Cấu trúc files sau triển khai

```
/opt/ml-service/
├── ml_service.py          # Flask API service
├── models/                # ML models
│   ├── svm_model.joblib
│   ├── kmeans_model.joblib
│   ├── scaler.joblib
│   └── kmeans_scaler.joblib
├── venv/                  # Python virtual environment
└── requirements.txt

/usr/local/openresty/nginx/conf/
├── nginx.conf             # Main nginx config
└── lua/                   # Lua scripts
    ├── predict_makespan.lua
    ├── predict_vm_cluster.lua
    ├── predict_batch.lua
    ├── vm_clusters_info.lua
    ├── models_info.lua
    └── demo_page.lua

/etc/systemd/system/
└── ml-service.service     # Systemd service
```

## 🔧 Cấu hình chi tiết

### ML Service (Flask API)

**File**: `/opt/ml-service/ml_service.py`

**Endpoints**:
- `GET /health` - Health check
- `POST /predict/makespan` - Dự đoán makespan
- `POST /predict/vm_cluster` - Dự đoán VM cluster
- `POST /predict/batch` - Dự đoán hàng loạt
- `GET /vm_clusters/info` - Thông tin cụm VM
- `GET /models/info` - Thông tin mô hình

### OpenResty Configuration

**File**: `/usr/local/openresty/nginx/conf/nginx.conf`

**Features**:
- API Gateway với caching
- CORS support
- Load balancing
- Error handling
- Performance optimization

### Systemd Services

**ML Service**: `/etc/systemd/system/ml-service.service`
- Auto-restart on failure
- Log management
- Environment isolation

## 🌐 Truy cập và sử dụng

### URLs chính

```bash
# Demo page (giao diện web)
http://your-ec2-ip/demo

# Health check
http://your-ec2-ip/health

# API endpoints
http://your-ec2-ip/predict/makespan
http://your-ec2-ip/predict/vm_cluster
http://your-ec2-ip/predict/batch
http://your-ec2-ip/vm_clusters/info
http://your-ec2-ip/models/info

# Direct ML service (nội bộ)
http://localhost:5000/health
```

### Ví dụ sử dụng API

#### 1. Dự đoán Makespan

```bash
curl -X POST http://your-ec2-ip/predict/makespan \
  -H "Content-Type: application/json" \
  -d '{
    "features": [4, 8, 100, 1000, 3]
  }'
```

**Response**:
```json
{
  "makespan": "small",
  "confidence": 2.325,
  "features": [4, 8, 100, 1000, 3],
  "timestamp": "2024-01-01T12:00:00"
}
```

#### 2. Dự đoán VM Cluster

```bash
curl -X POST http://your-ec2-ip/predict/vm_cluster \
  -H "Content-Type: application/json" \
  -d '{
    "vm_features": [0.7, 0.6, 0.4]
  }'
```

**Response**:
```json
{
  "cluster": 0,
  "distance": 0.720,
  "centroid": [1.013, -0.288, -0.884],
  "vm_features": [0.7, 0.6, 0.4],
  "timestamp": "2024-01-01T12:00:00"
}
```

#### 3. Batch Prediction

```bash
curl -X POST http://your-ec2-ip/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [[4, 8, 100, 1000, 3], [8, 16, 200, 2000, 4]],
    "vm_usages": [[0.7, 0.6, 0.4], [0.5, 0.8, 0.6]]
  }'
```

## 🔍 Monitoring và Logs

### Kiểm tra trạng thái services

```bash
# ML Service status
sudo systemctl status ml-service

# OpenResty status
sudo systemctl status openresty

# Check ports
sudo netstat -tlnp | grep -E ':(80|5000)'
```

### Xem logs

```bash
# ML Service logs
sudo journalctl -u ml-service -f

# OpenResty logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# System logs
sudo journalctl -f
```

### Performance monitoring

```bash
# Check memory usage
free -h

# Check CPU usage
top

# Check disk usage
df -h

# Check network connections
ss -tuln
```

## 🛠️ Troubleshooting

### ML Service không khởi động

```bash
# Check logs
sudo journalctl -u ml-service -n 50

# Check Python environment
cd /opt/ml-service
source venv/bin/activate
python ml_service.py

# Check file permissions
ls -la /opt/ml-service/
```

### OpenResty không khởi động

```bash
# Test nginx config
sudo /usr/local/openresty/nginx/sbin/nginx -t

# Check nginx logs
sudo tail -f /var/log/nginx/error.log

# Check port conflicts
sudo lsof -i :80
```

### API không phản hồi

```bash
# Test connectivity
curl -v http://localhost/health
curl -v http://localhost:5000/health

# Check firewall
sudo ufw status

# Test ML service directly
curl -X POST http://localhost:5000/predict/makespan \
  -H "Content-Type: application/json" \
  -d '{"features": [4, 8, 100, 1000, 3]}'
```

## 🔄 Maintenance

### Cập nhật ML models

```bash
# Stop ML service
sudo systemctl stop ml-service

# Backup old models
sudo cp -r /opt/ml-service/models /opt/ml-service/models.backup

# Copy new models
sudo cp new_models/* /opt/ml-service/models/

# Start ML service
sudo systemctl start ml-service

# Test new models
python3 test_integration.py
```

### Restart services

```bash
# Restart ML service
sudo systemctl restart ml-service

# Restart OpenResty
sudo systemctl restart openresty

# Restart both
sudo systemctl restart ml-service openresty
```

### Backup và restore

```bash
# Backup
sudo tar -czf ml-service-backup-$(date +%Y%m%d).tar.gz /opt/ml-service/

# Restore
sudo tar -xzf ml-service-backup-20240101.tar.gz -C /
```

## 📊 Performance Optimization

### Caching

- **ML Cache**: 10MB shared memory cho predictions
- **VM Info Cache**: 5MB shared memory cho cluster info
- **TTL**: 5 phút cho predictions, 10 phút cho info

### Load Balancing

- **Keep-alive**: 32 connections
- **Timeout**: 5s connect, 10s read/write
- **Connection pooling**: Enabled

### Monitoring

```bash
# Install monitoring tools
sudo apt install -y htop iotop nethogs

# Monitor real-time
htop
iotop
nethogs
```

## 🔒 Security

### Firewall configuration

```bash
# Allow only necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw deny 5000/tcp   # Block direct ML service access

# Enable firewall
sudo ufw enable
```

### SSL/TLS (Optional)

```bash
# Install certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 📈 Scaling

### Horizontal scaling

1. **Load Balancer**: Sử dụng AWS ALB
2. **Multiple ML Services**: Chạy nhiều instances
3. **Database**: Redis cho shared cache
4. **Monitoring**: CloudWatch metrics

### Vertical scaling

1. **Increase RAM**: Tăng shared memory cache
2. **Increase CPU**: Tối ưu model inference
3. **SSD Storage**: Tăng I/O performance

## 🆘 Support

### Logs location

- **ML Service**: `sudo journalctl -u ml-service`
- **OpenResty**: `/var/log/nginx/`
- **System**: `/var/log/syslog`

### Common issues

1. **Port 80 in use**: `sudo lsof -i :80`
2. **Permission denied**: `sudo chown -R $USER:$USER /opt/ml-service/`
3. **Python import error**: Check virtual environment
4. **Model not found**: Verify model files exist

### Contact

- **Documentation**: Check this README
- **Issues**: Create GitHub issue
- **Support**: Contact system administrator

---

**🎉 Chúc mừng! Hệ thống ML Service + OpenResty đã được triển khai thành công!** 