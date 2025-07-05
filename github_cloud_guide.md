# 🚀 Hướng dẫn sử dụng GitHub trên Cloud Server

## 📋 **Mục lục**
1. [Xóa repository cũ](#1-xóa-repository-cũ)
2. [Clone repository mới](#2-clone-repository-mới)
3. [Cập nhật repository](#3-cập-nhật-repository)
4. [Quản lý branches](#4-quản-lý-branches)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Xóa repository cũ

### **Bước 1: Backup dữ liệu quan trọng (nếu cần)**
```bash
# Tạo backup của config files
cp /opt/mccva/mccva_env.conf ~/mccva_env.conf.backup
cp /opt/mccva/nginx.conf ~/nginx.conf.backup

# Backup logs nếu cần
sudo cp -r /var/log/mccva ~/mccva_logs_backup
```

### **Bước 2: Dừng services**
```bash
# Dừng tất cả services
sudo systemctl stop openresty
sudo systemctl stop mccva-mock-servers
docker stop mccva-ml

# Kiểm tra không còn process nào chạy
sudo systemctl status openresty
sudo systemctl status mccva-mock-servers
docker ps
```

### **Bước 3: Xóa thư mục cũ**
```bash
# Di chuyển ra ngoài thư mục
cd /opt

# Xóa thư mục cũ
sudo rm -rf mccva

# Hoặc đổi tên để backup
sudo mv mccva mccva_old_$(date +%Y%m%d)
```

---

## 2. Clone repository mới

### **Bước 1: Clone từ GitHub**
```bash
# Clone repository
cd /opt
sudo git clone https://github.com/nhiwentwest/mccva.git

# Set quyền sở hữu
sudo chown -R ubuntu:ubuntu mccva
cd mccva
```

### **Bước 2: Kiểm tra repository**
```bash
# Kiểm tra trạng thái
git status
git log --oneline -5

# Kiểm tra files
ls -la
```

### **Bước 3: Restore config files (nếu cần)**
```bash
# Restore config files từ backup
cp ~/mccva_env.conf.backup mccva_env.conf
cp ~/nginx.conf.backup nginx.conf

# Hoặc tạo config mới
cat > mccva_env.conf << 'EOF'
NGINX_USER=ubuntu
LOGS_DIR=/usr/local/openresty/nginx/logs
MOCK_SERVER_COUNT=8
EOF
```

---

## 3. Cập nhật repository

### **Cách 1: Pull updates (nếu đã có repository)**
```bash
cd /opt/mccva

# Kiểm tra remote
git remote -v

# Pull updates mới nhất
git pull origin main

# Kiểm tra thay đổi
git log --oneline -3
```

### **Cách 2: Reset về trạng thái GitHub**
```bash
cd /opt/mccva

# Reset về trạng thái GitHub
git fetch origin
git reset --hard origin/main

# Xóa files không được track
git clean -fd
```

### **Cách 3: Stash changes và pull**
```bash
cd /opt/mccva

# Stash changes hiện tại
git stash -u

# Pull updates
git pull origin main

# Restore changes (nếu cần)
git stash pop
```

---

## 4. Quản lý branches

### **Tạo branch mới cho testing**
```bash
# Tạo và chuyển sang branch mới
git checkout -b testing-branch

# Push branch lên GitHub
git push origin testing-branch

# Quay về main branch
git checkout main
```

### **Merge changes**
```bash
# Merge branch vào main
git checkout main
git merge testing-branch

# Push lên GitHub
git push origin main

# Xóa branch local
git branch -d testing-branch
```

---

## 5. Script tự động hóa

### **Script backup và restore**
```bash
#!/bin/bash
# backup_mccva.sh

echo "🔄 Backup MCCVA system..."

# Backup configs
cp /opt/mccva/mccva_env.conf ~/mccva_env.conf.backup
cp /opt/mccva/nginx.conf ~/nginx.conf.backup

# Stop services
sudo systemctl stop openresty
sudo systemctl stop mccva-mock-servers
docker stop mccva-ml

echo "✅ Backup completed"
```

```bash
#!/bin/bash
# restore_mccva.sh

echo "🔄 Restore MCCVA system..."

# Restore configs
cp ~/mccva_env.conf.backup /opt/mccva/mccva_env.conf
cp ~/nginx.conf.backup /opt/mccva/nginx.conf

# Start services
cd /opt/mccva
./run.sh

echo "✅ Restore completed"
```

### **Script update từ GitHub**
```bash
#!/bin/bash
# update_mccva.sh

echo "🔄 Update MCCVA from GitHub..."

cd /opt/mccva

# Backup configs
cp mccva_env.conf ~/mccva_env.conf.backup

# Stop services
sudo systemctl stop openresty
sudo systemctl stop mccva-mock-servers
docker stop mccva-ml

# Pull updates
git fetch origin
git reset --hard origin/main

# Restore configs
cp ~/mccva_env.conf.backup mccva_env.conf

# Start services
./run.sh

echo "✅ Update completed"
```

---

## 6. Troubleshooting

### **Lỗi permission**
```bash
# Fix permission issues
sudo chown -R ubuntu:ubuntu /opt/mccva
chmod +x /opt/mccva/*.sh
```

### **Lỗi Git conflicts**
```bash
# Reset về trạng thái sạch
git reset --hard HEAD
git clean -fd

# Pull lại
git pull origin main
```

### **Lỗi services không start**
```bash
# Check logs
sudo journalctl -f -u openresty
sudo journalctl -f -u mccva-mock-servers
docker logs mccva-ml

# Restart services
sudo systemctl restart openresty
sudo systemctl restart mccva-mock-servers
docker restart mccva-ml
```

### **Lỗi Docker image**
```bash
# Rebuild Docker image
cd /opt/mccva
docker build -t mccva-ml-service .

# Remove old container
docker rm -f mccva-ml

# Start new container
docker run -d --name mccva-ml -p 5000:5000 --restart unless-stopped mccva-ml-service
```

---

## 7. Best Practices

### **✅ Nên làm:**
- Backup config files trước khi update
- Test trên branch riêng trước khi merge
- Kiểm tra services sau khi update
- Sử dụng git stash để lưu changes tạm thời

### **❌ Không nên làm:**
- Commit trực tiếp trên cloud server
- Xóa repository mà không backup
- Update khi services đang chạy
- Ignore error messages

---

## 8. Quick Commands Reference

```bash
# 🔄 Update từ GitHub
cd /opt/mccva && git pull origin main

# 🗑️ Xóa và clone lại
cd /opt && sudo rm -rf mccva && sudo git clone https://github.com/nhiwentwest/mccva.git

# 📊 Check status
cd /opt/mccva && git status && git log --oneline -3

# 🚀 Deploy
cd /opt/mccva && ./run.sh

# 🧪 Test
cd /opt/mccva && python3 test_ai_routing_simple.py
```

---

## 9. Monitoring và Logs

### **Check system status**
```bash
# Services status
sudo systemctl status openresty
sudo systemctl status mccva-mock-servers
docker ps

# Health checks
curl http://localhost/health
curl http://localhost:5000/health
```

### **View logs**
```bash
# OpenResty logs
sudo journalctl -f -u openresty

# Mock servers logs
sudo journalctl -f -u mccva-mock-servers

# ML Service logs
docker logs -f mccva-ml
```

---

**🎯 Lưu ý**: Luôn backup dữ liệu quan trọng trước khi thực hiện bất kỳ thao tác nào với repository! 