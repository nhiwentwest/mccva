# BÁO CÁO HUẤN LUYỆN MÔ HÌNH MACHINE LEARNING

## 1. Mô hình SVM (Support Vector Machine)

- **Kernel**: linear
- **C parameter**: 100
- **Gamma**: scale
- **Số support vectors**: 49
- **Số lớp**: 3

## 2. Mô hình K-Means

- **Số cụm**: 6
- **Inertia**: 1978.2949
- **Số lần lặp**: 15

## 3. Cách sử dụng mô hình

### SVM Model:
```python
import joblib

# Load model
svm_model = joblib.load('models/svm_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# Dự đoán
features = [cpu_cores, memory_gb, storage_gb, network_bandwidth, priority]
features_scaled = scaler.transform([features])
prediction = svm_model.predict(features_scaled)[0]
```

### K-Means Model:
```python
import joblib

# Load model
kmeans_model = joblib.load('models/kmeans_model.joblib')
kmeans_scaler = joblib.load('models/kmeans_scaler.joblib')

# Dự đoán cụm
vm_features = [cpu_usage, ram_usage, storage_usage]
vm_scaled = kmeans_scaler.transform([vm_features])
cluster = kmeans_model.predict(vm_scaled)[0]
```

## 4. Files được tạo

- `models/svm_model.joblib`: Mô hình SVM đã huấn luyện
- `models/kmeans_model.joblib`: Mô hình K-Means đã huấn luyện
- `models/scaler.joblib`: Scaler cho SVM
- `models/kmeans_scaler.joblib`: Scaler cho K-Means
- `models/svm_grid_search.joblib`: Kết quả grid search SVM
- `models/kmeans_info.joblib`: Thông tin cụm K-Means
