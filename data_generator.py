import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def generate_sample_data(n_samples=1000):
    """
    Tạo dữ liệu mẫu cho việc huấn luyện mô hình
    """
    np.random.seed(42)
    
    # Tạo dữ liệu cho các yêu cầu (requests)
    # Đặc trưng: cpu_cores, memory_gb, storage_gb, network_bandwidth, priority
    cpu_cores = np.random.randint(1, 17, n_samples)  # 1-16 cores
    memory_gb = np.random.randint(1, 65, n_samples)  # 1-64 GB
    storage_gb = np.random.randint(10, 1001, n_samples)  # 10-1000 GB
    network_bandwidth = np.random.randint(100, 10001, n_samples)  # 100-10000 Mbps
    priority = np.random.randint(1, 6, n_samples)  # 1-5 priority levels
    
    # Tạo nhãn makespan dựa trên đặc trưng
    # Tính điểm phức tạp dựa trên các đặc trưng
    complexity_score = (cpu_cores * 0.3 + 
                       memory_gb * 0.2 + 
                       storage_gb * 0.1 + 
                       network_bandwidth * 0.1 + 
                       priority * 0.3)
    
    # Phân loại thành 3 nhóm: nhỏ, vừa, lớn
    makespan_labels = []
    for score in complexity_score:
        if score < np.percentile(complexity_score, 33):
            makespan_labels.append('small')
        elif score < np.percentile(complexity_score, 67):
            makespan_labels.append('medium')
        else:
            makespan_labels.append('large')
    
    # Tạo DataFrame
    request_data = pd.DataFrame({
        'cpu_cores': cpu_cores,
        'memory_gb': memory_gb,
        'storage_gb': storage_gb,
        'network_bandwidth': network_bandwidth,
        'priority': priority,
        'makespan_label': makespan_labels
    })
    
    # Tạo dữ liệu VM
    vm_cpu_usage = np.random.uniform(0.1, 0.9, n_samples)  # 10-90% CPU usage
    vm_ram_usage = np.random.uniform(0.1, 0.9, n_samples)  # 10-90% RAM usage
    vm_storage_usage = np.random.uniform(0.2, 0.8, n_samples)  # 20-80% storage usage
    
    vm_data = pd.DataFrame({
        'cpu_usage': vm_cpu_usage,
        'ram_usage': vm_ram_usage,
        'storage_usage': vm_storage_usage
    })
    
    return request_data, vm_data

def prepare_training_data(request_data):
    """
    Chuẩn bị dữ liệu cho việc huấn luyện SVM
    """
    # Tách đặc trưng và nhãn
    X = request_data[['cpu_cores', 'memory_gb', 'storage_gb', 'network_bandwidth', 'priority']].values
    y = request_data['makespan_label'].values
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Lưu scaler để sử dụng sau này
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return X_scaled, y

if __name__ == "__main__":
    # Tạo thư mục models nếu chưa có
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Tạo dữ liệu mẫu
    print("Đang tạo dữ liệu mẫu...")
    request_data, vm_data = generate_sample_data(2000)
    
    # Lưu dữ liệu
    request_data.to_csv('data/request_data.csv', index=False)
    vm_data.to_csv('data/vm_data.csv', index=False)
    
    print(f"Đã tạo {len(request_data)} mẫu dữ liệu yêu cầu")
    print(f"Đã tạo {len(vm_data)} mẫu dữ liệu VM")
    print("Dữ liệu đã được lưu vào thư mục data/")
    
    # Hiển thị thống kê
    print("\nThống kê nhãn makespan:")
    print(request_data['makespan_label'].value_counts())
    
    print("\nThống kê dữ liệu VM:")
    print(vm_data.describe()) 