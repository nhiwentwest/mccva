#!/usr/bin/env python3
"""
Demo script để sử dụng các mô hình SVM và K-Means đã huấn luyện
"""

import joblib
import numpy as np
import pandas as pd

def demo_svm_model():
    """
    Demo sử dụng mô hình SVM để dự đoán makespan
    """
    print("=" * 60)
    print("🤖 DEMO SỬ DỤNG MÔ HÌNH SVM")
    print("=" * 60)
    
    try:
        # Load model và scaler
        svm_model = joblib.load('models/svm_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        
        print("✅ Đã load mô hình SVM thành công!")
        print(f"📊 Kernel: {svm_model.kernel}")
        print(f"📊 C parameter: {svm_model.C}")
        print(f"📊 Support vectors: {sum(svm_model.n_support_)}")
        
        # Tạo một số ví dụ dự đoán
        examples = [
            # [cpu_cores, memory_gb, storage_gb, network_bandwidth, priority]
            [2, 4, 50, 500, 1],    # Yêu cầu nhỏ
            [8, 16, 200, 2000, 3], # Yêu cầu vừa
            [16, 32, 500, 5000, 5], # Yêu cầu lớn
            [4, 8, 100, 1000, 2],   # Yêu cầu vừa
            [12, 24, 300, 3000, 4]  # Yêu cầu lớn
        ]
        
        print("\n🔮 DỰ ĐOÁN MAKESPAN:")
        print("-" * 50)
        
        for i, features in enumerate(examples, 1):
            # Chuẩn hóa dữ liệu
            features_scaled = scaler.transform([features])
            
            # Dự đoán
            prediction = svm_model.predict(features_scaled)[0]
            
            # Tính confidence score (khoảng cách đến decision boundary)
            decision_scores = svm_model.decision_function(features_scaled)
            # Nếu là mảng nhiều chiều (đa lớp), lấy giá trị lớn nhất
            if isinstance(decision_scores[0], np.ndarray):
                confidence = np.max(np.abs(decision_scores[0]))
            else:
                confidence = np.abs(decision_scores[0])
            
            print(f"Ví dụ {i}:")
            print(f"  CPU: {features[0]} cores, RAM: {features[1]} GB")
            print(f"  Storage: {features[2]} GB, Network: {features[3]} Mbps")
            print(f"  Priority: {features[4]}")
            print(f"  → Dự đoán: {prediction.upper()} (confidence: {confidence:.3f})")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi sử dụng SVM model: {e}")
        return False

def demo_kmeans_model():
    """
    Demo sử dụng mô hình K-Means để phân cụm VM
    """
    print("=" * 60)
    print("🎯 DEMO SỬ DỤNG MÔ HÌNH K-MEANS")
    print("=" * 60)
    
    try:
        # Load model và scaler
        kmeans_model = joblib.load('models/kmeans_model.joblib')
        kmeans_scaler = joblib.load('models/kmeans_scaler.joblib')
        cluster_info = joblib.load('models/kmeans_info.joblib')
        
        print("✅ Đã load mô hình K-Means thành công!")
        print(f"📊 Số cụm: {kmeans_model.n_clusters}")
        print(f"📊 Inertia: {kmeans_model.inertia_:.4f}")
        
        # Hiển thị thông tin các cụm
        print("\n📊 THÔNG TIN CÁC CỤM:")
        print("-" * 50)
        for i in range(kmeans_model.n_clusters):
            center = kmeans_model.cluster_centers_[i]
            print(f"Cụm {i}: CPU={center[0]:.3f}, RAM={center[1]:.3f}, Storage={center[2]:.3f}")
        
        # Tạo một số ví dụ VM
        vm_examples = [
            # [cpu_usage, ram_usage, storage_usage]
            [0.8, 0.3, 0.4],   # CPU cao, RAM thấp
            [0.2, 0.7, 0.6],   # CPU thấp, RAM cao
            [0.6, 0.6, 0.5],   # Cân bằng
            [0.3, 0.2, 0.3],   # Tất cả thấp
            [0.9, 0.8, 0.7]    # Tất cả cao
        ]
        
        print("\n🔮 PHÂN CỤM VM:")
        print("-" * 50)
        
        for i, vm_features in enumerate(vm_examples, 1):
            # Chuẩn hóa dữ liệu
            vm_scaled = kmeans_scaler.transform([vm_features])
            
            # Dự đoán cụm
            cluster = kmeans_model.predict(vm_scaled)[0]
            
            # Tính khoảng cách đến centroid
            distances = kmeans_model.transform(vm_scaled)[0]
            distance_to_center = distances[cluster]
            
            print(f"VM {i}:")
            print(f"  CPU: {vm_features[0]:.1%}, RAM: {vm_features[1]:.1%}, Storage: {vm_features[2]:.1%}")
            print(f"  → Thuộc cụm: {cluster} (distance: {distance_to_center:.3f})")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi sử dụng K-Means model: {e}")
        return False

def demo_integration():
    """
    Demo tích hợp cả hai mô hình
    """
    print("=" * 60)
    print("🔄 DEMO TÍCH HỢP HAI MÔ HÌNH")
    print("=" * 60)
    
    try:
        # Load cả hai model
        svm_model = joblib.load('models/svm_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        kmeans_model = joblib.load('models/kmeans_model.joblib')
        kmeans_scaler = joblib.load('models/kmeans_scaler.joblib')
        
        print("✅ Đã load cả hai mô hình thành công!")
        
        # Tạo kịch bản thực tế
        scenarios = [
            {
                'name': 'Web Server',
                'request': [4, 8, 100, 1000, 3],
                'vm_usage': [0.7, 0.6, 0.4]
            },
            {
                'name': 'Database Server',
                'request': [8, 16, 200, 2000, 4],
                'vm_usage': [0.5, 0.8, 0.6]
            },
            {
                'name': 'AI Training',
                'request': [16, 32, 500, 5000, 5],
                'vm_usage': [0.9, 0.7, 0.8]
            }
        ]
        
        print("\n🎯 PHÂN TÍCH KỊCH BẢN THỰC TẾ:")
        print("-" * 60)
        
        for scenario in scenarios:
            print(f"\n📋 {scenario['name']}:")
            
            # Dự đoán makespan cho yêu cầu
            request_scaled = scaler.transform([scenario['request']])
            makespan = svm_model.predict(request_scaled)[0]
            decision_score = svm_model.decision_function(request_scaled)[0]
            # Nếu là mảng nhiều chiều (đa lớp), lấy giá trị lớn nhất
            if isinstance(decision_score, np.ndarray):
                confidence = np.max(np.abs(decision_score))
            else:
                confidence = np.abs(decision_score)
            
            # Dự đoán cụm cho VM
            vm_scaled = kmeans_scaler.transform([scenario['vm_usage']])
            cluster = kmeans_model.predict(vm_scaled)[0]
            
            print(f"  Yêu cầu: {scenario['request'][0]} cores, {scenario['request'][1]} GB RAM")
            print(f"  → Makespan: {makespan.upper()} (confidence: {confidence:.3f})")
            print(f"  VM Usage: CPU {scenario['vm_usage'][0]:.1%}, RAM {scenario['vm_usage'][1]:.1%}")
            print(f"  → Thuộc cụm: {cluster}")
            
            # Đưa ra khuyến nghị
            if makespan == 'large' and cluster in [0, 3, 5]:  # Cụm có CPU cao
                print(f"  💡 Khuyến nghị: Phù hợp - VM có đủ tài nguyên cho yêu cầu lớn")
            elif makespan == 'small' and cluster in [1, 2, 4]:  # Cụm có tài nguyên thấp
                print(f"  💡 Khuyến nghị: Phù hợp - VM tiết kiệm cho yêu cầu nhỏ")
            else:
                print(f"  ⚠️  Khuyến nghị: Cần xem xét lại - Có thể không tối ưu")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi tích hợp mô hình: {e}")
        return False

def demo_batch_prediction():
    """
    Demo dự đoán hàng loạt
    """
    print("=" * 60)
    print("📊 DEMO DỰ ĐOÁN HÀNG LOẠT")
    print("=" * 60)
    
    try:
        # Load models
        svm_model = joblib.load('models/svm_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        kmeans_model = joblib.load('models/kmeans_model.joblib')
        kmeans_scaler = joblib.load('models/kmeans_scaler.joblib')
        
        # Tạo dữ liệu test
        np.random.seed(42)
        n_samples = 10
        
        # Tạo yêu cầu ngẫu nhiên
        requests = np.random.randint(1, 17, (n_samples, 1))  # CPU cores
        requests = np.hstack([
            requests,
            np.random.randint(1, 65, (n_samples, 1)),  # Memory GB
            np.random.randint(10, 1001, (n_samples, 1)),  # Storage GB
            np.random.randint(100, 10001, (n_samples, 1)),  # Network Mbps
            np.random.randint(1, 6, (n_samples, 1))  # Priority
        ])
        
        # Tạo VM usage ngẫu nhiên
        vm_usages = np.random.uniform(0.1, 0.9, (n_samples, 3))  # CPU, RAM, Storage
        
        print("🔮 DỰ ĐOÁN HÀNG LOẠT:")
        print("-" * 60)
        print(f"{'ID':<3} {'CPU':<4} {'RAM':<4} {'Storage':<8} {'Network':<8} {'Priority':<8} {'Makespan':<8} {'VM_CPU':<6} {'VM_RAM':<6} {'VM_Storage':<8} {'Cluster':<8}")
        print("-" * 100)
        
        for i in range(n_samples):
            # Dự đoán makespan
            request_scaled = scaler.transform([requests[i]])
            makespan = svm_model.predict(request_scaled)[0]
            
            # Dự đoán cluster
            vm_scaled = kmeans_scaler.transform([vm_usages[i]])
            cluster = kmeans_model.predict(vm_scaled)[0]
            
            print(f"{i+1:<3} {requests[i][0]:<4} {requests[i][1]:<4} {requests[i][2]:<8} {requests[i][3]:<8} {requests[i][4]:<8} {makespan.upper():<8} {vm_usages[i][0]:<6.1%} {vm_usages[i][1]:<6.1%} {vm_usages[i][2]:<8.1%} {cluster:<8}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán hàng loạt: {e}")
        return False

def main():
    """
    Hàm chính để chạy demo
    """
    print("🚀 DEMO SỬ DỤNG MÔ HÌNH MACHINE LEARNING")
    print("=" * 80)
    
    # Demo từng mô hình
    demo_svm_model()
    demo_kmeans_model()
    demo_integration()
    demo_batch_prediction()
    
    print("\n" + "=" * 80)
    print("✅ HOÀN THÀNH DEMO!")
    print("=" * 80)
    
    print("\n📚 HƯỚNG DẪN SỬ DỤNG:")
    print("1. SVM Model: Dự đoán makespan của yêu cầu tài nguyên")
    print("2. K-Means Model: Phân cụm VM theo mức sử dụng tài nguyên")
    print("3. Tích hợp: Kết hợp cả hai để đưa ra khuyến nghị tối ưu")
    print("4. Batch Prediction: Dự đoán hàng loạt cho nhiều yêu cầu")
    
    print("\n🔧 Cách sử dụng trong code:")
    print("""
# Load models
svm_model = joblib.load('models/svm_model.joblib')
kmeans_model = joblib.load('models/kmeans_model.joblib')
scaler = joblib.load('models/scaler.joblib')
kmeans_scaler = joblib.load('models/kmeans_scaler.joblib')

# Dự đoán makespan
features = [cpu_cores, memory_gb, storage_gb, network_bandwidth, priority]
features_scaled = scaler.transform([features])
makespan = svm_model.predict(features_scaled)[0]

# Dự đoán cụm VM
vm_features = [cpu_usage, ram_usage, storage_usage]
vm_scaled = kmeans_scaler.transform([vm_features])
cluster = kmeans_model.predict(vm_scaled)[0]
    """)

if __name__ == "__main__":
    main() 