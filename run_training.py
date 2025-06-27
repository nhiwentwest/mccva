#!/usr/bin/env python3
"""
Script chính để thực hiện Bước 5: Huấn luyện mô hình SVM và K-Means bên ngoài
"""

import os
import sys
import time
from datetime import datetime

def print_header():
    """In header cho script"""
    print("=" * 80)
    print("🚀 BƯỚC 5: HUẤN LUYỆN MÔ HÌNH SVM VÀ K-MEANS BÊN NGOÀI")
    print("=" * 80)
    print(f"Thời gian bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def check_dependencies():
    """Kiểm tra các thư viện cần thiết"""
    print("🔍 Kiểm tra dependencies...")
    
    required_packages = [
        'sklearn', 'pandas', 'numpy', 'joblib', 
        'matplotlib', 'seaborn', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} - KHÔNG TÌM THẤY")
    
    if missing_packages:
        print(f"\n❌ Thiếu các thư viện: {', '.join(missing_packages)}")
        print("Hãy cài đặt bằng lệnh: pip install -r requirements.txt")
        return False
    
    print("✅ Tất cả dependencies đã sẵn sàng!")
    return True

def create_directories():
    """Tạo các thư mục cần thiết"""
    print("\n📁 Tạo thư mục...")
    
    directories = ['data', 'models']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  ✅ Tạo thư mục: {directory}/")
        else:
            print(f"  ℹ️  Thư mục đã tồn tại: {directory}/")

def step_1_generate_data():
    """Bước 1: Tạo dữ liệu mẫu"""
    print("\n" + "="*50)
    print("📊 BƯỚC 1: TẠO DỮ LIỆU MẪU")
    print("="*50)
    
    start_time = time.time()
    
    try:
        from data_generator import generate_sample_data
        
        print("Đang tạo dữ liệu mẫu...")
        request_data, vm_data = generate_sample_data(2000)
        
        # Lưu dữ liệu
        request_data.to_csv('data/request_data.csv', index=False)
        vm_data.to_csv('data/vm_data.csv', index=False)
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ Hoàn thành tạo dữ liệu trong {elapsed_time:.2f} giây")
        print(f"📊 Đã tạo {len(request_data)} mẫu dữ liệu yêu cầu")
        print(f"📊 Đã tạo {len(vm_data)} mẫu dữ liệu VM")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi tạo dữ liệu: {e}")
        return False

def step_2_train_svm():
    """Bước 2: Huấn luyện mô hình SVM"""
    print("\n" + "="*50)
    print("🤖 BƯỚC 2: HUẤN LUYỆN MÔ HÌNH SVM")
    print("="*50)
    
    start_time = time.time()
    
    try:
        from train_svm_model import main as train_svm_main
        
        # Chạy huấn luyện SVM
        train_svm_main()
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ Hoàn thành huấn luyện SVM trong {elapsed_time:.2f} giây")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi huấn luyện SVM: {e}")
        return False

def step_3_train_kmeans():
    """Bước 3: Huấn luyện mô hình K-Means"""
    print("\n" + "="*50)
    print("🎯 BƯỚC 3: HUẤN LUYỆN MÔ HÌNH K-MEANS")
    print("="*50)
    
    start_time = time.time()
    
    try:
        from train_kmeans_model import main as train_kmeans_main
        
        # Chạy huấn luyện K-Means
        train_kmeans_main()
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ Hoàn thành huấn luyện K-Means trong {elapsed_time:.2f} giây")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi huấn luyện K-Means: {e}")
        return False

def step_4_evaluate_models():
    """Bước 4: Đánh giá và so sánh mô hình"""
    print("\n" + "="*50)
    print("📈 BƯỚC 4: ĐÁNH GIÁ VÀ SO SÁNH MÔ HÌNH")
    print("="*50)
    
    start_time = time.time()
    
    try:
        from model_evaluator import main as evaluate_main
        
        # Chạy đánh giá mô hình
        evaluate_main()
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ Hoàn thành đánh giá mô hình trong {elapsed_time:.2f} giây")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi đánh giá mô hình: {e}")
        return False

def check_results():
    """Kiểm tra kết quả sau khi huấn luyện"""
    print("\n" + "="*50)
    print("🔍 KIỂM TRA KẾT QUẢ")
    print("="*50)
    
    expected_files = [
        'data/request_data.csv',
        'data/vm_data.csv',
        'models/svm_model.joblib',
        'models/kmeans_model.joblib',
        'models/scaler.joblib',
        'models/kmeans_scaler.joblib',
        'models/svm_grid_search.joblib',
        'models/kmeans_info.joblib',
        'models/model_report.md'
    ]
    
    print("Kiểm tra các file đã được tạo:")
    
    all_files_exist = True
    for file_path in expected_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  ✅ {file_path} ({file_size:,} bytes)")
        else:
            print(f"  ❌ {file_path} - KHÔNG TÌM THẤY")
            all_files_exist = False
    
    if all_files_exist:
        print("\n🎉 TẤT CẢ FILE ĐÃ ĐƯỢC TẠO THÀNH CÔNG!")
    else:
        print("\n⚠️  MỘT SỐ FILE BỊ THIẾU!")
    
    return all_files_exist

def print_summary():
    """In tóm tắt kết quả"""
    print("\n" + "="*80)
    print("📋 TÓM TẮT KẾT QUẢ")
    print("="*80)
    
    print("✅ Đã hoàn thành Bước 5: Huấn luyện mô hình SVM và K-Means bên ngoài")
    print("\n📁 Các file quan trọng:")
    print("  • models/svm_model.joblib - Mô hình SVM đã huấn luyện")
    print("  • models/kmeans_model.joblib - Mô hình K-Means đã huấn luyện")
    print("  • models/scaler.joblib - Scaler cho SVM")
    print("  • models/kmeans_scaler.joblib - Scaler cho K-Means")
    print("  • models/model_report.md - Báo cáo chi tiết")
    
    print("\n🚀 Cách sử dụng mô hình:")
    print("""
# Sử dụng SVM model
import joblib
svm_model = joblib.load('models/svm_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# Dự đoán makespan
features = [cpu_cores, memory_gb, storage_gb, network_bandwidth, priority]
features_scaled = scaler.transform([features])
prediction = svm_model.predict(features_scaled)[0]  # 'small', 'medium', 'large'

# Sử dụng K-Means model
kmeans_model = joblib.load('models/kmeans_model.joblib')
kmeans_scaler = joblib.load('models/kmeans_scaler.joblib')

# Dự đoán cụm VM
vm_features = [cpu_usage, ram_usage, storage_usage]
vm_scaled = kmeans_scaler.transform([vm_features])
cluster = kmeans_model.predict(vm_scaled)[0]  # 0, 1, 2, ...
    """)
    
    print(f"\n⏰ Thời gian kết thúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

def main():
    """Hàm chính"""
    print_header()
    
    # Kiểm tra dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Tạo thư mục
    create_directories()
    
    # Thực hiện các bước
    steps = [
        ("Tạo dữ liệu mẫu", step_1_generate_data),
        ("Huấn luyện SVM", step_2_train_svm),
        ("Huấn luyện K-Means", step_3_train_kmeans),
        ("Đánh giá mô hình", step_4_evaluate_models)
    ]
    
    total_start_time = time.time()
    
    for step_name, step_func in steps:
        print(f"\n🔄 Đang thực hiện: {step_name}")
        
        if not step_func():
            print(f"❌ Lỗi ở bước: {step_name}")
            sys.exit(1)
    
    # Kiểm tra kết quả
    check_results()
    
    # In tóm tắt
    total_time = time.time() - total_start_time
    print(f"\n⏱️  Tổng thời gian thực hiện: {total_time:.2f} giây")
    
    print_summary()

if __name__ == "__main__":
    main() 