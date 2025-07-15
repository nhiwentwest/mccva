#!/usr/bin/env python3
"""
Advanced SVM Training with Full Dataset (train_svm_full.py)
Huấn luyện SVM với toàn bộ dataset từ thư mục dataset/modified
"""

import pandas as pd
import numpy as np
import os
import time
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

print("🚀 Bắt đầu huấn luyện SVM với toàn bộ dataset...")
start_time = time.time()

# Hàm ánh xạ nhãn
def map_class_names(class_name):
    mapping = {
        "'Very Low'": 'small',
        "'Low'": 'small',
        "'Medium'": 'medium',
        "'High'": 'large',
        "'Very High'": 'large'
    }
    return mapping.get(class_name, 'medium')

# Hàm huấn luyện SVM với GridSearchCV
def train_optimized_svm(X, y, type_val=None):
    """Huấn luyện SVM với GridSearchCV để tìm hyperparameters tối ưu"""
    print(f"  Huấn luyện SVM với GridSearchCV...")
    
    # Chuyển đổi nhãn chuỗi thành số nguyên nếu cần
    if isinstance(y.iloc[0], str):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        label_encoder = None
        y_encoded = y
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Tính class weights để xử lý mất cân bằng dữ liệu
    unique_classes = np.unique(y_encoded)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_encoded)
    class_weight_dict = {i: weight for i, weight in zip(unique_classes, class_weights)}
    print(f"  Class weights: {class_weight_dict}")
    
    # Tham số cho GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'poly']
    }
    
    # Huấn luyện SVM với GridSearchCV
    grid_search = GridSearchCV(
        SVC(probability=True, random_state=42, class_weight='balanced'),
        param_grid=param_grid,
        cv=5,
        scoring='f1_weighted',
        verbose=0,
        n_jobs=-1
    )
    
    grid_search.fit(X_scaled, y_encoded if label_encoder else y)
    
    # In thông tin về best parameters
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Lấy mô hình tốt nhất
    best_model = grid_search.best_estimator_
    
    return {
        'model': best_model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'type': type_val,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }

# Hàm dự đoán với mô hình đã huấn luyện
def predict_with_model(model_data, X):
    """Dự đoán với mô hình đã huấn luyện"""
    # Dự đoán với SVM
    X_scaled = model_data['scaler'].transform(X)
    y_pred = model_data['model'].predict(X_scaled)
    
    # Chuyển đổi lại nhãn số thành chuỗi nếu cần
    if model_data['label_encoder'] is not None:
        y_pred = model_data['label_encoder'].inverse_transform(y_pred)
    
    return y_pred

def main():
    """Hàm chính để huấn luyện và đánh giá mô hình"""
    # Tạo thư mục models nếu chưa có
    os.makedirs("models", exist_ok=True)
    
    # Bước 1: Load toàn bộ dataset
    print("\n📊 Đang load toàn bộ dataset...")
    dataset_dir = "dataset/modified"
    
    all_dfs = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".xlsx"):
            try:
                file_path = os.path.join(dataset_dir, file)
                df = pd.read_excel(file_path)
                all_dfs.append(df)
                print(f"✅ Đã load {len(df)} mẫu từ {file}")
            except Exception as e:
                print(f"❌ Lỗi khi đọc file {file}: {e}")
    
    if not all_dfs:
        print("❌ Không tìm thấy file dataset nào!")
        return
    
    # Kết hợp tất cả dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n📋 Tổng số mẫu: {len(df)}")
    
    # Hiển thị thông tin về dữ liệu
    print("\n📋 Thông tin về dữ liệu:")
    print(f"  Số lượng mẫu: {len(df)}")
    print(f"  Số lượng đặc trưng: {len(df.columns) - 2}")  # Trừ Class_Name và Type
    
    # Hiển thị phân phối nhãn
    print("\n📊 Phân phối nhãn:")
    class_counts = df['Class_Name'].value_counts()
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} mẫu ({count/len(df)*100:.1f}%)")
    
    # Hiển thị phân phối Type
    print("\n📊 Phân phối Type:")
    type_counts = df['Type'].value_counts()
    for typ, count in type_counts.items():
        print(f"  Type {typ}: {count} mẫu ({count/len(df)*100:.1f}%)")
    
    # Bước 2: Tiền xử lý dữ liệu
    print("\n🔄 Tiền xử lý dữ liệu...")
    
    # Ánh xạ nhãn
    df['target'] = df['Class_Name'].apply(map_class_names)
    
    # Chuyển đổi các cột thành số
    print("  Chuyển đổi các cột thành số...")
    numeric_cols = df.select_dtypes(include=['number']).columns
    X = df[numeric_cols].copy()
    
    # Kiểm tra và xử lý giá trị NaN
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"  ⚠️ Phát hiện {nan_count} giá trị NaN, đang thay thế bằng giá trị trung bình...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Bước 3: Huấn luyện mô hình chung cho tất cả các Type
    print("\n🧠 Huấn luyện mô hình chung cho tất cả các Type...")
    
    # Chia dữ liệu thành train/test
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Huấn luyện mô hình chung
    global_model_data = train_optimized_svm(X_train, y_train)
    
    # Đánh giá mô hình chung
    y_pred = predict_with_model(global_model_data, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📈 Kết quả đánh giá mô hình chung:")
    print(f"  Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Bước 4: Huấn luyện mô hình cho từng Type
    print("\n🧠 Huấn luyện mô hình cho từng Type...")
    
    type_models = {}
    for type_val in sorted(df['Type'].unique()):
        print(f"\n🔍 Huấn luyện mô hình cho Type {type_val}...")
        
        # Lấy dữ liệu cho Type cụ thể
        df_type = df[df['Type'] == type_val]
        X_type = df_type[numeric_cols].copy()
        y_type = df_type['target']
        
        print(f"  Type {type_val} có {len(df_type)} mẫu")
        
        # Hiển thị phân phối nhãn cho Type này
        print(f"  Phân phối nhãn cho Type {type_val}:")
        type_class_counts = y_type.value_counts()
        for cls, count in type_class_counts.items():
            print(f"    - {cls}: {count} mẫu ({count/len(df_type)*100:.1f}%)")
        
        # Kiểm tra và xử lý giá trị NaN
        nan_count = X_type.isna().sum().sum()
        if nan_count > 0:
            print(f"  ⚠️ Phát hiện {nan_count} giá trị NaN, đang thay thế bằng giá trị trung bình...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X_type = pd.DataFrame(imputer.fit_transform(X_type), columns=X_type.columns)
        
        # Chia dữ liệu thành train/test
        X_type_train, X_type_test, y_type_train, y_type_test = train_test_split(
            X_type, y_type, test_size=0.2, random_state=42)
        
        # Huấn luyện mô hình riêng cho Type này
        type_model_data = train_optimized_svm(X_type_train, y_type_train, type_val)
        
        # Đánh giá mô hình
        y_type_pred = predict_with_model(type_model_data, X_type_test)
        type_accuracy = accuracy_score(y_type_test, y_type_pred)
        print(f"  Accuracy cho Type {type_val}: {type_accuracy:.4f}")
        
        # Hiển thị phân phối nhãn trong tập test
        print(f"  Phân phối nhãn trong tập test:")
        for label, count in pd.Series(y_type_test).value_counts().items():
            print(f"    - {label}: {count} mẫu")
        
        # Hiển thị confusion matrix nếu có nhiều hơn 1 lớp
        if len(pd.Series(y_type_test).unique()) > 1:
            print(f"  Confusion Matrix:")
            print(confusion_matrix(y_type_test, y_type_pred))
        
        # Lưu mô hình
        type_models[type_val] = type_model_data
    
    # Bước 5: Lưu các mô hình
    print("\n💾 Lưu các mô hình...")
    
    # Lưu mô hình chung
    joblib.dump(global_model_data, "models/svm_global_model_full.joblib")
    
    # Lưu mô hình cho từng Type
    joblib.dump(type_models, "models/svm_type_models_full.joblib")
    
    # Lưu thông tin mô hình
    model_info = {
        'accuracy': float(accuracy),
        'n_features': X.shape[1],
        'n_samples': len(df),
        'n_types': len(df['Type'].unique()),
        'class_distribution': {k: int(v) for k, v in class_counts.items()},
        'type_distribution': {int(k): int(v) for k, v in type_counts.items()},
        'best_params': global_model_data['best_params'],
        'best_score': global_model_data['best_score'],
        'training_time': time.time() - start_time
    }
    joblib.dump(model_info, "models/svm_model_info_full.joblib")
    
    print("\n✅ Huấn luyện và lưu mô hình thành công!")
    print(f"⏱️ Tổng thời gian: {(time.time() - start_time):.2f} giây")

if __name__ == "__main__":
    main() 