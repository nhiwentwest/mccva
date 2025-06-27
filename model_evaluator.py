import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import time

def evaluate_svm_model():
    """
    Đánh giá mô hình SVM
    """
    print("=== ĐÁNH GIÁ MÔ HÌNH SVM ===")
    
    try:
        # Load model và dữ liệu
        svm_model = joblib.load('models/svm_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        
        # Đọc dữ liệu test
        request_data = pd.read_csv('data/request_data.csv')
        X = request_data[['cpu_cores', 'memory_gb', 'storage_gb', 'network_bandwidth', 'priority']].values
        y = request_data['makespan_label'].values
        
        # Chuẩn hóa dữ liệu
        X_scaled = scaler.transform(X)
        
        # Dự đoán
        start_time = time.time()
        y_pred = svm_model.predict(X_scaled)
        prediction_time = time.time() - start_time
        
        # Tính metrics
        accuracy = accuracy_score(y, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Thời gian dự đoán trung bình: {prediction_time/len(X):.6f} giây/mẫu")
        print(f"Tổng thời gian dự đoán {len(X)} mẫu: {prediction_time:.4f} giây")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['small', 'medium', 'large'],
                    yticklabels=['small', 'medium', 'large'])
        plt.title('Confusion Matrix - SVM Model Evaluation')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/svm_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'prediction_time': prediction_time,
            'avg_prediction_time': prediction_time/len(X)
        }
        
    except FileNotFoundError as e:
        print(f"Không tìm thấy file model: {e}")
        return None

def evaluate_kmeans_model():
    """
    Đánh giá mô hình K-Means
    """
    print("\n=== ĐÁNH GIÁ MÔ HÌNH K-MEANS ===")
    
    try:
        # Load model và dữ liệu
        kmeans_model = joblib.load('models/kmeans_model.joblib')
        kmeans_scaler = joblib.load('models/kmeans_scaler.joblib')
        
        # Đọc dữ liệu VM
        vm_data = pd.read_csv('data/vm_data.csv')
        features = ['cpu_usage', 'ram_usage', 'storage_usage']
        vm_features = vm_data[features]
        
        # Chuẩn hóa dữ liệu
        vm_scaled = kmeans_scaler.transform(vm_features)
        
        # Dự đoán cụm
        start_time = time.time()
        cluster_labels = kmeans_model.predict(vm_scaled)
        prediction_time = time.time() - start_time
        
        # Tính metrics
        silhouette_avg = silhouette_score(vm_scaled, cluster_labels)
        calinski_avg = calinski_harabasz_score(vm_scaled, cluster_labels)
        
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        print(f"Calinski-Harabasz Score: {calinski_avg:.4f}")
        print(f"Thời gian dự đoán trung bình: {prediction_time/len(vm_features):.6f} giây/mẫu")
        print(f"Tổng thời gian dự đoán {len(vm_features)} mẫu: {prediction_time:.4f} giây")
        
        # Phân tích cụm
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        print(f"\nPhân bố cụm:")
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(vm_features)) * 100
            print(f"  Cụm {cluster_id}: {count} VM ({percentage:.1f}%)")
        
        # Vẽ biểu đồ phân bố cụm
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.pie(cluster_counts.values, labels=[f'Cụm {i}' for i in cluster_counts.index], 
                autopct='%1.1f%%', startangle=90)
        plt.title('Phân bố VM theo cụm')
        
        plt.subplot(1, 2, 2)
        plt.bar(cluster_counts.index, cluster_counts.values, color='skyblue')
        plt.xlabel('Cluster ID')
        plt.ylabel('Số lượng VM')
        plt.title('Số lượng VM trong mỗi cụm')
        plt.xticks(range(len(cluster_counts)))
        
        plt.tight_layout()
        plt.savefig('models/kmeans_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'silhouette_score': silhouette_avg,
            'calinski_score': calinski_avg,
            'prediction_time': prediction_time,
            'avg_prediction_time': prediction_time/len(vm_features),
            'cluster_distribution': cluster_counts.to_dict()
        }
        
    except FileNotFoundError as e:
        print(f"Không tìm thấy file model: {e}")
        return None

def compare_models():
    """
    So sánh hiệu suất của các mô hình
    """
    print("\n=== SO SÁNH HIỆU SUẤT MÔ HÌNH ===")
    
    svm_results = evaluate_svm_model()
    kmeans_results = evaluate_kmeans_model()
    
    if svm_results and kmeans_results:
        # Tạo bảng so sánh
        comparison_data = {
            'Metric': ['Accuracy/Quality Score', 'Prediction Time (ms/sample)', 'Total Prediction Time (s)'],
            'SVM': [
                f"{svm_results['accuracy']:.4f}",
                f"{svm_results['avg_prediction_time']*1000:.2f}",
                f"{svm_results['prediction_time']:.4f}"
            ],
            'K-Means': [
                f"{kmeans_results['silhouette_score']:.4f}",
                f"{kmeans_results['avg_prediction_time']*1000:.2f}",
                f"{kmeans_results['prediction_time']:.4f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nBảng so sánh hiệu suất:")
        print(comparison_df.to_string(index=False))
        
        # Vẽ biểu đồ so sánh
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # So sánh thời gian dự đoán
        models = ['SVM', 'K-Means']
        prediction_times = [
            svm_results['avg_prediction_time']*1000,
            kmeans_results['avg_prediction_time']*1000
        ]
        
        axes[0].bar(models, prediction_times, color=['blue', 'orange'])
        axes[0].set_ylabel('Thời gian dự đoán (ms/mẫu)')
        axes[0].set_title('So sánh thời gian dự đoán')
        axes[0].set_ylim(0, max(prediction_times) * 1.2)
        
        # Thêm giá trị lên cột
        for i, v in enumerate(prediction_times):
            axes[0].text(i, v + max(prediction_times) * 0.01, f'{v:.2f}ms', 
                        ha='center', va='bottom')
        
        # So sánh chất lượng
        quality_scores = [svm_results['accuracy'], kmeans_results['silhouette_score']]
        quality_labels = ['Accuracy', 'Silhouette Score']
        
        axes[1].bar(models, quality_scores, color=['green', 'red'])
        axes[1].set_ylabel('Chất lượng mô hình')
        axes[1].set_title('So sánh chất lượng mô hình')
        axes[1].set_ylim(0, 1)
        
        # Thêm giá trị lên cột
        for i, v in enumerate(quality_scores):
            axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Lưu kết quả so sánh
        comparison_df.to_csv('models/model_comparison_results.csv', index=False)
        print("\n📊 Kết quả so sánh đã được lưu vào models/model_comparison_results.csv")

def generate_model_report():
    """
    Tạo báo cáo tổng hợp về các mô hình
    """
    print("\n=== TẠO BÁO CÁO MÔ HÌNH ===")
    
    report = []
    report.append("# BÁO CÁO HUẤN LUYỆN MÔ HÌNH MACHINE LEARNING")
    report.append("")
    report.append("## 1. Mô hình SVM (Support Vector Machine)")
    report.append("")
    
    try:
        svm_model = joblib.load('models/svm_model.joblib')
        report.append(f"- **Kernel**: {svm_model.kernel}")
        report.append(f"- **C parameter**: {svm_model.C}")
        report.append(f"- **Gamma**: {svm_model.gamma}")
        report.append(f"- **Số support vectors**: {sum(svm_model.n_support_)}")
        report.append(f"- **Số lớp**: {len(svm_model.classes_)}")
        report.append("")
    except:
        report.append("- Không tìm thấy mô hình SVM")
        report.append("")
    
    report.append("## 2. Mô hình K-Means")
    report.append("")
    
    try:
        kmeans_model = joblib.load('models/kmeans_model.joblib')
        report.append(f"- **Số cụm**: {kmeans_model.n_clusters}")
        report.append(f"- **Inertia**: {kmeans_model.inertia_:.4f}")
        report.append(f"- **Số lần lặp**: {kmeans_model.n_iter_}")
        report.append("")
    except:
        report.append("- Không tìm thấy mô hình K-Means")
        report.append("")
    
    report.append("## 3. Cách sử dụng mô hình")
    report.append("")
    report.append("### SVM Model:")
    report.append("```python")
    report.append("import joblib")
    report.append("")
    report.append("# Load model")
    report.append("svm_model = joblib.load('models/svm_model.joblib')")
    report.append("scaler = joblib.load('models/scaler.joblib')")
    report.append("")
    report.append("# Dự đoán")
    report.append("features = [cpu_cores, memory_gb, storage_gb, network_bandwidth, priority]")
    report.append("features_scaled = scaler.transform([features])")
    report.append("prediction = svm_model.predict(features_scaled)[0]")
    report.append("```")
    report.append("")
    
    report.append("### K-Means Model:")
    report.append("```python")
    report.append("import joblib")
    report.append("")
    report.append("# Load model")
    report.append("kmeans_model = joblib.load('models/kmeans_model.joblib')")
    report.append("kmeans_scaler = joblib.load('models/kmeans_scaler.joblib')")
    report.append("")
    report.append("# Dự đoán cụm")
    report.append("vm_features = [cpu_usage, ram_usage, storage_usage]")
    report.append("vm_scaled = kmeans_scaler.transform([vm_features])")
    report.append("cluster = kmeans_model.predict(vm_scaled)[0]")
    report.append("```")
    report.append("")
    
    report.append("## 4. Files được tạo")
    report.append("")
    report.append("- `models/svm_model.joblib`: Mô hình SVM đã huấn luyện")
    report.append("- `models/kmeans_model.joblib`: Mô hình K-Means đã huấn luyện")
    report.append("- `models/scaler.joblib`: Scaler cho SVM")
    report.append("- `models/kmeans_scaler.joblib`: Scaler cho K-Means")
    report.append("- `models/svm_grid_search.joblib`: Kết quả grid search SVM")
    report.append("- `models/kmeans_info.joblib`: Thông tin cụm K-Means")
    report.append("")
    
    # Lưu báo cáo
    with open('models/model_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("📄 Báo cáo đã được tạo: models/model_report.md")

def main():
    """
    Hàm chính để đánh giá mô hình
    """
    print("=== ĐÁNH GIÁ VÀ SO SÁNH MÔ HÌNH ===")
    
    # Đánh giá từng mô hình
    evaluate_svm_model()
    evaluate_kmeans_model()
    
    # So sánh mô hình
    compare_models()
    
    # Tạo báo cáo
    generate_model_report()
    
    print("\n✅ Hoàn thành đánh giá mô hình!")

if __name__ == "__main__":
    main() 