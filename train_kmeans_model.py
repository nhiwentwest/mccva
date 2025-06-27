import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def find_optimal_k(vm_data, max_k=10):
    """
    Tìm số cụm tối ưu bằng phương pháp Elbow và Silhouette
    """
    print("Đang tìm số cụm tối ưu...")
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    vm_scaled = scaler.fit_transform(vm_data)
    
    # Tính các metrics cho các giá trị k khác nhau
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(vm_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(vm_scaled, kmeans.labels_))
        calinski_scores.append(calinski_harabasz_score(vm_scaled, kmeans.labels_))
    
    # Vẽ biểu đồ Elbow
    plt.figure(figsize=(15, 5))
    
    # Elbow plot
    plt.subplot(1, 3, 1)
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Số cụm (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True)
    
    # Silhouette plot
    plt.subplot(1, 3, 2)
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.xlabel('Số cụm (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.grid(True)
    
    # Calinski-Harabasz plot
    plt.subplot(1, 3, 3)
    plt.plot(k_range, calinski_scores, 'go-')
    plt.xlabel('Số cụm (k)')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Method')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/kmeans_optimal_k.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Tìm k tối ưu (silhouette score cao nhất)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    print(f"Silhouette scores: {silhouette_scores}")
    print(f"Optimal k (Silhouette): {optimal_k}")
    
    return optimal_k, scaler

def train_kmeans_model(vm_data, n_clusters=3, scaler=None):
    """
    Huấn luyện mô hình K-Means
    """
    print(f"Đang huấn luyện K-Means với {n_clusters} cụm...")
    
    # Chuẩn hóa dữ liệu nếu chưa có scaler
    if scaler is None:
        scaler = StandardScaler()
        vm_scaled = scaler.fit_transform(vm_data)
    else:
        vm_scaled = scaler.transform(vm_data)
    
    # Huấn luyện K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(vm_scaled)
    
    # Tính các metrics
    silhouette_avg = silhouette_score(vm_scaled, kmeans.labels_)
    calinski_avg = calinski_harabasz_score(vm_scaled, kmeans.labels_)
    
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_avg:.4f}")
    
    return kmeans, scaler, vm_scaled

def visualize_clusters(vm_data, kmeans, scaler, save_path='models/kmeans_clusters.png'):
    """
    Trực quan hóa kết quả phân cụm
    """
    # Chuẩn hóa dữ liệu
    vm_scaled = scaler.transform(vm_data)
    
    # Giảm chiều dữ liệu để vẽ 2D
    pca = PCA(n_components=2)
    vm_pca = pca.fit_transform(vm_scaled)
    
    # Vẽ biểu đồ phân cụm
    plt.figure(figsize=(12, 5))
    
    # Biểu đồ phân cụm
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(vm_pca[:, 0], vm_pca[:, 1], c=kmeans.labels_, 
                         cmap='viridis', alpha=0.6, s=50)
    plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], 
               pca.transform(kmeans.cluster_centers_)[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('K-Means Clustering Results')
    plt.legend()
    plt.colorbar(scatter)
    
    # Biểu đồ phân bố cụm
    plt.subplot(1, 2, 2)
    cluster_counts = pd.Series(kmeans.labels_).value_counts().sort_index()
    plt.bar(cluster_counts.index, cluster_counts.values, color='skyblue')
    plt.xlabel('Cluster ID')
    plt.ylabel('Số lượng VM')
    plt.title('Phân bố VM theo cụm')
    plt.xticks(range(len(cluster_counts)))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_clusters(vm_data, kmeans, scaler):
    """
    Phân tích đặc điểm của từng cụm
    """
    vm_scaled = scaler.transform(vm_data)
    labels = kmeans.labels_
    
    # Thêm nhãn cụm vào dữ liệu
    vm_with_clusters = vm_data.copy()
    vm_with_clusters['cluster'] = labels
    
    print("\n=== PHÂN TÍCH CỤM ===")
    
    # Thống kê từng cụm
    for cluster_id in range(kmeans.n_clusters):
        cluster_data = vm_with_clusters[vm_with_clusters['cluster'] == cluster_id]
        
        print(f"\nCụm {cluster_id} ({len(cluster_data)} VM):")
        print(f"  CPU Usage: {cluster_data['cpu_usage'].mean():.3f} ± {cluster_data['cpu_usage'].std():.3f}")
        print(f"  RAM Usage: {cluster_data['ram_usage'].mean():.3f} ± {cluster_data['ram_usage'].std():.3f}")
        print(f"  Storage Usage: {cluster_data['storage_usage'].mean():.3f} ± {cluster_data['storage_usage'].std():.3f}")
    
    # Vẽ biểu đồ box plot cho từng đặc trưng
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    features = ['cpu_usage', 'ram_usage', 'storage_usage']
    feature_names = ['CPU Usage', 'RAM Usage', 'Storage Usage']
    
    for i, (feature, name) in enumerate(zip(features, feature_names)):
        vm_with_clusters.boxplot(column=feature, by='cluster', ax=axes[i])
        axes[i].set_title(f'{name} by Cluster')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel(name)
    
    plt.tight_layout()
    plt.savefig('models/kmeans_cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Hàm chính để huấn luyện K-Means model
    """
    print("=== HUẤN LUYỆN MÔ HÌNH K-MEANS ===")
    
    # Tạo thư mục models nếu chưa có
    import os
    os.makedirs('models', exist_ok=True)
    
    # Đọc dữ liệu VM
    try:
        vm_data = pd.read_csv('data/vm_data.csv')
        print(f"Đã đọc {len(vm_data)} mẫu dữ liệu VM từ data/vm_data.csv")
    except FileNotFoundError:
        print("Không tìm thấy file dữ liệu VM. Đang tạo dữ liệu mẫu...")
        from data_generator import generate_sample_data
        _, vm_data = generate_sample_data(2000)
        vm_data.to_csv('data/vm_data.csv', index=False)
    
    # Chọn đặc trưng cho phân cụm
    features = ['cpu_usage', 'ram_usage', 'storage_usage']
    vm_features = vm_data[features]
    
    print(f"Đặc trưng sử dụng: {features}")
    print(f"Kích thước dữ liệu: {vm_features.shape}")
    
    # Tìm số cụm tối ưu
    optimal_k, scaler = find_optimal_k(vm_features, max_k=8)
    
    # Huấn luyện model với k tối ưu
    kmeans, scaler, vm_scaled = train_kmeans_model(vm_features, n_clusters=optimal_k, scaler=scaler)
    
    # Trực quan hóa kết quả
    visualize_clusters(vm_features, kmeans, scaler)
    
    # Phân tích cụm
    analyze_clusters(vm_features, kmeans, scaler)
    
    # Lưu model và scaler
    joblib.dump(kmeans, 'models/kmeans_model.joblib')
    joblib.dump(scaler, 'models/kmeans_scaler.joblib')
    
    # Lưu thông tin cụm
    cluster_info = {
        'n_clusters': kmeans.n_clusters,
        'cluster_centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_,
        'labels': kmeans.labels_
    }
    joblib.dump(cluster_info, 'models/kmeans_info.joblib')
    
    print("\n=== KẾT QUẢ ===")
    print("✅ Mô hình K-Means đã được huấn luyện thành công!")
    print(f"📊 Số cụm tối ưu: {optimal_k}")
    print("📁 Model đã được lưu vào: models/kmeans_model.joblib")
    print("📁 Scaler đã được lưu vào: models/kmeans_scaler.joblib")
    print("📁 Thông tin cụm đã được lưu vào: models/kmeans_info.joblib")
    print("📊 Biểu đồ đã được lưu vào thư mục models/")
    
    # Hiển thị thông tin model
    print(f"\nThông tin model:")
    print(f"- Số cụm: {kmeans.n_clusters}")
    print(f"- Inertia: {kmeans.inertia_:.4f}")
    print(f"- Số lần lặp: {kmeans.n_iter_}")

if __name__ == "__main__":
    main() 