#!/usr/bin/env python3
"""
PHƯƠNG ÁN 2: Tích hợp mô hình ML đã huấn luyện sẵn trong OpenResty
Train SVM + K-Means locally → Deploy to cloud

🎯 Mục đích:
- Huấn luyện SVM cho phân loại yêu cầu (small/medium/large)
- Huấn luyện K-Means cho phân cụm VM 
- Save models dạng joblib để OpenResty sử dụng
- Triển khai lên cloud làm REST API server nhẹ
"""
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_datasets_from_local():
    """Load tất cả datasets từ thư mục dataset/"""
    print("📂 Loading datasets from local machine...")
    
    datasets = []
    dataset_dir = 'dataset'
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        return None, None
    
    # Load tất cả file Excel trong thư mục dataset
    excel_files = [f for f in os.listdir(dataset_dir) if f.endswith('.xlsx')]
    
    if not excel_files:
        print("❌ No Excel files found in dataset directory!")
        return None, None
    
    print(f"📊 Found {len(excel_files)} Excel files:")
    
    request_data = []
    vm_data = []
    
    for file in excel_files:
        file_path = os.path.join(dataset_dir, file)
        try:
            print(f"  ✅ Loading {file}")
            df = pd.read_excel(file_path)
            print(f"     - Rows: {len(df)}, Columns: {len(df.columns)}")
            
            # Tách data cho SVM (request classification) và K-Means (VM clustering)
            request_data.append(df)
            
            # Extract VM data cho K-Means (giả sử có columns về VM resources)
            vm_columns = ['cpu_cores', 'memory_mb', 'cpu_speed', 'network_receive', 'network_transmit']
            available_vm_cols = [col for col in vm_columns if col in df.columns]
            
            if available_vm_cols:
                vm_subset = df[available_vm_cols].copy()
                vm_data.append(vm_subset)
                
        except Exception as e:
            print(f"  ❌ Error loading {file}: {e}")
    
    if not request_data:
        print("❌ No valid data loaded!")
        return None, None
    
    # Combine all datasets
    combined_requests = pd.concat(request_data, ignore_index=True)
    combined_vms = pd.concat(vm_data, ignore_index=True) if vm_data else None
    
    print(f"\n📈 Combined Data Summary:")
    print(f"  Request data: {len(combined_requests)} samples")
    print(f"  VM data: {len(combined_vms) if combined_vms is not None else 0} samples")
    
    return combined_requests, combined_vms

def prepare_svm_data(df):
    """Chuẩn bị data cho SVM classification"""
    print("\n🎯 Preparing SVM Data (Request Classification)...")
    
    def classify_request_makespan(row):
        """Phân loại makespan theo SVM-friendly boundaries"""
        cpu = row.get('cpu_cores', 0)
        memory = row.get('memory_mb', 0) / 1024  # Convert to GB
        jobs = row.get('jobs_1min', 0)
        network = row.get('network_receive', 0) + row.get('network_transmit', 0)
        
        # Clear decision boundaries cho SVM
        # Small: low resource requests
        if cpu <= 4 and memory <= 8 and jobs <= 10:
            return 'small'
        
        # Large: high resource requests  
        elif cpu >= 12 or memory >= 32 or jobs >= 20:
            return 'large'
        
        # Medium: everything else
        else:
            return 'medium'
    
    # Apply classification
    df['makespan_class'] = df.apply(classify_request_makespan, axis=1)
    
    # Check distribution
    class_counts = df['makespan_class'].value_counts()
    print("Request Class Distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # Feature engineering cho SVM
    feature_columns = [
        'cpu_cores', 'memory_mb', 'jobs_1min', 'jobs_5min',
        'network_receive', 'network_transmit', 'cpu_speed'
    ]
    
    # Handle missing values
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Derived features
    df['memory_gb'] = df['memory_mb'] / 1024
    df['network_total'] = df['network_receive'] + df['network_transmit']
    df['resource_density'] = df['memory_gb'] / (df['cpu_cores'] + 0.1)
    df['workload_intensity'] = df['jobs_1min'] / (df['cpu_cores'] + 0.1)
    
    # Final feature set (10 features theo spec)
    svm_features = [
        'cpu_cores', 'memory_gb', 'jobs_1min', 'jobs_5min',
        'network_receive', 'network_transmit', 'cpu_speed',
        'network_total', 'resource_density', 'workload_intensity'
    ]
    
    # Balance classes
    min_count = min(class_counts)
    balanced_samples = []
    
    for class_name in ['small', 'medium', 'large']:
        if class_name in class_counts:
            class_data = df[df['makespan_class'] == class_name].sample(
                min(min_count, len(df[df['makespan_class'] == class_name])), 
                random_state=42
            )
            balanced_samples.append(class_data)
    
    balanced_df = pd.concat(balanced_samples, ignore_index=True)
    
    X = balanced_df[svm_features].values
    y = balanced_df['makespan_class'].values
    
    print(f"✅ SVM Features: {X.shape}")
    print(f"✅ Feature names: {svm_features}")
    
    return X, y, svm_features

def prepare_kmeans_data(df):
    """Chuẩn bị data cho K-Means VM clustering"""
    print("\n🎯 Preparing K-Means Data (VM Clustering)...")
    
    if df is None or len(df) == 0:
        print("❌ No VM data available for K-Means!")
        return None, None
    
    # VM resource features cho clustering
    vm_features = ['cpu_cores', 'memory_mb', 'cpu_speed', 'network_receive', 'network_transmit']
    
    # Handle missing values
    for col in vm_features:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Feature engineering
    df['memory_gb'] = df['memory_mb'] / 1024
    df['network_total'] = df['network_receive'] + df['network_transmit']
    df['cpu_memory_ratio'] = df['cpu_cores'] / (df['memory_gb'] + 0.1)
    
    # Final features for clustering
    kmeans_features = [
        'cpu_cores', 'memory_gb', 'cpu_speed', 'network_total', 'cpu_memory_ratio'
    ]
    
    # Ensure all features exist
    available_features = [f for f in kmeans_features if f in df.columns]
    
    if len(available_features) < 3:
        print("❌ Not enough features for VM clustering!")
        return None, None
    
    X_vm = df[available_features].values
    
    print(f"✅ VM Features: {X_vm.shape}")
    print(f"✅ Feature names: {available_features}")
    
    return X_vm, available_features

def train_svm_model(X, y, feature_names):
    """Train SVM model cho request classification"""
    print("\n🚀 Training SVM Model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # SVM Grid Search
    print("🎯 SVM Grid Search...")
    svm_grid = GridSearchCV(
        SVC(random_state=42, probability=True),
        {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear', 'poly']
        },
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    svm_grid.fit(X_train_scaled, y_train_encoded)
    best_svm = svm_grid.best_estimator_
    
    # Evaluate
    y_pred = best_svm.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test_encoded, y_pred)
    
    print(f"\n📊 SVM Results:")
    print(f"Best params: {svm_grid.best_params_}")
    print(f"CV Score: {svm_grid.best_score_:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    
    # Classification report
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_labels))
    
    return best_svm, scaler, label_encoder, svm_grid, test_accuracy

def train_kmeans_model(X_vm, vm_features):
    """Train K-Means model cho VM clustering"""
    print("\n🚀 Training K-Means Model...")
    
    if X_vm is None:
        return None, None, None
    
    # Standard scaling for K-Means
    vm_scaler = StandardScaler()
    X_vm_scaled = vm_scaler.fit_transform(X_vm)
    
    # Find optimal number of clusters
    print("🔍 Finding optimal clusters...")
    silhouette_scores = []
    K_range = range(2, min(8, len(X_vm_scaled)))
    
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans_temp.fit_predict(X_vm_scaled)
        silhouette_avg = silhouette_score(X_vm_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"  k={k}: silhouette={silhouette_avg:.3f}")
    
    # Choose best k
    best_k = K_range[np.argmax(silhouette_scores)]
    best_silhouette = max(silhouette_scores)
    
    print(f"✅ Optimal clusters: {best_k} (silhouette: {best_silhouette:.3f})")
    
    # Train final K-Means
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_vm_scaled)
    
    print(f"\n📊 K-Means Results:")
    print(f"Clusters: {best_k}")
    print(f"Silhouette score: {best_silhouette:.3f}")
    
    # Cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("Cluster distribution:")
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} VMs")
    
    return kmeans_final, vm_scaler, best_silhouette

def test_models_scenarios(svm_model, svm_scaler, label_encoder, kmeans_model, vm_scaler):
    """Test cả 2 models với realistic scenarios"""
    print("\n🧪 Testing Models on Realistic Scenarios...")
    
    # Test SVM scenarios
    print("\n📊 SVM Test Scenarios:")
    svm_scenarios = [
        {'name': 'Light Web Server', 'features': [2, 4, 5, 3, 100, 80, 2.4, 180, 2.0, 2.5], 'expected': 'small'},
        {'name': 'Production API', 'features': [8, 16, 12, 10, 500, 400, 3.2, 900, 2.0, 1.5], 'expected': 'medium'},
        {'name': 'ML Training Job', 'features': [16, 64, 25, 20, 1000, 800, 4.0, 1800, 4.0, 1.56], 'expected': 'large'},
        {'name': 'Database Server', 'features': [6, 32, 15, 12, 600, 500, 3.0, 1100, 5.33, 2.5], 'expected': 'medium'},
        {'name': 'Video Processing', 'features': [12, 48, 20, 18, 800, 700, 3.8, 1500, 4.0, 1.67], 'expected': 'large'}
    ]
    
    svm_correct = 0
    for scenario in svm_scenarios:
        features_scaled = svm_scaler.transform([scenario['features']])
        pred_encoded = svm_model.predict(features_scaled)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]
        
        # Confidence
        probabilities = svm_model.predict_proba(features_scaled)[0]
        confidence = max(probabilities)
        
        is_correct = prediction == scenario['expected']
        status = "✅" if is_correct else "❌"
        
        print(f"  {status} {scenario['name']}: {prediction} (conf: {confidence:.2f}) - expected: {scenario['expected']}")
        
        if is_correct:
            svm_correct += 1
    
    svm_accuracy = svm_correct / len(svm_scenarios)
    print(f"\n🎯 SVM Scenario Accuracy: {svm_correct}/{len(svm_scenarios)} = {svm_accuracy:.1%}")
    
    # Test K-Means scenarios
    if kmeans_model and vm_scaler:
        print("\n📊 K-Means Test Scenarios:")
        vm_scenarios = [
            {'name': 'Low-end VM', 'features': [2, 4, 2.4, 150, 0.5]},
            {'name': 'Mid-range VM', 'features': [8, 16, 3.2, 800, 0.5]},
            {'name': 'High-end VM', 'features': [16, 64, 4.0, 1500, 0.25]},
            {'name': 'Memory-optimized', 'features': [4, 32, 2.8, 600, 0.125]},
            {'name': 'CPU-optimized', 'features': [24, 32, 4.2, 1000, 0.75]}
        ]
        
        for scenario in vm_scenarios:
            features_scaled = vm_scaler.transform([scenario['features']])
            cluster = kmeans_model.predict(features_scaled)[0]
            print(f"  📍 {scenario['name']}: Cluster {cluster}")
    
    return svm_accuracy

def save_models_for_deployment(svm_model, svm_scaler, label_encoder, svm_grid, 
                              kmeans_model, vm_scaler, svm_features, vm_features,
                              svm_accuracy, kmeans_silhouette):
    """Save models cho deployment lên cloud"""
    print("\n💾 Saving Models for Cloud Deployment...")
    
    os.makedirs('models', exist_ok=True)
    
    # Save SVM models
    joblib.dump(svm_model, 'models/svm_model.joblib')
    joblib.dump(svm_scaler, 'models/svm_scaler.joblib') 
    joblib.dump(label_encoder, 'models/svm_label_encoder.joblib')
    joblib.dump(svm_grid, 'models/svm_grid_search.joblib')
    
    # Save K-Means models (nếu có)
    if kmeans_model and vm_scaler:
        joblib.dump(kmeans_model, 'models/kmeans_model.joblib')
        joblib.dump(vm_scaler, 'models/kmeans_scaler.joblib')
    
    # Training info
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'models': {
            'svm': {
                'type': 'SVM (Support Vector Machine)',
                'kernel': svm_model.kernel,
                'C': svm_model.C,
                'gamma': svm_model.gamma,
                'test_accuracy': svm_accuracy,
                'feature_names': svm_features,
                'label_mapping': dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
            },
            'kmeans': {
                'type': 'K-Means Clustering',
                'n_clusters': kmeans_model.n_clusters if kmeans_model else None,
                'silhouette_score': kmeans_silhouette,
                'feature_names': vm_features if vm_features else None
            } if kmeans_model else None
        },
        'deployment_plan': 'Phương án 2: Tích hợp mô hình ML đã huấn luyện sẵn trong OpenResty',
        'usage': {
            'svm': 'POST /predict - Phân loại yêu cầu (small/medium/large)',
            'kmeans': 'GET /vm_clusters - Phân cụm VM resources'
        },
        'integration_methods': [
            'HTTP API (Flask/FastAPI)',
            'FFI (Foreign Function Interface)', 
            'Subprocess calls via lua-resty-shell'
        ]
    }
    
    joblib.dump(training_info, 'models/training_info.joblib')
    
    print("✅ Models saved for deployment:")
    for file in sorted(os.listdir('models')):
        size = os.path.getsize(f'models/{file}')
        print(f"  📄 {file} ({size} bytes)")
    
    return True

def main():
    """Main training pipeline"""
    print("🚀 PHƯƠNG ÁN 2: TRAIN SVM + K-MEANS LOCAL")
    print("=" * 60)
    print("🎯 Huấn luyện trên local machine → Deploy lên cloud")
    print("📊 SVM: Request classification (small/medium/large)")
    print("🔧 K-Means: VM clustering cho load balancing")
    print("⚡ Output: Models sẵn sàng cho OpenResty integration")
    print()
    
    start_time = datetime.now()
    
    try:
        # Load datasets
        request_df, vm_df = load_datasets_from_local()
        if request_df is None:
            return False
        
        # Prepare SVM data
        X_svm, y_svm, svm_features = prepare_svm_data(request_df)
        
        # Prepare K-Means data  
        X_vm, vm_features = prepare_kmeans_data(vm_df)
        
        # Train SVM
        svm_model, svm_scaler, label_encoder, svm_grid, svm_accuracy = train_svm_model(X_svm, y_svm, svm_features)
        
        # Train K-Means
        kmeans_model, vm_scaler, kmeans_silhouette = train_kmeans_model(X_vm, vm_features)
        
        # Test scenarios
        scenario_accuracy = test_models_scenarios(svm_model, svm_scaler, label_encoder, kmeans_model, vm_scaler)
        
        # Save models
        save_models_for_deployment(
            svm_model, svm_scaler, label_encoder, svm_grid,
            kmeans_model, vm_scaler, svm_features, vm_features,
            svm_accuracy, kmeans_silhouette
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED!")
        print("=" * 60)
        print(f"✅ SVM Model: {svm_model.kernel} kernel, accuracy: {svm_accuracy:.1%}")
        if kmeans_model:
            print(f"✅ K-Means Model: {kmeans_model.n_clusters} clusters, silhouette: {kmeans_silhouette:.3f}")
        print(f"⏱️  Total time: {duration:.1f} minutes")
        print(f"🎯 Ready for OpenResty deployment!")
        
        print(f"\n🚀 Next Steps:")
        print("1. Upload models/ folder to cloud server")
        print("2. Setup Flask/FastAPI service với joblib.load()")
        print("3. Configure OpenResty để gọi ML API")
        print("4. Test endpoints: POST /predict, GET /vm_clusters")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Models ready for cloud deployment!")
        print("📦 Upload folder 'models/' to cloud server")
        print("🔧 Setup ML service với joblib models")
        print("🌐 Integrate với OpenResty theo Phương án 2")
    else:
        print("\n❌ Training failed!") 