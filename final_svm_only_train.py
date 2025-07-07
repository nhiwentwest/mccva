#!/usr/bin/env python3
"""
FINAL: SVM-Only Training cho PHƯƠNG ÁN 2
✅ 64% accuracy SVM model cho OpenResty
❌ Skip K-Means (insufficient data variance)
"""
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_datasets_from_local():
    """Load datasets với column names đúng"""
    print("📂 Loading datasets from local machine...")
    
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        return None
    
    excel_files = [f for f in os.listdir(dataset_dir) if f.endswith('.xlsx')]
    if not excel_files:
        print("❌ No Excel files found!")
        return None
    
    print(f"📊 Found {len(excel_files)} Excel files:")
    
    request_data = []
    
    for file in excel_files:
        file_path = os.path.join(dataset_dir, file)
        try:
            print(f"  ✅ Loading {file}")
            df = pd.read_excel(file_path)
            print(f"     - Rows: {len(df)}, Columns: {len(df.columns)}")
            
            # Standardize column names
            df.columns = df.columns.str.strip().str.replace(' ', '_')
            
            # Add for request classification
            request_data.append(df)
                
        except Exception as e:
            print(f"  ❌ Error loading {file}: {e}")
    
    if not request_data:
        print("❌ No valid data loaded!")
        return None
    
    # Combine datasets
    combined_requests = pd.concat(request_data, ignore_index=True)
    
    print(f"\n📈 Combined Data Summary:")
    print(f"  Request data: {len(combined_requests)} samples")
    
    # Show column info
    print(f"📋 Columns: {list(combined_requests.columns)}")
    
    return combined_requests

def prepare_svm_data(df):
    """Chuẩn bị SVM data với columns thực tế"""
    print("\n🎯 Preparing SVM Data (Request Classification)...")
    
    # Map column names to standard names
    column_mapping = {
        'Jobs_per__1Minute': 'jobs_1min',
        'Jobs_per__5_Minutes': 'jobs_5min', 
        'Jobs_per__15Minutes': 'jobs_15min',
        'Mem_capacity': 'memory_mb',
        'Disk_capacity_GB': 'disk_gb',
        'Num_of_CPU_Cores': 'cpu_cores',
        'CPU_speed_per_Core': 'cpu_speed',
        'Avg_Recieve_Kbps': 'network_receive',
        'Avg_Transmit_Kbps': 'network_transmit',
        'Class_Name': 'class_name'
    }
    
    # Apply mapping
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Use existing Class_Name
    print("✅ Using existing Class_Name column")
    # Clean class names
    df['class_name'] = df['class_name'].astype(str).str.strip().str.replace("'", "")
    
    # Map to standard classes
    class_mapping = {
        'Very Low': 'small',
        'Low': 'small', 
        'Medium': 'medium',
        'High': 'large',
        'Very High': 'large'
    }
    
    df['makespan_class'] = df['class_name'].map(class_mapping)
    
    # Fill missing mappings with 'medium'
    df['makespan_class'] = df['makespan_class'].fillna('medium')
    
    # Check distribution
    class_counts = df['makespan_class'].value_counts()
    print("Class Distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # Feature engineering
    feature_columns = ['jobs_1min', 'jobs_5min', 'memory_mb', 'cpu_cores', 'cpu_speed', 
                      'network_receive', 'network_transmit']
    
    # Handle missing columns and values
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Derived features
    df['memory_gb'] = df['memory_mb'] / 1024
    df['network_total'] = df['network_receive'] + df['network_transmit']
    df['resource_density'] = df['memory_gb'] / (df['cpu_cores'] + 0.1)
    df['workload_intensity'] = df['jobs_1min'] / (df['cpu_cores'] + 0.1)
    
    # Final feature set
    svm_features = [
        'jobs_1min', 'jobs_5min', 'memory_gb', 'cpu_cores', 'cpu_speed',
        'network_receive', 'network_transmit', 'network_total', 
        'resource_density', 'workload_intensity'
    ]
    
    # Balance classes cho tốt hơn
    min_count = min(class_counts)
    print(f"📊 Minimum class count: {min_count}")
    
    if min_count > 100:  # Balance if enough samples
        balanced_samples = []
        target_size = min(min_count, 1000)  # Cap tại 1000 samples mỗi class
        
        for class_name in ['small', 'medium', 'large']:
            if class_name in class_counts:
                class_data = df[df['makespan_class'] == class_name].sample(
                    min(target_size, len(df[df['makespan_class'] == class_name])), 
                    random_state=42
                )
                balanced_samples.append(class_data)
                print(f"  📌 {class_name}: {len(class_data)} samples")
        
        df = pd.concat(balanced_samples, ignore_index=True)
        print(f"✅ Balanced dataset: {len(df)} total samples")
    
    X = df[svm_features].values
    y = df['makespan_class'].values
    
    print(f"✅ SVM Features: {X.shape}")
    print(f"✅ Feature names: {svm_features}")
    
    return X, y, svm_features

def train_svm_model(X, y, feature_names):
    """Train SVM model với optimization tốt hơn"""
    print("\n🚀 Training SVM Model...")
    
    # Check for multiple classes
    unique_classes = np.unique(y)
    print(f"🎯 Classes found: {unique_classes}")
    
    if len(unique_classes) < 2:
        print("❌ Need at least 2 classes for SVM!")
        return None, None, None, None, 0
    
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
    
    # SVM training với extended grid search
    print("🎯 SVM Training with Grid Search...")
    svm_grid = GridSearchCV(
        SVC(random_state=42, probability=True),
        {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly']
        },
        cv=5,  # Increased CV folds
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
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_labels, labels=['small', 'medium', 'large'])
    print("     small  medium  large")
    for i, label in enumerate(['small', 'medium', 'large']):
        print(f"{label:>6}: {cm[i]}")
    
    return best_svm, scaler, label_encoder, svm_grid, test_accuracy

def test_svm_scenarios(svm_model, scaler, label_encoder, feature_names):
    """Test SVM với các scenarios thực tế"""
    print("\n🧪 Testing SVM with Real Scenarios...")
    
    # Test scenarios for OpenResty
    test_scenarios = [
        {
            'name': 'Light Web Request',
            'jobs_1min': 2.0, 'jobs_5min': 8.0, 'memory_gb': 0.5, 'cpu_cores': 1.0,
            'cpu_speed': 2400.0, 'network_receive': 100.0, 'network_transmit': 50.0,
            'network_total': 150.0, 'resource_density': 0.5, 'workload_intensity': 2.0
        },
        {
            'name': 'Medium API Call',
            'jobs_1min': 15.0, 'jobs_5min': 60.0, 'memory_gb': 2.0, 'cpu_cores': 4.0,
            'cpu_speed': 3200.0, 'network_receive': 500.0, 'network_transmit': 300.0,
            'network_total': 800.0, 'resource_density': 0.5, 'workload_intensity': 3.75
        },
        {
            'name': 'Heavy Processing',
            'jobs_1min': 45.0, 'jobs_5min': 180.0, 'memory_gb': 8.0, 'cpu_cores': 8.0,
            'cpu_speed': 3600.0, 'network_receive': 2000.0, 'network_transmit': 1500.0,
            'network_total': 3500.0, 'resource_density': 1.0, 'workload_intensity': 5.625
        }
    ]
    
    for scenario in test_scenarios:
        name = scenario.pop('name')
        features = np.array([[scenario[fname] for fname in feature_names]])
        features_scaled = scaler.transform(features)
        
        prediction = svm_model.predict(features_scaled)[0]
        probabilities = svm_model.predict_proba(features_scaled)[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        
        print(f"\n📋 {name}:")
        print(f"   Features: {[f'{k}={v}' for k,v in scenario.items()]}")
        print(f"   ✅ Prediction: {predicted_class}")
        print(f"   📊 Probabilities: {dict(zip(label_encoder.classes_, probabilities))}")

def save_svm_model_for_deployment(svm_model, svm_scaler, label_encoder, svm_grid, 
                                  svm_features, svm_accuracy):
    """Save SVM model cho cloud deployment"""
    print("\n💾 Saving SVM Model for Cloud Deployment...")
    
    os.makedirs('models', exist_ok=True)
    
    # Save SVM models
    joblib.dump(svm_model, 'models/svm_model.joblib')
    joblib.dump(svm_scaler, 'models/svm_scaler.joblib') 
    joblib.dump(label_encoder, 'models/svm_label_encoder.joblib')
    joblib.dump(svm_features, 'models/svm_feature_names.joblib')
    joblib.dump(svm_grid, 'models/svm_grid_search.joblib')
    
    # Training info
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'model': {
            'type': 'SVM (Support Vector Machine)',
            'kernel': svm_model.kernel,
            'C': svm_model.C,
            'gamma': svm_model.gamma,
            'test_accuracy': svm_accuracy,
            'feature_names': svm_features,
            'classes': list(label_encoder.classes_),
            'label_mapping': dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        },
        'deployment_plan': 'Phương án 2: SVM-only cho OpenResty',
        'usage': 'POST /predict - Phân loại yêu cầu (small/medium/large)',
        'notes': {
            'kmeans_status': 'Skipped - Insufficient data variance',
            'optimization': 'Optimized cho OpenResty real-time prediction',
            'memory_footprint': 'Lightweight SVM model'
        }
    }
    
    joblib.dump(training_info, 'models/training_info.joblib')
    
    print("✅ SVM Model saved for deployment:")
    for file in sorted(os.listdir('models')):
        size = os.path.getsize(f'models/{file}')
        print(f"  📄 {file} ({size:,} bytes)")
    
    return True

def main():
    """Main SVM training pipeline"""
    print("🚀 FINAL: SVM-ONLY TRAINING cho PHƯƠNG ÁN 2")
    print("=" * 60)
    print("🎯 Train SVM với dataset thực tế")
    print("📊 SVM: Request classification (small/medium/large)")
    print("❌ Skip K-Means: Insufficient data variance")
    print()
    
    start_time = datetime.now()
    
    try:
        # Load datasets
        request_df = load_datasets_from_local()
        if request_df is None:
            return False
        
        # Prepare SVM data
        X_svm, y_svm, svm_features = prepare_svm_data(request_df)
        
        # Train SVM
        svm_model, svm_scaler, label_encoder, svm_grid, svm_accuracy = train_svm_model(X_svm, y_svm, svm_features)
        
        if svm_model is None:
            print("❌ SVM training failed!")
            return False
        
        # Test scenarios
        test_svm_scenarios(svm_model, svm_scaler, label_encoder, svm_features)
        
        # Save model
        save_svm_model_for_deployment(
            svm_model, svm_scaler, label_encoder, svm_grid,
            svm_features, svm_accuracy
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "=" * 60)
        print("🎉 SVM TRAINING COMPLETED!")
        print("=" * 60)
        print(f"✅ SVM Model: {svm_model.kernel} kernel")
        print(f"📊 Test Accuracy: {svm_accuracy:.1%}")
        print(f"🎯 Classes: {list(label_encoder.classes_)}")
        print(f"⏱️  Total time: {duration:.1f} minutes")
        print(f"🚀 Ready for OpenResty deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ SVM Model ready for cloud deployment!")
        print("📦 Upload folder 'models/' to cloud server")
        print("🎯 Use SVM for real-time request classification in OpenResty")
    else:
        print("\n❌ Training failed!") 