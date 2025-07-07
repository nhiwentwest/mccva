#!/usr/bin/env python3
"""
FORCE SVM ONLY TRAINING - Đúng với thiết kế ban đầu
Stick với SVM theo Phương án 2: Tích hợp mô hình SVM đã huấn luyện sẵn trong OpenResty
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

def load_real_datasets():
    """Load real Excel datasets"""
    print("📂 Loading real datasets...")
    
    datasets = []
    files = ['mmc2.xlsx', 'mmc3.xlsx', 'mmc4.xlsx']
    
    for file in files:
        file_path = f'dataset/{file}'
        if os.path.exists(file_path):
            print(f"  ✅ Loading {file}")
            df = pd.read_excel(file_path)
            datasets.append(df)
        else:
            print(f"  ❌ File not found: {file}")
    
    if not datasets:
        print("❌ No datasets found!")
        return None
    
    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    print(f"📊 Total samples: {len(combined_df)}")
    
    return combined_df

def svm_class_assignment(df):
    """SVM-optimized class assignment với clear boundaries"""
    print("\n🎯 SVM Class Assignment (optimized for linear separation)...")
    
    def classify_svm_optimal(row):
        """SVM-friendly classification với clear decision boundaries"""
        cpu = row.get('cpu_cores', 0)
        memory = row.get('memory_mb', 0) / 1024  # Convert to GB
        jobs = row.get('jobs_1min', 0)
        
        # SVM-optimized thresholds for better linear separation
        # Small: low resource usage
        if cpu <= 4 and memory <= 0.025 and jobs <= 8:
            return 'small'
        
        # Large: high resource usage (OR condition for SVM hyperplane)
        elif cpu >= 8 or memory >= 0.045 or jobs >= 12:
            return 'large'
        
        # Medium: everything else
        else:
            return 'medium'
    
    # Apply classification
    df['makespan'] = df.apply(classify_svm_optimal, axis=1)
    
    # Check distribution
    class_counts = df['makespan'].value_counts()
    print(f"Class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # Balance classes for SVM training
    min_count = min(class_counts)
    balanced_samples = []
    
    for class_name in ['small', 'medium', 'large']:
        class_data = df[df['makespan'] == class_name].sample(min_count, random_state=42)
        balanced_samples.append(class_data)
    
    balanced_df = pd.concat(balanced_samples, ignore_index=True)
    
    final_counts = balanced_df['makespan'].value_counts()
    print(f"\n✅ Balanced for SVM:")
    for class_name, count in final_counts.items():
        print(f"  {class_name}: {count}")
    
    return balanced_df

def svm_feature_engineering(df):
    """Feature engineering optimized for SVM"""
    print("\n🔧 SVM Feature Engineering...")
    
    # Core features for SVM
    feature_columns = [
        'cpu_cores', 'memory_mb', 'jobs_1min', 'jobs_5min',
        'network_receive', 'network_transmit', 'cpu_speed'
    ]
    
    # Handle missing values
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # SVM-specific feature engineering
    df['memory_gb'] = df['memory_mb'] / 1024  # Convert to GB for better scaling
    df['network_total'] = df['network_receive'] + df['network_transmit']
    df['resource_ratio'] = df['memory_gb'] / (df['cpu_cores'] + 0.1)  # Avoid division by 0
    
    # Final feature set for SVM (10 features as per original design)
    svm_features = [
        'cpu_cores', 'memory_gb', 'jobs_1min', 'jobs_5min',
        'network_receive', 'network_transmit', 'cpu_speed',
        'network_total', 'resource_ratio', 'priority'
    ]
    
    # Add priority if missing (for 10 features total)
    if 'priority' not in df.columns:
        # Derive priority from resource usage
        df['priority'] = np.clip(
            (df['cpu_cores'] / 4 + df['memory_gb'] / 16 + df['jobs_1min'] / 10).astype(int), 
            1, 5
        )
    
    # Prepare features matrix
    X = df[svm_features].values
    y = df['makespan'].values
    
    print(f"✅ SVM Features shape: {X.shape}")
    print(f"✅ SVM Feature names: {svm_features}")
    
    return X, y, svm_features

def svm_only_training(X, y, feature_names):
    """SVM ONLY training - theo thiết kế ban đầu"""
    print("\n🚀 SVM ONLY Training (theo Phương án 2)...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Standard scaling (quan trọng cho SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # SVM ONLY - theo spec ban đầu
    print("🎯 Training SVM (RBF kernel)...")
    svm_grid = GridSearchCV(
        SVC(random_state=42, probability=True),  # probability=True for confidence scores
        {
            'C': [0.1, 1, 10, 100],              # Regularization parameter
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Kernel coefficient
            'kernel': ['rbf', 'linear', 'poly']   # Kernel types
        },
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    svm_grid.fit(X_train_scaled, y_train_encoded)
    
    # Get best SVM model
    best_svm = svm_grid.best_estimator_
    
    print(f"\n🏆 Best SVM Parameters: {svm_grid.best_params_}")
    print(f"🎯 Best CV Score: {svm_grid.best_score_:.3f}")
    
    # Test accuracy
    y_pred = best_svm.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test_encoded, y_pred)
    
    print(f"\n📊 SVM Performance:")
    print(f"Training accuracy: {best_svm.score(X_train_scaled, y_train_encoded):.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    
    # Classification report
    print("\n📋 SVM Classification Report:")
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    print(classification_report(y_test, y_pred_labels))
    
    # Confusion matrix
    print("\n🔢 SVM Confusion Matrix:")
    cm = confusion_matrix(y_test_encoded, y_pred)
    print("       small  medium  large")
    for i, class_name in enumerate(['small', 'medium', 'large']):
        print(f"{class_name:>6} {cm[i]}")
    
    return best_svm, scaler, label_encoder, svm_grid

def test_svm_scenarios(svm_model, scaler, label_encoder):
    """Test SVM với realistic scenarios"""
    print(f"\n🧪 Testing SVM on realistic scenarios...")
    
    # Test scenarios theo thiết kế OpenResty
    scenarios = [
        # Small workloads - for lightweight OpenResty processing
        {'name': 'Micro Service', 'features': [2, 0.004, 6, 4, 100, 80, 2.4, 180, 0.002, 1], 'expected': 'small'},
        {'name': 'Basic Web Server', 'features': [4, 0.008, 7, 5, 200, 150, 2.8, 350, 0.002, 2], 'expected': 'small'},
        {'name': 'Development Server', 'features': [3, 0.006, 8, 6, 150, 120, 2.6, 270, 0.002, 1], 'expected': 'small'},
        
        # Medium workloads - standard OpenResty load balancing
        {'name': 'Production API', 'features': [6, 0.016, 9, 7, 600, 500, 3.2, 1100, 0.0027, 3], 'expected': 'medium'},
        {'name': 'Web Application', 'features': [5, 0.012, 10, 8, 500, 400, 3.0, 900, 0.0024, 3], 'expected': 'medium'},
        {'name': 'Database Server', 'features': [6, 0.020, 11, 9, 700, 600, 3.4, 1300, 0.0033, 4], 'expected': 'medium'},
        
        # Large workloads - high-performance routing
        {'name': 'ML Training Server', 'features': [16, 0.064, 18, 15, 2000, 1800, 4.0, 3800, 0.004, 5], 'expected': 'large'},
        {'name': 'Video Processing', 'features': [12, 0.048, 15, 12, 1500, 1200, 3.8, 2700, 0.004, 4], 'expected': 'large'},
        {'name': 'Enterprise Server', 'features': [20, 0.080, 20, 18, 2500, 2200, 4.2, 4700, 0.004, 5], 'expected': 'large'}
    ]
    
    correct = 0
    total = len(scenarios)
    
    print("  SVM Scenario Results:")
    for scenario in scenarios:
        features_scaled = scaler.transform([scenario['features']])
        pred_encoded = svm_model.predict(features_scaled)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]
        
        # Get confidence (probability)
        probabilities = svm_model.predict_proba(features_scaled)[0]
        confidence = max(probabilities)
        
        is_correct = prediction == scenario['expected']
        status = "✅" if is_correct else "❌"
        
        print(f"    {status} {scenario['name']}: {prediction} (conf: {confidence:.3f}, expected: {scenario['expected']})")
        
        if is_correct:
            correct += 1
    
    accuracy = correct / total
    print(f"\n🎯 SVM Scenario Accuracy: {correct}/{total} = {accuracy:.1%}")
    
    return accuracy

def save_svm_models(svm_model, scaler, label_encoder, feature_names, grid_search, scenario_accuracy):
    """Save SVM models theo thiết kế ban đầu"""
    print("\n💾 Saving SVM models...")
    
    os.makedirs('models', exist_ok=True)
    
    # Save models - theo spec joblib format
    joblib.dump(svm_model, 'models/svm_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(label_encoder, 'models/label_encoder.joblib')
    joblib.dump(feature_names, 'models/feature_names.joblib')
    joblib.dump(grid_search, 'models/svm_grid_search.joblib')
    
    # Training info theo thiết kế OpenResty
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'SVM (Support Vector Machine)',
        'kernel': svm_model.kernel,
        'C': svm_model.C,
        'gamma': svm_model.gamma,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'scenario_accuracy': scenario_accuracy,
        'feature_names': feature_names,
        'label_mapping': dict(zip(label_encoder.classes_, 
                                label_encoder.transform(label_encoder.classes_))),
        'design_purpose': 'Phương án 2: Tích hợp mô hình SVM đã huấn luyện sẵn trong OpenResty',
        'advantages': [
            'Lighter model (ít memory hơn)',
            'Faster prediction (quan trọng cho OpenResty)',
            'Phù hợp với real-time classification',
            'Đúng thiết kế ban đầu'
        ]
    }
    
    joblib.dump(training_info, 'models/training_info.joblib')
    
    print("✅ SVM models saved (theo thiết kế ban đầu):")
    for file in os.listdir('models'):
        print(f"  📄 {file}")
    
    return True

def main():
    """Main SVM-only training pipeline"""
    print("🚀 FORCE SVM ONLY TRAINING")
    print("=" * 60)
    print("🎯 Mục đích: Đúng với thiết kế ban đầu - Phương án 2")
    print("📊 Model: SVM (Support Vector Machine) ONLY")
    print("⚡ Tối ưu: Lighter, Faster, Real-time cho OpenResty")
    print()
    
    start_time = datetime.now()
    
    try:
        # Load real datasets
        df = load_real_datasets()
        if df is None:
            return False
        
        # SVM class assignment
        balanced_df = svm_class_assignment(df)
        if balanced_df is None:
            return False
        
        # SVM feature engineering
        X, y, feature_names = svm_feature_engineering(balanced_df)
        
        # SVM ONLY training
        svm_model, scaler, label_encoder, grid_search = svm_only_training(X, y, feature_names)
        
        # Test SVM scenarios
        scenario_accuracy = test_svm_scenarios(svm_model, scaler, label_encoder)
        
        # Save SVM models
        save_svm_models(svm_model, scaler, label_encoder, feature_names, grid_search, scenario_accuracy)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "=" * 60)
        print("🎉 SVM ONLY Training Completed!")
        print("=" * 60)
        print(f"✅ Model: SVM ({svm_model.kernel} kernel)")
        print(f"✅ Parameters: C={svm_model.C}, gamma={svm_model.gamma}")
        print(f"✅ Scenario Accuracy: {scenario_accuracy:.1%}")
        print(f"⏱️  Training time: {duration:.1f} minutes")
        print(f"🎯 Design: Đúng với Phương án 2 - OpenResty integration")
        
        if scenario_accuracy >= 0.8:
            print(f"\n🎉 EXCELLENT! SVM ready for OpenResty deployment!")
            print("🚀 Next: Integrate với OpenResty theo thiết kế ban đầu")
        else:
            print(f"\n🔧 SVM cần fine-tuning thêm...")
        
        return True
        
    except Exception as e:
        print(f"❌ SVM Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ SVM ONLY training hoàn thành!")
        print("📋 Ready for OpenResty integration:")
        print("   1. Load svm_model.joblib")
        print("   2. Use scaler.joblib for preprocessing")
        print("   3. Use label_encoder.joblib for output")
    else:
        print("\n❌ SVM training failed!") 