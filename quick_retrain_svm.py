#!/usr/bin/env python3
"""
QUICK RETRAIN: Fast Balanced SVM
Optimized for speed - smaller grid search, sampling
"""
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from collections import Counter

def load_and_sample_datasets():
    """Load datasets và sample để training nhanh hơn"""
    print("📂 Loading and sampling datasets...")
    
    dataset_dir = 'dataset'
    excel_files = [f for f in os.listdir(dataset_dir) if f.endswith('.xlsx')]
    
    all_data = []
    for file in excel_files[:3]:  # Chỉ lấy 3 files đầu cho nhanh
        file_path = os.path.join(dataset_dir, file)
        try:
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip().str.replace(' ', '_')
            # Sample 500 rows từ mỗi file
            if len(df) > 500:
                df = df.sample(n=500, random_state=42)
            all_data.append(df)
            print(f"  ✅ Sampled {len(df)} rows from {file}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"📊 Total samples: {len(combined_df)}")
    
    return combined_df

def prepare_quick_svm_data(df):
    """Prepare data nhanh"""
    print("\n🎯 Preparing Quick SVM Data...")
    
    # Map columns
    column_mapping = {
        'Jobs_per__1Minute': 'jobs_1min',
        'Jobs_per__5_Minutes': 'jobs_5min', 
        'Mem_capacity': 'memory_mb',
        'Num_of_CPU_Cores': 'cpu_cores',
        'CPU_speed_per_Core': 'cpu_speed',
        'Avg_Recieve_Kbps': 'network_receive',
        'Avg_Transmit_Kbps': 'network_transmit',
        'Class_Name': 'class_name'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Clean class names
    df['class_name'] = df['class_name'].astype(str).str.strip().str.replace("'", "")
    
    print("📊 Original class distribution:")
    print(df['class_name'].value_counts())
    
    # Map to balanced classes
    class_mapping = {
        'Very Low': 'small',
        'Low': 'small',
        'Medium': 'medium', 
        'High': 'large',
        'Very High': 'large'
    }
    
    df['makespan_class'] = df['class_name'].map(class_mapping)
    df['makespan_class'] = df['makespan_class'].fillna('small')
    
    print("\n📊 Mapped class distribution:")
    print(df['makespan_class'].value_counts())
    
    # Features - simplified
    feature_columns = ['jobs_1min', 'jobs_5min', 'memory_mb', 'cpu_cores', 'cpu_speed', 
                      'network_receive', 'network_transmit']
    
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convert and create features
    df['memory_gb'] = df['memory_mb'] / 1024
    df['network_total'] = df['network_receive'] + df['network_transmit']
    df['resource_density'] = df['memory_gb'] / (df['cpu_cores'] + 0.1)
    df['workload_intensity'] = df['jobs_1min'] / (df['cpu_cores'] + 0.1)
    
    svm_features = [
        'jobs_1min', 'jobs_5min', 'memory_gb', 'cpu_cores', 'cpu_speed',
        'network_receive', 'network_transmit', 'network_total', 
        'resource_density', 'workload_intensity'
    ]
    
    df = df.dropna(subset=svm_features + ['makespan_class'])
    
    X = df[svm_features].values
    y = df['makespan_class'].values
    
    print(f"\n✅ Features: {X.shape}")
    print(f"✅ Final distribution:")
    for class_name, count in pd.Series(y).value_counts().items():
        print(f"  {class_name}: {count}")
    
    return X, y, svm_features

def train_quick_svm(X, y, feature_names):
    """Train SVM nhanh với simplified grid"""
    print("\n🚀 Quick SVM Training...")
    
    unique_classes = np.unique(y)
    print(f"🎯 Classes: {unique_classes}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # SMOTE - only if we have enough samples
    if min(Counter(y_train_encoded).values()) > 1:
        print("🎯 Applying SMOTE...")
        smote = SMOTE(random_state=42, k_neighbors=min(3, min(Counter(y_train_encoded).values()) - 1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_encoded)
        print(f"After SMOTE: {Counter(y_train_balanced)}")
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train_encoded
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_balanced), y=y_train_balanced)
    class_weight_dict = dict(zip(np.unique(y_train_balanced), class_weights))
    
    # Simple SVM - no grid search for speed
    print("🎯 Training SVM with balanced classes...")
    svm_model = SVC(
        C=10,                    # Good default
        gamma='scale',           # Good default  
        kernel='rbf',           # Proven to work well
        random_state=42, 
        probability=True,
        class_weight=class_weight_dict
    )
    
    svm_model.fit(X_train_balanced, y_train_balanced)
    
    # Test
    y_pred = svm_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    print(f"\n📊 Quick SVM Results:")
    print(f"Test accuracy: {accuracy:.3f}")
    
    # Test scenarios
    print("\n🧪 Testing scenarios:")
    test_scenarios = [
        ([1, 5, 0.5, 2, 2.0, 100, 50, 150, 0.24, 0.48], "Light"),
        ([15, 60, 4.0, 4, 2.5, 500, 300, 800, 0.98, 3.66], "Medium"),
        ([50, 200, 16.0, 8, 3.0, 2000, 1500, 3500, 1.95, 6.10], "Heavy")
    ]
    
    for features, name in test_scenarios:
        features_scaled = scaler.transform([features])
        pred_numeric = svm_model.predict(features_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_numeric])[0]
        proba = svm_model.predict_proba(features_scaled)[0]
        confidence = np.max(proba)
        print(f"  {name}: {pred_label} (confidence: {confidence:.3f})")
    
    return svm_model, scaler, label_encoder, accuracy

def save_quick_model(svm_model, scaler, label_encoder, feature_names, accuracy):
    """Save model"""
    print("\n💾 Saving Quick Model...")
    
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(svm_model, 'models/svm_model.joblib')
    joblib.dump(scaler, 'models/svm_scaler.joblib') 
    joblib.dump(label_encoder, 'models/svm_label_encoder.joblib')
    joblib.dump(feature_names, 'models/svm_feature_names.joblib')
    
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'model': {
            'type': 'Quick Balanced SVM',
            'kernel': svm_model.kernel,
            'C': svm_model.C,
            'gamma': svm_model.gamma,
            'test_accuracy': accuracy,
            'feature_names': feature_names,
            'classes': list(label_encoder.classes_),
            'balancing': 'SMOTE + Class Weights (Quick)'
        }
    }
    
    joblib.dump(training_info, 'models/training_info.joblib')
    print("✅ Quick model saved!")

def main():
    print("🚀 QUICK RETRAIN - BALANCED SVM")
    print("=" * 50)
    print("⚡ Optimized for speed")
    
    start_time = datetime.now()
    
    try:
        # Load sampled data
        df = load_and_sample_datasets()
        
        # Prepare data
        X, y, feature_names = prepare_quick_svm_data(df)
        
        # Train
        svm_model, scaler, label_encoder, accuracy = train_quick_svm(X, y, feature_names)
        
        # Save
        save_quick_model(svm_model, scaler, label_encoder, feature_names, accuracy)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "=" * 50)
        print("🎉 QUICK TRAINING COMPLETED!")
        print(f"✅ Accuracy: {accuracy:.1%}")
        print(f"⏱️  Time: {duration:.1f} seconds")
        print("🚀 Ready to test!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 