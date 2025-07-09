#!/usr/bin/env python3
"""
Retrained Balanced SVM Model for OpenResty Integration
WITH REAL-TIME PROGRESS TRACKING
Fixes class imbalance using SMOTE + Class Weights
Accuracy target: >75% (improved from 64%)
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import time
import sys

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from collections import Counter

# PROGRESS TRACKING CLASS
class ProgressTracker:
    def __init__(self, total_fits):
        self.total_fits = total_fits
        self.current_fit = 0
        self.start_time = time.time()
        
    def update(self, cv_results=None):
        self.current_fit += 1
        elapsed = time.time() - self.start_time
        progress = (self.current_fit / self.total_fits) * 100
        
        # Estimate remaining time
        if self.current_fit > 0:
            avg_time_per_fit = elapsed / self.current_fit
            remaining_fits = self.total_fits - self.current_fit
            eta_seconds = remaining_fits * avg_time_per_fit
            eta_minutes = eta_seconds / 60
            
            print(f"🔄 Progress: {self.current_fit}/{self.total_fits} fits ({progress:.1f}%) - "
                  f"Elapsed: {elapsed/60:.1f}min - ETA: {eta_minutes:.1f}min")
        else:
            print(f"🔄 Progress: {self.current_fit}/{self.total_fits} fits ({progress:.1f}%)")

def load_datasets_from_local():
    """Load datasets from local dataset directory"""
    print("📂 Loading datasets from local machine...")
    
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        return None
    
    # List all Excel files
    excel_files = [f for f in os.listdir(dataset_dir) if f.endswith('.xlsx')]
    if not excel_files:
        print(f"❌ No Excel files found in '{dataset_dir}'!")
        return None
    
    print(f"✅ Found {len(excel_files)} Excel files:")
    
    # Load and combine all datasets
    dataframes = []
    total_rows = 0
    
    for i, file in enumerate(excel_files):
        file_path = os.path.join(dataset_dir, file)
        try:
            df = pd.read_excel(file_path)
            dataframes.append(df)
            print(f"  {i+1}. {file}: {len(df)} rows, {len(df.columns)} columns")
            total_rows += len(df)
        except Exception as e:
            print(f"⚠️  Error loading {file}: {e}")
    
    if not dataframes:
        print("❌ No valid datasets loaded!")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Standardize column names
    column_mapping = {
        'Class': 'Class_Name',
        'class': 'Class_Name', 
        'CLASS': 'Class_Name',
        'Label': 'Class_Name'
    }
    
    combined_df.rename(columns=column_mapping, inplace=True)
    
    print(f"📊 Combined data summary: {len(combined_df)} samples, {len(combined_df.columns)} features")
    print(f"✅ Total samples available for training: {total_rows:,}")
    
    return combined_df

def prepare_balanced_svm_data(df):
    """Prepare data for balanced SVM training with SMOTE"""
    print("\n🔧 Preparing Balanced SVM Data...")
    
    # Map actual column names to standardized names
    feature_mapping = {
        'Jobs_per_ 1Minute': 'jobs_1min',
        'Jobs_per_ 5 Minutes': 'jobs_5min',
        'Jobs_per_ 15Minutes': 'jobs_15min',
        'Mem capacity': 'memory_mb',
        'Disk_capacity_GB': 'disk_capacity',
        'Num_of_CPU_Cores': 'cpu_cores',
        'CPU_speed_per_Core': 'cpu_speed',
        'Avg_Recieve_Kbps': 'network_receive',
        'Avg_Transmit_Kbps': 'network_transmit',
        # Legacy mappings for backward compatibility
        'CPU_cores': 'cpu_cores',
        'Memory_MB': 'memory_mb', 
        'Jobs_1min': 'jobs_1min',
        'Jobs_5min': 'jobs_5min',
        'CPU_Speed_GHz': 'cpu_speed',
        'Network_In_Kbps': 'network_receive',
        'Network_Out_Kbps': 'network_transmit'
    }
    
    df_clean = df.rename(columns=feature_mapping).copy()
    
    # Clean class names
    if 'Class_Name' in df_clean.columns:
        df_clean['Class_Name'] = df_clean['Class_Name'].astype(str).str.strip().str.replace("'", "")
        print(f"📊 Original class distribution:")
        print(df_clean['Class_Name'].value_counts())
        
        # Map to standard makespan classes
        class_mapping = {
            'Very Low': 'small',
            'Low': 'small', 
            'Medium': 'medium',
            'High': 'large',
            'Very High': 'large'
        }
        
        df_clean['makespan_class'] = df_clean['Class_Name'].map(class_mapping)
        df_clean = df_clean.dropna(subset=['makespan_class'])
        
        print(f"\n📊 Mapped class distribution:")
        print(df_clean['makespan_class'].value_counts())
    else:
        print("❌ No Class_Name column found!")
        return None, None, None
    
    # Feature engineering - create 10 features
    if 'memory_mb' in df_clean.columns:
        df_clean['memory_gb'] = df_clean['memory_mb'] / 1024
    
    # Fill missing values and engineer features
    numeric_cols = ['jobs_1min', 'jobs_5min', 'jobs_15min', 'memory_gb', 'cpu_cores', 'cpu_speed', 
                   'network_receive', 'network_transmit', 'disk_capacity']
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    # Create derived features
    df_clean['network_total'] = df_clean.get('network_receive', 0) + df_clean.get('network_transmit', 0)
    df_clean['resource_density'] = (df_clean.get('memory_gb', 1) * df_clean.get('cpu_cores', 1)) / (df_clean.get('cpu_speed', 1) + 0.1)
    df_clean['workload_intensity'] = (df_clean.get('jobs_1min', 1) * df_clean.get('jobs_5min', 1)) / (df_clean.get('cpu_cores', 1) + 0.1)
    
    # Select final features (exactly 10) - use available columns
    available_features = []
    potential_features = ['jobs_1min', 'jobs_5min', 'jobs_15min', 'memory_gb', 'cpu_cores', 'cpu_speed', 
                         'network_receive', 'network_transmit', 'disk_capacity', 'network_total', 'resource_density', 'workload_intensity']
    
    for feature in potential_features:
        if feature in df_clean.columns:
            available_features.append(feature)
        if len(available_features) >= 10:
            break
    
    feature_names = available_features[:10]  # Take first 10 available features
    
    # Create feature matrix
    X = df_clean[feature_names].copy()
    y = df_clean['makespan_class'].copy()
    
    # Handle any remaining missing values
    X = X.fillna(0)
    
    print(f"✅ Features prepared: {X.shape}")
    print(f"✅ Feature names: {feature_names}")
    
    print(f"\n📊 Final class distribution:")
    class_counts = y.value_counts()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    return X, y, feature_names

def train_balanced_svm_model(X, y, feature_names):
    """Train balanced SVM with progress tracking"""
    print("\n🚀 Training Balanced SVM Model...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"🎯 Classes found: {label_encoder.classes_}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training: {len(X_train)}, Test: {len(X_test)}")
    
    # Encode training labels
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print(f"Label mapping: {label_mapping}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE for balancing
    print(f"\n🎯 Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_encoded)
    
    print(f"After SMOTE: {len(X_train_balanced)} samples")
    print(f"Balanced distribution: {Counter(y_train_balanced)}")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_balanced), y=y_train_balanced)
    class_weight_dict = dict(zip(np.unique(y_train_balanced), class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # GridSearchCV with progress tracking
    print(f"\n🎯 SVM Training with Balanced Classes...")
    
    # Simplified grid for faster training
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    
    total_combinations = len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])
    cv_folds = 3
    total_fits = total_combinations * cv_folds
    
    print(f"🔄 GridSearchCV: {total_combinations} combinations × {cv_folds} folds = {total_fits} fits")
    print(f"⏱️ Estimated time: 3-5 minutes per fit = {total_fits*4/60:.1f}-{total_fits*8/60:.1f} hours")
    print(f"🚀 Starting GridSearchCV...")
    
    # Initialize progress tracker
    tracker = ProgressTracker(total_fits)
    
    # Custom scorer with progress tracking
    def progress_scorer(estimator, X, y):
        tracker.update()
        from sklearn.metrics import f1_score
        y_pred = estimator.predict(X)
        return f1_score(y, y_pred, average='macro')
    
    svm_grid = GridSearchCV(
        SVC(random_state=42, probability=True, class_weight=class_weight_dict),
        param_grid,
        cv=cv_folds,
        scoring=progress_scorer,  # Custom scorer with progress
        n_jobs=1,  # Sequential to track progress properly
        verbose=3,   # Maximum verbosity
        return_train_score=True
    )
    
    print(f"⏰ Training started at: {datetime.now().strftime('%H:%M:%S')}")
    start_time = time.time()
    
    # Fit with progress updates
    svm_grid.fit(X_train_balanced, y_train_balanced)
    
    training_time = time.time() - start_time
    print(f"⏰ Training completed at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"⏱️ Total training time: {training_time/60:.1f} minutes")
    
    best_svm = svm_grid.best_estimator_
    
    # Evaluate on original test set
    y_pred = best_svm.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test_encoded, y_pred)
    
    print(f"\n📊 Balanced SVM Results:")
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

def test_svm_scenarios(svm_model, svm_scaler, label_encoder, feature_names):
    """Test SVM với scenarios thực tế"""
    print("\n🧪 Testing SVM Scenarios...")
    
    test_scenarios = [
        {
            'name': 'Very Light Load',
            'features': [1, 5, 0.5, 2, 2.0, 100, 50, 150, 0.24, 0.48]
        },
        {
            'name': 'Medium Load', 
            'features': [15, 60, 4.0, 4, 2.5, 500, 300, 800, 0.98, 3.66]
        },
        {
            'name': 'Heavy Load',
            'features': [50, 200, 16.0, 8, 3.0, 2000, 1500, 3500, 1.95, 6.10]
        }
    ]
    
    for scenario in test_scenarios:
        features = [scenario['features']]
        features_scaled = svm_scaler.transform(features)
        
        pred_numeric = svm_model.predict(features_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_numeric])[0]
        
        # Get prediction probabilities
        proba = svm_model.predict_proba(features_scaled)[0]
        confidence = np.max(proba)
        
        print(f"  {scenario['name']}: {pred_label} (confidence: {confidence:.3f})")

def save_balanced_svm_model(svm_model, scaler, label_encoder, grid_search, feature_names, accuracy):
    """Save the balanced SVM model"""
    print("\n💾 Saving Balanced SVM Model...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save all components
    joblib.dump(svm_model, 'models/svm_model.joblib')
    joblib.dump(scaler, 'models/svm_scaler.joblib')
    joblib.dump(label_encoder, 'models/svm_label_encoder.joblib')
    joblib.dump(feature_names, 'models/svm_feature_names.joblib')
    joblib.dump(grid_search, 'models/svm_grid_search.joblib')
    
    # Save training info
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'model': {
            'type': 'Balanced SVM (Support Vector Machine)',
            'kernel': svm_model.kernel,
            'C': svm_model.C,
            'gamma': svm_model.gamma,
            'test_accuracy': accuracy,
            'feature_names': feature_names,
            'classes': list(label_encoder.classes_),
            'label_mapping': dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))),
            'balancing': 'SMOTE + Class Weights'
        },
        'deployment_plan': 'Balanced SVM for OpenResty',
        'usage': 'POST /predict/makespan - Classification with proper class balance'
    }
    
    joblib.dump(training_info, 'models/training_info.joblib')
    
    print("✅ Balanced SVM model saved successfully!")

def main():
    """Main balanced SVM training pipeline"""
    print("🚀 RETRAIN: BALANCED SVM MODEL")
    print("=" * 60)
    print("🎯 Fix class imbalance with SMOTE + Class Weights")
    print("📊 Proper handling of Very Low/Low/Medium/High classes")
    print()
    
    start_time = datetime.now()
    
    try:
        # Load datasets
        df = load_datasets_from_local()
        if df is None:
            return False
        
        # Prepare balanced data
        X, y, feature_names = prepare_balanced_svm_data(df)
        
        # Train balanced SVM
        svm_model, scaler, label_encoder, grid_search, accuracy = train_balanced_svm_model(X, y, feature_names)
        
        if svm_model is None:
            print("❌ Balanced SVM training failed!")
            return False
        
        # Test scenarios
        test_svm_scenarios(svm_model, scaler, label_encoder, feature_names)
        
        # Save model
        save_balanced_svm_model(svm_model, scaler, label_encoder, grid_search, feature_names, accuracy)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "=" * 60)
        print("🎉 BALANCED SVM TRAINING COMPLETED!")
        print("=" * 60)
        print(f"✅ Model: {svm_model.kernel} kernel with class balancing")
        print(f"📊 Test Accuracy: {accuracy:.1%}")
        print(f"🎯 Classes: {list(label_encoder.classes_)}")
        print(f"⏱️  Total time: {duration:.1f} minutes")
        print(f"🚀 Ready for deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 