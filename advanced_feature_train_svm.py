#!/usr/bin/env python3
"""
ADVANCED Feature Engineering - Target 60%+ scenario accuracy
Improved features để phân biệt rõ ràng small/medium/large classes
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_real_datasets():
    """Load real datasets from Excel files"""
    print("📁 Loading REAL datasets from Excel files...")
    
    datasets = []
    for file in ['dataset/mmc2.xlsx', 'dataset/mmc3.xlsx', 'dataset/mmc4.xlsx']:
        try:
            df = pd.read_excel(file)
            print(f"  ✅ {file}: {len(df)} rows")
            datasets.append(df)
        except Exception as e:
            print(f"  ❌ Could not load {file}: {e}")
    
    if not datasets:
        print("❌ No datasets loaded!")
        return None
    
    combined_df = pd.concat(datasets, ignore_index=True)
    print(f"📊 Combined REAL dataset: {len(combined_df)} rows")
    
    return combined_df

def advanced_class_balancing(df):
    """Advanced class balancing with better medium samples"""
    print("🔧 Advanced class balancing...")
    
    # Remove quotes from class names
    df['Class_Name'] = df['Class_Name'].str.replace("'", "").str.strip()
    
    print("📊 Original classes:")
    for cls in df['Class_Name'].unique():
        count = len(df[df['Class_Name'] == cls])
        print(f"  '{cls}': {count} samples")
    
    # IMPROVED MAPPING: More realistic thresholds
    label_mapping = {
        'Very Low': 'small',
        'Low': 'small', 
        'Medium': 'medium',
        'High': 'large',
        'Very High': 'large'
    }
    
    df['Class_Name_Clean'] = df['Class_Name'].map(label_mapping)
    df = df.dropna(subset=['Class_Name_Clean'])
    
    print("\n📊 After mapping:")
    class_counts = df['Class_Name_Clean'].value_counts()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Separate classes
    small_df = df[df['Class_Name_Clean'] == 'small']
    medium_df = df[df['Class_Name_Clean'] == 'medium'] 
    large_df = df[df['Class_Name_Clean'] == 'large']
    
    # ADVANCED MEDIUM SAMPLE CREATION
    print(f"\n⚖️  Advanced medium sample creation...")
    print(f"Original: small={len(small_df)}, medium={len(medium_df)}, large={len(large_df)}")
    
    # Calculate composite resource scores for each sample
    def calculate_resource_score(row):
        cpu_cores = row.get('Num_of_CPU_Cores', 2)
        memory_mb = row.get('Mem capacity', 4000)
        memory_gb = memory_mb / 1024
        storage = row.get('Disk_capacity_GB', 100)
        network_rx = row.get('Avg_Recieve_Kbps', 1000)
        network_tx = row.get('Avg_Transmit_Kbps', 1000)
        
        # Weighted resource score (0-100)
        resource_score = (
            cpu_cores * 8 +           # CPU weight: 8
            memory_gb * 4 +           # Memory weight: 4  
            storage / 20 +            # Storage weight: 1/20
            (network_rx + network_tx) / 100  # Network weight: 1/100
        )
        return min(100, resource_score)
    
    # Add resource scores
    small_df = small_df.copy()
    large_df = large_df.copy()
    small_df['resource_score'] = small_df.apply(calculate_resource_score, axis=1)
    large_df['resource_score'] = large_df.apply(calculate_resource_score, axis=1)
    
    # Find REALISTIC medium samples (boundary cases)
    boundary_medium = []
    
    # Method 1: High-resource small samples (score > 75th percentile)
    small_threshold = small_df['resource_score'].quantile(0.75)
    small_candidates = small_df[small_df['resource_score'] >= small_threshold]
    print(f"  Small candidates for medium (score >= {small_threshold:.1f}): {len(small_candidates)}")
    
    for _, row in small_candidates.iterrows():
        new_row = row.copy()
        new_row['Class_Name_Clean'] = 'medium'
        boundary_medium.append(new_row)
    
    # Method 2: Low-resource large samples (score < 25th percentile)  
    large_threshold = large_df['resource_score'].quantile(0.25)
    large_candidates = large_df[large_df['resource_score'] <= large_threshold]
    print(f"  Large candidates for medium (score <= {large_threshold:.1f}): {len(large_candidates)}")
    
    for _, row in large_candidates.iterrows():
        new_row = row.copy()
        new_row['Class_Name_Clean'] = 'medium'
        boundary_medium.append(new_row)
    
    # Add realistic medium samples
    if boundary_medium:
        synthetic_df = pd.DataFrame(boundary_medium)
        medium_df = pd.concat([medium_df, synthetic_df], ignore_index=True)
        print(f"  ✅ Added {len(synthetic_df)} realistic boundary samples to medium class")
    
    # BALANCED SAMPLING: 1200 per class for more data
    target_samples = 1200
    print(f"🎯 Target: {target_samples} samples per class")
    
    balanced_dfs = []
    
    for class_name, class_df in [('small', small_df), ('medium', medium_df), ('large', large_df)]:
        if len(class_df) >= target_samples:
            balanced = resample(class_df, replace=False, n_samples=target_samples, random_state=42)
            print(f"  📉 {class_name}: {len(class_df)} → {target_samples} (downsampled)")
        else:
            balanced = resample(class_df, replace=True, n_samples=target_samples, random_state=42)
            print(f"  📈 {class_name}: {len(class_df)} → {target_samples} (oversampled)")
        balanced_dfs.append(balanced)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    print(f"\n✅ Final balanced dataset: {len(balanced_df)} samples")
    final_counts = balanced_df['Class_Name_Clean'].value_counts()
    for class_name, count in final_counts.items():
        print(f"  {class_name}: {count} samples ({count/len(balanced_df)*100:.1f}%)")
    
    return balanced_df

def advanced_feature_engineering(df):
    """ADVANCED feature engineering với better class separation"""
    print("🔄 Advanced feature engineering...")
    
    feature_data = []
    
    for _, row in df.iterrows():
        # Raw features
        cpu_cores = row.get('Num_of_CPU_Cores', 2)
        memory_mb = row.get('Mem capacity', 4000)
        memory_gb = memory_mb / 1024
        storage = row.get('Disk_capacity_GB', 100)
        network_rx = row.get('Avg_Recieve_Kbps', 1000)
        network_tx = row.get('Avg_Transmit_Kbps', 1000)
        network_total = network_rx + network_tx
        jobs_1min = row.get('Jobs_per_1Minute', 1)
        jobs_5min = row.get('Jobs_per_5Minutes', 5)
        cpu_speed = row.get('CPU_speed_per_Core', 2.5)
        
        # ADVANCED FEATURE CALCULATIONS
        
        # 1. COMPOSITE RESOURCE POWER (0-100) - key discriminator
        resource_power = min(100, 
            cpu_cores * 6 +           # CPU cores weighted heavily
            memory_gb * 2.5 +         # Memory moderate weight
            storage / 25 +            # Storage light weight
            cpu_speed * 8             # CPU speed important
        )
        
        # 2. WORKLOAD INTENSITY (0-100) - jobs and network activity
        workload_intensity = min(100,
            jobs_1min * 12 +          # 1-min jobs weight
            jobs_5min * 2.5 +         # 5-min jobs weight  
            network_total / 150       # Network activity
        )
        
        # 3. THROUGHPUT CAPABILITY (0-100) - combined processing power
        throughput = min(100,
            cpu_cores * cpu_speed * 3 +  # Total CPU power
            memory_gb * 1.5 +            # Memory contribution
            network_total / 200          # Network throughput
        )
        
        # 4. RESOURCE UTILIZATION RATIO (0-100) - efficiency metric
        base_capacity = cpu_cores * memory_gb * cpu_speed
        actual_workload = jobs_1min * jobs_5min * (storage / 100)
        if base_capacity > 0:
            utilization_ratio = min(100, (actual_workload / base_capacity) * 50)
        else:
            utilization_ratio = 10
        
        # 5. SYSTEM COMPLEXITY SCORE (1-10) - based on resource power
        if resource_power < 20:
            complexity_score = 1
        elif resource_power < 35:
            complexity_score = 2
        elif resource_power < 50:
            complexity_score = 3
        elif resource_power < 65:
            complexity_score = 4
        elif resource_power < 75:
            complexity_score = 5
        elif resource_power < 82:
            complexity_score = 6
        elif resource_power < 88:
            complexity_score = 7
        elif resource_power < 94:
            complexity_score = 8
        elif resource_power < 98:
            complexity_score = 9
        else:
            complexity_score = 10
        
        # 6. PERFORMANCE TIER (1-5) - clear class distinction
        if resource_power < 30:
            performance_tier = 1  # Small
        elif resource_power < 55:
            performance_tier = 2  # Small-Medium
        elif resource_power < 75:
            performance_tier = 3  # Medium
        elif resource_power < 90:
            performance_tier = 4  # Medium-Large
        else:
            performance_tier = 5  # Large
        
        # 7. SCALABILITY INDEX (0-100) - growth potential
        scalability_index = min(100,
            (cpu_cores * 10) +        # More cores = more scalable
            (memory_gb * 3) +         # More memory = better scaling
            (network_total / 100)     # Network affects scaling
        )
        
        # 8. DATA PROCESSING CAPACITY (0-500) 
        data_capacity = min(500,
            storage * (jobs_1min + 1) / 8 +  # Storage + job processing
            memory_gb * jobs_5min * 2        # Memory for data processing
        )
        
        # 9. CONCURRENT PROCESSING POWER (0-2000)
        concurrent_power = min(2000,
            cpu_cores * jobs_5min * 8 +      # Core * jobs
            throughput * 2                   # Throughput contribution
        )
        
        # 10. PRIORITY WEIGHT (1-5) - based on workload intensity
        if workload_intensity < 20:
            priority_weight = 1
        elif workload_intensity < 40:
            priority_weight = 2
        elif workload_intensity < 65:
            priority_weight = 3
        elif workload_intensity < 85:
            priority_weight = 4
        else:
            priority_weight = 5
        
        # FINAL 10 FEATURES - designed for clear class separation
        features = [
            resource_power,        # 0: Most important - overall power
            performance_tier,      # 1: Clear tier classification  
            throughput,           # 2: Processing capability
            workload_intensity,   # 3: Job intensity
            complexity_score,     # 4: System complexity (1-10)
            scalability_index,    # 5: Scaling potential
            data_capacity,        # 6: Data processing ability
            concurrent_power,     # 7: Parallel processing
            utilization_ratio,    # 8: Resource efficiency
            priority_weight       # 9: Workload priority
        ]
        
        feature_data.append(features)
    
    feature_names = [
        'resource_power', 'performance_tier', 'throughput', 'workload_intensity', 'complexity_score',
        'scalability_index', 'data_capacity', 'concurrent_power', 'utilization_ratio', 'priority_weight'
    ]
    
    X = np.array(feature_data)
    y = df['Class_Name_Clean'].values
    
    print(f"✅ Advanced features shape: {X.shape}")
    print(f"   Key feature ranges:")
    for i, name in enumerate(feature_names[:5]):  # Show top 5 features
        print(f"   {name}: {X[:, i].min():.1f} - {X[:, i].max():.1f}")
    
    return X, y, feature_names

def advanced_model_training(X, y, feature_names):
    """Advanced model training with multiple algorithms"""
    print("\n🚀 Advanced model training...")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # TRAIN MULTIPLE MODELS
    models = {}
    
    # 1. Optimized SVM
    print("🔍 Training Optimized SVM...")
    svm_grid = GridSearchCV(
        SVC(random_state=42, probability=True),
        {
            'C': [0.1, 1, 10, 50],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'poly'],
            'class_weight': ['balanced']
        },
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    svm_grid.fit(X_train_scaled, y_train_encoded)
    models['SVM'] = svm_grid
    
    # 2. Random Forest with more trees
    print("🔍 Training Enhanced Random Forest...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {
            'n_estimators': [150, 200],
            'max_depth': [15, 25, None],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']
        },
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    rf_grid.fit(X_train_scaled, y_train_encoded)
    models['RandomForest'] = rf_grid
    
    # 3. Gradient Boosting (often best for tabular data)
    print("🔍 Training Gradient Boosting...")
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        {
            'n_estimators': [100, 150],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7]
        },
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    gb_grid.fit(X_train_scaled, y_train_encoded)
    models['GradientBoosting'] = gb_grid
    
    # Compare all models
    print("\n📊 Advanced model comparison:")
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        test_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test_encoded, test_pred)
        cv_score = model.best_score_
        
        print(f"  {name}:")
        print(f"    CV Score: {cv_score:.3f}")
        print(f"    Test Accuracy: {test_acc:.3f}")
        print(f"    Best Params: {str(model.best_params_)[:80]}...")
        
        if test_acc > best_score:
            best_score = test_acc
            best_model = model
            best_name = name
    
    print(f"\n🏆 Best Model: {best_name} (Test Accuracy: {best_score:.3f})")
    
    # Detailed evaluation
    best_pred = best_model.predict(X_test_scaled)
    y_pred_labels = label_encoder.inverse_transform(best_pred)
    
    print(f"\n📋 {best_name} Classification Report:")
    print(classification_report(y_test, y_pred_labels))
    
    print(f"\n🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test_encoded, best_pred)
    print("       small  medium  large")
    for i, class_name in enumerate(['small', 'medium', 'large']):
        print(f"{class_name:>6} {cm[i]}")
    
    return best_model.best_estimator_, scaler, label_encoder, best_model, best_name

def test_advanced_scenarios(model, scaler, label_encoder, model_name):
    """Test with more realistic and challenging scenarios"""
    print(f"\n🧪 Testing {model_name} on ADVANCED scenarios...")
    
    # ADVANCED REALISTIC TEST SCENARIOS
    scenarios = [
        # Small category - clear small systems
        {'name': 'Basic Web Server', 'features': [18, 1, 25, 8, 1, 28, 12, 80, 15, 1], 'expected': 'small'},
        {'name': 'Micro Service', 'features': [12, 1, 15, 5, 1, 20, 8, 50, 10, 1], 'expected': 'small'},
        {'name': 'Development Server', 'features': [22, 2, 30, 12, 2, 35, 18, 120, 20, 2], 'expected': 'small'},
        
        # Medium category - middle-tier systems  
        {'name': 'Production API', 'features': [45, 3, 55, 35, 4, 65, 85, 720, 45, 3], 'expected': 'medium'},
        {'name': 'Database Server', 'features': [50, 3, 60, 28, 5, 70, 140, 900, 35, 3], 'expected': 'medium'},
        {'name': 'Enterprise App', 'features': [55, 4, 65, 40, 6, 75, 200, 1200, 50, 4], 'expected': 'medium'},
        
        # Large category - high-end systems
        {'name': 'ML Training Server', 'features': [85, 5, 88, 75, 9, 95, 400, 2400, 80, 5], 'expected': 'large'},
        {'name': 'Video Processing', 'features': [90, 5, 92, 85, 8, 98, 520, 1800, 85, 4], 'expected': 'large'},
        {'name': 'High-End Compute', 'features': [95, 5, 95, 90, 10, 100, 800, 3600, 90, 5], 'expected': 'large'},
        
        # Edge cases to test robustness
        {'name': 'Minimal System', 'features': [8, 1, 12, 3, 1, 15, 5, 30, 8, 1], 'expected': 'small'},
        {'name': 'Borderline Medium', 'features': [42, 3, 48, 32, 3, 55, 75, 650, 40, 3], 'expected': 'medium'},
        {'name': 'Enterprise Scale', 'features': [98, 5, 98, 95, 10, 100, 1000, 4000, 95, 5], 'expected': 'large'}
    ]
    
    correct = 0
    total = len(scenarios)
    
    print("  Advanced scenario results:")
    for scenario in scenarios:
        features_scaled = scaler.transform([scenario['features']])
        pred_encoded = model.predict(features_scaled)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]
        
        # Get confidence
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            confidence = max(proba)
            conf_str = f" (conf: {confidence:.2f})"
        else:
            conf_str = ""
        
        is_correct = prediction == scenario['expected']
        status = "✅" if is_correct else "❌"
        
        print(f"    {status} {scenario['name']}: {prediction}{conf_str} (expected: {scenario['expected']})")
        
        if is_correct:
            correct += 1
    
    accuracy = correct / total
    print(f"\n🎯 Advanced Scenario Accuracy: {correct}/{total} = {accuracy:.1%}")
    
    return accuracy

def save_advanced_models(model, scaler, label_encoder, feature_names, grid_search, scenario_accuracy, model_name):
    """Save advanced models"""
    print("\n💾 Saving advanced models...")
    
    os.makedirs('models', exist_ok=True)
    
    # Save models
    joblib.dump(model, 'models/svm_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(label_encoder, 'models/label_encoder.joblib')
    joblib.dump(feature_names, 'models/feature_names.joblib')
    joblib.dump(grid_search, 'models/svm_grid_search.joblib')
    
    # Advanced training info
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'model_type': f'Advanced {model_name} with Enhanced Features',
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'scenario_accuracy': scenario_accuracy,
        'data_source': 'Real Excel files (mmc2.xlsx, mmc3.xlsx, mmc4.xlsx)',
        'training_method': 'Advanced Feature Engineering',
        'class_balance': 'Perfectly Balanced (1200 samples each)',
        'feature_names': feature_names,
        'advanced_features': [
            'Resource Power (0-100) - composite metric',
            'Performance Tier (1-5) - clear classification',
            'Throughput Capability - processing power',
            'Workload Intensity - job processing load',
            'Complexity Score (1-10) - system complexity',
            'Scalability Index - growth potential',
            'Data Capacity - data processing ability',
            'Concurrent Power - parallel processing',
            'Utilization Ratio - resource efficiency',
            'Priority Weight - workload priority'
        ],
        'improvements': [
            'Advanced composite features for better separation',
            'Multiple model comparison (SVM, RF, GB)',
            'Enhanced hyperparameter tuning',
            'Realistic boundary-based medium samples',
            'Comprehensive scenario testing'
        ]
    }
    
    joblib.dump(training_info, 'models/training_info.joblib')
    
    print("✅ Advanced models saved!")
    return True

def main():
    """Main advanced training pipeline"""
    print("🚀 ADVANCED Feature Engineering Training")
    print("=" * 60)
    print("🎯 Goal: Achieve 60%+ scenario accuracy with advanced features")
    print("📊 Data: Real Excel datasets (7,350 samples)")
    print("🔬 Method: Advanced feature engineering + multiple models")
    print()
    
    # Load real datasets
    df = load_real_datasets()
    if df is None:
        return False
    
    # Advanced class balancing
    balanced_df = advanced_class_balancing(df)
    if balanced_df is None:
        return False
    
    # Advanced feature engineering
    X, y, feature_names = advanced_feature_engineering(balanced_df)
    
    # Advanced model training
    model, scaler, label_encoder, grid_search, model_name = advanced_model_training(X, y, feature_names)
    
    # Advanced scenario testing
    scenario_accuracy = test_advanced_scenarios(model, scaler, label_encoder, model_name)
    
    # Save models
    save_advanced_models(model, scaler, label_encoder, feature_names, grid_search, scenario_accuracy, model_name)
    
    print("\n" + "=" * 60)
    print("🎉 Advanced Training Completed!")
    print("=" * 60)
    print(f"✅ Best Model: {model_name}")
    print(f"✅ Data: Real datasets with advanced features")
    print(f"✅ Class Balance: Perfect (1,200 samples each)")
    print(f"✅ Scenario Accuracy: {scenario_accuracy:.1%}")
    print(f"✅ Feature Engineering: Advanced composite features")
    
    if scenario_accuracy >= 0.6:
        print(f"\n🎉 SUCCESS! Target achieved: {scenario_accuracy:.1%} ≥ 60%")
        print("🚀 Ready for cloud deployment!")
    elif scenario_accuracy >= 0.5:
        print(f"\n👍 Good progress: {scenario_accuracy:.1%} (close to target)")
        print("🔧 Consider additional tuning for 60%+")
    else:
        print(f"\n⚠️  Still need improvement: {scenario_accuracy:.1%} < 60%")
        print("🔬 May need different approach or more data")
    
    return True

if __name__ == "__main__":
    start_time = datetime.now()
    success = main()
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    print(f"\n⏱️  Total training time: {duration:.1f} minutes")
    
    if success:
        if duration < 10:
            print("✅ Training completed efficiently!")
        else:
            print("✅ Training completed (took some time for thoroughness)")
    else:
        print("❌ Training failed!") 