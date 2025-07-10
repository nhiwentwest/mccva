#!/usr/bin/env python3
"""
Retrained Balanced SVM Model for OpenResty Integration
REAL-WORLD VERSION 2.1 - Handles natural class imbalance while ensuring model effectiveness

This version preserves the natural class distribution found in real VM workloads:
- Very Low: ~39.5% (most common)
- High: ~43.1% (second most common)  
- Low: ~6.6% (less common)
- Medium: ~0.4% (rare)
- Very High: ~0.05% (very rare)

Imbalance handling strategy:
- Preserves real-world distribution (reflects actual usage patterns)
- Ensures minimum viable samples for rare classes (50+ samples)
- Uses class weights + selective SMOTE for model balance
- Optimized for balanced accuracy (better metric for imbalanced data)
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import time
import sys
import psutil
import gc
import signal
import atexit
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from collections import Counter

# Global variables for cleanup
temp_files = []

def cleanup_resources():
    """Clean up resources on exit"""
    print("\nCleaning up resources...")
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass
    print("  Cleanup completed")

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print(f"\nSignal {signum} received - shutting down...")
    cleanup_resources()
    sys.exit(0)

# Register cleanup
atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class SimpleProgressTracker:
    """Simple progress tracker that only reports on actual fit completion"""
    def __init__(self, total_fits, description="Training"):
        self.total_fits = total_fits
        self.current_fit = 0
        self.start_time = time.time()
        self.description = description
        self.last_report_time = 0
        
    def report_progress(self, fit_number):
        """Report progress for completed fit"""
        current_time = time.time()
        
        # Only report if significant time passed or new fit completed
        if (fit_number > self.current_fit) or (current_time - self.last_report_time > 120):
            self.current_fit = fit_number
            self.last_report_time = current_time
            
            elapsed = current_time - self.start_time
            progress = (fit_number / self.total_fits) * 100
            memory_usage = psutil.virtual_memory().percent
            
            if fit_number < self.total_fits:
                # Calculate ETA based on completed fits
                if fit_number > 0:
                    avg_time_per_fit = elapsed / fit_number
                    remaining_fits = self.total_fits - fit_number
                    eta_minutes = (remaining_fits * avg_time_per_fit) / 60
                    
                    print(f"[{self.description}] Completed: {fit_number}/{self.total_fits} fits ({progress:.1f}%) - "
                          f"Elapsed: {elapsed/60:.1f}min - ETA: {eta_minutes:.1f}min - RAM: {memory_usage:.1f}%")
                else:
                    print(f"[{self.description}] Starting... RAM: {memory_usage:.1f}%")
            else:
                print(f"[{self.description}] COMPLETED: {fit_number} fits - "
                      f"Total time: {elapsed/60:.1f}min - Final RAM: {memory_usage:.1f}%")

def check_memory_and_system():
    """Check system resources and recommend optimal settings"""
    print("\n=== SYSTEM RESOURCE CHECK ===")
    
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    memory_percent = memory.percent
    
    print(f"Total RAM: {memory_gb:.1f}GB")
    print(f"Available RAM: {available_gb:.1f}GB ({100-memory_percent:.1f}% free)")
    print(f"Current RAM usage: {memory_percent:.1f}%")
    
    cpu_count = psutil.cpu_count()
    print(f"CPU cores: {cpu_count}")
    
    # SIGNIFICANTLY INCREASED dataset sizes to use more of 94k available samples
    if available_gb < 3:
        recommended_jobs = 1
        recommended_samples = 40000  # Increased from 20K
        print("LOW MEMORY: n_jobs=1, samples=40K")
    elif available_gb < 5:
        recommended_jobs = 2
        recommended_samples = 60000  # Increased from 35K  
        print("MODERATE MEMORY: n_jobs=2, samples=60K")
    elif available_gb < 8:
        recommended_jobs = 4
        recommended_samples = 80000  # Increased from 50K
        print("GOOD MEMORY: n_jobs=4, samples=80K")
    else:
        recommended_jobs = 4
        recommended_samples = 90000  # Use almost all data (94K available)
        print("EXCELLENT MEMORY: n_jobs=4, samples=90K")
        
    return recommended_jobs, recommended_samples

def stratified_sampling(df, target_samples=80000):
    """REAL-WORLD stratified sampling - preserves natural imbalance with minimum viable samples"""
    print(f"\n=== REAL-WORLD STRATIFIED SAMPLING ===")
    print(f"Input dataset: {len(df):,} samples")
    
    # Use actual column name
    class_counts = df['Class_Name'].value_counts()
    print(f"Original class distribution:\n{class_counts}")
    print(f"Original percentages:")
    for cls, count in class_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {cls}: {count:,} ({pct:.1f}%)")
    
    # REAL-WORLD STRATEGY: Preserve natural proportions with minimum viable samples
    min_viable_samples = 50   # Minimum for meaningful training
    
    sampled_dfs = []
    total_allocated = 0
    
    # Calculate proportional sizes first
    all_classes = class_counts.index.tolist()
    proportional_sizes = {}
    
    for class_name in all_classes:
        class_df = df[df['Class_Name'] == class_name]
        class_size = len(class_df)
        
        if class_size == 0:
            continue
            
        # Calculate exact proportional size
        original_proportion = class_size / len(df)
        proportional_size = int(target_samples * original_proportion)
        
        # Ensure minimum viable samples for very rare classes
        if proportional_size < min_viable_samples and class_size >= min_viable_samples:
            proportional_size = min_viable_samples
            print(f"  BOOST {class_name}: Boosted to minimum viable {min_viable_samples} samples")
        elif class_size < min_viable_samples:
            proportional_size = class_size  # Use all available
            print(f"  WARNING {class_name}: Only {class_size} samples available (less than minimum viable)")
        
        proportional_sizes[class_name] = min(proportional_size, class_size)
    
    # Check if total exceeds target - if so, scale down proportionally
    total_planned = sum(proportional_sizes.values())
    if total_planned > target_samples:
        scale_factor = target_samples / total_planned
        print(f"  SCALING: Scaling down by factor {scale_factor:.3f} to fit target")
        
        for class_name in proportional_sizes:
            old_size = proportional_sizes[class_name]
            new_size = max(min_viable_samples, int(old_size * scale_factor))
            proportional_sizes[class_name] = min(new_size, len(df[df['Class_Name'] == class_name]))
    
    # Sample each class
    for class_name, sample_size in proportional_sizes.items():
        class_df = df[df['Class_Name'] == class_name]
        class_size = len(class_df)
        
        if sample_size > 0:
            sampled_class = class_df.sample(n=sample_size, random_state=42)
            sampled_dfs.append(sampled_class)
            total_allocated += sample_size
            
            original_pct = (class_size / len(df)) * 100
            new_pct = (sample_size / target_samples) * 100
            print(f"  {class_name}: {class_size:,} -> {sample_size:,} samples ({original_pct:.1f}% -> {new_pct:.1f}%)")
    
    if not sampled_dfs:
        raise ValueError("No samples collected during stratified sampling")
    
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal sampled dataset: {len(sampled_df):,} samples")
    print(f"Target was: {target_samples:,} samples")
    print(f"Final class distribution (preserving real-world imbalance):")
    final_counts = sampled_df['Class_Name'].value_counts()
    for cls, count in final_counts.items():
        pct = (count / len(sampled_df)) * 100
        print(f"  {cls}: {count:,} ({pct:.1f}%)")
    
    print(f"\nIMBALANCE HANDLING STRATEGY:")
    print(f"  - Preserved natural class distribution (reflects real-world usage)")
    print(f"  - Ensured minimum {min_viable_samples} samples per class for training")
    print(f"  - Will use class weights + SMOTE for model balancing")
    
    return sampled_df

def load_datasets_from_local():
    """Load datasets from local Excel files"""
    print("\n=== LOADING DATASETS ===")
    
    # Updated file names based on actual files
    expected_files = [
        'mmc2.xlsx', 'mmc3.xlsx', 'mmc4.xlsx', 'mmc5.xlsx', 'mmc6.xlsx',
        'mmc7.xlsx', 'mmc8.xlsx', 'mmc9.xlsx', 'mmc10.xlsx', 'mmc11.xlsx'
    ]
    
    dataframes = []
    
    for filename in expected_files:
        filepath = os.path.join('dataset', filename)  # Changed from 'datasets' to 'dataset'
        if os.path.exists(filepath):
            try:
                df = pd.read_excel(filepath)
                dataframes.append(df)
                print(f"LOADED {filename}: {len(df):,} samples")
            except Exception as e:
                print(f"FAILED to load {filename}: {e}")
        else:
            print(f"FILE NOT FOUND: {filepath}")
    
    if not dataframes:
        raise FileNotFoundError("No dataset files found in dataset/ directory")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nCOMBINED DATASET: {len(combined_df):,} total samples")
    
    return combined_df

def prepare_balanced_svm_data(df):
    """Prepare and clean data for SVM training"""
    print("\n=== DATA PREPARATION ===")
    print(f"Input data shape: {df.shape}")
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_count - len(df):,} duplicate rows")
    
    # Required columns based on actual dataset structure
    required_columns = [
        'Jobs_per_ 1Minute', 'Jobs_per_ 5 Minutes', 'Jobs_per_ 15Minutes',
        'Mem capacity', 'Disk_capacity_GB', 'Num_of_CPU_Cores', 'CPU_speed_per_Core',
        'Avg_Recieve_Kbps', 'Avg_Transmit_Kbps', 'Class_Name'
    ]
    
    df_clean = df.dropna(subset=required_columns)
    print(f"Removed {len(df) - len(df_clean):,} rows with missing values")
    print(f"Final clean data shape: {df_clean.shape}")
    
    # Create features based on actual dataset columns
    feature_columns = [
        'Jobs_per_ 1Minute', 'Jobs_per_ 5 Minutes', 'Jobs_per_ 15Minutes',
        'Mem capacity', 'Disk_capacity_GB', 'Num_of_CPU_Cores', 'CPU_speed_per_Core',
        'Avg_Recieve_Kbps', 'Avg_Transmit_Kbps'
    ]
    
    # Add derived features
    df_clean['total_jobs'] = (
        df_clean['Jobs_per_ 1Minute'] + 
        df_clean['Jobs_per_ 5 Minutes'] + 
        df_clean['Jobs_per_ 15Minutes']
    ) / 3  # Average jobs rate
    
    df_clean['total_network'] = df_clean['Avg_Recieve_Kbps'] + df_clean['Avg_Transmit_Kbps']
    
    df_clean['resource_intensity'] = (
        df_clean['total_jobs'] * 0.3 + 
        df_clean['Mem capacity'] * 0.2 + 
        df_clean['Disk_capacity_GB'] * 0.2 +
        df_clean['Num_of_CPU_Cores'] * df_clean['CPU_speed_per_Core'] * 0.3
    )
    
    df_clean['network_ratio'] = np.where(
        df_clean['Avg_Transmit_Kbps'] > 0,
        df_clean['Avg_Recieve_Kbps'] / df_clean['Avg_Transmit_Kbps'],
        df_clean['Avg_Recieve_Kbps']
    )
    
    feature_columns.extend(['total_jobs', 'total_network', 'resource_intensity', 'network_ratio'])
    
    # Remove outliers (simple IQR method)
    for col in feature_columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            before_count = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            outliers_removed = before_count - len(df_clean)
            
            if outliers_removed > 0:
                print(f"Removed {outliers_removed:,} outliers from {col}")
    
    print(f"Final processed data shape: {df_clean.shape}")
    print(f"Class distribution:\n{df_clean['Class_Name'].value_counts()}")
    
    X = df_clean[feature_columns]
    y = df_clean['Class_Name']  # Use actual target column
    
    return X, y, feature_columns

def train_balanced_svm_model(X, y, feature_names, n_jobs=4):
    """Train SVM with fixed progress tracking and memory management"""
    print(f"\n=== SVM TRAINING (n_jobs={n_jobs}) ===")
    
    memory_before = psutil.virtual_memory().percent
    print(f"Memory usage before training: {memory_before:.1f}%")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Classes: {label_encoder.classes_}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training: {len(X_train):,}, Test: {len(X_test):,}")
    
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Memory cleanup
    del X_train, X_test
    gc.collect()
    
    # Conservative SMOTE to prevent memory explosion
    print(f"\nApplying REAL-WORLD SMOTE strategy...")
    class_counts = Counter(y_train_encoded)
    min_class_size = min(class_counts.values())
    max_class_size = max(class_counts.values())
    print(f"Class sizes before SMOTE: {dict(class_counts)}")
    print(f"Imbalance ratio: {max_class_size/min_class_size:.1f}:1")
    
    k_neighbors = min(3, min_class_size - 1) if min_class_size > 1 else 1
    
    # REAL-WORLD SMOTE: Only boost very small classes to reasonable minimums
    if min_class_size < 100:  # Only apply SMOTE if really needed
        # Target: bring smallest classes to 100-200 samples, not full balance
        target_min = min(200, min_class_size * 3)  # Conservative boost
        
        sampling_strategy = {}
        for label, count in class_counts.items():
            if count < target_min:
                sampling_strategy[label] = target_min
                print(f"  BOOST Class {label}: {count} -> {target_min} samples")
            else:
                print(f"  OK Class {label}: {count} samples (no SMOTE needed)")
        
        if sampling_strategy:
            print(f"SMOTE strategy: {sampling_strategy}")
        else:
            print("No SMOTE needed - all classes have sufficient samples")
            sampling_strategy = None
    else:
        sampling_strategy = None
        print("All classes have >100 samples - skipping SMOTE")
    
    try:
        if sampling_strategy:
            smote = SMOTE(
                random_state=42, 
                k_neighbors=k_neighbors,
                sampling_strategy=sampling_strategy
            )
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_encoded)
            print(f"After SMOTE: {len(X_train_balanced):,} samples")
            print(f"New distribution: {Counter(y_train_balanced)}")
            
            # Calculate new imbalance ratio
            new_counts = Counter(y_train_balanced)
            new_min = min(new_counts.values())
            new_max = max(new_counts.values())
            print(f"New imbalance ratio: {new_max/new_min:.1f}:1 (improved from {max_class_size/min_class_size:.1f}:1)")
        else:
            X_train_balanced, y_train_balanced = X_train_scaled, y_train_encoded
            print("Using original distribution (natural imbalance preserved)")
            
    except Exception as e:
        print(f"SMOTE failed: {e}. Using original data...")
        X_train_balanced, y_train_balanced = X_train_scaled, y_train_encoded
    
    del X_train_scaled
    gc.collect()
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_balanced), y=y_train_balanced)
    class_weight_dict = dict(zip(np.unique(y_train_balanced), class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # Grid Search with reasonable parameters
    print("=== GRID SEARCH ===")
    param_grid = {
        'C': [1, 10, 100],
        'gamma': ['scale', 0.1],
        'kernel': ['rbf']
    }
    
    total_combinations = len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])
    cv_folds = 3
    total_fits = total_combinations * cv_folds
    
    print(f"Grid parameters: {param_grid}")
    print(f"Total: {total_combinations} combinations × {cv_folds} folds = {total_fits} fits")
    print(f"Using n_jobs={n_jobs} for parallel processing")
    
    # Initialize simple progress tracker
    tracker = SimpleProgressTracker(total_fits, "SVM GridSearch")
    tracker.report_progress(0)  # Initial report
    
    try:
        print(f"Starting GridSearchCV at: {datetime.now().strftime('%H:%M:%S')}")
        start_time = time.time()
        
        svm_grid = GridSearchCV(
            SVC(random_state=42, probability=True, class_weight=class_weight_dict),
            param_grid,
            cv=cv_folds,
            scoring='balanced_accuracy',  # Better for imbalanced classes than f1_macro
            n_jobs=n_jobs,
            verbose=0,  # No verbose output to prevent spam
            return_train_score=False,  # Save memory
            error_score=0.0  # Handle failed fits gracefully
        )
        
        # Fit the model
        svm_grid.fit(X_train_balanced, y_train_balanced)
        
        # Report completion
        tracker.report_progress(total_fits)
        
        training_time = time.time() - start_time
        print(f"Training completed at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Total training time: {training_time/60:.1f} minutes")
        
    except Exception as e:
        print(f"GridSearchCV failed: {e}")
        print("Using fallback SVM...")
        
        # Fallback to simple SVM
        fallback_svm = SVC(
            C=10, gamma='scale', kernel='rbf',
            random_state=42, probability=True, 
            class_weight=class_weight_dict
        )
        fallback_svm.fit(X_train_balanced, y_train_balanced)
        
        class MockGrid:
            def __init__(self, estimator):
                self.best_estimator_ = estimator
                self.best_params_ = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
                self.best_score_ = 0.0
                self.cv_results_ = {}
        
        svm_grid = MockGrid(fallback_svm)
    
    # Evaluation
    print(f"\n=== MODEL EVALUATION ===")
    best_svm = svm_grid.best_estimator_
    print(f"Best parameters: {svm_grid.best_params_}")
    
    # Efficient prediction for test set
    print(f"Evaluating on {len(X_test_scaled):,} test samples...")
    
    if len(X_test_scaled) > 10000:
        # Batch prediction for very large test sets
        batch_size = 5000
        predictions = []
        
        print("Using batch prediction for large test set...")
        for i in range(0, len(X_test_scaled), batch_size):
            batch = X_test_scaled[i:i+batch_size]
            batch_pred = best_svm.predict(batch)
            predictions.extend(batch_pred)
            print(f"Batch {i//batch_size + 1}/{(len(X_test_scaled)-1)//batch_size + 1} completed")
        
        y_pred = np.array(predictions)
    else:
        y_pred = best_svm.predict(X_test_scaled)
    
    # Convert back to original labels
    y_test_original = label_encoder.inverse_transform(y_test_encoded)
    y_pred_original = label_encoder.inverse_transform(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_original, y_pred_original)
    from sklearn.metrics import balanced_accuracy_score, f1_score
    balanced_acc = balanced_accuracy_score(y_test_original, y_pred_original)
    f1_macro = f1_score(y_test_original, y_pred_original, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test_original, y_pred_original, average='weighted', zero_division=0)
    
    print(f"\nMODEL PERFORMANCE ON IMBALANCED DATA:")
    print(f"  Standard Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%) <- Key metric for imbalanced data")
    print(f"  F1-Score (Macro): {f1_macro:.4f}")
    print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
    
    # Calculate per-class performance
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_original, y_pred_original, labels=label_encoder.classes_, zero_division=0
    )
    
    print(f"\nPER-CLASS PERFORMANCE:")
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 55)
    for i, cls in enumerate(label_encoder.classes_):
        print(f"{cls:<12} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f} {support[i]:<8}")
    
    # Identify problematic classes
    print(f"\nCLASS ANALYSIS:")
    for i, cls in enumerate(label_encoder.classes_):
        if recall[i] < 0.3:
            print(f"  WARNING {cls}: Low recall ({recall[i]:.3f}) - model struggles to detect this class")
        elif recall[i] > 0.8:
            print(f"  GOOD {cls}: Good recall ({recall[i]:.3f}) - model detects this class well")
        else:
            print(f"  OK {cls}: Moderate recall ({recall[i]:.3f}) - acceptable performance")
    
    return best_svm, scaler, label_encoder, svm_grid, balanced_acc

def test_svm_scenarios(svm_model, scaler, label_encoder, feature_names):
    """Test SVM with example scenarios"""
    print(f"\n=== TESTING SVM SCENARIOS ===")
    
    # Updated test scenarios based on actual feature structure
    test_scenarios = [
        # [Jobs_1min, Jobs_5min, Jobs_15min, Mem, Disk, CPU_Cores, CPU_Speed, Recv_Kbps, Trans_Kbps, total_jobs, total_network, resource_intensity, network_ratio]
        [3.0, 4.0, 5.0, 8192, 500.0, 4, 2400, 10.0, 5.0, 4.0, 15.0, 2500.0, 2.0],     # Very Low workload
        [50.0, 55.0, 60.0, 16384, 1000.0, 8, 3200, 100.0, 80.0, 55.0, 180.0, 15000.0, 1.25],  # Low workload  
        [150.0, 160.0, 170.0, 32768, 2000.0, 16, 3600, 500.0, 400.0, 160.0, 900.0, 45000.0, 1.25],  # Medium workload
        [300.0, 320.0, 350.0, 65536, 4000.0, 32, 4000, 1000.0, 800.0, 323.3, 1800.0, 95000.0, 1.25],  # High workload
    ]
    
    expected_classes = ['\'Very Low\'', '\'Low\'', '\'Medium\'', '\'High\'']
    
    for i, (scenario, expected) in enumerate(zip(test_scenarios, expected_classes)):
        scenario_df = pd.DataFrame([scenario], columns=feature_names)
        scenario_scaled = scaler.transform(scenario_df)
        
        prediction = svm_model.predict(scenario_scaled)[0]
        probabilities = svm_model.predict_proba(scenario_scaled)[0]
        prediction_label = label_encoder.inverse_transform([prediction])[0]
        
        print(f"\nScenario {i+1}: {expected} workload")
        print(f"  Predicted: {prediction_label}")
        print(f"  Probabilities: {dict(zip(label_encoder.classes_, probabilities))}")
        print(f"  CORRECT" if prediction_label == expected else f"  WRONG (expected {expected})")

def save_balanced_svm_model(svm_model, scaler, label_encoder, grid_search, feature_names, balanced_accuracy):
    """Save trained model and metadata"""
    print(f"\n=== SAVING MODEL ===")
    
    os.makedirs('models', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    accuracy_str = f"{balanced_accuracy*100:.1f}pct"
    
    model_data = {
        'svm_model': svm_model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'best_params': grid_search.best_params_ if hasattr(grid_search, 'best_params_') else {},
        'balanced_accuracy': balanced_accuracy,
        'timestamp': timestamp,
        'version': '2.1-realworld'
    }
    
    model_filename = f'models/balanced_svm_model_{accuracy_str}_{timestamp}.pkl'
    joblib.dump(model_data, model_filename, compress=3)
    print(f"Model saved: {model_filename}")
    
    # Create latest symlink
    latest_link = 'models/balanced_svm_model_latest.pkl'
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(os.path.basename(model_filename), latest_link)
    print(f"Latest model link: {latest_link}")
    
    # Save metadata
    metadata = {
        'balanced_accuracy': balanced_accuracy,
        'timestamp': timestamp,
        'feature_names': feature_names,
        'best_params': model_data['best_params'],
        'classes': list(label_encoder.classes_),
        'version': '2.1-realworld'
    }
    
    metadata_filename = f'models/svm_metadata_{timestamp}.json'
    import json
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_filename}")
    
    return model_filename

def main():
    """Main training pipeline"""
    print("MCCVA SVM Training - Real-World Version 2.1")
    print("   Handles natural class imbalance for production VM workloads")
    print("=" * 70)
    
    # Check system resources and get optimal settings
    optimal_n_jobs, optimal_samples = check_memory_and_system()
    
    try:
        # Load datasets
        df = load_datasets_from_local()
        
        # Apply stratified sampling  
        df_sampled = stratified_sampling(df, target_samples=optimal_samples)
        
        # Prepare data
        X, y, feature_names = prepare_balanced_svm_data(df_sampled)
        
        print(f"\nFINAL TRAINING DATA:")
        print(f"  Samples: {len(X):,}")
        print(f"  Features: {len(feature_names)}")
        print(f"  Feature names: {feature_names}")
        
        # Train model
        svm_model, scaler, label_encoder, grid_search, balanced_accuracy = train_balanced_svm_model(
            X, y, feature_names, n_jobs=optimal_n_jobs
        )
        
        # Test scenarios
        test_svm_scenarios(svm_model, scaler, label_encoder, feature_names)
        
        # Save model
        model_filename = save_balanced_svm_model(
            svm_model, scaler, label_encoder, grid_search, feature_names, balanced_accuracy
        )
        
        print(f"\nTRAINING COMPLETED SUCCESSFULLY!")
        print(f"Balanced accuracy: {balanced_accuracy:.4f} ({balanced_accuracy*100:.2f}%) - Key metric for imbalanced data")
        print(f"Model saved: {model_filename}")
        print(f"\nREAL-WORLD IMBALANCE HANDLING:")
        print(f"  - Preserved natural class distribution (reflects actual VM workload patterns)")
        print(f"  - Used class weights + selective SMOTE for model balance")
        print(f"  - Optimized for balanced accuracy (better than standard accuracy for imbalanced data)")
        print(f"  - Ready for integration with OpenResty load balancer")
        
        # Final memory check
        final_memory = psutil.virtual_memory().percent
        print(f"Final memory usage: {final_memory:.1f}%")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 