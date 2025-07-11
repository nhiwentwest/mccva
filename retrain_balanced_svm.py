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

def map_to_three_classes(y_original):
    """
    Map 5-class system to 3-class system for ML service compatibility
    ['Very Low', 'Low'] -> 'small'
    ['Medium'] -> 'medium' 
    ['High', 'Very High'] -> 'large'
    
    Note: Dataset has quoted class names like "'Very Low'" so we need to handle that
    """
    class_mapping = {
        "'Very Low'": 'small',
        "'Low'": 'small', 
        "'Medium'": 'medium',
        "'High'": 'large',
        "'Very High'": 'large',
        # Also handle versions without quotes in case they exist
        'Very Low': 'small',
        'Low': 'small',
        'Medium': 'medium', 
        'High': 'large',
        'Very High': 'large'
    }
    
    y_mapped = [class_mapping.get(cls, 'medium') for cls in y_original]
    return np.array(y_mapped)

def prepare_balanced_svm_data(df):
    """Prepare SVM training data với balanced sampling approach"""
    print(f"\n=== DATA PREPARATION ===")
    
    # Remove rows with missing values in critical columns  
    critical_columns = ['Class_Name']  # Use actual column name
    df_clean = df.dropna(subset=critical_columns)
    print(f"After removing missing values: {len(df_clean):,} samples")
    
    # Use actual column names from dataset
    feature_columns = [
        'Jobs_per_ 1Minute', 'Jobs_per_ 5 Minutes', 'Jobs_per_ 15Minutes',
        'Mem capacity', 'Disk_capacity_GB', 'Num_of_CPU_Cores', 'CPU_speed_per_Core',
        'Avg_Recieve_Kbps', 'Avg_Transmit_Kbps'
    ]
    
    # Check if all required columns exist
    missing_cols = [col for col in feature_columns if col not in df_clean.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        print(f"Available columns: {list(df_clean.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Extract features and target
    X = df_clean[feature_columns].copy()
    y_original = df_clean['Class_Name'].copy()  # Use actual column name
    
    # Map to 3-class system for ML service compatibility
    y_mapped = map_to_three_classes(y_original)
    
    print(f"\nClass mapping applied:")
    print(f"  Original classes: {sorted(y_original.unique())}")
    print(f"  Mapped classes: {sorted(np.unique(y_mapped))}")
    
    # Check class distribution after mapping
    mapped_dist = Counter(y_mapped)
    print(f"\nMapped class distribution:")
    total_samples = len(y_mapped)
    for cls, count in sorted(mapped_dist.items()):
        percentage = (count / total_samples) * 100
        print(f"  {cls}: {count:,} samples ({percentage:.2f}%)")
    
    # Handle missing values in features
    print(f"\nHandling missing values...")
    for col in feature_columns:
        missing_count = X[col].isnull().sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} missing values -> filled with median")
            X[col] = X[col].fillna(X[col].median())
    
    # Remove outliers using IQR method for each feature
    print(f"\nRemoving outliers...")
    initial_size = len(X)
    for col in feature_columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Create mask for valid values
        valid_mask = (X[col] >= lower_bound) & (X[col] <= upper_bound)
        X = X[valid_mask]
        y_mapped = y_mapped[valid_mask]
    
    final_size = len(X)
    removed_count = initial_size - final_size
    print(f"  Removed {removed_count:,} outliers ({(removed_count/initial_size)*100:.2f}%)")
    print(f"  Final dataset size: {final_size:,} samples")
    
    # Final class distribution check
    final_dist = Counter(y_mapped)
    print(f"\nFinal class distribution after cleaning:")
    for cls, count in sorted(final_dist.items()):
        percentage = (count / len(y_mapped)) * 100
        print(f"  {cls}: {count:,} samples ({percentage:.2f}%)")
    
    return X.values, y_mapped, feature_columns

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
    """Test SVM với realistic scenarios using 3-class system"""
    print(f"\n=== TESTING SVM SCENARIOS ===")
    
    # Updated test scenarios for 3-class system using actual feature names
    # [Jobs_per_ 1Minute, Jobs_per_ 5 Minutes, Jobs_per_ 15Minutes, Mem capacity, Disk_capacity_GB, Num_of_CPU_Cores, CPU_speed_per_Core, Avg_Recieve_Kbps, Avg_Transmit_Kbps]
    test_scenarios = [
        # Small workload - Low job rate, small memory/disk
        [3.0, 4.2, 5.0, 8192, 500.0, 4, 2400, 10.0, 5.0],     
        # Small workload - Slightly higher but still small
        [12.0, 13.7, 15.0, 16384, 1000.0, 8, 3200, 50.0, 25.0],  
        # Medium workload - Moderate job rate and resources
        [150.0, 160.0, 170.0, 32768, 2000.0, 16, 3600, 500.0, 400.0],  
        # Large workload - High job rate, large memory/disk  
        [300.0, 320.0, 350.0, 65536, 4000.0, 32, 4000, 1000.0, 800.0],  
    ]
    
    expected_classes = ['small', 'small', 'medium', 'large']
    
    for i, (scenario, expected) in enumerate(zip(test_scenarios, expected_classes)):
        scenario_df = pd.DataFrame([scenario], columns=feature_names)
        scenario_scaled = scaler.transform(scenario_df)
        
        prediction = svm_model.predict(scenario_scaled)[0]
        probabilities = svm_model.predict_proba(scenario_scaled)[0]
        prediction_label = label_encoder.inverse_transform([prediction])[0]
        
        print(f"\nScenario {i+1}: {expected} workload")
        print(f"  Predicted: {prediction_label}")
        print(f"  Probabilities: {dict(zip(label_encoder.classes_, probabilities))}")
        print(f"  ✅ CORRECT" if prediction_label == expected else f"  ❌ WRONG (expected {expected})")

def save_balanced_svm_model(svm_model, scaler, label_encoder, grid_search, feature_names, balanced_accuracy):
    """Save trained model and metadata with ML service compatible filenames"""
    print(f"\n=== SAVING MODEL ===")
    
    os.makedirs('models', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    accuracy_str = f"{balanced_accuracy*100:.1f}pct"
    
    # Save individual components with ML service expected filenames
    joblib.dump(svm_model, 'models/svm_model.joblib')
    joblib.dump(scaler, 'models/svm_scaler.joblib') 
    joblib.dump(label_encoder, 'models/svm_label_encoder.joblib')
    joblib.dump(feature_names, 'models/svm_feature_names.joblib')
    
    print(f"✅ SVM Model saved: models/svm_model.joblib")
    print(f"✅ SVM Scaler saved: models/svm_scaler.joblib")
    print(f"✅ SVM Label Encoder saved: models/svm_label_encoder.joblib")
    print(f"✅ SVM Features saved: models/svm_feature_names.joblib")
    
    # Also save backup with timestamp
    model_data = {
        'svm_model': svm_model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'best_params': grid_search.best_params_ if hasattr(grid_search, 'best_params_') else {},
        'balanced_accuracy': balanced_accuracy,
        'timestamp': timestamp,
        'version': '3.0-compatible',
        'class_system': '3-class (small/medium/large)'
    }
    
    backup_filename = f'models/svm_backup_{accuracy_str}_{timestamp}.pkl'
    joblib.dump(model_data, backup_filename, compress=3)
    print(f"📦 Backup saved: {backup_filename}")
    
    # Save metadata
    metadata = {
        'balanced_accuracy': balanced_accuracy,
        'timestamp': timestamp,
        'feature_names': feature_names,
        'best_params': model_data['best_params'],
        'classes': list(label_encoder.classes_),
        'version': '3.0-compatible',
        'class_system': '3-class (small/medium/large)',
        'compatible_with': 'ml_service.py'
    }
    
    # Save SVM info for integration tests
    joblib.dump(metadata, 'models/svm_info.joblib')
    print(f"📋 SVM Info saved: models/svm_info.joblib")
    
    # Save JSON metadata for human reading
    metadata_filename = f'models/svm_metadata_{timestamp}.json'
    import json
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 JSON Metadata saved: {metadata_filename}")
    
    return backup_filename

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