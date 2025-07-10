#!/usr/bin/env python3
"""
Retrained Optimized K-Means Model for OpenResty Integration
REAL-WORLD VERSION 2.1 - VM Resource Clustering for Meta-Learning

This script handles VM resource clustering as part of the MCCVA Meta-Learning architecture:
- Uses same 94K dataset as SVM (consistent data pipeline)
- Clusters VM utilization patterns for ensemble learning
- Optimized for production deployment with OpenResty
- Memory efficient with proper resource management
- No emoji/icons for clean logging

Architecture: SVM (workload classification) + K-Means (VM clustering) -> Meta-Learning NN
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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
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
    """Simple progress tracker that only reports on significant updates"""
    def __init__(self, total_iterations, description="K-Means Training"):
        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.start_time = time.time()
        self.description = description
        self.last_report_time = 0
        
    def update(self, iteration=None):
        """Update progress and report if significant change"""
        current_time = time.time()
        
        if iteration is not None:
            self.current_iteration = iteration
        else:
            self.current_iteration += 1
        
        # Only report every 2 minutes or on completion
        if (current_time - self.last_report_time > 120) or (self.current_iteration >= self.total_iterations):
            self.last_report_time = current_time
            elapsed = current_time - self.start_time
            progress = min((self.current_iteration / self.total_iterations) * 100, 100.0)
            memory_usage = psutil.virtual_memory().percent
            
            if self.current_iteration < self.total_iterations:
                if self.current_iteration > 0:
                    avg_time_per_iteration = elapsed / self.current_iteration
                    remaining = self.total_iterations - self.current_iteration
                    eta_minutes = (remaining * avg_time_per_iteration) / 60
                    print(f"[{self.description}] Progress: {self.current_iteration}/{self.total_iterations} ({progress:.1f}%) - "
                          f"Elapsed: {elapsed/60:.1f}min - ETA: {eta_minutes:.1f}min - RAM: {memory_usage:.1f}%")
                else:
                    print(f"[{self.description}] Starting... RAM: {memory_usage:.1f}%")
            else:
                print(f"[{self.description}] COMPLETED: {self.current_iteration} iterations - "
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
    
    # Use larger dataset sizes like SVM (K-Means can handle more data efficiently)
    if available_gb < 3:
        recommended_samples = 40000
        print("LOW MEMORY: samples=40K")
    elif available_gb < 5:
        recommended_samples = 60000
        print("MODERATE MEMORY: samples=60K")
    elif available_gb < 8:
        recommended_samples = 80000
        print("GOOD MEMORY: samples=80K")
    else:
        recommended_samples = 90000
        print("EXCELLENT MEMORY: samples=90K")
        
    return recommended_samples

def stratified_vm_sampling(df, target_samples=80000):
    """REAL-WORLD stratified sampling for VM clustering - preserves resource distribution"""
    print(f"\n=== REAL-WORLD VM STRATIFIED SAMPLING ===")
    print(f"Input dataset: {len(df):,} samples")
    print(f"Target samples: {target_samples:,}")
    
    if len(df) <= target_samples:
        print("Dataset already smaller than target - using full dataset")
        return df
    
    # Create resource utilization bins for stratification based on actual data
    def create_resource_bins(df):
        """Create bins based on actual VM resource utilization patterns"""
        # Use actual columns from dataset
        required_cols = ['Mem capacity', 'Disk_capacity_GB', 'Num_of_CPU_Cores', 'CPU_speed_per_Core']
        
        # Check if all required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"WARNING: Missing columns {missing_cols}, using available data")
        
        # Calculate normalized resource scores using available columns
        df_scored = df.copy()
        
        if 'Mem capacity' in df.columns:
            df_scored['memory_norm'] = df['Mem capacity'] / df['Mem capacity'].max()
        else:
            df_scored['memory_norm'] = 0.5
            
        if 'Num_of_CPU_Cores' in df.columns and 'CPU_speed_per_Core' in df.columns:
            cpu_power = df['Num_of_CPU_Cores'] * df['CPU_speed_per_Core']
            df_scored['cpu_norm'] = cpu_power / cpu_power.max()
        else:
            df_scored['cpu_norm'] = 0.5
            
        if 'Disk_capacity_GB' in df.columns:
            df_scored['disk_norm'] = df['Disk_capacity_GB'] / df['Disk_capacity_GB'].max()
        else:
            df_scored['disk_norm'] = 0.5
        
        # Create composite resource score
        df_scored['resource_score'] = (
            df_scored['memory_norm'] * 0.4 + 
            df_scored['cpu_norm'] * 0.4 + 
            df_scored['disk_norm'] * 0.2
        )
        
        # Create resource tiers: Low (0-0.33), Medium (0.33-0.67), High (0.67-1.0)
        df_scored['resource_tier'] = pd.cut(
            df_scored['resource_score'], 
            bins=[0, 0.33, 0.67, 1.0], 
            labels=['low', 'medium', 'high'],
            include_lowest=True
        )
        
        return df_scored
    
    # Create resource bins
    df_binned = create_resource_bins(df)
    
    # Get current tier distribution
    tier_counts = df_binned['resource_tier'].value_counts()
    print(f"\nOriginal VM resource tier distribution:")
    for tier_name, count in tier_counts.items():
        pct = (count / len(df_binned)) * 100
        print(f"  {tier_name}: {count:,} ({pct:.1f}%)")
    
    # Calculate proportional sampling
    sampling_ratio = target_samples / len(df_binned)
    print(f"\nSampling ratio: {sampling_ratio:.3f}")
    
    # Stratified sampling maintaining proportions
    sampled_dfs = []
    total_sampled = 0
    
    for tier_name in ['low', 'medium', 'high']:
        tier_df = df_binned[df_binned['resource_tier'] == tier_name]
        if len(tier_df) == 0:
            continue
            
        target_tier_samples = max(1, int(len(tier_df) * sampling_ratio))
        
        if len(tier_df) > target_tier_samples:
            sampled_tier_df = tier_df.sample(n=target_tier_samples, random_state=42)
        else:
            sampled_tier_df = tier_df
        
        # Remove helper columns
        cols_to_remove = ['memory_norm', 'cpu_norm', 'disk_norm', 'resource_score', 'resource_tier']
        for col in cols_to_remove:
            if col in sampled_tier_df.columns:
                sampled_tier_df = sampled_tier_df.drop(col, axis=1)
        
        sampled_dfs.append(sampled_tier_df)
        total_sampled += len(sampled_tier_df)
        
        original_pct = (len(tier_df) / len(df_binned)) * 100
        new_pct = (len(sampled_tier_df) / target_samples) * 100
        print(f"  {tier_name}: {len(tier_df):,} -> {len(sampled_tier_df):,} ({original_pct:.1f}% -> {new_pct:.1f}%)")
    
    # Combine sampled data
    if not sampled_dfs:
        print("WARNING: No valid samples collected, using original data")
        return df
        
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"\nFinal sampled dataset: {len(sampled_df):,} samples")
    print(f"Memory reduction: {(1 - len(sampled_df)/len(df))*100:.1f}%")
    print(f"STRATEGY: Preserved VM resource distribution for accurate clustering")
    
    return sampled_df

def load_datasets_from_local():
    """Load datasets from local Excel files (same as SVM pipeline)"""
    print("\n=== LOADING DATASETS ===")
    
    # Updated file names based on actual files (same as SVM)
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

def prepare_kmeans_vm_data(df):
    """Prepare VM resource data for K-Means clustering using actual dataset structure"""
    print("\n=== DATA PREPARATION FOR K-MEANS ===")
    print(f"Input data shape: {df.shape}")
    
    # Remove duplicates (same as SVM)
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_count - len(df):,} duplicate rows")
    
    # Required columns based on actual dataset structure (same as SVM)
    required_columns = [
        'Jobs_per_ 1Minute', 'Jobs_per_ 5 Minutes', 'Jobs_per_ 15Minutes',
        'Mem capacity', 'Disk_capacity_GB', 'Num_of_CPU_Cores', 'CPU_speed_per_Core',
        'Avg_Recieve_Kbps', 'Avg_Transmit_Kbps'
    ]
    
    df_clean = df.dropna(subset=required_columns)
    print(f"Removed {len(df) - len(df_clean):,} rows with missing values")
    print(f"Final clean data shape: {df_clean.shape}")
    
    # Create VM resource utilization features for clustering
    # Focus on resource capacity and utilization patterns
    
    # Memory utilization (normalize by max)
    max_memory = df_clean['Mem capacity'].max()
    df_clean['memory_utilization'] = df_clean['Mem capacity'] / max_memory
    
    # CPU power (cores * speed, normalized)
    df_clean['cpu_power'] = df_clean['Num_of_CPU_Cores'] * df_clean['CPU_speed_per_Core']
    max_cpu_power = df_clean['cpu_power'].max()
    df_clean['cpu_utilization'] = df_clean['cpu_power'] / max_cpu_power
    
    # Storage utilization (normalized)
    max_storage = df_clean['Disk_capacity_GB'].max()
    df_clean['storage_utilization'] = df_clean['Disk_capacity_GB'] / max_storage
    
    # Network utilization (combined receive + transmit, normalized)
    df_clean['total_network'] = df_clean['Avg_Recieve_Kbps'] + df_clean['Avg_Transmit_Kbps']
    max_network = df_clean['total_network'].max()
    df_clean['network_utilization'] = df_clean['total_network'] / max_network
    
    # Workload intensity (average job rate, normalized)
    df_clean['avg_job_rate'] = (
        df_clean['Jobs_per_ 1Minute'] + 
        df_clean['Jobs_per_ 5 Minutes'] + 
        df_clean['Jobs_per_ 15Minutes']
    ) / 3
    max_job_rate = df_clean['avg_job_rate'].max()
    df_clean['workload_intensity'] = df_clean['avg_job_rate'] / max_job_rate
    
    # Select K-Means features for VM clustering
    # Focus on resource utilization patterns that distinguish VM types
    kmeans_features = [
        'memory_utilization',    # How much memory the VM uses
        'cpu_utilization',       # How much CPU power the VM has
        'storage_utilization',   # How much storage the VM uses
        'network_utilization',   # How much network bandwidth the VM uses
        'workload_intensity'     # How intensive the VM's workload is
    ]
    
    # Ensure all features exist and are properly bounded
    for feature in kmeans_features:
        if feature not in df_clean.columns:
            print(f"WARNING: Feature {feature} not found, setting to 0.5")
            df_clean[feature] = 0.5
    
    # Create feature matrix - ensure values are between 0 and 1
    X = df_clean[kmeans_features].copy()
    X = X.clip(0, 1)  # Ensure values are between 0 and 1
    
    # Handle any remaining missing values
    X = X.fillna(0.5)  # Default to 50% utilization
    
    # Remove outliers (simple IQR method, same as SVM)
    for col in kmeans_features:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        before_count = len(X)
        valid_rows = (X[col] >= lower_bound) & (X[col] <= upper_bound)
        X = X[valid_rows]
        outliers_removed = before_count - len(X)
        
        if outliers_removed > 0:
            print(f"Removed {outliers_removed:,} outliers from {col}")
    
    print(f"Final processed data shape: {X.shape}")
    print(f"K-Means features: {kmeans_features}")
    
    # Show feature statistics
    print(f"\nVM Resource Utilization Statistics:")
    for feature in kmeans_features:
        values = X[feature]
        print(f"  {feature}: min={values.min():.3f}, mean={values.mean():.3f}, max={values.max():.3f}, std={values.std():.3f}")
    
    return X, kmeans_features, df_clean

def determine_optimal_clusters(X, max_clusters=10):
    """Determine optimal number of clusters using multiple methods"""
    print(f"\n=== DETERMINING OPTIMAL CLUSTERS ===")
    print(f"Testing cluster counts from 2 to {max_clusters}")
    
    # Limit data for cluster analysis if too large
    if len(X) > 10000:
        print(f"Dataset large ({len(X):,} samples), using sample of 10,000 for cluster analysis")
        X_sample = X.sample(n=10000, random_state=42)
    else:
        X_sample = X
    
    cluster_range = range(2, max_clusters + 1)
    
    # Method 1: Elbow Method (Within-cluster sum of squares)
    wcss = []
    # Method 2: Silhouette Score
    silhouette_scores = []
    # Method 3: Calinski-Harabasz Score
    calinski_scores = []
    # Method 4: Davies-Bouldin Score
    davies_bouldin_scores = []
    
    print("Analyzing cluster counts:")
    
    for k in cluster_range:
        print(f"  Testing k={k}...", end=" ")
        
        # Fit K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(X_sample)
        
        # Calculate metrics
        wcss.append(kmeans.inertia_)
        
        if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for these metrics
            sil_score = silhouette_score(X_sample, cluster_labels)
            cal_score = calinski_harabasz_score(X_sample, cluster_labels)
            db_score = davies_bouldin_score(X_sample, cluster_labels)
            
            silhouette_scores.append(sil_score)
            calinski_scores.append(cal_score)
            davies_bouldin_scores.append(db_score)
            
            print(f"Silhouette: {sil_score:.3f}, Calinski: {cal_score:.1f}, DB: {db_score:.3f}")
        else:
            silhouette_scores.append(0)
            calinski_scores.append(0)
            davies_bouldin_scores.append(float('inf'))
            print("Invalid clustering")
    
    # Find optimal k using different methods
    optimal_k_methods = {}
    
    # Elbow method - look for the "elbow" in WCSS
    # Simple heuristic: largest drop in WCSS
    wcss_drops = [wcss[i] - wcss[i+1] for i in range(len(wcss)-1)]
    optimal_k_methods['elbow'] = cluster_range[wcss_drops.index(max(wcss_drops))]
    
    # Silhouette - highest score
    if silhouette_scores:
        optimal_k_methods['silhouette'] = cluster_range[silhouette_scores.index(max(silhouette_scores))]
    
    # Calinski-Harabasz - highest score
    if calinski_scores:
        optimal_k_methods['calinski'] = cluster_range[calinski_scores.index(max(calinski_scores))]
    
    # Davies-Bouldin - lowest score
    if davies_bouldin_scores:
        optimal_k_methods['davies_bouldin'] = cluster_range[davies_bouldin_scores.index(min(davies_bouldin_scores))]
    
    print(f"\nOptimal cluster recommendations:")
    for method, k in optimal_k_methods.items():
        print(f"  {method}: k={k}")
    
    # Consensus decision - most common recommendation, with preference for simplicity
    recommendations = list(optimal_k_methods.values())
    from collections import Counter
    vote_counts = Counter(recommendations)
    
    # If there's a tie, prefer smaller k (simpler model)
    optimal_k = min(vote_counts, key=lambda x: (-vote_counts[x], x))
    
    print(f"\nConsensus optimal k: {optimal_k}")
    print(f"Votes: {dict(vote_counts)}")
    
    return optimal_k, {
        'wcss': wcss,
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'cluster_range': list(cluster_range)
    }

def train_optimized_kmeans_model(X, feature_names, optimal_k=None):
    """Train optimized K-Means with comprehensive analysis"""
    print("\n=== K-MEANS TRAINING WITH OPTIMIZATION ===")
    
    # Memory check before training
    memory_before = psutil.virtual_memory().percent
    print(f"Memory usage before training: {memory_before:.1f}%")
    
    # Determine optimal number of clusters if not provided
    if optimal_k is None:
        optimal_k, cluster_analysis = determine_optimal_clusters(X)
    else:
        print(f"Using provided optimal k: {optimal_k}")
        cluster_analysis = None
    
    # Split data for validation
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    print(f"Training: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Memory cleanup
    del X_train, X_test
    gc.collect()
    
    # Train K-Means with multiple initializations for robustness
    print(f"\nTraining K-Means with k={optimal_k}")
    print("Using multiple initializations for robustness...")
    
    # Progress tracking
    n_init_runs = 20  # Number of different initializations
    max_iter = 300    # Maximum iterations per run
    total_iterations = n_init_runs * max_iter
    
    tracker = SimpleProgressTracker(total_iterations, "K-Means Training")
    
    # Custom K-Means with progress tracking
    best_kmeans = None
    best_inertia = float('inf')
    
    print(f"Running {n_init_runs} initializations with up to {max_iter} iterations each")
    
    start_time = time.time()
    
    for init_run in range(n_init_runs):
        print(f"Initialization {init_run + 1}/{n_init_runs}...")
        
        # Create K-Means model
        kmeans = KMeans(
            n_clusters=optimal_k,
            random_state=42 + init_run,  # Different random seed for each run
            n_init=1,  # Single initialization per run (we control multiple runs)
            max_iter=max_iter,
            tol=1e-4,
            algorithm='lloyd'  # Classical K-Means algorithm
        )
        
        # Fit the model
        kmeans.fit(X_train_scaled)
        
        # Track progress
        tracker.update(init_run * max_iter + kmeans.n_iter_)
        
        # Check if this is the best model so far
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_kmeans = kmeans
            print(f"  New best inertia: {best_inertia:.3f}")
        
        # Memory check during training
        current_memory = psutil.virtual_memory().percent
        if current_memory > 90:
            print(f"WARNING: High memory usage ({current_memory:.1f}%) during training")
    
    training_time = time.time() - start_time
    print(f"Training completed at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total training time: {training_time/60:.1f} minutes")
    print(f"Best inertia (WCSS): {best_inertia:.3f}")
    print(f"Converged in {best_kmeans.n_iter_} iterations")
    
    # Evaluate model
    print(f"\n=== MODEL EVALUATION ===")
    
    # Predict on training data for internal validation
    train_clusters = best_kmeans.predict(X_train_scaled)
    train_silhouette = silhouette_score(X_train_scaled, train_clusters)
    train_calinski = calinski_harabasz_score(X_train_scaled, train_clusters)
    train_davies_bouldin = davies_bouldin_score(X_train_scaled, train_clusters)
    
    # Predict on test data
    test_clusters = best_kmeans.predict(X_test_scaled)
    test_silhouette = silhouette_score(X_test_scaled, test_clusters)
    test_calinski = calinski_harabasz_score(X_test_scaled, test_clusters)
    test_davies_bouldin = davies_bouldin_score(X_test_scaled, test_clusters)
    
    print(f"Training Metrics:")
    print(f"  Silhouette Score: {train_silhouette:.3f}")
    print(f"  Calinski-Harabasz Score: {train_calinski:.1f}")
    print(f"  Davies-Bouldin Score: {train_davies_bouldin:.3f}")
    
    print(f"Test Metrics:")
    print(f"  Silhouette Score: {test_silhouette:.3f}")
    print(f"  Calinski-Harabasz Score: {test_calinski:.1f}")
    print(f"  Davies-Bouldin Score: {test_davies_bouldin:.3f}")
    
    # Cluster analysis
    print(f"\nCluster Analysis:")
    cluster_counts = Counter(train_clusters)
    for cluster_id in sorted(cluster_counts.keys()):
        count = cluster_counts[cluster_id]
        percentage = count / len(train_clusters) * 100
        print(f"  Cluster {cluster_id}: {count:,} samples ({percentage:.1f}%)")
    
    # Cluster centers analysis
    print(f"\nCluster Centers (scaled coordinates):")
    for cluster_id, center in enumerate(best_kmeans.cluster_centers_):
        print(f"  Cluster {cluster_id}: {[f'{coord:.3f}' for coord in center]}")
    
    # Transform centers back to original scale for interpretation
    print(f"\nCluster Centers (original scale):")
    original_centers = scaler.inverse_transform(best_kmeans.cluster_centers_)
    for cluster_id, center in enumerate(original_centers):
        print(f"  Cluster {cluster_id}:")
        for feature_idx, feature_name in enumerate(feature_names):
            print(f"    {feature_name}: {center[feature_idx]:.3f}")
    
    # Final memory check
    memory_final = psutil.virtual_memory().percent
    print(f"\nMemory usage at end: {memory_final:.1f}%")
    
    # Model quality assessment
    quality_score = train_silhouette * 0.5 + (1 - train_davies_bouldin) * 0.3 + min(train_calinski / 1000, 1) * 0.2
    print(f"Overall Quality Score: {quality_score:.3f} (0-1, higher is better)")
    
    return best_kmeans, scaler, {
        'optimal_k': optimal_k,
        'inertia': best_inertia,
        'training_time': training_time,
        'train_silhouette': train_silhouette,
        'test_silhouette': test_silhouette,
        'train_calinski': train_calinski,
        'test_calinski': test_calinski,
        'train_davies_bouldin': train_davies_bouldin,
        'test_davies_bouldin': test_davies_bouldin,
        'cluster_distribution': dict(cluster_counts),
        'quality_score': quality_score,
        'cluster_analysis': cluster_analysis
    }

def test_kmeans_scenarios(kmeans_model, kmeans_scaler, feature_names):
    """Test K-Means with realistic VM scenarios"""
    print("\nTesting K-Means Scenarios...")
    
    # Test scenarios representing different VM resource utilization patterns
    # All 5 features: [memory_utilization, cpu_utilization, storage_utilization, network_utilization, workload_intensity]
    test_scenarios = [
        {
            'name': 'Low Resource VM',
            'description': 'Basic VM with minimal resource usage',
            'features': [0.2, 0.3, 0.1, 0.2, 0.1]  # Low across all metrics
        },
        {
            'name': 'Balanced VM',
            'description': 'Well-balanced resource utilization',
            'features': [0.5, 0.6, 0.4, 0.5, 0.5]  # Medium utilization across resources
        },
        {
            'name': 'CPU-Intensive VM',
            'description': 'High CPU usage, moderate other resources',
            'features': [0.4, 0.9, 0.3, 0.4, 0.8]  # High CPU and workload
        },
        {
            'name': 'Memory-Intensive VM',
            'description': 'High memory usage, moderate other resources',
            'features': [0.9, 0.5, 0.2, 0.3, 0.6]  # High memory, medium others
        },
        {
            'name': 'Storage-Intensive VM',
            'description': 'High storage usage, moderate other resources',
            'features': [0.3, 0.4, 0.9, 0.6, 0.4]  # High storage and network
        },
        {
            'name': 'Network-Intensive VM',
            'description': 'High network usage, moderate other resources',
            'features': [0.4, 0.5, 0.3, 0.9, 0.7]  # High network and workload
        },
        {
            'name': 'High Resource VM',
            'description': 'Heavy utilization across all resources',
            'features': [0.8, 0.9, 0.7, 0.8, 0.9]  # High utilization everywhere
        }
    ]
    
    for scenario in test_scenarios:
        features = [scenario['features']]
        features_scaled = kmeans_scaler.transform(features)
        
        # Predict cluster
        cluster = kmeans_model.predict(features_scaled)[0]
        
        # Calculate distance to cluster center
        distances = kmeans_model.transform(features_scaled)[0]
        distance_to_center = distances[cluster]
        
        # Calculate distance to all cluster centers for confidence
        closest_distances = sorted(distances)
        confidence = (closest_distances[1] - closest_distances[0]) / closest_distances[1] if len(closest_distances) > 1 else 1.0
        
        print(f"  {scenario['name']}:")
        print(f"    Description: {scenario['description']}")
        print(f"    Features: {scenario['features']}")
        print(f"    Cluster: {cluster}")
        print(f"    Distance to center: {distance_to_center:.3f}")
        print(f"    Confidence: {confidence:.3f}")
        print()

def save_optimized_kmeans_model(kmeans_model, scaler, feature_names, training_results):
    """Save the optimized K-Means model with comprehensive metadata"""
    print("\nSaving Optimized K-Means Model...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model components
    joblib.dump(kmeans_model, 'models/kmeans_model.joblib')
    joblib.dump(scaler, 'models/kmeans_scaler.joblib')
    joblib.dump(feature_names, 'models/kmeans_feature_names.joblib')
    
    # Save comprehensive training info
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'model': {
            'type': 'Optimized K-Means Clustering',
            'algorithm': 'Lloyd K-Means',
            'n_clusters': kmeans_model.n_clusters,
            'n_init_runs': 20,
            'max_iter': 300,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'inertia': training_results['inertia'],
            'training_time_minutes': training_results['training_time'] / 60
        },
        'performance': {
            'train_silhouette_score': training_results['train_silhouette'],
            'test_silhouette_score': training_results['test_silhouette'],
            'train_calinski_harabasz_score': training_results['train_calinski'],
            'test_calinski_harabasz_score': training_results['test_calinski'],
            'train_davies_bouldin_score': training_results['train_davies_bouldin'],
            'test_davies_bouldin_score': training_results['test_davies_bouldin'],
            'overall_quality_score': training_results['quality_score']
        },
        'clusters': {
            'optimal_k': training_results['optimal_k'],
            'cluster_distribution': training_results['cluster_distribution'],
            'cluster_centers_scaled': kmeans_model.cluster_centers_.tolist(),
            'cluster_centers_original': scaler.inverse_transform(kmeans_model.cluster_centers_).tolist()
        },
        'deployment': {
            'usage': 'VM resource clustering for load balancing',
            'api_endpoint': '/predict/vm_cluster',
            'input_format': 'vm_features: [cpu_utilization, memory_utilization, storage_utilization]',
            'output_format': 'cluster: int, distance: float, centroid: array',
            'integration': 'Ensemble with SVM for MCCVA routing'
        },
        'optimization': {
            'memory_efficient': True,
            'progress_tracking': True,
            'multiple_initializations': True,
            'comprehensive_evaluation': True
        }
    }
    
    joblib.dump(training_info, 'models/kmeans_info.joblib')
    
    print("Optimized K-Means model saved successfully!")
    print(f"Model files saved in 'models/' directory:")
    print(f"  - kmeans_model.joblib")
    print(f"  - kmeans_scaler.joblib") 
    print(f"  - kmeans_feature_names.joblib")
    print(f"  - kmeans_info.joblib")

def main():
    """Main optimized K-Means training pipeline"""
    print("RETRAIN: OPTIMIZED K-MEANS MODEL")
    print("=" * 60)
    print("Advanced VM resource clustering with comprehensive optimization")
    print("Memory efficient • Progress tracking • Multiple evaluations")
    print()
    
    start_time = datetime.now()
    
    try:
        # System resource check
        recommended_samples = check_memory_and_system()
        if recommended_samples <= 0:
            print("ERROR: Insufficient system resources for training")
            return False
        
        # Load datasets
        df = load_datasets_from_local()
        if df is None:
            return False
        
        # Apply stratified sampling if dataset is large
        if len(df) > recommended_samples:
            df = stratified_vm_sampling(df, target_samples=recommended_samples)
        
        # Prepare K-Means data
        X, feature_names, df_processed = prepare_kmeans_vm_data(df)
        
        if X is None or len(X) == 0:
            print("ERROR: No valid data prepared for K-Means training")
            return False
        
        # Train optimized K-Means
        kmeans_model, scaler, training_results = train_optimized_kmeans_model(X, feature_names)
        
        if kmeans_model is None:
            print("ERROR: K-Means training failed!")
            return False
        
        # Test scenarios
        test_kmeans_scenarios(kmeans_model, scaler, feature_names)
        
        # Save model
        save_optimized_kmeans_model(kmeans_model, scaler, feature_names, training_results)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "=" * 60)
        print("OPTIMIZED K-MEANS TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Model: {kmeans_model.n_clusters}-cluster K-Means")
        print(f"Features: {feature_names}")
        print(f"Quality Score: {training_results['quality_score']:.3f}/1.0")
        print(f"Silhouette Score: {training_results['test_silhouette']:.3f}")
        print(f"Total time: {duration:.1f} minutes")
        print(f"Ready for ensemble deployment with SVM!")
        
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nK-Means Model ready for cloud deployment!")
        print("Upload 'models/' folder to cloud server")
        print("Use K-Means for VM clustering in ensemble system")
    else:
        print("\nK-Means training failed!") 