#!/usr/bin/env python3
"""
Optimized K-Means Training for MCCVA VM Clustering
- Processes full dataset from dataset/modified
- Uses silhouette score analysis to determine optimal number of clusters
- Implements stratified sampling to preserve resource distribution
- Optimized for memory efficiency and performance
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
    """Stratified sampling for VM clustering - preserves resource distribution"""
    print(f"\n=== STRATIFIED VM SAMPLING ===")
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
    
    return sampled_df

def load_datasets():
    """Load datasets from Excel files"""
    print("\n=== LOADING DATASETS ===")
    
    # Tạo thư mục models nếu chưa có
    os.makedirs("models", exist_ok=True)
    
    # Load toàn bộ dataset
    dataset_dir = "dataset/modified"
    
    all_dfs = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".xlsx"):
            try:
                file_path = os.path.join(dataset_dir, file)
                df = pd.read_excel(file_path)
                all_dfs.append(df)
                print(f"Loaded {len(df)} samples from {file}")
            except Exception as e:
                print(f"Error reading file {file}: {e}")
    
    if not all_dfs:
        print("No dataset files found!")
        sys.exit(1)
    
    # Kết hợp tất cả dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal samples: {len(df)}")
    
    return df

def prepare_kmeans_data(df):
    """Prepare data for K-Means clustering"""
    print("\n=== PREPARING DATA FOR K-MEANS ===")
    
    # Select numeric columns for clustering
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Remove any target columns if they exist
    exclude_cols = ['Class_Name', 'Type', 'target', 'label', 'cluster']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"Selected {len(feature_cols)} numeric features for clustering")
    
    # Extract features
    X = df[feature_cols].copy()
    
    # Check for NaN values
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values, replacing with mean values")
        X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    
    return X_scaled, scaler, feature_cols

def determine_optimal_clusters(X, max_clusters=10):
    """Determine optimal number of clusters using silhouette score"""
    print("\n=== DETERMINING OPTIMAL NUMBER OF CLUSTERS ===")
    
    # Initialize variables to track best results
    best_k = 5  # Default value
    best_score = -1
    results = {}
    
    # Progress tracker
    progress = SimpleProgressTracker(max_clusters - 1, "Cluster Analysis")
    
    # Test different numbers of clusters
    for k in range(2, max_clusters + 1):
        progress.update(k - 1)
        
        # Use efficient KMeans implementation with early stopping
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-4,
            random_state=42,
            algorithm='auto'
        )
        
        # Fit model
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        try:
            # Use sample of data for silhouette calculation if dataset is large
            if X.shape[0] > 10000:
                sample_indices = np.random.choice(X.shape[0], 10000, replace=False)
                silhouette_avg = silhouette_score(X[sample_indices], kmeans.labels_[sample_indices])
            else:
                silhouette_avg = silhouette_score(X, kmeans.labels_)
                
            # Calculate other metrics
            calinski_avg = calinski_harabasz_score(X, kmeans.labels_)
            davies_avg = davies_bouldin_score(X, kmeans.labels_)
            
            # Store results
            results[k] = {
                'silhouette': silhouette_avg,
                'calinski_harabasz': calinski_avg,
                'davies_bouldin': davies_avg,
                'inertia': kmeans.inertia_
            }
            
            print(f"  k={k}: Silhouette={silhouette_avg:.3f}, CH={calinski_avg:.1f}, DB={davies_avg:.3f}")
            
            # Update best score
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_k = k
                
        except Exception as e:
            print(f"  Error calculating metrics for k={k}: {e}")
    
    print(f"\nOptimal number of clusters: k={best_k} (silhouette score: {best_score:.3f})")
    
    # Free memory
    gc.collect()
    
    return best_k, results

def train_kmeans_model(X, feature_names, optimal_k=None):
    """Train K-Means model with optimal parameters"""
    print(f"\n=== TRAINING K-MEANS MODEL (k={optimal_k}) ===")
    
    # Use default k=5 if optimal_k is not provided
    if optimal_k is None:
        optimal_k = 5
        print("Using default k=5 clusters")
    
    # Initialize KMeans with optimal parameters
    kmeans = KMeans(
        n_clusters=optimal_k,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=42,
        algorithm='auto'
    )
    
    # Train model
    start_time = time.time()
    kmeans.fit(X)
    training_time = time.time() - start_time
    
    # Get cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Analyze clusters
    cluster_counts = Counter(labels)
    total_samples = len(labels)
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final inertia: {kmeans.inertia_:.2f}")
    
    print("\nCluster distribution:")
    for cluster_id, count in sorted(cluster_counts.items()):
        percentage = (count / total_samples) * 100
        print(f"  Cluster {cluster_id}: {count} samples ({percentage:.1f}%)")
    
    # Analyze cluster centers
    print("\nCluster centers (top features):")
    for i, center in enumerate(cluster_centers):
        # Get top influential features for this cluster
        feature_importance = [(feature_names[j], center[j]) for j in range(len(feature_names))]
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_features = feature_importance[:3]  # Top 3 features
        print(f"  Cluster {i}: " + ", ".join([f"{name}={value:.2f}" for name, value in top_features]))
    
    # Calculate silhouette score on sample
    if X.shape[0] > 10000:
        sample_indices = np.random.choice(X.shape[0], 10000, replace=False)
        silhouette_avg = silhouette_score(X[sample_indices], kmeans.labels_[sample_indices])
    else:
        silhouette_avg = silhouette_score(X, kmeans.labels_)
    
    print(f"\nFinal silhouette score: {silhouette_avg:.3f}")
    
    # Prepare results
    training_results = {
        'n_clusters': optimal_k,
        'inertia': float(kmeans.inertia_),
        'silhouette_score': float(silhouette_avg),
        'training_time': float(training_time),
        'n_samples': int(total_samples),
        'n_features': len(feature_names),
        'cluster_distribution': {int(k): int(v) for k, v in cluster_counts.items()},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return kmeans, training_results

def save_kmeans_model(kmeans_model, scaler, feature_names, training_results):
    """Save K-Means model and related artifacts"""
    print("\n=== SAVING K-MEANS MODEL ===")
    
    # Create directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model
    joblib.dump(kmeans_model, "models/kmeans_model.joblib")
    print("Saved model to models/kmeans_model.joblib")
    
    # Save scaler
    joblib.dump(scaler, "models/kmeans_scaler.joblib")
    print("Saved scaler to models/kmeans_scaler.joblib")
    
    # Save feature names
    joblib.dump(feature_names, "models/kmeans_feature_names.joblib")
    print("Saved feature names to models/kmeans_feature_names.joblib")
    
    # Save model info
    joblib.dump(training_results, "models/kmeans_info.joblib")
    print("Saved model info to models/kmeans_info.joblib")
    
    print("\nModel saving completed")

def main():
    """Main function"""
    print("=== OPTIMIZED K-MEANS TRAINING FOR MCCVA VM CLUSTERING ===")
    start_time = time.time()
    
    # Check system resources
    recommended_samples = check_memory_and_system()
    
    try:
        # Load datasets
        df = load_datasets()
        
        # Sample data if needed
        if len(df) > recommended_samples:
            df = stratified_vm_sampling(df, target_samples=recommended_samples)
        
        # Prepare data for K-Means
        X, scaler, feature_names = prepare_kmeans_data(df)
        
        # Determine optimal number of clusters
        optimal_k, cluster_results = determine_optimal_clusters(X, max_clusters=10)
        
        # Train K-Means model
        kmeans_model, training_results = train_kmeans_model(X, feature_names, optimal_k)
        
        # Save model and artifacts
        save_kmeans_model(kmeans_model, scaler, feature_names, training_results)
        
        # Final report
        total_time = time.time() - start_time
        print(f"\n=== TRAINING COMPLETED IN {total_time/60:.2f} MINUTES ===")
        print(f"Optimal clusters: {optimal_k}")
        print(f"Silhouette score: {training_results['silhouette_score']:.3f}")
        print(f"Total samples processed: {len(df)}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up resources
        cleanup_resources()

if __name__ == "__main__":
    main() 