#!/usr/bin/env python3
"""
Debug feature calculation to understand why SVM predictions are wrong
"""

import numpy as np

def calculate_features(cpu, memory, storage, network, priority):
    """Calculate derived features matching ml_service.py"""
    cpu_memory_ratio = cpu / (memory + 1e-6)
    storage_memory_ratio = storage / (memory + 1e-6)
    network_cpu_ratio = network / (cpu + 1e-6)
    resource_intensity = (cpu * memory * storage) / 1000
    priority_weighted_cpu = cpu * priority
    
    return [cpu, memory, storage, network, priority,
            cpu_memory_ratio, storage_memory_ratio, network_cpu_ratio,
            resource_intensity, priority_weighted_cpu]

def analyze_test_scenarios():
    """Analyze the actual test scenarios to understand feature distributions"""
    test_scenarios = [
        ([2, 4, 50, 500, 1], "small", "Web Server"),
        ([4, 8, 100, 1000, 3], "medium", "Database Server"),
        ([12, 32, 500, 5000, 5], "large", "ML Training"),
        ([16, 64, 800, 8000, 4], "large", "Video Rendering"),
        ([1, 2, 20, 2000, 2], "small", "API Gateway"),
        ([6, 12, 200, 1500, 3], "medium", "File Server")
    ]
    
    print("Feature Analysis of Test Scenarios:")
    print("=" * 60)
    
    for base_features, expected, name in test_scenarios:
        cpu, memory, storage, network, priority = base_features
        features = calculate_features(cpu, memory, storage, network, priority)
        
        print(f"\n{name} ({expected}):")
        print(f"  CPU: {cpu}, Memory: {memory}, Storage: {storage}, Network: {network}, Priority: {priority}")
        print(f"  CPU/Memory ratio: {features[5]:.3f}")
        print(f"  Storage/Memory ratio: {features[6]:.3f}")
        print(f"  Network/CPU ratio: {features[7]:.3f}")
        print(f"  Resource intensity: {features[8]:.1f}")
        print(f"  Priority weighted CPU: {features[9]:.1f}")
    
    # Analyze feature ranges by category
    print("\n\nFeature Ranges by Category:")
    print("=" * 40)
    
    small_features = []
    medium_features = []
    large_features = []
    
    for base_features, expected, name in test_scenarios:
        features = calculate_features(*base_features)
        if expected == "small":
            small_features.append(features)
        elif expected == "medium":
            medium_features.append(features)
        elif expected == "large":
            large_features.append(features)
    
    # Convert to numpy arrays for analysis
    small_arr = np.array(small_features)
    medium_arr = np.array(medium_features)
    large_arr = np.array(large_features)
    
    feature_names = ['cpu_cores', 'memory', 'storage', 'network_bandwidth', 'priority',
                    'cpu_memory_ratio', 'storage_memory_ratio', 'network_cpu_ratio', 
                    'resource_intensity', 'priority_weighted_cpu']
    
    print("\nSMALL workloads:")
    for i, name in enumerate(feature_names):
        if len(small_arr) > 0:
            print(f"  {name}: {small_arr[:, i].min():.3f} - {small_arr[:, i].max():.3f}")
    
    print("\nMEDIUM workloads:")
    for i, name in enumerate(feature_names):
        if len(medium_arr) > 0:
            print(f"  {name}: {medium_arr[:, i].min():.3f} - {medium_arr[:, i].max():.3f}")
    
    print("\nLARGE workloads:")
    for i, name in enumerate(feature_names):
        if len(large_arr) > 0:
            print(f"  {name}: {large_arr[:, i].min():.3f} - {large_arr[:, i].max():.3f}")

if __name__ == "__main__":
    analyze_test_scenarios() 