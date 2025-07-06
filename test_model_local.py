#!/usr/bin/env python3
"""
Test current model để hiểu vấn đề
"""

import joblib
import numpy as np

def test_current_model():
    """Test model hiện tại với test cases"""
    print("🔍 Testing Current Model")
    print("="*50)
    
    # Load models
    svm = joblib.load('models/svm_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    label_encoder = joblib.load('models/label_encoder.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    
    print(f"SVM Model: {svm}")
    print(f"Classes: {label_encoder.classes_}")
    print(f"Support vectors: {len(svm.support_vectors_)}")
    print(f"Feature names: {feature_names}")
    print()
    
    # Test cases với 5 features cơ bản
    test_cases = [
        {
            "name": "Web Server (Small)",
            "input": [2, 4, 50, 500, 1],
            "expected": "small"
        },
        {
            "name": "Database Server (Medium)", 
            "input": [4, 8, 100, 1000, 3],
            "expected": "medium"
        },
        {
            "name": "ML Training (Large)",
            "input": [12, 32, 500, 5000, 5], 
            "expected": "large"
        },
        {
            "name": "Video Rendering (Large)",
            "input": [16, 64, 800, 8000, 4],
            "expected": "large"
        }
    ]
    
    correct = 0
    total = len(test_cases)
    
    for case in test_cases:
        # Create enhanced features (10 features)
        cpu_cores, memory_gb, storage_gb, network_bandwidth, priority = case["input"]
        
        # Calculate enhanced features
        cpu_memory_ratio = cpu_cores / (memory_gb + 1e-6)
        storage_memory_ratio = storage_gb / (memory_gb + 1e-6)
        network_cpu_ratio = network_bandwidth / (cpu_cores + 1e-6)
        resource_intensity = (cpu_cores * memory_gb * storage_gb) / 1000
        priority_weighted_cpu = cpu_cores * priority
        
        # Combine all features
        enhanced_features = [
            cpu_cores, memory_gb, storage_gb, network_bandwidth, priority,
            cpu_memory_ratio, storage_memory_ratio, network_cpu_ratio,
            resource_intensity, priority_weighted_cpu
        ]
        
        # Scale features
        scaled = scaler.transform([enhanced_features])
        
        # Predict
        prediction = svm.predict(scaled)[0]
        prediction_class = label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence
        proba = svm.predict_proba(scaled)[0]
        confidence = np.max(proba)
        
        # Check accuracy
        is_correct = prediction_class == case["expected"]
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} {case['name']}")
        print(f"   Input: {case['input']}")
        print(f"   Enhanced: {enhanced_features}")
        print(f"   Expected: {case['expected']}")
        print(f"   Predicted: {prediction_class}")
        print(f"   Confidence: {confidence:.3f}")
        print()
    
    accuracy = (correct / total) * 100
    print(f"📊 Model Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    return accuracy

def analyze_training_data():
    """Phân tích training data để hiểu vấn đề"""
    print("\n🔍 Analyzing Training Data")
    print("="*50)
    
    # Load training info
    try:
        training_info = joblib.load('models/training_info.joblib')
        print("Training Information:")
        print(f"  Model Type: {training_info.get('model_type', 'Unknown')}")
        print(f"  Kernel: {training_info.get('kernel', 'Unknown')}")
        print(f"  C Parameter: {training_info.get('C', 'Unknown')}")
        print(f"  Gamma: {training_info.get('gamma', 'Unknown')}")
        print(f"  Support Vectors: {training_info.get('n_support_vectors', 'Unknown')}")
        print(f"  Classes: {training_info.get('classes', 'Unknown')}")
        print(f"  Features: {len(training_info.get('feature_names', []))}")
        print(f"  Training Timestamp: {training_info.get('timestamp', 'Unknown')}")
    except:
        print("Training info not available")
    
    # Generate sample data để xem distribution
    np.random.seed(42)
    data = []
    
    for _ in range(100):
        cpu = np.random.randint(1, 17)
        memory = np.random.randint(1, 65)
        storage = np.random.randint(10, 1001)
        network = np.random.randint(100, 10001)
        priority = np.random.randint(1, 6)
        
        # Enhanced feature calculation
        cpu_memory_ratio = cpu / (memory + 1e-6)
        storage_memory_ratio = storage / (memory + 1e-6)
        network_cpu_ratio = network / (cpu + 1e-6)
        resource_intensity = (cpu * memory * storage) / 1000
        priority_weighted_cpu = cpu * priority
        
        # Current logic (có thể sai)
        if cpu <= 2 and memory <= 4 and storage <= 50:
            makespan = 'small'
        elif cpu <= 8 and memory <= 16 and storage <= 200:
            makespan = 'medium'
        else:
            makespan = 'large'
        
        data.append([cpu, memory, storage, network, priority, makespan])
    
    # Analyze distribution
    makespans = [row[5] for row in data]
    from collections import Counter
    dist = Counter(makespans)
    
    print("\nTraining Data Distribution:")
    for makespan, count in dist.items():
        print(f"  {makespan}: {count} ({count/len(data)*100:.1f}%)")
    
    # Check specific cases
    print("\nSpecific Cases Analysis:")
    for case in data:
        cpu, memory, storage, network, priority, makespan = case
        if cpu == 4 and memory == 8 and storage == 100:  # Database Server case
            print(f"  Database-like: {case} -> {makespan}")
        elif cpu == 12 and memory == 32 and storage == 500:  # ML Training case
            print(f"  ML-like: {case} -> {makespan}")

if __name__ == "__main__":
    test_current_model()
    analyze_training_data() 