#!/usr/bin/env python3
"""
MCCVA AI Accuracy Test - Enhanced Version
Tests both direct SVM (10 features) and enhanced endpoint (5 features)
"""
import requests
import json
import sys

def convert_5_to_10_features(features):
    """Convert 5 features to 10 features using same logic as ml_service.py"""
    cpu_cores, memory, storage, network_bandwidth, priority = features
    
    # Calculate derived features EXACTLY as in ml_service.py
    cpu_memory_ratio = cpu_cores / (memory + 1e-6)
    storage_memory_ratio = storage / (memory + 1e-6)
    network_cpu_ratio = network_bandwidth / (cpu_cores + 1e-6)
    resource_intensity = (cpu_cores * memory * storage) / 1000
    priority_weighted_cpu = priority * cpu_cores
    
    return [
        cpu_cores, memory, storage, network_bandwidth, priority,
        cpu_memory_ratio, storage_memory_ratio, network_cpu_ratio,
        resource_intensity, priority_weighted_cpu
    ]

def test_svm_direct(features_10, expected, name):
    """Test direct SVM endpoint with 10 features"""
    try:
        response = requests.post("http://localhost:5000/predict/makespan", 
                               json={"features": features_10}, timeout=5)
        result = response.json()
        prediction = result["makespan"]
        confidence = result.get("confidence", 0)
        correct = (prediction == expected)
        return prediction, confidence, correct, None
    except Exception as e:
        return None, 0, False, str(e)

def test_enhanced_endpoint(features_5, expected, name):
    """Test enhanced endpoint with 5 features"""
    try:
        response = requests.post("http://localhost:5000/predict/enhanced", 
                               json={"features": features_5}, timeout=5)
        result = response.json()
        svm_pred = result["model_contributions"]["svm"]["prediction"]
        final_pred = result["makespan"]
        confidence = result.get("confidence", 0)
        correct = (svm_pred == expected)
        return svm_pred, confidence, correct, final_pred
    except Exception as e:
        return None, 0, False, str(e)

def main():
    print("\n🧪 MCCVA AI ACCURACY TEST - ENHANCED")
    print("=" * 90)
    
    # Test cases: [5_features, expected_class, name]
    test_cases = [
        ([2, 4, 50, 500, 1], "small", "Web Server"),
        ([4, 8, 100, 1000, 3], "medium", "Database"), 
        ([12, 32, 500, 5000, 5], "large", "ML Training"),
        ([1, 2, 20, 100, 1], "small", "API Gateway"),
        ([6, 16, 200, 2000, 4], "medium", "Cache Server"),
        ([8, 24, 300, 3000, 4], "large", "Compute Node")
    ]
    
    print("TEST 1: Direct SVM Endpoint (10 features)")
    print("-" * 90)
    print(f"{'Server Type':15} | {'SVM Pred':8} | {'Confidence':10} | {'Expected':8} | {'Status'}")
    print("-" * 90)
    
    svm_correct = 0
    for features_5, expected, name in test_cases:
        features_10 = convert_5_to_10_features(features_5)
        pred, conf, correct, error = test_svm_direct(features_10, expected, name)
        
        if error:
            print(f"{name:15} | ERROR: {error}")
        else:
            print(f"{name:15} | {pred:8} | {conf:10.3f} | {expected:8} | {'✅' if correct else '❌'}")
            if correct:
                svm_correct += 1
    
    print("\nTEST 2: Enhanced Endpoint (5 features)")
    print("-" * 90)
    print(f"{'Server Type':15} | {'SVM Pred':8} | {'Final':8} | {'Expected':8} | {'Status'}")
    print("-" * 90)
    
    enhanced_correct = 0
    for features_5, expected, name in test_cases:
        pred, conf, correct, final = test_enhanced_endpoint(features_5, expected, name)
        
        if pred is None:
            print(f"{name:15} | ERROR: {final}")
        else:
            print(f"{name:15} | {pred:8} | {final:8} | {expected:8} | {'✅' if correct else '❌'}")
            if correct:
                enhanced_correct += 1
    
    # Results
    total = len(test_cases)
    svm_accuracy = (svm_correct / total) * 100
    enhanced_accuracy = (enhanced_correct / total) * 100
    
    print("-" * 90)
    print(f"\n📊 RESULTS:")
    print(f"Direct SVM Accuracy:    {svm_correct}/{total} = {svm_accuracy:.1f}%")
    print(f"Enhanced API Accuracy:  {enhanced_correct}/{total} = {enhanced_accuracy:.1f}%")
    
    if max(svm_accuracy, enhanced_accuracy) >= 80:
        print("🎉 EXCELLENT! Model is working correctly!")
        sys.exit(0)
    elif max(svm_accuracy, enhanced_accuracy) >= 60:
        print("✅ GOOD! Significant improvement achieved!")
        sys.exit(0)
    else:
        print("⚠️  Still needs improvement...")
        
        # Debug info
        print(f"\n🔧 DEBUG INFO:")
        print("Check feature conversion logic or retrain model")
        sys.exit(1)

if __name__ == "__main__":
    main()
