#!/usr/bin/env python3
"""
MCCVA AI Accuracy Test - Enhanced Version
Tests both direct SVM (10 features) and enhanced endpoint (5 features)
"""
import requests
import json
import sys

def convert_5_to_10_features(features):
    """Convert 5 features to 10 features matching direct SVM API format"""
    cpu_cores, memory, storage, network_bandwidth, priority = features
    
    # Generate the missing 5 features based on realistic mappings
    # task_complexity (1-5): based on cpu cores 
    task_complexity = min(5, max(1, (cpu_cores // 2) + 1))
    
    # data_size (1-1000): based on storage
    data_size = min(1000, max(1, storage // 10))
    
    # io_intensity (1-100): based on storage and network
    io_intensity = min(100, max(1, (storage + network_bandwidth) // 100))
    
    # parallel_degree (100-2000): based on cpu cores and priority
    parallel_degree = min(2000, max(100, cpu_cores * 100 + priority * 50))
    
    # deadline_urgency (1-5): same as priority
    deadline_urgency = priority
    
    return [
        cpu_cores, memory, storage, network_bandwidth, priority,
        task_complexity, data_size, io_intensity, parallel_degree, deadline_urgency
    ]

def test_svm_direct(features_10, expected, name):
    """Test direct SVM endpoint with 10 features"""
    try:
        response = requests.post("http://localhost:5000/predict/makespan", 
                               json={"features": features_10}, timeout=5)
        result = response.json()
        
        # Debug: print full response if error
        if "error" in result:
            return None, 0, False, f"API Error: {result['error']}"
        
        prediction = result.get("makespan")
        if prediction is None:
            return None, 0, False, f"No 'makespan' in response: {result}"
            
        confidence = result.get("confidence", 0)
        correct = (prediction == expected)
        return prediction, confidence, correct, None
    except requests.exceptions.RequestException as e:
        return None, 0, False, f"Request failed: {e}"
    except json.JSONDecodeError as e:
        return None, 0, False, f"JSON decode error: {e}"
    except Exception as e:
        return None, 0, False, f"Unexpected error: {e}"

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
