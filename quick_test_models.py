#!/usr/bin/env python3
"""
Quick Model Testing Script
Test nhanh models trước khi deploy lên cloud
"""

import joblib
import numpy as np
import json
import os

def test_models():
    """Test tất cả models đã train"""
    print("🚀 Quick Model Testing")
    print("="*40)
    
    # Test cases từ real scenarios
    test_cases = [
        # Small tasks
        {"name": "Web Server", "features": [2, 4, 50, 500, 1], "expected": "small"},
        {"name": "Light API", "features": [1, 2, 30, 300, 1], "expected": "small"},
        
        # Medium tasks
        {"name": "Database Server", "features": [4, 8, 100, 1000, 3], "expected": "medium"},
        {"name": "Application Server", "features": [6, 12, 150, 1200, 3], "expected": "medium"},
        
        # Large tasks
        {"name": "Big Data Processing", "features": [8, 16, 200, 2000, 5], "expected": "large"},
        {"name": "ML Training", "features": [12, 32, 500, 5000, 5], "expected": "large"},
        {"name": "Video Rendering", "features": [16, 64, 800, 8000, 4], "expected": "large"},
    ]
    
    models_to_test = ['svm', 'rf', 'ensemble']
    results = {}
    
    for model_name in models_to_test:
        model_file = f'models/{model_name}_model_improved.joblib'
        scaler_file = 'models/scaler_improved.joblib'
        
        if not os.path.exists(model_file):
            print(f"❌ {model_name} model not found: {model_file}")
            continue
            
        print(f"\n📊 Testing {model_name.upper()} Model:")
        print("-" * 30)
        
        try:
            # Load model and scaler
            model = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
            
            correct = 0
            total = len(test_cases)
            
            for case in test_cases:
                # Extract enhanced features (same as training)
                cpu, memory, storage, network, priority = case["features"]
                
                enhanced_features = [
                    cpu, memory, storage, network, priority,  # Original features
                    cpu / memory if memory > 0 else 0,  # compute_intensity
                    storage / memory if memory > 0 else 0,  # storage_intensity
                    network / cpu if cpu > 0 else 0,  # network_intensity
                    (cpu * memory) / storage if storage > 0 else 0,  # resource_ratio
                    priority / 5.0,  # priority_weight
                    int(cpu / memory > 0.5 if memory > 0 else 0),  # is_compute_intensive
                    int(memory > 16),  # is_memory_intensive
                    int(storage > 500),  # is_storage_intensive
                    int(network > 5000),  # is_network_intensive
                    int(priority >= 4),  # high_priority
                    int(priority <= 2),  # low_priority
                    int(abs(cpu - memory/4) < 2),  # balanced_resources
                    int(storage > (cpu * memory * 2)),  # storage_heavy
                    int(network > (cpu * 1000))  # network_heavy
                ]
                
                # Scale features
                features_scaled = scaler.transform([enhanced_features])
                
                # Predict
                prediction = model.predict(features_scaled)[0]
                
                # Get confidence (if available)
                try:
                    if hasattr(model, 'predict_proba'):
                        confidence = max(model.predict_proba(features_scaled)[0])
                    elif hasattr(model, 'decision_function'):
                        confidence = abs(model.decision_function(features_scaled)[0])
                    else:
                        confidence = 1.0
                except:
                    confidence = 1.0
                
                is_correct = prediction == case["expected"]
                if is_correct:
                    correct += 1
                
                status = "✅" if is_correct else "❌"
                print(f"{status} {case['name']}: {prediction} (expected: {case['expected']}, conf: {confidence:.3f})")
            
            accuracy = (correct / total) * 100
            print(f"\n{model_name.upper()} Accuracy: {accuracy:.1f}% ({correct}/{total})")
            
            results[model_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Summary
    print("\n" + "="*40)
    print("📊 TEST SUMMARY")
    print("="*40)
    
    best_model = None
    best_accuracy = 0
    
    for model_name, result in results.items():
        if 'accuracy' in result:
            print(f"{model_name.upper()}: {result['accuracy']:.1f}%")
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model = model_name
        else:
            print(f"{model_name.upper()}: ERROR - {result.get('error', 'Unknown error')}")
    
    if best_model:
        print(f"\n🏆 Best Model: {best_model.upper()} ({best_accuracy:.1f}%)")
        
        if best_accuracy >= 70:
            print("✅ Model ready for deployment!")
        elif best_accuracy >= 50:
            print("⚠️ Model needs improvement before deployment")
        else:
            print("❌ Model needs significant improvement")
    
    # Save results
    with open('quick_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to quick_test_results.json")
    return results

if __name__ == "__main__":
    test_models() 