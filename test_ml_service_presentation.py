#!/usr/bin/env python3
"""
Test script for ML Service endpoints that match Final Presentation
Test các endpoints được cải tiến cho demo presentation
"""

import requests
import json
import time
from datetime import datetime

# Configuration
ML_SERVICE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing Health Endpoint...")
    try:
        response = requests.get(f"{ML_SERVICE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Health: {data['status']}")
            print(f"  📊 Models loaded: {data['models_loaded']}")
            return True
        else:
            print(f"  ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Health check error: {e}")
        return False

def test_ensemble_endpoint():
    """Test simplified ensemble endpoint (matches presentation)"""
    print("\n🎯 Testing Simplified Ensemble Endpoint...")
    
    # Test scenarios from presentation
    test_cases = [
        {
            "name": "Light Web Traffic",
            "features": [2, 8, 0.5, 2, 2.0, 100, 50, 150, 0.24, 0.48],
            "expected": "small"
        },
        {
            "name": "API Processing", 
            "features": [15, 60, 2.0, 4, 2.5, 500, 300, 800, 0.98, 3.66],
            "expected": "medium"
        },
        {
            "name": "Data Heavy Processing",
            "features": [45, 180, 8.0, 8, 3.0, 2000, 1500, 3500, 2.67, 11.25],
            "expected": "large"
        }
    ]
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            payload = {"features": test_case["features"]}
            response = requests.post(f"{ML_SERVICE_URL}/predict/ensemble", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                predicted = result["decision_final"]
                expected = test_case["expected"]
                is_correct = predicted == expected
                
                if is_correct:
                    correct_predictions += 1
                
                print(f"  {i}. {test_case['name']}:")
                print(f"     Expected: {expected} | Predicted: {predicted} {'✅' if is_correct else '❌'}")
                print(f"     Algorithm: {result['algorithm']}")
                print(f"     Confidence: {result['ensemble_confidence']:.3f}")
                print(f"     SVM: {result['svm_prediction']} (conf: {result['svm_confidence']:.3f})")
                
            else:
                print(f"  {i}. {test_case['name']}: ❌ Error {response.status_code}")
                
        except Exception as e:
            print(f"  {i}. {test_case['name']}: ❌ Exception: {e}")
    
    accuracy = (correct_predictions / total_tests) * 100
    print(f"\n  📊 Ensemble Accuracy: {correct_predictions}/{total_tests} = {accuracy:.1f}%")
    return accuracy >= 60  # 60% minimum for demo

def test_scenarios_endpoint():
    """Test automated scenarios endpoint"""
    print("\n🧪 Testing Automated Scenarios Endpoint...")
    try:
        response = requests.get(f"{ML_SERVICE_URL}/test/scenarios")
        if response.status_code == 200:
            data = response.json()
            summary = data["summary"]
            print(f"  ✅ Scenarios tested: {summary['total_scenarios']}")
            print(f"  📊 Accuracy: {summary['accuracy']}")
            print(f"  🎯 Correct: {summary['correct_predictions']}")
            
            # Show details
            for result in data["test_results"][:3]:  # Show first 3
                status = "✅" if result["correct"] else "❌"
                print(f"    {status} {result['scenario']}: {result['predicted']} (expected: {result['expected']})")
            
            return float(summary['accuracy'].replace('%', '')) >= 60
        else:
            print(f"  ❌ Scenarios test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Scenarios test error: {e}")
        return False

def test_performance_endpoint():
    """Test performance metrics endpoint"""
    print("\n📈 Testing Performance Metrics Endpoint...")
    try:
        response = requests.get(f"{ML_SERVICE_URL}/models/performance")
        if response.status_code == 200:
            data = response.json()
            
            print(f"  📊 Individual SVM Accuracy: {data['individual_models']['svm']['accuracy']:.1%}")
            print(f"  🤖 Individual K-Means Accuracy: {data['individual_models']['kmeans']['accuracy']:.1%}")
            print(f"  🎯 Ensemble Accuracy: {data['ensemble']['accuracy']:.1%}")
            print(f"  ⚡ Ensemble Response Time: {data['ensemble']['response_time_ms']}ms")
            print(f"  📈 Accuracy Improvement: {data['ensemble']['improvement']['accuracy']}")
            print(f"  🚀 Production Ready: {data['deployment_status']['production_ready']}")
            
            return True
        else:
            print(f"  ❌ Performance test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Performance test error: {e}")
        return False

def main():
    """Main test runner"""
    print("🧪 TESTING ML SERVICE FOR FINAL PRESENTATION")
    print("=" * 60)
    print(f"⏰ Test started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"🎯 Target URL: {ML_SERVICE_URL}")
    print()
    
    tests_passed = 0
    total_tests = 4
    
    # Run tests
    if test_health():
        tests_passed += 1
    
    if test_ensemble_endpoint():
        tests_passed += 1
        
    if test_scenarios_endpoint():
        tests_passed += 1
        
    if test_performance_endpoint():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Tests passed: {tests_passed}/{total_tests}")
    print(f"📊 Success rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🚀 ALL TESTS PASSED - Ready for presentation!")
    elif tests_passed >= 3:
        print("⚠️  Mostly ready - Minor issues detected")
    else:
        print("❌ Major issues - Need debugging before presentation")
    
    print(f"⏰ Test completed at: {datetime.now().strftime('%H:%M:%S')}")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    main() 