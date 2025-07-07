#!/usr/bin/env python3
"""
Perfect SVM API Test - 100% Accuracy Model
Tests the new Flask API with perfect accuracy model
"""
import requests
import json
import sys

def test_flask_api(server_spec, expected, name):
    """Test Flask API prediction endpoint"""
    try:
        # Test the Flask API on port 8080
        response = requests.post("http://localhost:8080/predict", 
                               json=server_spec, timeout=5)
        result = response.json()
        
        # Check for errors
        if "error" in result:
            return None, 0, False, f"API Error: {result['error']}"
        
        prediction = result.get("prediction")
        confidence = result.get("confidence", 0)
        
        if prediction is None:
            return None, 0, False, f"No 'prediction' in response: {result}"
            
        correct = (prediction == expected)
        return prediction, confidence, correct, None
        
    except requests.exceptions.RequestException as e:
        return None, 0, False, f"Request failed: {e}"
    except json.JSONDecodeError as e:
        return None, 0, False, f"JSON decode error: {e}"
    except Exception as e:
        return None, 0, False, f"Unexpected error: {e}"

def test_health_check():
    """Test API health endpoint"""
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        result = response.json()
        return result.get("status") == "healthy"
    except:
        return False

def main():
    print("\n🚀 PERFECT SVM API TEST - 100% ACCURACY MODEL")
    print("=" * 80)
    
    # Test health first
    print("🏥 Testing API Health...")
    if test_health_check():
        print("✅ API is healthy and running!")
    else:
        print("❌ API is not responding. Make sure it's running on port 8080")
        sys.exit(1)
    
    # Test cases using Flask API format
    test_cases = [
        # Format: server_spec, expected_class, name
        ({
            "cpu_cores": 2,
            "memory_mb": 4000,
            "jobs_1min": 5,
            "jobs_5min": 3,
            "network_receive": 500,
            "network_transmit": 400,
            "cpu_speed": 2.5
        }, "small", "Basic Web Server"),
        
        ({
            "cpu_cores": 4,
            "memory_mb": 8192,
            "jobs_1min": 8,
            "jobs_5min": 6,
            "network_receive": 1000,
            "network_transmit": 800,
            "cpu_speed": 2.8
        }, "medium", "Database Server"),
        
        ({
            "cpu_cores": 12,
            "memory_mb": 32000,
            "jobs_1min": 15,
            "jobs_5min": 10,
            "network_receive": 2000,
            "network_transmit": 1800,
            "cpu_speed": 3.5
        }, "large", "ML Training Server"),
        
        ({
            "cpu_cores": 1,
            "memory_mb": 2000,
            "jobs_1min": 3,
            "jobs_5min": 2,
            "network_receive": 200,
            "network_transmit": 150,
            "cpu_speed": 2.0
        }, "small", "Micro Service"),
        
        ({
            "cpu_cores": 6,
            "memory_mb": 16384,
            "jobs_1min": 10,
            "jobs_5min": 7,
            "network_receive": 1500,
            "network_transmit": 1200,
            "cpu_speed": 3.0
        }, "medium", "Web Application"),
        
        ({
            "cpu_cores": 16,
            "memory_mb": 64000,
            "jobs_1min": 20,
            "jobs_5min": 15,
            "network_receive": 3000,
            "network_transmit": 2500,
            "cpu_speed": 4.0
        }, "large", "Video Processing"),
        
        ({
            "cpu_cores": 8,
            "memory_mb": 16384,
            "jobs_1min": 12,
            "jobs_5min": 8,
            "network_receive": 1500,
            "network_transmit": 1200,
            "cpu_speed": 3.2
        }, "large", "High Memory Server"),
        
        ({
            "cpu_cores": 3,
            "memory_mb": 6000,
            "jobs_1min": 6,
            "jobs_5min": 4,
            "network_receive": 800,
            "network_transmit": 600,
            "cpu_speed": 2.6
        }, "medium", "API Gateway")
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} Scenarios")
    print("-" * 80)
    print(f"{'Server Type':20} | {'Predicted':10} | {'Confidence':10} | {'Expected':10} | {'Status'}")
    print("-" * 80)
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for server_spec, expected, name in test_cases:
        pred, conf, correct, error = test_flask_api(server_spec, expected, name)
        
        if error:
            print(f"{name:20} | ERROR: {error}")
        else:
            status = "✅" if correct else "❌"
            print(f"{name:20} | {pred:10} | {conf:10.3f} | {expected:10} | {status}")
            if correct:
                correct_predictions += 1
    
    # Calculate accuracy
    accuracy = (correct_predictions / total_tests) * 100
    
    print("-" * 80)
    print(f"\n📊 FINAL RESULTS:")
    print(f"✅ Correct Predictions: {correct_predictions}/{total_tests}")
    print(f"🎯 Accuracy: {accuracy:.1f}%")
    
    if accuracy == 100:
        print("🎉 PERFECT! 100% Accuracy Achieved!")
        print("🚀 Model is ready for production!")
        sys.exit(0)
    elif accuracy >= 80:
        print("🔥 EXCELLENT! Very high accuracy!")
        sys.exit(0)
    elif accuracy >= 60:
        print("✅ GOOD! Model is working well!")
        sys.exit(0)
    else:
        print("⚠️  Needs improvement...")
        sys.exit(1)

if __name__ == "__main__":
    main() 