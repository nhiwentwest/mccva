#!/usr/bin/env python3
"""
Simple AI Routing Test for Cloud Server
Test AI prediction accuracy và server distribution
"""

import requests
import json
import time
from datetime import datetime

def test_ai_routing():
    """Test AI routing với các scenarios thực tế"""
    print("🧪 Testing AI Routing on Cloud Server")
    print("="*50)
    
    # Test scenarios
    test_cases = [
        {
            "name": "Web Server (Small)",
            "data": {
                "cpu_cores": 2,
                "memory": 4,
                "storage": 50,
                "network_bandwidth": 500,
                "priority": 1
            },
            "expected": "small"
        },
        {
            "name": "Database Server (Medium)",
            "data": {
                "cpu_cores": 4,
                "memory": 8,
                "storage": 100,
                "network_bandwidth": 1000,
                "priority": 3
            },
            "expected": "medium"
        },
        {
            "name": "ML Training (Large)",
            "data": {
                "cpu_cores": 12,
                "memory": 32,
                "storage": 500,
                "network_bandwidth": 5000,
                "priority": 5
            },
            "expected": "large"
        },
        {
            "name": "Video Rendering (Large)",
            "data": {
                "cpu_cores": 16,
                "memory": 64,
                "storage": 800,
                "network_bandwidth": 8000,
                "priority": 4
            },
            "expected": "large"
        }
    ]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(test_cases),
        "correct_predictions": 0,
        "incorrect_predictions": 0,
        "test_cases": [],
        "server_distribution": {},
        "average_response_time": 0
    }
    
    total_response_time = 0
    
    print(f"Running {len(test_cases)} test cases...")
    print()
    
    for i, case in enumerate(test_cases, 1):
        print(f"Test {i}: {case['name']}")
        print(f"  Input: {case['data']}")
        print(f"  Expected: {case['expected']}")
        
        try:
            start_time = time.time()
            
            # Send request to OpenResty
            response = requests.post(
                "http://localhost/routing",
                json=case['data'],
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            response_time = time.time() - start_time
            total_response_time += response_time
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('prediction', 'unknown')
                server = result.get('server', 'unknown')
                confidence = result.get('confidence', 0)
                
                is_correct = prediction == case['expected']
                
                if is_correct:
                    results["correct_predictions"] += 1
                    status = "✅"
                else:
                    results["incorrect_predictions"] += 1
                    status = "❌"
                
                print(f"  {status} Prediction: {prediction}")
                print(f"     Server: {server}")
                print(f"     Confidence: {confidence:.3f}")
                print(f"     Response time: {response_time:.3f}s")
                
                # Track server distribution
                if server not in results["server_distribution"]:
                    results["server_distribution"][server] = 0
                results["server_distribution"][server] += 1
                
                # Store test case result
                results["test_cases"].append({
                    "name": case['name'],
                    "input": case['data'],
                    "expected": case['expected'],
                    "prediction": prediction,
                    "server": server,
                    "confidence": confidence,
                    "response_time": response_time,
                    "correct": is_correct
                })
                
            else:
                print(f"  ❌ HTTP Error: {response.status_code}")
                results["incorrect_predictions"] += 1
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Request failed: {e}")
            results["incorrect_predictions"] += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results["incorrect_predictions"] += 1
        
        print()
    
    # Calculate accuracy
    total_tests = results["total_tests"]
    correct = results["correct_predictions"]
    accuracy = (correct / total_tests) * 100 if total_tests > 0 else 0
    results["accuracy"] = accuracy
    results["average_response_time"] = total_response_time / total_tests if total_tests > 0 else 0
    
    # Print summary
    print("="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    print(f"Total tests: {total_tests}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {results['incorrect_predictions']}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Average response time: {results['average_response_time']:.3f}s")
    
    print("\nServer Distribution:")
    for server, count in results["server_distribution"].items():
        percentage = (count / total_tests) * 100
        print(f"  {server}: {count} requests ({percentage:.1f}%)")
    
    # Performance assessment
    print("\nPerformance Assessment:")
    if accuracy >= 80:
        print("  🏆 Excellent AI performance!")
    elif accuracy >= 60:
        print("  ✅ Good AI performance")
    elif accuracy >= 40:
        print("  ⚠️ Moderate AI performance - needs improvement")
    else:
        print("  ❌ Poor AI performance - significant improvement needed")
    
    if results["average_response_time"] < 0.1:
        print("  ⚡ Fast response times")
    elif results["average_response_time"] < 0.5:
        print("  ✅ Acceptable response times")
    else:
        print("  ⏱️ Slow response times - check system performance")
    
    # Save results
    with open('cloud_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to cloud_test_results.json")
    return results

if __name__ == "__main__":
    test_ai_routing() 