#!/usr/bin/env python3
"""
Test script cho MCCVA Algorithm
Makespan Classification & Clustering VM Algorithm
Chạy: python test_mccva.py
"""

import requests
import json
import time

def test_mccva_algorithm():
    """Test MCCVA Algorithm với các test cases khác nhau"""
    print("🤖 Testing MCCVA Algorithm")
    print("Makespan Classification & Clustering VM Algorithm")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Small Task (Low Resource)",
            "features": [2, 4, 50, 500, 1],  # [cpu_cores, memory_gb, storage_gb, network_bandwidth, priority]
            "vm_features": [0.3, 0.2, 0.1],  # [cpu_usage, ram_usage, storage_usage]
            "expected_makespan": "small"
        },
        {
            "name": "Medium Task (Balanced)",
            "features": [8, 16, 200, 2000, 3],
            "vm_features": [0.6, 0.5, 0.4],
            "expected_makespan": "medium"
        },
        {
            "name": "Large Task (High Resource)",
            "features": [16, 32, 500, 5000, 5],
            "vm_features": [0.9, 0.8, 0.7],
            "expected_makespan": "large"
        },
        {
            "name": "CPU Intensive Task",
            "features": [12, 8, 100, 1000, 4],
            "vm_features": [0.9, 0.3, 0.2],
            "expected_makespan": "medium"
        },
        {
            "name": "Storage Intensive Task",
            "features": [4, 8, 800, 1000, 2],
            "vm_features": [0.2, 0.3, 0.9],
            "expected_makespan": "medium"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print("-" * 40)
        
        request_data = {
            "features": test_case["features"],
            "vm_features": test_case["vm_features"]
        }
        
        try:
            # Test MCCVA routing
            start_time = time.time()
            response = requests.post(
                "http://localhost/mccva/route",
                json=request_data,
                timeout=10
            )
            routing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"  ✅ MCCVA routing successful!")
                print(f"     Routing time: {routing_time:.3f}s")
                print(f"     Target VM: {result.get('target_vm', 'N/A')}")
                
                # Check MCCVA decision
                mccva_decision = result.get('mccva_decision', {})
                routing_info = result.get('routing_info', {})
                
                print(f"     Makespan: {mccva_decision.get('makespan_prediction', 'N/A')}")
                print(f"     Cluster: {mccva_decision.get('cluster_prediction', 'N/A')}")
                print(f"     Confidence: {mccva_decision.get('confidence_score', 'N/A'):.3f}")
                print(f"     Algorithm: {routing_info.get('algorithm', 'N/A')}")
                print(f"     Method: {routing_info.get('method', 'N/A')}")
                
                # Validate MCCVA logic
                makespan = mccva_decision.get('makespan_prediction')
                if makespan == test_case['expected_makespan']:
                    print(f"     ✅ Makespan prediction correct: {makespan}")
                else:
                    print(f"     ⚠️  Makespan prediction: {makespan} (expected: {test_case['expected_makespan']})")
                
                # Check VM selection logic
                target_vm = result.get('target_vm', '')
                if '8081' in target_vm and makespan == 'small':
                    print(f"     ✅ VM selection correct (small task → low load VM)")
                elif '8083' in target_vm and makespan == 'medium':
                    print(f"     ✅ VM selection correct (medium task → medium load VM)")
                elif '8085' in target_vm and makespan == 'large':
                    print(f"     ✅ VM selection correct (large task → high load VM)")
                else:
                    print(f"     ⚠️  VM selection: {target_vm}")
                
            else:
                print(f"  ❌ MCCVA routing failed: HTTP {response.status_code}")
                print(f"     Response: {response.text}")
                
        except Exception as e:
            print(f"  ❌ MCCVA routing error: {e}")
    
    print("\n" + "=" * 60)

def test_individual_components():
    """Test từng component của MCCVA"""
    print("\n🔧 Testing Individual MCCVA Components")
    print("-" * 40)
    
    # Test SVM Classification
    print("\n📊 Testing SVM Classification (Makespan Prediction):")
    try:
        response = requests.post(
            "http://localhost/predict/makespan",
            json={"features": [4, 8, 100, 1000, 3]},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ SVM working: {result.get('makespan')} (confidence: {result.get('confidence', 0):.3f})")
        else:
            print(f"  ❌ SVM failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"  ❌ SVM error: {e}")
    
    # Test K-Means Clustering
    print("\n📊 Testing K-Means Clustering (VM Clustering):")
    try:
        response = requests.post(
            "http://localhost/predict/vm_cluster",
            json={"vm_features": [0.5, 0.5, 0.5]},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ K-Means working: Cluster {result.get('cluster')}")
        else:
            print(f"  ❌ K-Means failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"  ❌ K-Means error: {e}")

def test_performance():
    """Test performance của MCCVA"""
    print("\n⚡ Testing MCCVA Performance")
    print("-" * 40)
    
    test_data = {
        "features": [4, 8, 100, 1000, 3],
        "vm_features": [0.5, 0.5, 0.5]
    }
    
    times = []
    success_count = 0
    total_requests = 10
    
    print(f"Running {total_requests} MCCVA routing requests...")
    
    for i in range(total_requests):
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost/mccva/route",
                json=test_data,
                timeout=5
            )
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
                success_count += 1
                print(f"  Request {i+1}: {times[-1]:.3f}s")
            else:
                print(f"  Request {i+1}: Failed (HTTP {response.status_code})")
                
        except Exception as e:
            print(f"  Request {i+1}: Error - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 Performance Results:")
        print(f"  Success rate: {success_count}/{total_requests} ({success_count/total_requests*100:.1f}%)")
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Min time: {min_time:.3f}s")
        print(f"  Max time: {max_time:.3f}s")
        
        if avg_time < 0.1:
            print(f"  ✅ Performance: Excellent (< 100ms)")
        elif avg_time < 0.5:
            print(f"  ✅ Performance: Good (< 500ms)")
        else:
            print(f"  ⚠️  Performance: Slow (> 500ms)")
    else:
        print(f"  ❌ No successful requests to measure performance")

def main():
    """Main test function"""
    print("🤖 MCCVA Algorithm Test Suite")
    print("Testing Makespan Classification & Clustering VM Algorithm")
    print("=" * 60)
    
    # Wait for services to be ready
    print("⏳ Waiting for services to be ready...")
    time.sleep(2)
    
    # Run tests
    test_individual_components()
    test_mccva_algorithm()
    test_performance()
    
    print("\n✅ MCCVA test completed!")
    print("\n📋 Test Summary:")
    print("   • SVM Classification: Makespan prediction")
    print("   • K-Means Clustering: VM clustering")
    print("   • MCCVA Algorithm: AI-based VM selection")
    print("   • Performance: Routing speed measurement")
    
    print("\n🌐 Access URLs:")
    print("   • MCCVA Routing: POST http://localhost/mccva/route")
    print("   • SVM Prediction: POST http://localhost/predict/makespan")
    print("   • K-Means Prediction: POST http://localhost/predict/vm_cluster")
    print("   • Health Check: http://localhost/health")

if __name__ == "__main__":
    main() 