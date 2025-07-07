#!/usr/bin/env python3
"""
Cloud Deployment Test Script
Kiểm tra mô hình SVM đã huấn luyện sau khi deploy lên cloud
Usage: python3 test_cloud_deployment.py [cloud_ip]
"""

import requests
import json
import sys
import time
from datetime import datetime

def test_api_endpoint(base_url, endpoint="/health"):
    """Test một endpoint cụ thể"""
    try:
        url = f"{base_url}{endpoint}"
        response = requests.get(url, timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def test_svm_prediction(base_url, test_features):
    """Test SVM prediction với dữ liệu mẫu thông qua ml_service.py"""
    try:
        url = f"{base_url}/predict/makespan"
        # ml_service.py expects {"features": [10 features]}
        data = {"features": test_features}
        response = requests.post(url, json=data, timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def convert_server_specs_to_features(server_specs):
    """Convert server specs to 10 features for ml_service.py"""
    cpu_cores = server_specs.get('cpu_cores', 2)
    memory_gb = server_specs.get('memory_mb', 4000) / 1024  # Convert MB to GB
    jobs_1min = server_specs.get('jobs_1min', 1)
    jobs_5min = server_specs.get('jobs_5min', 5)
    network_receive = server_specs.get('network_receive', 1000)
    network_transmit = server_specs.get('network_transmit', 1000)
    cpu_speed = server_specs.get('cpu_speed', 2.5)
    
    # Generate 10 features for ml_service.py
    # [cpu_cores, memory_gb, storage_gb, network_bandwidth, priority, task_complexity, data_size, io_intensity, parallel_degree, deadline_urgency]
    features = [
        cpu_cores,                                          # cpu_cores
        memory_gb,                                         # memory_gb  
        min(max(memory_gb * 10, 10), 1000),              # storage_gb (estimated)
        network_receive + network_transmit,                # network_bandwidth
        min(max(int(cpu_cores / 2), 1), 5),              # priority (estimated)
        min(max(int((jobs_1min + jobs_5min) / 5), 1), 5), # task_complexity
        min(max(jobs_1min * 10, 1), 1000),               # data_size
        min(max(int(network_receive / 100), 1), 100),     # io_intensity
        min(max(int(cpu_cores * cpu_speed * 100), 100), 2000), # parallel_degree
        min(max(int(jobs_1min / 10), 1), 5)              # deadline_urgency
    ]
    
    return features

def run_comprehensive_test(base_url):
    """Chạy test toàn diện cho deployed ML service"""
    print("🚀 CLOUD DEPLOYMENT TEST - ML SERVICE")
    print("=" * 50)
    print(f"Testing URL: {base_url}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("Using ml_service.py endpoints")
    print()
    
    # Test 1: Health Check
    print("🏥 Test 1: Health Check")
    success, data = test_api_endpoint(base_url, "/health")
    if success:
        print("✅ Health check passed!")
        print(f"   Status: {data.get('status', 'unknown')}")
        print(f"   Service: {data.get('service', 'unknown')}")
        print(f"   SVM loaded: {data.get('models_loaded', {}).get('svm', False)}")
        print(f"   K-Means loaded: {data.get('models_loaded', {}).get('kmeans', False)}")
    else:
        print("❌ Health check failed!")
        print(f"   Error: {data.get('error', 'Unknown error')}")
        return False
    
    print()
    
    # Test 2: Models Info
    print("📊 Test 2: Models Information")
    success, data = test_api_endpoint(base_url, "/models/info")
    if success:
        print("✅ Models info retrieved!")
        svm_info = data.get('svm_model', {})
        kmeans_info = data.get('kmeans_model', {})
        print(f"   SVM kernel: {svm_info.get('kernel', 'unknown')}")
        print(f"   SVM classes: {svm_info.get('classes', [])}")
        print(f"   K-Means clusters: {kmeans_info.get('n_clusters', 'unknown')}")
    else:
        print("⚠️  Models info not available")
    
    print()
    
    # Test 3: SVM Predictions với scenarios từ training
    print("🧪 Test 3: SVM Prediction Scenarios (via ml_service.py)")
    
    test_scenarios = [
        {
            'name': 'Light Web Request (Small)',
            'specs': {
                'cpu_cores': 2,
                'memory_mb': 512,
                'jobs_1min': 2,
                'jobs_5min': 8,
                'network_receive': 100,
                'network_transmit': 50,
                'cpu_speed': 2.4
            },
            'expected': 'small'
        },
        {
            'name': 'Medium API Call (Medium)',
            'specs': {
                'cpu_cores': 4,
                'memory_mb': 2048,
                'jobs_1min': 15,
                'jobs_5min': 60,
                'network_receive': 500,
                'network_transmit': 300,
                'cpu_speed': 3.2
            },
            'expected': 'medium'
        },
        {
            'name': 'Heavy Processing (Large)',
            'specs': {
                'cpu_cores': 8,
                'memory_mb': 8192,
                'jobs_1min': 45,
                'jobs_5min': 180,
                'network_receive': 2000,
                'network_transmit': 1500,
                'cpu_speed': 3.6
            },
            'expected': 'large'
        },
        {
            'name': 'Edge Case - High CPU Only',
            'specs': {
                'cpu_cores': 12,
                'memory_mb': 1024,
                'jobs_1min': 5,
                'jobs_5min': 20,
                'network_receive': 200,
                'network_transmit': 100,
                'cpu_speed': 3.0
            },
            'expected': 'large'  # High CPU should trigger large
        },
        {
            'name': 'Edge Case - High Memory Only',
            'specs': {
                'cpu_cores': 2,
                'memory_mb': 16384,  # 16GB
                'jobs_1min': 3,
                'jobs_5min': 12,
                'network_receive': 150,
                'network_transmit': 75,
                'cpu_speed': 2.8
            },
            'expected': 'large'  # High memory should trigger large
        }
    ]
    
    passed_tests = 0
    total_tests = len(test_scenarios)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n   Scenario {i}: {scenario['name']}")
        
        # Convert server specs to 10 features
        features = convert_server_specs_to_features(scenario['specs'])
        print(f"   Features: {features}")
        
        success, result = test_svm_prediction(base_url, features)
        
        if success:
            prediction = result.get('makespan', 'unknown')
            confidence = result.get('confidence', 0)
            
            if prediction == scenario['expected']:
                print(f"   ✅ PASSED: {prediction} (confidence: {confidence:.3f})")
                passed_tests += 1
            else:
                print(f"   ❌ FAILED: Expected {scenario['expected']}, got {prediction}")
                print(f"      Confidence: {confidence:.3f}")
                
        else:
            print(f"   ❌ ERROR: {result.get('error', 'Unknown error')}")
    
    print()
    print(f"📊 Prediction Test Results: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
    
    # Test 4: Performance Test
    print("\n⚡ Test 4: Performance Test")
    start_time = time.time()
    
    quick_test_specs = {
        'cpu_cores': 4,
        'memory_mb': 4096,
        'jobs_1min': 10,
        'jobs_5min': 40,
        'network_receive': 800,
        'network_transmit': 600,
        'cpu_speed': 2.8
    }
    
    quick_test_features = convert_server_specs_to_features(quick_test_specs)
    
    num_requests = 10
    successful_requests = 0
    
    for i in range(num_requests):
        success, _ = test_svm_prediction(base_url, quick_test_features)
        if success:
            successful_requests += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_requests * 1000  # ms
    
    print(f"   Requests: {successful_requests}/{num_requests} successful")
    print(f"   Average response time: {avg_time:.1f}ms")
    print(f"   Total time: {total_time:.2f}s")
    
    if avg_time < 1000 and successful_requests == num_requests:
        print("   ✅ Performance test passed!")
    else:
        print("   ⚠️  Performance could be improved")
    
    # Test 5: K-Means VM Clustering (if available)
    print("\n🔄 Test 5: K-Means VM Clustering")
    vm_test_data = {"vm_features": [0.6, 0.7, 0.5]}  # cpu, ram, storage usage
    
    try:
        url = f"{base_url}/predict/vm_cluster"
        response = requests.post(url, json=vm_test_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            cluster = result.get('cluster', 'unknown')
            distance = result.get('distance', 0)
            print(f"   ✅ K-Means test passed: Cluster {cluster} (distance: {distance:.3f})")
        else:
            print(f"   ⚠️  K-Means test failed: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  K-Means not available: {e}")
    
    # Final Summary
    print("\n" + "=" * 50)
    print("📋 FINAL TEST SUMMARY")
    print("=" * 50)
    
    if passed_tests == total_tests and successful_requests == num_requests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ ML Service is working correctly on cloud")
        print("✅ SVM model predictions are accurate")
        print("✅ API endpoints are responsive")
        print("✅ Ready for OpenResty integration")
        return True
    else:
        print("⚠️  SOME TESTS FAILED")
        print(f"   SVM prediction accuracy: {passed_tests}/{total_tests}")
        print(f"   API reliability: {successful_requests}/{num_requests}")
        return False

def main():
    """Main function"""
    # Get cloud IP từ command line hoặc sử dụng localhost
    if len(sys.argv) > 1:
        cloud_ip = sys.argv[1]
    else:
        cloud_ip = "localhost"
    
    # Xây dựng base URL
    if cloud_ip in ["localhost", "127.0.0.1"]:
        base_url = f"http://{cloud_ip}:5000"
    else:
        base_url = f"http://{cloud_ip}:5000"  # Hoặc port khác tùy setup
    
    print(f"Testing deployed ML Service at: {base_url}")
    print("Note: Make sure ml_service.py is running on the target server!")
    print()
    
    # Chạy test
    try:
        success = run_comprehensive_test(base_url)
        
        if success:
            print("\n🚀 Cloud deployment is working perfectly!")
            print("📱 ML Service ready for OpenResty integration")
            print("🔗 Use endpoints: /predict/makespan, /predict/vm_cluster")
            sys.exit(0)
        else:
            print("\n❌ Cloud deployment has issues")
            print("🔧 Check server logs and model files")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 