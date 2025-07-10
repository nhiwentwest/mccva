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

def test_meta_learning_endpoints(base_url):
    """Test Meta-Learning specific endpoints"""
    print("\n🧠 Test 4: Meta-Learning Endpoints")
    
    # Test Meta-Learning direct endpoint
    print("   Testing /predict/meta_learning...")
    meta_features = [
        0.8, 0.7, 0.6,  # svm_conf, kmeans_conf, rule_conf
        0.0, 1.0, 0.0,  # svm_small, svm_medium, svm_large
        2.0, 0.3,       # cluster_id, cluster_distance_norm
        0.6, 0.4, 0.3,  # compute_intensity, memory_intensity, storage_intensity
        1.0, 0.7        # is_high_priority, resource_balance_score
    ]
    
    try:
        response = requests.post(f"{base_url}/predict/meta_learning", 
                               json={"features": meta_features}, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Meta-Learning: {result.get('makespan')} (confidence: {result.get('confidence', 0):.3f})")
            print(f"      Method: {result.get('method')}")
            print(f"      Test Accuracy: {result.get('model_info', {}).get('test_accuracy', 'Unknown')}")
        else:
            print(f"   ❌ Meta-Learning endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Meta-Learning test error: {e}")
    
    # Test Complete MCCVA pipeline
    print("\n   Testing /predict/mccva_complete...")
    complete_input = {
        "cpu_cores": 4,
        "memory_gb": 8,
        "storage_gb": 200,
        "network_bandwidth": 5000,
        "priority": 4,
        "vm_cpu_usage": 0.7,
        "vm_memory_usage": 0.6,
        "vm_storage_usage": 0.5
    }
    
    try:
        response = requests.post(f"{base_url}/predict/mccva_complete", 
                               json=complete_input, timeout=15)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ MCCVA Complete: {result.get('makespan')} (confidence: {result.get('confidence', 0):.3f})")
            print(f"      Method: {result.get('method')}")
            
            # Show stage results
            stages = result.get('stage_results', {})
            print(f"      Stage 1 (SVM): {stages.get('stage1_svm', {}).get('prediction', 'Unknown')}")
            print(f"      Stage 2 (K-Means): Cluster {stages.get('stage2_kmeans', {}).get('cluster', 'Unknown')}")
            print(f"      Stage 3 (Meta-Learning): {stages.get('stage3_metalearning', {}).get('prediction', 'Unknown')}")
            
            # Model info
            model_info = result.get('model_info', {})
            print(f"      Meta-Learning Accuracy: {model_info.get('metalearning_accuracy', 'Unknown')}")
        else:
            print(f"   ❌ MCCVA Complete endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ MCCVA Complete test error: {e}")

def convert_server_specs_to_features(server_specs):
    """Convert server specs to 10 features that match the ACTUAL trained model"""
    cpu_cores = server_specs.get('cpu_cores', 2)
    memory_mb = server_specs.get('memory_mb', 4000)
    jobs_1min = server_specs.get('jobs_1min', 1)
    jobs_5min = server_specs.get('jobs_5min', 5)
    network_receive = server_specs.get('network_receive', 1000)
    network_transmit = server_specs.get('network_transmit', 1000)
    cpu_speed = server_specs.get('cpu_speed', 2.5)
    
    # EXACT FEATURES from actual trained model:
    # ['jobs_1min', 'jobs_5min', 'memory_gb', 'cpu_cores', 'cpu_speed', 
    #  'network_receive', 'network_transmit', 'network_total', 'resource_density', 'workload_intensity']
    
    memory_gb = memory_mb / 1024
    network_total = network_receive + network_transmit
    resource_density = memory_gb / (cpu_cores + 0.1)  # Avoid division by zero
    workload_intensity = jobs_1min / (cpu_cores + 0.1)
    
    features = [
        jobs_1min,           # jobs_1min
        jobs_5min,           # jobs_5min  
        memory_gb,           # memory_gb
        cpu_cores,           # cpu_cores
        cpu_speed,           # cpu_speed
        network_receive,     # network_receive
        network_transmit,    # network_transmit
        network_total,       # network_total
        resource_density,    # resource_density
        workload_intensity   # workload_intensity
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
        print(f"   Version: {data.get('version', 'unknown')}")
        models_loaded = data.get('models_loaded', {})
        print(f"   SVM loaded: {models_loaded.get('svm', False)}")
        print(f"   K-Means loaded: {models_loaded.get('kmeans', False)}")
        print(f"   Meta-Learning loaded: {models_loaded.get('meta_learning', False)}")
        print(f"   Meta-Learning Ready: {data.get('meta_learning_ready', False)}")
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
    
    print(f"\n   📊 SVM Test Results: {passed_tests}/{total_tests} = {(passed_tests/total_tests*100):.1f}%")
    
    # Test 4: Meta-Learning Endpoints
    test_meta_learning_endpoints(base_url)
    
    print("\n" + "="*50)
    print(f"🎯 OVERALL RESULT: {'✅ PASSED' if passed_tests >= total_tests * 0.8 else '❌ FAILED'}")
    print(f"   Minimum threshold: 80% of SVM tests must pass")
    print(f"   Meta-Learning integration tested separately")
    print(f"   Completed at: {datetime.now().isoformat()}")
    
    return passed_tests >= total_tests * 0.8

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