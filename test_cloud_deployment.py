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

def test_svm_prediction(base_url, test_data):
    """Test SVM prediction với dữ liệu mẫu"""
    try:
        url = f"{base_url}/predict"
        response = requests.post(url, json=test_data, timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def run_comprehensive_test(base_url):
    """Chạy test toàn diện cho deployed model"""
    print("🚀 CLOUD DEPLOYMENT TEST")
    print("=" * 50)
    print(f"Testing URL: {base_url}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Test 1: Health Check
    print("🏥 Test 1: Health Check")
    success, data = test_api_endpoint(base_url, "/health")
    if success:
        print("✅ Health check passed!")
        print(f"   Status: {data.get('status', 'unknown')}")
        print(f"   Model loaded: {data.get('model_loaded', False)}")
        print(f"   API version: {data.get('api_version', 'unknown')}")
    else:
        print("❌ Health check failed!")
        print(f"   Error: {data.get('error', 'Unknown error')}")
        return False
    
    print()
    
    # Test 2: Model Info
    print("📊 Test 2: Model Information")
    success, data = test_api_endpoint(base_url, "/model_info")
    if success:
        print("✅ Model info retrieved!")
        print(f"   Model type: {data.get('model_type', 'unknown')}")
        print(f"   Features: {len(data.get('feature_names', []))}")
        print(f"   Classes: {data.get('classes', [])}")
    else:
        print("⚠️  Model info not available (might not be implemented)")
    
    print()
    
    # Test 3: SVM Predictions với scenarios từ training
    print("🧪 Test 3: SVM Prediction Scenarios")
    
    test_scenarios = [
        {
            'name': 'Light Web Request (Small)',
            'data': {
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
            'data': {
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
            'data': {
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
            'data': {
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
            'data': {
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
        success, result = test_svm_prediction(base_url, scenario['data'])
        
        if success:
            prediction = result.get('prediction', 'unknown')
            confidence = result.get('confidence', 0)
            
            if prediction == scenario['expected']:
                print(f"   ✅ PASSED: {prediction} (confidence: {confidence:.2%})")
                passed_tests += 1
            else:
                print(f"   ❌ FAILED: Expected {scenario['expected']}, got {prediction}")
                print(f"      Confidence: {confidence:.2%}")
                
        else:
            print(f"   ❌ ERROR: {result.get('error', 'Unknown error')}")
    
    print()
    print(f"📊 Prediction Test Results: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
    
    # Test 4: Performance Test
    print("\n⚡ Test 4: Performance Test")
    start_time = time.time()
    
    quick_test = {
        'cpu_cores': 4,
        'memory_mb': 4096,
        'jobs_1min': 10,
        'jobs_5min': 40,
        'network_receive': 800,
        'network_transmit': 600,
        'cpu_speed': 2.8
    }
    
    num_requests = 10
    successful_requests = 0
    
    for i in range(num_requests):
        success, _ = test_svm_prediction(base_url, quick_test)
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
    
    # Final Summary
    print("\n" + "=" * 50)
    print("📋 FINAL TEST SUMMARY")
    print("=" * 50)
    
    if passed_tests == total_tests and successful_requests == num_requests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Model is working correctly on cloud")
        print("✅ API endpoints are responsive")
        print("✅ SVM predictions are accurate")
        return True
    else:
        print("⚠️  SOME TESTS FAILED")
        print(f"   Prediction accuracy: {passed_tests}/{total_tests}")
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
    
    print(f"Testing deployed model at: {base_url}")
    print("Note: Make sure Flask app is running on the target server!")
    print()
    
    # Chạy test
    try:
        success = run_comprehensive_test(base_url)
        
        if success:
            print("\n🚀 Cloud deployment is working perfectly!")
            print("📱 You can now integrate with OpenResty")
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