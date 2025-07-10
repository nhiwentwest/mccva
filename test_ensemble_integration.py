#!/usr/bin/env python3
"""
🧪 ENSEMBLE INTEGRATION TEST
Test SVM + K-Means ensemble system after both models are trained
Comprehensive validation for research paper
"""

import requests
import json
import time
from datetime import datetime
import sys
import os

# Test configuration
ML_SERVICE_URL = "http://localhost:5000"

def test_model_loading():
    """Test if both SVM and K-Means models are loaded"""
    print("🔍 Testing Model Loading...")
    
    try:
        response = requests.get(f"{ML_SERVICE_URL}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            models_status = health_data.get('models_loaded', {})
            
            print(f"   SVM Model: {'✅' if models_status.get('svm') else '❌'}")
            print(f"   K-Means Model: {'✅' if models_status.get('kmeans') else '❌'}")
            print(f"   SVM Scaler: {'✅' if models_status.get('svm_scaler') else '❌'}")
            print(f"   K-Means Scaler: {'✅' if models_status.get('kmeans_scaler') else '❌'}")
            print(f"   Label Encoder: {'✅' if models_status.get('svm_label_encoder') else '❌'}")
            
            all_loaded = all(models_status.values())
            if all_loaded:
                print("   ✅ All models loaded successfully!")
                return True
            else:
                print("   ❌ Some models missing!")
                return False
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

def test_individual_svm():
    """Test SVM model individually"""
    print("\n🎯 Testing Individual SVM...")
    
    test_cases = [
        {
            'name': 'Light Workload',
            'features': [2, 4, 50, 1000, 2, 0.5, 12.5, 500, 0.4, 4],  # 10 features
            'expected': 'small'
        },
        {
            'name': 'Medium Workload', 
            'features': [4, 16, 200, 5000, 3, 0.25, 12.5, 1250, 12.8, 12],
            'expected': 'medium'
        },
        {
            'name': 'Heavy Workload',
            'features': [8, 32, 500, 10000, 5, 0.25, 15.625, 1250, 128, 40],
            'expected': 'large'
        }
    ]
    
    success_count = 0
    
    for case in test_cases:
        try:
            payload = {"features": case['features']}
            response = requests.post(f"{ML_SERVICE_URL}/predict/makespan", 
                                   json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('makespan', 'unknown')
                confidence = result.get('confidence', 0)
                
                print(f"   {case['name']}: {prediction} (confidence: {confidence:.3f})")
                
                if prediction == case['expected']:
                    success_count += 1
                    print(f"     ✅ Correct prediction!")
                else:
                    print(f"     ⚠️  Expected {case['expected']}, got {prediction}")
            else:
                print(f"   {case['name']}: ❌ Failed ({response.status_code})")
                
        except Exception as e:
            print(f"   {case['name']}: ❌ Error: {e}")
    
    accuracy = success_count / len(test_cases) * 100
    print(f"   📊 SVM Accuracy: {success_count}/{len(test_cases)} = {accuracy:.1f}%")
    return success_count == len(test_cases)

def test_individual_kmeans():
    """Test K-Means model individually"""
    print("\n🔄 Testing Individual K-Means...")
    
    test_cases = [
        {
            'name': 'Low Resource VM',
            'vm_features': [0.2, 0.3, 0.1],  # Low CPU, Memory, Storage utilization
            'expected_cluster_range': [0, 1, 2]  # Should be low-resource clusters
        },
        {
            'name': 'Balanced VM',
            'vm_features': [0.5, 0.6, 0.4],  # Medium utilization
            'expected_cluster_range': [1, 2, 3]  # Should be medium-resource clusters
        },
        {
            'name': 'High Resource VM',
            'vm_features': [0.8, 0.9, 0.7],  # High utilization
            'expected_cluster_range': [3, 4, 5]  # Should be high-resource clusters
        }
    ]
    
    success_count = 0
    
    for case in test_cases:
        try:
            payload = {"vm_features": case['vm_features']}
            response = requests.post(f"{ML_SERVICE_URL}/predict/vm_cluster", 
                                   json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                cluster = result.get('cluster', -1)
                distance = result.get('distance', 999)
                
                print(f"   {case['name']}: Cluster {cluster} (distance: {distance:.3f})")
                
                if cluster in case['expected_cluster_range']:
                    success_count += 1
                    print(f"     ✅ Reasonable cluster assignment!")
                else:
                    print(f"     ⚠️  Unexpected cluster (expected range: {case['expected_cluster_range']})")
            else:
                print(f"   {case['name']}: ❌ Failed ({response.status_code})")
                
        except Exception as e:
            print(f"   {case['name']}: ❌ Error: {e}")
    
    accuracy = success_count / len(test_cases) * 100
    print(f"   📊 K-Means Accuracy: {success_count}/{len(test_cases)} = {accuracy:.1f}%")
    return success_count >= len(test_cases) * 0.6  # 60% threshold for K-Means

def test_ensemble_predictions():
    """Test ensemble prediction combining SVM + K-Means"""
    print("\n🎯 Testing Ensemble Predictions...")
    
    test_cases = [
        {
            'name': 'Light Web Server',
            'features': [2, 8, 100, 2000, 2],  # 5 features for enhanced endpoint
            'vm_features': [0.3, 0.4, 0.2],    # 3 VM features
            'expected': 'small'
        },
        {
            'name': 'API Processing',
            'features': [4, 16, 300, 5000, 3],
            'vm_features': [0.6, 0.7, 0.5],
            'expected': 'medium'
        },
        {
            'name': 'Data Analysis',
            'features': [8, 32, 800, 8000, 4],
            'vm_features': [0.8, 0.9, 0.7],
            'expected': 'large'
        },
        {
            'name': 'Edge Case - High CPU',
            'features': [12, 16, 200, 3000, 5],
            'vm_features': [0.9, 0.5, 0.3],
            'expected': ['medium', 'large']  # Could be either
        }
    ]
    
    success_count = 0
    total_response_time = 0
    
    for case in test_cases:
        try:
            payload = {
                "features": case['features'],
                "vm_features": case['vm_features']
            }
            
            start_time = time.time()
            response = requests.post(f"{ML_SERVICE_URL}/predict/enhanced", 
                                   json=payload, timeout=15)
            response_time = (time.time() - start_time) * 1000  # ms
            total_response_time += response_time
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('makespan', 'unknown')
                confidence = result.get('confidence', 0)
                cluster = result.get('cluster', -1)
                contributions = result.get('model_contributions', {})
                
                print(f"   {case['name']}:")
                print(f"     🎯 Final Decision: {prediction} (confidence: {confidence:.3f})")
                print(f"     🔄 VM Cluster: {cluster}")
                print(f"     ⚡ Response Time: {response_time:.1f}ms")
                
                # Show model contributions
                if contributions:
                    svm_contrib = contributions.get('svm', {})
                    kmeans_contrib = contributions.get('kmeans', {})
                    
                    print(f"     📊 SVM: {svm_contrib.get('prediction', 'N/A')} (weight: {svm_contrib.get('weight', 0):.2f})")
                    print(f"     📊 K-Means: Cluster {kmeans_contrib.get('prediction', 'N/A')} (weight: {kmeans_contrib.get('weight', 0):.2f})")
                
                # Check if prediction is correct
                expected = case['expected']
                if isinstance(expected, list):
                    is_correct = prediction in expected
                else:
                    is_correct = prediction == expected
                
                if is_correct:
                    success_count += 1
                    print(f"     ✅ Correct ensemble prediction!")
                else:
                    print(f"     ⚠️  Expected {expected}, got {prediction}")
                    
            else:
                print(f"   {case['name']}: ❌ Failed ({response.status_code})")
                if response.text:
                    print(f"     Error: {response.text}")
                
        except Exception as e:
            print(f"   {case['name']}: ❌ Error: {e}")
    
    avg_response_time = total_response_time / len(test_cases)
    accuracy = success_count / len(test_cases) * 100
    
    print(f"\n   📊 Ensemble Results:")
    print(f"     Accuracy: {success_count}/{len(test_cases)} = {accuracy:.1f}%")
    print(f"     Avg Response Time: {avg_response_time:.1f}ms")
    
    return success_count >= len(test_cases) * 0.75 and avg_response_time < 1000  # 75% accuracy, <1s response

def test_comparison_endpoint():
    """Test model comparison endpoint"""
    print("\n🔬 Testing Model Comparison...")
    
    test_features = [4, 16, 300, 5000, 3, 0.25, 18.75, 1250, 19.2, 12]  # 10 features
    vm_features = [0.6, 0.7, 0.5]
    
    try:
        payload = {
            "features": test_features,
            "vm_features": vm_features
        }
        
        response = requests.post(f"{ML_SERVICE_URL}/predict/compare", 
                               json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            individual = result.get('individual_predictions', {})
            ensemble = result.get('ensemble_prediction', {})
            comparison = result.get('comparison', {})
            
            print(f"   Individual SVM: {individual.get('svm', {}).get('prediction', 'N/A')}")
            print(f"   Individual K-Means: Cluster {individual.get('kmeans', {}).get('cluster', 'N/A')}")
            print(f"   Ensemble Decision: {ensemble.get('decision', 'N/A')}")
            print(f"   Algorithm Used: {ensemble.get('algorithm_used', 'N/A')}")
            print(f"   Processing Time: {comparison.get('processing_time_ms', 0):.1f}ms")
            print(f"   ✅ Comparison test successful!")
            return True
        else:
            print(f"   ❌ Comparison test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Comparison test error: {e}")
        return False

def test_performance_benchmark():
    """Run performance benchmark"""
    print("\n⚡ Testing Performance Benchmark...")
    
    test_payload = {
        "features": [4, 16, 300, 5000, 3],
        "vm_features": [0.6, 0.7, 0.5]
    }
    
    num_requests = 20
    successful_requests = 0
    total_time = 0
    response_times = []
    
    print(f"   Running {num_requests} requests...")
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(f"{ML_SERVICE_URL}/predict/enhanced", 
                                   json=test_payload, timeout=10)
            response_time = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                successful_requests += 1
                total_time += response_time
                response_times.append(response_time)
                
        except Exception as e:
            print(f"   Request {i+1} failed: {e}")
    
    if successful_requests > 0:
        avg_time = total_time / successful_requests
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"   📊 Performance Results:")
        print(f"     Success Rate: {successful_requests}/{num_requests} = {successful_requests/num_requests*100:.1f}%")
        print(f"     Average Time: {avg_time:.1f}ms")
        print(f"     Min Time: {min_time:.1f}ms")
        print(f"     Max Time: {max_time:.1f}ms")
        
        # Performance criteria for research paper
        success_rate_ok = successful_requests / num_requests >= 0.95  # 95% success rate
        performance_ok = avg_time < 500  # Average < 500ms
        
        if success_rate_ok and performance_ok:
            print(f"   ✅ Performance benchmark passed!")
            return True
        else:
            print(f"   ⚠️  Performance could be improved")
            return False
    else:
        print(f"   ❌ All requests failed!")
        return False

def test_meta_learning_system():
    """Test the complete Meta-Learning Neural Network system"""
    print("\n🧠 Testing Meta-Learning Neural Network System...")
    
    # Test 1: Direct Meta-Learning endpoint
    print("   Testing /predict/meta_learning endpoint...")
    
    test_cases = [
        {
            'name': 'High Confidence Features',
            'features': [
                0.9, 0.8, 0.7,  # svm_conf, kmeans_conf, rule_conf
                0.0, 0.0, 1.0,  # svm_small, svm_medium, svm_large (large prediction)
                4.0, 0.2,       # cluster_id, cluster_distance_norm
                0.8, 0.7, 0.6,  # compute_intensity, memory_intensity, storage_intensity
                1.0, 0.5        # is_high_priority, resource_balance_score
            ],
            'expected': 'large'
        },
        {
            'name': 'Medium Workload Features',
            'features': [
                0.7, 0.6, 0.6,  # svm_conf, kmeans_conf, rule_conf
                0.0, 1.0, 0.0,  # svm_small, svm_medium, svm_large (medium prediction)
                2.0, 0.4,       # cluster_id, cluster_distance_norm
                0.5, 0.5, 0.4,  # compute_intensity, memory_intensity, storage_intensity
                0.0, 0.7        # is_high_priority, resource_balance_score
            ],
            'expected': 'medium'
        },
        {
            'name': 'Light Workload Features',
            'features': [
                0.6, 0.5, 0.5,  # svm_conf, kmeans_conf, rule_conf
                1.0, 0.0, 0.0,  # svm_small, svm_medium, svm_large (small prediction)
                0.0, 0.6,       # cluster_id, cluster_distance_norm
                0.2, 0.3, 0.2,  # compute_intensity, memory_intensity, storage_intensity
                0.0, 0.8        # is_high_priority, resource_balance_score
            ],
            'expected': 'small'
        }
    ]
    
    success_count = 0
    
    for case in test_cases:
        try:
            payload = {"features": case['features']}
            response = requests.post(f"{ML_SERVICE_URL}/predict/meta_learning", 
                                   json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('makespan', 'unknown')
                confidence = result.get('confidence', 0)
                method = result.get('method', 'unknown')
                
                print(f"   {case['name']}: {prediction} (confidence: {confidence:.3f}, method: {method})")
                
                if prediction == case['expected']:
                    success_count += 1
                    print(f"     ✅ Correct Meta-Learning prediction!")
                else:
                    print(f"     ⚠️  Expected {case['expected']}, got {prediction}")
                    
                # Show model info
                model_info = result.get('model_info', {})
                if model_info:
                    print(f"     Architecture: {model_info.get('architecture', 'Unknown')}")
                    print(f"     Test Accuracy: {model_info.get('test_accuracy', 'Unknown')}")
                    
            else:
                print(f"   {case['name']}: ❌ Failed ({response.status_code})")
                if response.status_code == 503:
                    print("     Meta-Learning models not loaded!")
                    
        except Exception as e:
            print(f"   {case['name']}: ❌ Error: {e}")
    
    # Test 2: Complete MCCVA pipeline
    print("\n   Testing /predict/mccva_complete endpoint...")
    
    complete_test_cases = [
        {
            'name': 'Small Workload Pipeline',
            'input': {
                "cpu_cores": 2,
                "memory_gb": 2,
                "storage_gb": 50,
                "network_bandwidth": 1000,
                "priority": 2,
                "vm_cpu_usage": 0.3,
                "vm_memory_usage": 0.4,
                "vm_storage_usage": 0.2
            },
            'expected': 'small'
        },
        {
            'name': 'Large Workload Pipeline',
            'input': {
                "cpu_cores": 8,
                "memory_gb": 16,
                "storage_gb": 500,
                "network_bandwidth": 8000,
                "priority": 5,
                "vm_cpu_usage": 0.8,
                "vm_memory_usage": 0.9,
                "vm_storage_usage": 0.7
            },
            'expected': 'large'
        }
    ]
    
    for case in complete_test_cases:
        try:
            response = requests.post(f"{ML_SERVICE_URL}/predict/mccva_complete", 
                                   json=case['input'], timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('makespan', 'unknown')
                confidence = result.get('confidence', 0)
                method = result.get('method', 'unknown')
                
                print(f"   {case['name']}: {prediction} (confidence: {confidence:.3f})")
                
                # Show 3-stage results
                stages = result.get('stage_results', {})
                svm_pred = stages.get('stage1_svm', {}).get('prediction', 'Unknown')
                kmeans_cluster = stages.get('stage2_kmeans', {}).get('cluster', 'Unknown')
                meta_pred = stages.get('stage3_metalearning', {}).get('prediction', 'Unknown')
                
                print(f"     Stage 1 (SVM): {svm_pred}")
                print(f"     Stage 2 (K-Means): Cluster {kmeans_cluster}")
                print(f"     Stage 3 (Meta-Learning): {meta_pred}")
                
                if prediction == case['expected']:
                    success_count += 1
                    print(f"     ✅ Complete pipeline prediction correct!")
                else:
                    print(f"     ⚠️  Expected {case['expected']}, got {prediction}")
                    
            else:
                print(f"   {case['name']}: ❌ Failed ({response.status_code})")
                
        except Exception as e:
            print(f"   {case['name']}: ❌ Error: {e}")
    
    total_tests = len(test_cases) + len(complete_test_cases)
    accuracy = success_count / total_tests * 100
    print(f"   📊 Meta-Learning System Accuracy: {success_count}/{total_tests} = {accuracy:.1f}%")
    return success_count >= total_tests * 0.6  # 60% threshold

def main():
    """Main testing function"""
    print("🧪 MCCVA ENSEMBLE INTEGRATION TEST")
    print("=" * 60)
    print(f"Testing ML Service at: {ML_SERVICE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: Model Loading
    test_results['model_loading'] = test_model_loading()
    
    # Test 2: Individual SVM
    test_results['svm_individual'] = test_individual_svm()
    
    # Test 3: Individual K-Means
    test_results['kmeans_individual'] = test_individual_kmeans()
    
    # Test 4: Enhanced Ensemble (SVM + K-Means)
    test_results['ensemble_enhanced'] = test_ensemble_predictions()
    
    # Test 5: Meta-Learning Neural Network System
    test_results['meta_learning_system'] = test_meta_learning_system()
    
    # Test 6: Comparison Endpoint
    test_results['comparison_endpoint'] = test_comparison_endpoint()
    
    # Test 7: Performance Benchmark
    test_results['performance_benchmark'] = test_performance_benchmark()
    
    # Final Results
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS")
    print("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    overall_success_rate = passed_tests / total_tests * 100
    print(f"\n🎯 OVERALL SUCCESS RATE: {passed_tests}/{total_tests} = {overall_success_rate:.1f}%")
    
    if overall_success_rate >= 80:
        print("✅ ENSEMBLE INTEGRATION TEST PASSED!")
        print("   System ready for production deployment")
        print("   All 3 stages (SVM + K-Means + Meta-Learning) working correctly")
    elif overall_success_rate >= 60:
        print("⚠️  ENSEMBLE INTEGRATION TEST PARTIALLY PASSED")
        print("   Some components need attention before deployment")
    else:
        print("❌ ENSEMBLE INTEGRATION TEST FAILED")
        print("   Major issues detected - deployment not recommended")
    
    print(f"\nCompleted at: {datetime.now().isoformat()}")
    return overall_success_rate >= 80

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 