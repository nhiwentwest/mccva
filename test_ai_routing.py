#!/usr/bin/env python3
"""
AI Routing Logic Test Script
Kiểm tra việc phân phối request có tận dụng được AI model không
Tập trung vào: makespan prediction, server selection, load balancing
Enhanced version với better test scenarios
"""

import requests
import json
import time
import statistics
from collections import defaultdict, Counter
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

class AIRoutingTester:
    def __init__(self, base_url="http://localhost"):
        self.base_url = base_url
        self.results = {
            "total_requests": 0,
            "successful_requests": 0,
            "ai_predictions": [],
            "server_distribution": defaultdict(int),
            "response_times": [],
            "error_logs": []
        }
        
        # Enhanced test scenarios với better data distribution
        self.test_scenarios = [
            {
                "name": "Web Server (Small)",
                "data": {
                    "cpu_cores": 2, 
                    "memory": 4, 
                    "storage": 50, 
                    "network_bandwidth": 500, 
                    "priority": 1,
                    "task_complexity": 1,     # Simple task
                    "data_size": 20,          # Small data
                    "io_intensity": 10,       # Low I/O
                    "parallel_degree": 100,   # Low parallelism
                    "deadline_urgency": 1     # Low urgency
                },
                "vm_features": [0.3, 0.4, 0.2],  # Low VM usage
                "expected_makespan": "small",
                "expected_servers": [1, 2, 3]  # Low capacity servers
            },
            {
                "name": "Database Server (Medium)",
                "data": {
                    "cpu_cores": 4, 
                    "memory": 8, 
                    "storage": 100, 
                    "network_bandwidth": 1000, 
                    "priority": 3,
                    "task_complexity": 2,     # Medium complexity
                    "data_size": 50,          # Medium data
                    "io_intensity": 25,       # Medium I/O
                    "parallel_degree": 500,   # Medium parallelism
                    "deadline_urgency": 2     # Medium urgency
                },
                "vm_features": [0.6, 0.7, 0.5],  # Medium VM usage
                "expected_makespan": "medium", 
                "expected_servers": [3, 4, 5, 6]  # Medium capacity servers
            },
            {
                "name": "ML Training (Large)",
                "data": {
                    "cpu_cores": 12, 
                    "memory": 32, 
                    "storage": 500, 
                    "network_bandwidth": 5000, 
                    "priority": 5,
                    "task_complexity": 4,     # High complexity
                    "data_size": 200,         # Large data
                    "io_intensity": 75,       # High I/O
                    "parallel_degree": 1500,  # High parallelism
                    "deadline_urgency": 4     # High urgency
                },
                "vm_features": [0.8, 0.9, 0.7],  # High VM usage
                "expected_makespan": "large",
                "expected_servers": [5, 6, 7, 8]  # High capacity servers
            },
            {
                "name": "Video Rendering (Large)",
                "data": {
                    "cpu_cores": 16, 
                    "memory": 64, 
                    "storage": 800, 
                    "network_bandwidth": 8000, 
                    "priority": 4,
                    "task_complexity": 5,     # Very high complexity
                    "data_size": 500,         # Very large data
                    "io_intensity": 90,       # Very high I/O
                    "parallel_degree": 2000,  # Max parallelism
                    "deadline_urgency": 5     # Max urgency
                },
                "vm_features": [0.9, 0.8, 0.6],  # High VM usage
                "expected_makespan": "large",
                "expected_servers": [5, 6, 7, 8]
            },
            {
                "name": "API Gateway (Small)",
                "data": {
                    "cpu_cores": 1, 
                    "memory": 2, 
                    "storage": 20, 
                    "network_bandwidth": 2000, 
                    "priority": 2,
                    "task_complexity": 1,     # Simple task
                    "data_size": 10,          # Small data
                    "io_intensity": 5,        # Very low I/O
                    "parallel_degree": 100,   # Low parallelism
                    "deadline_urgency": 1     # Low urgency
                },
                "vm_features": [0.4, 0.3, 0.1],  # Low VM usage
                "expected_makespan": "small",
                "expected_servers": [1, 2, 3]
            },
            {
                "name": "File Server (Medium)",
                "data": {
                    "cpu_cores": 6, 
                    "memory": 12, 
                    "storage": 200, 
                    "network_bandwidth": 1500, 
                    "priority": 3,
                    "task_complexity": 3,     # Medium-high complexity
                    "data_size": 100,         # Medium-large data
                    "io_intensity": 50,       # Medium-high I/O
                    "parallel_degree": 800,   # Medium-high parallelism
                    "deadline_urgency": 3     # Medium urgency
                },
                "vm_features": [0.5, 0.6, 0.8],  # Medium VM usage
                "expected_makespan": "medium",
                "expected_servers": [3, 4, 5, 6]
            }
        ]
    
    def test_enhanced_prediction_accuracy(self):
        """Test độ chính xác của Enhanced AI prediction"""
        print("🤖 Testing Enhanced AI Prediction Accuracy")
        print("=" * 60)
        
        correct_predictions = 0
        total_predictions = 0
        
        for scenario in self.test_scenarios:
            print(f"\n📋 Testing: {scenario['name']}")
            print(f"   Expected makespan: {scenario['expected_makespan']}")
            print(f"   Input: {scenario['data']}")
            print(f"   VM Features: {scenario['vm_features']}")
            
            try:
                start_time = time.time()
                
                # Test enhanced prediction endpoint
                enhanced_data = {
                    "features": [
                        scenario['data']['cpu_cores'],
                        scenario['data']['memory'],
                        scenario['data']['storage'],
                        scenario['data']['network_bandwidth'],
                        scenario['data']['priority']
                    ],
                    "vm_features": scenario['vm_features']
                }
                
                # Test enhanced endpoint directly
                enhanced_response = requests.post(
                    f"{self.base_url}:5000/predict/enhanced",
                    json=enhanced_data,
                    timeout=10
                )
                
                if enhanced_response.status_code == 200:
                    enhanced_result = enhanced_response.json()
                    enhanced_makespan = enhanced_result.get('makespan', 'unknown')
                    enhanced_confidence = enhanced_result.get('confidence', 0)
                    model_contributions = enhanced_result.get('model_contributions', {})
                    
                    print(f"   🔬 Enhanced Prediction: {enhanced_makespan} (confidence: {enhanced_confidence:.3f})")
                    print(f"   📊 Model Contributions:")
                    for model, contrib in model_contributions.items():
                        print(f"     - {model}: {contrib.get('prediction', 'N/A')} (weight: {contrib.get('weight', 0):.3f})")
                
                # Test routing endpoint
                response = requests.post(
                    f"{self.base_url}/mccva/route",
                    json=scenario["data"],
                    timeout=10
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Parse response theo format thực tế của OpenResty
                    prediction_data = result.get('prediction', {})
                    makespan_prediction = prediction_data.get('makespan', 'unknown')
                    confidence = prediction_data.get('confidence', 0)
                    target_vm = result.get('server', 'unknown')
                    
                    # Check if mccva_decision exists, otherwise use prediction data
                    mccva_decision = result.get('mccva_decision', {})
                    if not mccva_decision:
                        mccva_decision = {
                            'makespan_prediction': makespan_prediction,
                            'confidence_score': confidence,
                            'algorithm_used': 'Enhanced Prediction'
                        }
                    
                    print(f"   ✅ Routing Prediction: {makespan_prediction} (confidence: {confidence:.3f})")
                    print(f"   🎯 Target VM: {target_vm}")
                    print(f"   ⏱️  Response time: {response_time:.3f}s")
                    
                    # Check prediction accuracy
                    if makespan_prediction == scenario['expected_makespan']:
                        correct_predictions += 1
                        print(f"   ✅ Prediction CORRECT")
                    else:
                        print(f"   ❌ Prediction WRONG (expected: {scenario['expected_makespan']})")
                    
                    total_predictions += 1
                    
                    # Store results
                    self.results["ai_predictions"].append({
                        "scenario": scenario["name"],
                        "expected": scenario["expected_makespan"],
                        "predicted": makespan_prediction,
                        "enhanced_predicted": enhanced_makespan if enhanced_response.status_code == 200 else "unknown",
                        "confidence": confidence,
                        "enhanced_confidence": enhanced_confidence if enhanced_response.status_code == 200 else 0,
                        "target_vm": target_vm,
                        "response_time": response_time,
                        "correct": makespan_prediction == scenario['expected_makespan'],
                        "enhanced_correct": enhanced_makespan == scenario['expected_makespan'] if enhanced_response.status_code == 200 else False
                    })
                    
                else:
                    print(f"   ❌ Request failed: HTTP {response.status_code}")
                    print(f"   Error: {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
        
        # Calculate accuracy
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        print(f"\n📊 AI Prediction Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
        
        # Calculate enhanced accuracy if available
        enhanced_correct = sum(1 for pred in self.results["ai_predictions"] if pred.get("enhanced_correct", False))
        enhanced_accuracy = (enhanced_correct / total_predictions * 100) if total_predictions > 0 else 0
        print(f"📊 Enhanced Prediction Accuracy: {enhanced_accuracy:.1f}% ({enhanced_correct}/{total_predictions})")
        
        return accuracy, enhanced_accuracy
    
    def test_server_distribution(self, requests_per_scenario=10):
        """Test phân phối server với nhiều requests"""
        print(f"\n🔄 Testing Server Distribution ({requests_per_scenario} requests per scenario)")
        print("=" * 70)
        
        server_distribution = defaultdict(int)
        scenario_results = defaultdict(lambda: {"servers": [], "predictions": []})
        
        for scenario in self.test_scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            print(f"   Expected servers: {scenario['expected_servers']}")
            
            for i in range(requests_per_scenario):
                try:
                    response = requests.post(
                        f"{self.base_url}/mccva/route",
                        json=scenario["data"],
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        target_vm = result.get('target_vm', 'unknown')
                        mccva_decision = result.get('mccva_decision', {})
                        makespan = mccva_decision.get('makespan_prediction', 'unknown')
                        
                        # Extract server number from target_vm
                        server_num = None
                        if 'server_' in target_vm:
                            server_num = int(target_vm.split('_')[-1])
                        elif 'mock_server_' in target_vm:
                            server_num = int(target_vm.split('_')[-1])
                        elif '808' in target_vm:
                            server_num = int(target_vm.split(':')[-1]) - 8080
                        
                        if server_num:
                            server_distribution[server_num] += 1
                            scenario_results[scenario['name']]["servers"].append(server_num)
                            scenario_results[scenario['name']]["predictions"].append(makespan)
                            
                            if i < 3:  # Show first 3 results
                                print(f"   Request {i+1}: Server {server_num} (makespan: {makespan})")
                    
                except Exception as e:
                    print(f"   Request {i+1}: Error - {e}")
        
        # Analyze distribution
        print(f"\n📊 Overall Server Distribution:")
        for server_num in sorted(server_distribution.keys()):
            count = server_distribution[server_num]
            percentage = (count / (len(self.test_scenarios) * requests_per_scenario)) * 100
            print(f"   Server {server_num}: {count} requests ({percentage:.1f}%)")
        
        # Check if AI routing is working
        print(f"\n🤖 AI Routing Analysis:")
        for scenario_name, results in scenario_results.items():
            servers_used = set(results["servers"])
            predictions = set(results["predictions"])
            
            print(f"\n   {scenario_name}:")
            print(f"     Servers used: {sorted(servers_used)}")
            print(f"     Predictions: {predictions}")
            
            # Check if servers align with makespan prediction
            if len(predictions) == 1:  # Consistent prediction
                makespan = list(predictions)[0]
                if makespan == "small" and all(s <= 3 for s in servers_used):
                    print(f"     ✅ AI routing working: small tasks → low capacity servers")
                elif makespan == "medium" and all(3 <= s <= 6 for s in servers_used):
                    print(f"     ✅ AI routing working: medium tasks → medium capacity servers")
                elif makespan == "large" and all(s >= 5 for s in servers_used):
                    print(f"     ✅ AI routing working: large tasks → high capacity servers")
                else:
                    print(f"     ⚠️  AI routing may not be optimal")
            else:
                print(f"     ⚠️  Inconsistent predictions: {predictions}")
        
        self.results["server_distribution"] = dict(server_distribution)
        return server_distribution
    
    def test_load_balancing_efficiency(self, total_requests=100):
        """Test hiệu quả load balancing"""
        print(f"\n⚖️ Testing Load Balancing Efficiency ({total_requests} requests)")
        print("=" * 50)
        
        # Use medium priority requests for load balancing test
        test_data = {
            "cpu_cores": 4,
            "memory": 8, 
            "storage": 100,
            "network_bandwidth": 1000,
            "priority": 3,
            "task_complexity": 2,
            "data_size": 50,
            "io_intensity": 25,
            "parallel_degree": 500,
            "deadline_urgency": 2
        }
        
        server_counts = defaultdict(int)
        response_times = []
        
        print("Sending requests...")
        for i in range(total_requests):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/mccva/route",
                    json=test_data,
                    timeout=10
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    target_vm = result.get('target_vm', 'unknown')
                    
                    # Extract server number
                    server_num = None
                    if 'mock_server_' in target_vm:
                        server_num = int(target_vm.split('_')[-1])
                    elif '808' in target_vm:
                        server_num = int(target_vm.split(':')[-1]) - 8080
                    
                    if server_num:
                        server_counts[server_num] += 1
                        response_times.append(response_time)
                        
                        if i < 10:  # Show first 10 results
                            print(f"   Request {i+1}: Server {server_num} ({response_time:.3f}s)")
                
            except Exception as e:
                print(f"   Request {i+1}: Error - {e}")
        
        # Calculate load balancing metrics
        if server_counts:
            counts = list(server_counts.values())
            mean_count = statistics.mean(counts)
            std_count = statistics.stdev(counts) if len(counts) > 1 else 0
            cv = (std_count / mean_count) if mean_count > 0 else 0  # Coefficient of variation
            
            print(f"\n📊 Load Balancing Results:")
            print(f"   Total requests: {total_requests}")
            print(f"   Servers used: {len(server_counts)}")
            print(f"   Mean requests per server: {mean_count:.1f}")
            print(f"   Standard deviation: {std_count:.1f}")
            print(f"   Coefficient of variation: {cv:.3f}")
            
            # Evaluate load balancing
            if cv < 0.3:
                print(f"   ✅ Excellent load balancing (CV < 0.3)")
            elif cv < 0.5:
                print(f"   ✅ Good load balancing (CV < 0.5)")
            elif cv < 0.7:
                print(f"   ⚠️  Fair load balancing (CV < 0.7)")
            else:
                print(f"   ❌ Poor load balancing (CV >= 0.7)")
            
            # Show distribution
            print(f"\n   Server Distribution:")
            for server_num in sorted(server_counts.keys()):
                count = server_counts[server_num]
                percentage = (count / total_requests) * 100
                print(f"     Server {server_num}: {count} ({percentage:.1f}%)")
        
        # Response time analysis
        if response_times:
            avg_time = statistics.mean(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            print(f"\n⏱️ Response Time Analysis:")
            print(f"   Average: {avg_time:.3f}s")
            print(f"   Min: {min_time:.3f}s")
            print(f"   Max: {max_time:.3f}s")
            
            if avg_time < 0.1:
                print(f"   ✅ Excellent performance")
            elif avg_time < 0.5:
                print(f"   ✅ Good performance")
            else:
                print(f"   ⚠️  Performance needs improvement")
        
        return {
            "server_counts": dict(server_counts),
            "response_times": response_times,
            "load_balancing_cv": cv if server_counts else 0
        }
    
    def test_ai_vs_random_routing(self, requests_per_test=50):
        """So sánh AI routing với random routing"""
        print(f"\n🎲 AI vs Random Routing Comparison ({requests_per_test} requests each)")
        print("=" * 60)
        
        test_data = {
            "cpu_cores": 4,
            "memory": 8,
            "storage": 100, 
            "network_bandwidth": 1000,
            "priority": 3,
            "task_complexity": 2,
            "data_size": 50,
            "io_intensity": 25,
            "parallel_degree": 500,
            "deadline_urgency": 2
        }
        
        # Test AI routing
        print("🤖 Testing AI Routing...")
        ai_servers = []
        ai_response_times = []
        
        for i in range(requests_per_test):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/mccva/route",
                    json=test_data,
                    timeout=10
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    target_vm = result.get('target_vm', 'unknown')
                    
                    server_num = None
                    if 'mock_server_' in target_vm:
                        server_num = int(target_vm.split('_')[-1])
                    elif '808' in target_vm:
                        server_num = int(target_vm.split(':')[-1]) - 8080
                    
                    if server_num:
                        ai_servers.append(server_num)
                        ai_response_times.append(response_time)
                
            except Exception as e:
                print(f"   AI Request {i+1}: Error - {e}")
        
        # Simulate random routing (theoretical)
        print("🎲 Simulating Random Routing...")
        import random
        random_servers = [random.randint(1, 8) for _ in range(requests_per_test)]
        
        # Analyze results
        ai_distribution = Counter(ai_servers)
        random_distribution = Counter(random_servers)
        
        print(f"\n📊 Comparison Results:")
        print(f"   AI Routing Distribution:")
        for server in sorted(ai_distribution.keys()):
            count = ai_distribution[server]
            percentage = (count / len(ai_servers)) * 100 if ai_servers else 0
            print(f"     Server {server}: {count} ({percentage:.1f}%)")
        
        print(f"   Random Routing Distribution:")
        for server in sorted(random_distribution.keys()):
            count = random_distribution[server]
            percentage = (count / len(random_servers)) * 100
            print(f"     Server {server}: {count} ({percentage:.1f}%)")
        
        # Calculate metrics
        ai_cv = statistics.stdev(list(ai_distribution.values())) / statistics.mean(list(ai_distribution.values())) if ai_distribution else 0
        random_cv = statistics.stdev(list(random_distribution.values())) / statistics.mean(list(random_distribution.values())) if random_distribution else 0
        
        print(f"\n📈 Load Balancing Metrics:")
        print(f"   AI Routing CV: {ai_cv:.3f}")
        print(f"   Random Routing CV: {random_cv:.3f}")
        
        if ai_cv < random_cv:
            improvement = ((random_cv - ai_cv) / random_cv) * 100
            print(f"   ✅ AI routing improves load balancing by {improvement:.1f}%")
        else:
            print(f"   ⚠️  AI routing does not improve load balancing")
        
        return {
            "ai_distribution": dict(ai_distribution),
            "random_distribution": dict(random_distribution),
            "ai_cv": ai_cv,
            "random_cv": random_cv
        }
    
    def generate_report(self):
        """Tạo báo cáo tổng hợp"""
        print(f"\n📋 AI Routing Test Report")
        print("=" * 50)
        
        # AI Prediction Accuracy
        accuracy, enhanced_accuracy = self.test_enhanced_prediction_accuracy()
        
        # Server Distribution
        distribution = self.test_server_distribution(requests_per_scenario=10)
        
        # Load Balancing Efficiency
        lb_results = self.test_load_balancing_efficiency(total_requests=80)
        
        # AI vs Random Comparison
        comparison = self.test_ai_vs_random_routing(requests_per_test=40)
        
        # Summary
        print(f"\n🎯 Summary:")
        print(f"   AI Prediction Accuracy: {accuracy:.1f}%")
        print(f"   Enhanced Prediction Accuracy: {enhanced_accuracy:.1f}%")
        print(f"   Load Balancing CV: {lb_results['load_balancing_cv']:.3f}")
        print(f"   AI vs Random Improvement: {((comparison['random_cv'] - comparison['ai_cv']) / comparison['random_cv'] * 100):.1f}%" if comparison['random_cv'] > 0 else "N/A")
        
        # Overall assessment
        if accuracy >= 80 and lb_results['load_balancing_cv'] < 0.5:
            print(f"   ✅ AI routing system is working effectively")
        elif accuracy >= 60 and lb_results['load_balancing_cv'] < 0.7:
            print(f"   ⚠️  AI routing system needs improvement")
        else:
            print(f"   ❌ AI routing system needs significant improvement")
        
        return {
            "accuracy": accuracy,
            "enhanced_accuracy": enhanced_accuracy,
            "load_balancing_cv": lb_results['load_balancing_cv'],
            "ai_vs_random_improvement": ((comparison['random_cv'] - comparison['ai_cv']) / comparison['random_cv'] * 100) if comparison['random_cv'] > 0 else 0
        }

def main():
    """Main function"""
    print("🚀 AI Routing Logic Test")
    print("Testing MCCVA AI-powered load balancing system")
    print("=" * 60)
    
    # Check if system is running
    try:
        response = requests.get("http://localhost/health", timeout=5)
        if response.status_code != 200:
            print("❌ System not ready. Please start the system first:")
            print("   cd /opt/mccva && ./run.sh")
            return
    except:
        print("❌ Cannot connect to system. Please start the system first:")
        print("   cd /opt/mccva && ./run.sh")
        return
    
    # Run tests
    tester = AIRoutingTester()
    results = tester.generate_report()
    
    print(f"\n✅ Test completed successfully!")
    print(f"   AI Accuracy: {results['accuracy']:.1f}%")
    print(f"   Enhanced AI Accuracy: {results['enhanced_accuracy']:.1f}%")
    print(f"   Load Balancing Quality: {'Good' if results['load_balancing_cv'] < 0.5 else 'Needs Improvement'}")
    print(f"   AI Improvement: {results['ai_vs_random_improvement']:.1f}%")

if __name__ == "__main__":
    main() 