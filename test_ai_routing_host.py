#!/usr/bin/env python3
"""
AI Routing Logic Test Script (Host Environment)
Kiểm tra việc phân phối request có tận dụng được AI model không
Phân biệt môi trường host và Docker
"""

import requests
import json
import time
import statistics
import subprocess
import os
from collections import defaultdict, Counter
from typing import Dict, List, Any

class HostAIRoutingTester:
    def __init__(self):
        self.results = {
            "total_requests": 0,
            "successful_requests": 0,
            "ai_predictions": [],
            "server_distribution": defaultdict(int),
            "response_times": [],
            "error_logs": []
        }
        
        # Test scenarios với different resource requirements
        self.test_scenarios = [
            {
                "name": "Small Task (Low Resource)",
                "data": {"cpu_cores": 2, "memory": 4, "storage": 50, "network_bandwidth": 500, "priority": 1},
                "expected_makespan": "small",
                "expected_servers": [1, 2, 3]  # Low capacity servers
            },
            {
                "name": "Medium Task (Balanced)",
                "data": {"cpu_cores": 4, "memory": 8, "storage": 100, "network_bandwidth": 1000, "priority": 3},
                "expected_makespan": "medium", 
                "expected_servers": [3, 4, 5, 6]  # Medium capacity servers
            },
            {
                "name": "Large Task (High Resource)",
                "data": {"cpu_cores": 8, "memory": 16, "storage": 200, "network_bandwidth": 2000, "priority": 5},
                "expected_makespan": "large",
                "expected_servers": [5, 6, 7, 8]  # High capacity servers
            },
            {
                "name": "CPU Intensive Task",
                "data": {"cpu_cores": 12, "memory": 8, "storage": 100, "network_bandwidth": 1000, "priority": 4},
                "expected_makespan": "large",
                "expected_servers": [5, 6, 7, 8]
            },
            {
                "name": "Storage Intensive Task", 
                "data": {"cpu_cores": 4, "memory": 8, "storage": 800, "network_bandwidth": 1000, "priority": 2},
                "expected_makespan": "medium",
                "expected_servers": [3, 4, 5, 6]
            }
        ]
    
    def check_services_health(self):
        """Kiểm tra health của tất cả services"""
        print("🏥 Checking Services Health")
        print("=" * 50)
        
        services_status = {}
        
        # Check OpenResty
        try:
            response = requests.get("http://localhost/health", timeout=5)
            services_status["openresty"] = response.status_code == 200
            print(f"   OpenResty: {'✅' if response.status_code == 200 else '❌'} (HTTP {response.status_code})")
        except Exception as e:
            services_status["openresty"] = False
            print(f"   OpenResty: ❌ ({e})")
        
        # Check ML Service (Docker)
        try:
            response = requests.get("http://localhost:5000/health", timeout=5)
            services_status["ml_service"] = response.status_code == 200
            print(f"   ML Service: {'✅' if response.status_code == 200 else '❌'} (HTTP {response.status_code})")
            if response.status_code == 200:
                data = response.json()
                models_loaded = data.get("models_loaded", {})
                print(f"     Models: {models_loaded}")
        except Exception as e:
            services_status["ml_service"] = False
            print(f"   ML Service: ❌ ({e})")
        
        # Check Mock Servers
        mock_servers_ok = 0
        for port in range(8081, 8089):
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=3)
                if response.status_code == 200:
                    mock_servers_ok += 1
            except:
                pass
        services_status["mock_servers"] = mock_servers_ok >= 6  # At least 6 servers working
        print(f"   Mock Servers: {'✅' if mock_servers_ok >= 6 else '❌'} ({mock_servers_ok}/8 working)")
        
        # Check Docker container
        try:
            result = subprocess.run(['docker', 'ps', '--filter', 'name=mccva-ml', '--format', '{{.Status}}'], 
                                  capture_output=True, text=True, timeout=5)
            services_status["docker_container"] = "Up" in result.stdout
            print(f"   Docker Container: {'✅' if 'Up' in result.stdout else '❌'}")
        except Exception as e:
            services_status["docker_container"] = False
            print(f"   Docker Container: ❌ ({e})")
        
        return services_status
    
    def test_ml_service_directly(self):
        """Test ML Service trực tiếp"""
        print("\n🤖 Testing ML Service Directly")
        print("=" * 40)
        
        test_data = {
            "features": [4, 8, 100, 1000, 3]  # [cpu_cores, memory, storage, network_bandwidth, priority]
        }
        
        try:
            response = requests.post(
                "http://localhost:5000/predict/makespan",
                json=test_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                makespan = result.get('makespan', 'unknown')
                confidence = result.get('confidence', 0)
                print(f"   ✅ ML Service working: {makespan} (confidence: {confidence:.3f})")
                return True
            else:
                print(f"   ❌ ML Service failed: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ ML Service error: {e}")
            return False
    
    def test_openresty_routing(self):
        """Test OpenResty routing endpoint"""
        print("\n🌐 Testing OpenResty Routing")
        print("=" * 40)
        
        test_data = {
            "cpu_cores": 4,
            "memory": 8,
            "storage": 100,
            "network_bandwidth": 1000,
            "priority": 3
        }
        
        try:
            response = requests.post(
                "http://localhost/mccva/route",
                json=test_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                target_vm = result.get('target_vm', 'unknown')
                mccva_decision = result.get('mccva_decision', {})
                makespan = mccva_decision.get('makespan_prediction', 'unknown')
                confidence = mccva_decision.get('confidence_score', 0)
                
                print(f"   ✅ OpenResty routing working:")
                print(f"     Target VM: {target_vm}")
                print(f"     Makespan: {makespan}")
                print(f"     Confidence: {confidence:.3f}")
                
                # Check if AI prediction is working
                if makespan != 'unknown' and confidence > 0:
                    print(f"     ✅ AI prediction is working")
                    return True
                else:
                    print(f"     ⚠️  AI prediction may not be working properly")
                    return False
            else:
                print(f"   ❌ OpenResty routing failed: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ OpenResty routing error: {e}")
            return False
    
    def test_ai_prediction_accuracy(self):
        """Test độ chính xác của AI prediction"""
        print("\n🤖 Testing AI Prediction Accuracy")
        print("=" * 50)
        
        correct_predictions = 0
        total_predictions = 0
        
        for scenario in self.test_scenarios:
            print(f"\n📋 Testing: {scenario['name']}")
            print(f"   Expected makespan: {scenario['expected_makespan']}")
            
            try:
                start_time = time.time()
                response = requests.post(
                    "http://localhost/mccva/route",
                    json=scenario["data"],
                    timeout=10
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    mccva_decision = result.get('mccva_decision', {})
                    makespan_prediction = mccva_decision.get('makespan_prediction', 'unknown')
                    confidence = mccva_decision.get('confidence_score', 0)
                    target_vm = result.get('target_vm', 'unknown')
                    
                    print(f"   ✅ AI Prediction: {makespan_prediction} (confidence: {confidence:.3f})")
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
                        "confidence": confidence,
                        "target_vm": target_vm,
                        "response_time": response_time,
                        "correct": makespan_prediction == scenario['expected_makespan']
                    })
                    
                else:
                    print(f"   ❌ Request failed: HTTP {response.status_code}")
                    print(f"   Error: {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
        
        # Calculate accuracy
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        print(f"\n📊 AI Prediction Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
        
        return accuracy
    
    def test_server_distribution(self, requests_per_scenario=10):
        """Test phân phối server với nhiều requests"""
        print(f"\n🔄 Testing Server Distribution ({requests_per_scenario} requests per scenario)")
        print("=" * 60)
        
        server_distribution = defaultdict(int)
        scenario_results = defaultdict(lambda: {"servers": [], "predictions": []})
        
        for scenario in self.test_scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            print(f"   Expected servers: {scenario['expected_servers']}")
            
            for i in range(requests_per_scenario):
                try:
                    response = requests.post(
                        "http://localhost/mccva/route",
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
                        if 'mock_server_' in target_vm:
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
    
    def generate_diagnostic_report(self):
        """Tạo báo cáo chẩn đoán"""
        print(f"\n📋 Diagnostic Report")
        print("=" * 50)
        
        # Check services health
        services_status = self.check_services_health()
        
        # Test ML Service directly
        ml_service_ok = self.test_ml_service_directly()
        
        # Test OpenResty routing
        openresty_ok = self.test_openresty_routing()
        
        # Test AI prediction accuracy
        accuracy = self.test_ai_prediction_accuracy()
        
        # Test server distribution
        distribution = self.test_server_distribution(requests_per_scenario=5)
        
        # Summary
        print(f"\n🎯 Summary:")
        print(f"   Services Health: {sum(services_status.values())}/{len(services_status)} services OK")
        print(f"   ML Service: {'✅' if ml_service_ok else '❌'}")
        print(f"   OpenResty Routing: {'✅' if openresty_ok else '❌'}")
        print(f"   AI Prediction Accuracy: {accuracy:.1f}%")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if not services_status["ml_service"]:
            print(f"   🔧 Fix ML Service: Check Docker container and port 5000")
        if not services_status["openresty"]:
            print(f"   🔧 Fix OpenResty: Check nginx configuration and port 80")
        if not services_status["mock_servers"]:
            print(f"   🔧 Fix Mock Servers: Check systemd service and ports 8081-8088")
        if accuracy < 60:
            print(f"   🔧 Improve AI Accuracy: Check ML models and training data")
        
        return {
            "services_health": services_status,
            "ml_service_ok": ml_service_ok,
            "openresty_ok": openresty_ok,
            "accuracy": accuracy
        }

def main():
    """Main function"""
    print("🚀 AI Routing Logic Test (Host Environment)")
    print("Testing MCCVA AI-powered load balancing system")
    print("=" * 60)
    
    # Run diagnostic tests
    tester = HostAIRoutingTester()
    results = tester.generate_diagnostic_report()
    
    print(f"\n✅ Diagnostic completed!")
    print(f"   Overall Status: {'✅ Healthy' if results['ml_service_ok'] and results['openresty_ok'] else '❌ Needs Attention'}")

if __name__ == "__main__":
    main() 