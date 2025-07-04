#!/usr/bin/env python3
"""
Advanced Test Suite cho MCCVA System
Comprehensive testing với stress testing, failure scenarios, và load balancing validation
"""

import requests
import json
import time
import threading
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import socket

class MCCVAAdvancedTester:
    def __init__(self, host_ip="172.17.0.1"):
        self.host_ip = host_ip
        self.base_url = f"http://{host_ip}"
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "performance_metrics": {},
            "error_logs": []
        }
    
    def log_test(self, test_name, success, details=""):
        """Log test results"""
        self.results["total_tests"] += 1
        if success:
            self.results["passed"] += 1
            print(f"✅ {test_name}: PASSED")
            if details:
                print(f"   {details}")
        else:
            self.results["failed"] += 1
            print(f"❌ {test_name}: FAILED")
            if details:
                print(f"   {details}")
            self.results["error_logs"].append(f"{test_name}: {details}")
    
    def test_ml_service_health(self):
        """Test ML Service health và model loading"""
        print("\n=== ML Service Health Test ===")
        
        try:
            response = requests.get(f"{self.base_url}:5000/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models_loaded = data.get("models_loaded", {})
                
                # Check all models are loaded
                all_loaded = all(models_loaded.values())
                self.log_test("ML Service Health", all_loaded, 
                            f"Models: {models_loaded}")
                    
            else:
                self.log_test("ML Service Health", False, 
                            f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("ML Service Health", False, str(e))
    
    def test_routing_consistency(self):
        """Test routing consistency với cùng input"""
        print("\n=== Routing Consistency Test ===")
        
        test_data = {
            "cpu_cores": 4,
            "memory": 8,
            "storage": 100,
            "network_bandwidth": 1000,
            "priority": 3
        }
        
        servers_used = set()
        predictions = []
        
        for i in range(10):
            try:
                response = requests.post(f"{self.base_url}/mccva/route", 
                                       json=test_data, timeout=15)
                if response.status_code == 200:
                    result = response.json()
                    servers_used.add(result.get("server"))
                    predictions.append(result.get("prediction", {}).get("makespan"))
                else:
                    self.log_test(f"Routing Consistency {i+1}", False, 
                                f"Status: {response.status_code}")
                    return
            except Exception as e:
                self.log_test(f"Routing Consistency {i+1}", False, str(e))
                return
        
        # Check consistency
        consistent_routing = len(servers_used) == 1
        consistent_prediction = len(set(predictions)) == 1
        
        self.log_test("Routing Consistency", consistent_routing, 
                     f"Servers used: {servers_used}")
        self.log_test("Prediction Consistency", consistent_prediction, 
                     f"Predictions: {set(predictions)}")
    
    def test_load_balancing(self):
        """Test load balancing với different priorities"""
        print("\n=== Load Balancing Test ===")
        
        server_counts = {}
        
        # Test different priority levels
        for priority in [1, 2, 3, 4, 5]:
            test_data = {
                "cpu_cores": 4,
                "memory": 8,
                "storage": 100,
                "network_bandwidth": 1000,
                "priority": priority
            }
            
            for i in range(5):  # 5 requests per priority
                try:
                    response = requests.post(f"{self.base_url}/mccva/route", 
                                           json=test_data, timeout=15)
                    if response.status_code == 200:
                        result = response.json()
                        server = result.get("server")
                        server_counts[server] = server_counts.get(server, 0) + 1
                    else:
                        self.log_test(f"Load Balancing Priority {priority}", False, 
                                    f"Status: {response.status_code}")
                        return
                except Exception as e:
                    self.log_test(f"Load Balancing Priority {priority}", False, str(e))
                    return
        
        # Check load distribution
        total_requests = sum(server_counts.values())
        balanced = total_requests > 0 and len(server_counts) > 1
        
        self.log_test("Load Balancing Distribution", balanced, 
                     f"Server distribution: {server_counts}")
    
    def test_stress_loading(self):
        """Stress test với concurrent requests"""
        print("\n=== Stress Test ===")
        
        def make_request(request_id):
            test_data = {
                "cpu_cores": random.randint(1, 16),
                "memory": random.randint(1, 64),
                "storage": random.randint(10, 1000),
                "network_bandwidth": random.randint(100, 10000),
                "priority": random.randint(1, 5)
            }
            
            start_time = time.time()
            try:
                response = requests.post(f"{self.base_url}/mccva/route", 
                                       json=test_data, timeout=30)
                end_time = time.time()
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "response_time": end_time - start_time,
                        "server": response.json().get("server")
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Status: {response.status_code}",
                        "response_time": end_time - start_time
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    "success": False,
                    "error": str(e),
                    "response_time": end_time - start_time
                }
        
        # Concurrent stress test
        concurrent_requests = 20
        print(f"Running {concurrent_requests} concurrent requests...")
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request, i) for i in range(concurrent_requests)]
            results = [future.result() for future in as_completed(futures)]
        end_time = time.time()
        
        # Analyze results
        successful = sum(1 for r in results if r["success"])
        response_times = [r["response_time"] for r in results if r["success"]]
        servers_used = set(r["server"] for r in results if r["success"] and r["server"])
        
        # Performance metrics
        avg_response_time = statistics.mean(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        
        success_rate = (successful / concurrent_requests) * 100
        total_time = end_time - start_time
        
        self.log_test("Stress Test Success Rate", success_rate >= 90, 
                     f"Success rate: {success_rate:.1f}% ({successful}/{concurrent_requests})")
        self.log_test("Stress Test Response Time", avg_response_time < 5, 
                     f"Avg: {avg_response_time:.2f}s, Max: {max_response_time:.2f}s, Min: {min_response_time:.2f}s")
        self.log_test("Stress Test Load Distribution", len(servers_used) > 1, 
                     f"Servers used: {servers_used}")
        
        # Store performance metrics
        self.results["performance_metrics"]["stress_test"] = {
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "min_response_time": min_response_time,
            "total_time": total_time,
            "servers_used": list(servers_used)
        }
    
    def test_failure_scenarios(self):
        """Test failure scenarios và fallback mechanisms"""
        print("\n=== Failure Scenarios Test ===")
        
        # Test 1: Invalid JSON
        try:
            response = requests.post(f"{self.base_url}/mccva/route", 
                                   data="invalid json", 
                                   headers={"Content-Type": "application/json"}, 
                                   timeout=10)
            self.log_test("Invalid JSON Handling", response.status_code == 400, 
                         f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Invalid JSON Handling", False, str(e))
        
        # Test 2: Missing required fields
        test_data = {"cpu_cores": 4}  # Missing other fields
        try:
            response = requests.post(f"{self.base_url}/mccva/route", 
                                   json=test_data, timeout=10)
            self.log_test("Missing Fields Handling", response.status_code == 400, 
                         f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Missing Fields Handling", False, str(e))
    
    def test_edge_cases(self):
        """Test edge cases và boundary conditions"""
        print("\n=== Edge Cases Test ===")
        
        edge_cases = [
            {
                "name": "Minimum Values",
                "data": {"cpu_cores": 1, "memory": 1, "storage": 10, "network_bandwidth": 100, "priority": 1}
            },
            {
                "name": "Maximum Values", 
                "data": {"cpu_cores": 16, "memory": 64, "storage": 1000, "network_bandwidth": 10000, "priority": 5}
            },
            {
                "name": "High Priority Low Resources",
                "data": {"cpu_cores": 1, "memory": 1, "storage": 10, "network_bandwidth": 100, "priority": 5}
            },
            {
                "name": "Low Priority High Resources",
                "data": {"cpu_cores": 16, "memory": 64, "storage": 1000, "network_bandwidth": 10000, "priority": 1}
            }
        ]
        
        for case in edge_cases:
            try:
                response = requests.post(f"{self.base_url}/mccva/route", 
                                       json=case["data"], timeout=15)
                if response.status_code == 200:
                    result = response.json()
                    self.log_test(f"Edge Case: {case['name']}", True, 
                                f"Server: {result.get('server')}")
                else:
                    self.log_test(f"Edge Case: {case['name']}", False, 
                                f"Status: {response.status_code}")
            except Exception as e:
                self.log_test(f"Edge Case: {case['name']}", False, str(e))
    
    def test_system_resilience(self):
        """Test system resilience với rapid requests"""
        print("\n=== System Resilience Test ===")
        
        def rapid_request():
            test_data = {
                "cpu_cores": random.randint(1, 16),
                "memory": random.randint(1, 64),
                "storage": random.randint(10, 1000),
                "network_bandwidth": random.randint(100, 10000),
                "priority": random.randint(1, 5)
            }
            
            try:
                response = requests.post(f"{self.base_url}/mccva/route", 
                                       json=test_data, timeout=10)
                return response.status_code == 200
            except:
                return False
        
        # Rapid fire test
        rapid_requests = 50
        successful = 0
        
        start_time = time.time()
        for i in range(rapid_requests):
            if rapid_request():
                successful += 1
            time.sleep(0.1)  # 100ms between requests
        end_time = time.time()
        
        success_rate = (successful / rapid_requests) * 100
        total_time = end_time - start_time
        
        self.log_test("System Resilience", success_rate >= 95, 
                     f"Success rate: {success_rate:.1f}% ({successful}/{rapid_requests}) in {total_time:.2f}s")
        
        self.results["performance_metrics"]["resilience_test"] = {
            "success_rate": success_rate,
            "total_requests": rapid_requests,
            "successful_requests": successful,
            "total_time": total_time
        }
    
    def run_all_tests(self):
        """Chạy tất cả advanced tests"""
        print("🚀 Starting Advanced MCCVA Test Suite...")
        print(f"Target host: {self.host_ip}")
        
        start_time = time.time()
        
        # Run all test categories
        self.test_ml_service_health()
        self.test_routing_consistency()
        self.test_load_balancing()
        self.test_stress_loading()
        self.test_failure_scenarios()
        self.test_edge_cases()
        self.test_system_resilience()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 ADVANCED TEST SUITE SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"Passed: {self.results['passed']}")
        print(f"Failed: {self.results['failed']}")
        print(f"Success Rate: {(self.results['passed']/self.results['total_tests'])*100:.1f}%")
        print(f"Total Time: {total_time:.2f}s")
        
        # Performance metrics
        if self.results["performance_metrics"]:
            print(f"\n📈 PERFORMANCE METRICS:")
            for test_name, metrics in self.results["performance_metrics"].items():
                print(f"  {test_name}:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.2f}")
                    else:
                        print(f"    {key}: {value}")
        
        # Error logs
        if self.results["error_logs"]:
            print(f"\n❌ ERROR LOGS:")
            for error in self.results["error_logs"]:
                print(f"  - {error}")
        
        # Final assessment
        overall_success = self.results["failed"] == 0
        print(f"\n🎯 FINAL ASSESSMENT: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        
        if overall_success:
            print("🎉 System is production-ready!")
        else:
            print("⚠️  System needs improvements before production deployment")
        
        return overall_success

def get_host_ip():
    """Lấy IP của host từ Docker container"""
    try:
        # Thử các cách khác nhau để kết nối đến host
        host_ip = None
        
        # Cách 1: host.docker.internal (macOS/Windows)
        try:
            socket.gethostbyname('host.docker.internal')
            return 'host.docker.internal'
        except:
            pass
        
        # Cách 2: gateway IP
        try:
            import subprocess
            result = subprocess.run(['ip', 'route', 'show', 'default'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gateway = result.stdout.split()[2]
                return gateway
        except:
            pass
        
        # Cách 3: 172.17.0.1 (Docker default gateway)
        return '172.17.0.1'
        
    except Exception as e:
        print(f"Error getting host IP: {e}")
        return '172.17.0.1'  # fallback

if __name__ == "__main__":
    host_ip = get_host_ip()
    print(f"Host IP detected: {host_ip}")
    
    tester = MCCVAAdvancedTester(host_ip)
    success = tester.run_all_tests()
    
    exit(0 if success else 1) 