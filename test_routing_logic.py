#!/usr/bin/env python3
"""
Test script cho MCCVA routing logic và retry/fallback
Kiểm tra các kịch bản: bình thường, server lỗi, concurrent requests
"""

import requests
import json
import time
import threading
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

class MCCVATester:
    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url
        self.mock_ports = list(range(8081, 8089))  # 8081-8088
        self.test_data = {
            "cpu_cores": 4,
            "memory": 8,
            "storage": 100,
            "network_bandwidth": 1000,
            "priority": 3
        }
    
    def test_health_endpoints(self) -> Dict[str, bool]:
        """Test health của tất cả mock servers"""
        print("🔍 Testing health endpoints...")
        results = {}
        
        for port in self.mock_ports:
            try:
                response = requests.get(f"{self.base_url}:{port}/health", timeout=5)
                results[f"port_{port}"] = response.status_code == 200
                status = "✅" if response.status_code == 200 else "❌"
                print(f"  {status} Port {port}: {response.status_code}")
            except Exception as e:
                results[f"port_{port}"] = False
                print(f"  ❌ Port {port}: Connection failed - {e}")
        
        return results
    
    def test_mccva_routing(self, test_name: str = "Normal") -> Dict[str, Any]:
        """Test MCCVA routing endpoint"""
        print(f"\n🚀 Testing MCCVA routing: {test_name}")
        
        try:
            response = requests.post(
                f"{self.base_url}/mccva/route",
                json=self.test_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            result = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.json() if response.status_code == 200 else None,
                "error": response.text if response.status_code != 200 else None
            }
            
            if result["success"]:
                print(f"  ✅ Success: {response.status_code}")
                if "routing_info" in result["response"]:
                    print(f"     Method: {result['response']['routing_info'].get('method', 'unknown')}")
                    print(f"     Target VM: {result['response'].get('target_vm', 'unknown')}")
                    if result["response"].get("retry"):
                        print(f"     🔄 Retry: {result['response'].get('retry')}")
                        print(f"     Tried VMs: {len(result['response'].get('tried_vms', []))}")
            else:
                print(f"  ❌ Failed: {response.status_code}")
                print(f"     Error: {result['error']}")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            return {
                "status_code": 0,
                "success": False,
                "response": None,
                "error": str(e)
            }
    
    def kill_mock_server(self, port: int) -> bool:
        """Tắt mock server trên port cụ thể"""
        try:
            subprocess.run(f"sudo fuser -k {port}/tcp", shell=True, check=True)
            print(f"  🔴 Killed mock server on port {port}")
            time.sleep(2)  # Đợi process tắt hoàn toàn
            return True
        except subprocess.CalledProcessError:
            print(f"  ⚠️  No process found on port {port}")
            return False
    
    def test_retry_fallback(self) -> List[Dict[str, Any]]:
        """Test retry/fallback logic"""
        print("\n🔄 Testing retry/fallback logic...")
        results = []
        
        # Test 1: Tắt 1 server (primary)
        print("\n  Test 1: Kill primary server (8081)")
        self.kill_mock_server(8081)
        result1 = self.test_mccva_routing("Primary server down")
        result1["test_case"] = "primary_down"
        results.append(result1)
        
        # Test 2: Tắt thêm 1 server nữa (backup)
        print("\n  Test 2: Kill backup server (8082)")
        self.kill_mock_server(8082)
        result2 = self.test_mccva_routing("Primary + backup down")
        result2["test_case"] = "primary_backup_down"
        results.append(result2)
        
        # Test 3: Tắt thêm server nữa
        print("\n  Test 3: Kill more servers (8083, 8084)")
        self.kill_mock_server(8083)
        self.kill_mock_server(8084)
        result3 = self.test_mccva_routing("Multiple servers down")
        result3["test_case"] = "multiple_down"
        results.append(result3)
        
        return results
    
    def test_concurrent_requests(self, num_requests: int = 10) -> List[Dict[str, Any]]:
        """Test concurrent requests"""
        print(f"\n⚡ Testing {num_requests} concurrent requests...")
        
        def make_request():
            return self.test_mccva_routing("Concurrent")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in futures]
        
        success_count = sum(1 for r in results if r["success"])
        print(f"  📊 Results: {success_count}/{num_requests} successful")
        
        return results
    
    def test_ml_service(self) -> Dict[str, Any]:
        """Test ML Service (Flask API)"""
        print("\n🤖 Testing ML Service...")
        
        try:
            # Test health
            health_response = requests.get(f"{self.base_url}:5000/health", timeout=5)
            print(f"  Health: {health_response.status_code}")
            
            # Test makespan prediction
            makespan_response = requests.post(
                f"{self.base_url}:5000/predict/makespan",
                json={"features": [4, 8, 100, 1000, 3]},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            result = {
                "health_ok": health_response.status_code == 200,
                "makespan_ok": makespan_response.status_code == 200,
                "makespan_result": makespan_response.json() if makespan_response.status_code == 200 else None
            }
            
            if result["makespan_ok"]:
                print(f"  ✅ Makespan prediction: {result['makespan_result']}")
            else:
                print(f"  ❌ Makespan failed: {makespan_response.status_code}")
            
            return result
            
        except Exception as e:
            print(f"  ❌ ML Service test failed: {e}")
            return {"health_ok": False, "makespan_ok": False, "error": str(e)}
    
    def run_full_test_suite(self):
        """Chạy toàn bộ test suite"""
        print("🧪 MCCVA Routing Logic Test Suite")
        print("=" * 50)
        
        # Test 1: Health check
        health_results = self.test_health_endpoints()
        
        # Test 2: ML Service
        ml_results = self.test_ml_service()
        
        # Test 3: Normal routing
        normal_result = self.test_mccva_routing("Normal")
        
        # Test 4: Retry/fallback
        retry_results = self.test_retry_fallback()
        
        # Test 5: Concurrent requests
        concurrent_results = self.test_concurrent_requests(5)
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)
        
        healthy_servers = sum(health_results.values())
        print(f"Mock Servers: {healthy_servers}/{len(self.mock_ports)} healthy")
        print(f"ML Service: {'✅' if ml_results['health_ok'] else '❌'}")
        print(f"Normal Routing: {'✅' if normal_result['success'] else '❌'}")
        
        retry_success = sum(1 for r in retry_results if r['success'])
        print(f"Retry/Fallback: {retry_success}/{len(retry_results)} successful")
        
        concurrent_success = sum(1 for r in concurrent_results if r['success'])
        print(f"Concurrent: {concurrent_success}/{len(concurrent_results)} successful")
        
        return {
            "health": health_results,
            "ml_service": ml_results,
            "normal_routing": normal_result,
            "retry_fallback": retry_results,
            "concurrent": concurrent_results
        }

def main():
    """Main function"""
    tester = MCCVATester()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "health":
            tester.test_health_endpoints()
        elif sys.argv[1] == "routing":
            tester.test_mccva_routing()
        elif sys.argv[1] == "retry":
            tester.test_retry_fallback()
        elif sys.argv[1] == "concurrent":
            tester.test_concurrent_requests()
        elif sys.argv[1] == "ml":
            tester.test_ml_service()
        else:
            print("Usage: python test_routing_logic.py [health|routing|retry|concurrent|ml]")
    else:
        # Run full test suite
        tester.run_full_test_suite()

if __name__ == "__main__":
    main() 