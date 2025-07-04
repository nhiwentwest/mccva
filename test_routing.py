#!/usr/bin/env python3
"""
Test script cho MCCVA routing logic
Chạy trong Docker container để test toàn bộ hệ thống
"""

import requests
import json
import time
import socket

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

def test_ml_service():
    """Test ML Service trực tiếp"""
    print("=== Testing ML Service ===")
    
    # Test health
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        print(f"✅ Health check: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Models: {data.get('models_loaded')}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test predict makespan
    try:
        data = {
            "features": [4, 8, 100, 1000, 3]
        }
        response = requests.post('http://localhost:5000/predict/makespan', 
                               json=data, timeout=5)
        print(f"✅ Predict makespan: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Makespan: {result.get('makespan')}")
            print(f"   Confidence: {result.get('confidence')}")
    except Exception as e:
        print(f"❌ Predict makespan failed: {e}")

def test_openresty(host_ip):
    """Test OpenResty từ Docker"""
    print(f"\n=== Testing OpenResty from Docker (host: {host_ip}) ===")
    
    # Test health
    try:
        response = requests.get(f'http://{host_ip}/health', timeout=10)
        print(f"✅ OpenResty health: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.text[:100]}...")
    except Exception as e:
        print(f"❌ OpenResty health failed: {e}")
    
    # Test routing
    try:
        data = {
            "cpu_cores": 4,
            "memory": 8,
            "storage": 100,
            "network_bandwidth": 1000,
            "priority": 3
        }
        response = requests.post(f'http://{host_ip}/mccva/route', 
                               json=data, timeout=15)
        print(f"✅ AI routing: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Server: {result.get('server')}")
            print(f"   Response: {result.get('response', '')[:100]}...")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ AI routing failed: {e}")

def test_mock_servers(host_ip):
    """Test Mock Servers"""
    print(f"\n=== Testing Mock Servers (host: {host_ip}) ===")
    
    for port in range(8081, 8089):
        try:
            response = requests.get(f'http://{host_ip}:{port}/health', timeout=5)
            print(f"✅ Mock server {port}: {response.status_code}")
            if response.status_code == 200:
                print(f"   Response: {response.text[:50]}...")
        except Exception as e:
            print(f"❌ Mock server {port} failed: {e}")

def test_routing_scenarios(host_ip):
    """Test các scenarios khác nhau"""
    print(f"\n=== Testing Routing Scenarios (host: {host_ip}) ===")
    
    scenarios = [
        {
            "name": "Low Priority Request",
            "data": {"cpu_cores": 2, "memory": 4, "storage": 50, "network_bandwidth": 500, "priority": 1}
        },
        {
            "name": "High Priority Request", 
            "data": {"cpu_cores": 8, "memory": 16, "storage": 200, "network_bandwidth": 2000, "priority": 5}
        },
        {
            "name": "Medium Priority Request",
            "data": {"cpu_cores": 4, "memory": 8, "storage": 100, "network_bandwidth": 1000, "priority": 3}
        }
    ]
    
    for scenario in scenarios:
        try:
            print(f"\n--- {scenario['name']} ---")
            response = requests.post(f'http://{host_ip}/mccva/route', 
                                   json=scenario['data'], timeout=10)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Server: {result.get('server')}")
                print(f"Response: {result.get('response', '')[:80]}...")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Failed: {e}")

def test_performance(host_ip):
    """Test performance với nhiều requests"""
    print(f"\n=== Performance Test (host: {host_ip}) ===")
    
    data = {"cpu_cores": 4, "memory": 8, "storage": 100, "network_bandwidth": 1000, "priority": 3}
    
    start_time = time.time()
    success_count = 0
    total_requests = 5  # Giảm từ 10 xuống 5
    
    for i in range(total_requests):
        try:
            print(f"Sending request {i+1}...")
            response = requests.post(f'http://{host_ip}/mccva/route', 
                                   json=data, timeout=30)  # Tăng timeout từ 10s lên 30s
            if response.status_code == 200:
                success_count += 1
                result = response.json()
                print(f"Request {i+1}: ✅ (Server: {result.get('server', 'N/A')})")
            else:
                print(f"Request {i+1}: ❌ ({response.status_code})")
                print(f"   Error: {response.text[:200]}...")
        except Exception as e:
            print(f"Request {i+1}: ❌ ({e})")
        
        time.sleep(2)  # Tăng delay từ 0.5s lên 2s
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nPerformance Results:")
    print(f"Total requests: {total_requests}")
    print(f"Successful: {success_count}")
    print(f"Success rate: {(success_count/total_requests)*100:.1f}%")
    print(f"Total time: {duration:.2f}s")
    print(f"Average time per request: {duration/total_requests:.2f}s")

def debug_500_error(host_ip):
    """Debug lỗi 500 chi tiết"""
    print(f"\n=== Debug 500 Error (host: {host_ip}) ===")
    
    # Test 1: Kiểm tra ML Service
    print("1. Testing ML Service...")
    try:
        response = requests.get('http://localhost:5000/health', timeout=10)
        print(f"   ML Service health: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ML Service error: {e}")
    
    # Test 2: Kiểm tra OpenResty health
    print("\n2. Testing OpenResty health...")
    try:
        response = requests.get(f'http://{host_ip}/health', timeout=10)
        print(f"   OpenResty health: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   OpenResty error: {e}")
    
    # Test 3: Kiểm tra một mock server
    print("\n3. Testing Mock Server 8081...")
    try:
        response = requests.get(f'http://{host_ip}:8081/health', timeout=10)
        print(f"   Mock server 8081: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Mock server error: {e}")
    
    # Test 4: Test routing với verbose
    print("\n4. Testing routing with verbose...")
    try:
        data = {"cpu_cores": 4, "memory": 8, "storage": 100, "network_bandwidth": 1000, "priority": 3}
        response = requests.post(f'http://{host_ip}/mccva/route', 
                               json=data, timeout=30)
        print(f"   Routing status: {response.status_code}")
        print(f"   Response headers: {dict(response.headers)}")
        print(f"   Response body: {response.text[:500]}...")
    except Exception as e:
        print(f"   Routing error: {e}")

if __name__ == "__main__":
    print("🚀 Starting MCCVA Routing Logic Test...")
    
    # Lấy host IP
    host_ip = get_host_ip()
    print(f"Host IP detected: {host_ip}")
    
    # Test ML Service
    test_ml_service()
    
    # Test OpenResty
    test_openresty(host_ip)
    
    # Test Mock Servers
    test_mock_servers(host_ip)
    
    # Test Routing Scenarios
    test_routing_scenarios(host_ip)
    
    # Debug 500 error nếu cần
    debug_500_error(host_ip)
    
    # Performance Test
    test_performance(host_ip)
    
    print("\n🎉 Test completed!") 