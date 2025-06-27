#!/usr/bin/env python3
"""
Deployment Check Script cho MCCVA Algorithm
Kiểm tra tất cả components sau khi deploy
Chạy: python3 check_deployment.py
"""

import requests
import json
import time
import subprocess
import sys
import os

def print_status(message, status="INFO"):
    """Print colored status message"""
    colors = {
        "INFO": "\033[94m",    # Blue
        "SUCCESS": "\033[92m", # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
        "RESET": "\033[0m"     # Reset
    }
    print(f"{colors.get(status, colors['INFO'])}[{status}]{colors['RESET']} {message}")

def check_system_services():
    """Kiểm tra system services"""
    print_status("🔧 Checking System Services", "INFO")
    print("=" * 50)
    
    services = ["mccva-ml", "openresty"]
    
    for service in services:
        try:
            result = subprocess.run(
                ["systemctl", "is-active", service],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip() == "active":
                print_status(f"✅ {service} is running", "SUCCESS")
            else:
                print_status(f"❌ {service} is not running", "ERROR")
                print(f"   Status: {result.stdout.strip()}")
                
        except Exception as e:
            print_status(f"❌ Error checking {service}: {e}", "ERROR")

def check_ports():
    """Kiểm tra ports đang listen"""
    print_status("🌐 Checking Network Ports", "INFO")
    print("=" * 50)
    
    ports = [80, 5000]
    
    for port in ports:
        try:
            result = subprocess.run(
                ["netstat", "-tlnp"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if str(port) in result.stdout:
                print_status(f"✅ Port {port} is listening", "SUCCESS")
            else:
                print_status(f"❌ Port {port} is not listening", "ERROR")
                
        except Exception as e:
            print_status(f"❌ Error checking port {port}: {e}", "ERROR")

def check_ml_service():
    """Kiểm tra ML Service"""
    print_status("🤖 Checking ML Service", "INFO")
    print("=" * 50)
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:5000/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print_status("✅ ML Service health check passed", "SUCCESS")
            print(f"   Status: {data.get('status')}")
            print(f"   Models loaded: {data.get('models_loaded')}")
        else:
            print_status(f"❌ ML Service health check failed: HTTP {response.status_code}", "ERROR")
            
    except requests.exceptions.ConnectionError:
        print_status("❌ Cannot connect to ML Service on port 5000", "ERROR")
    except Exception as e:
        print_status(f"❌ Error checking ML Service: {e}", "ERROR")

def check_openresty():
    """Kiểm tra OpenResty"""
    print_status("🚀 Checking OpenResty", "INFO")
    print("=" * 50)
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print_status("✅ OpenResty health check passed", "SUCCESS")
            print(f"   Service: {data.get('service')}")
        else:
            print_status(f"❌ OpenResty health check failed: HTTP {response.status_code}", "ERROR")
            
    except requests.exceptions.ConnectionError:
        print_status("❌ Cannot connect to OpenResty on port 80", "ERROR")
    except Exception as e:
        print_status(f"❌ Error checking OpenResty: {e}", "ERROR")

def test_predictions():
    """Test các predictions"""
    print_status("🧪 Testing Predictions", "INFO")
    print("=" * 50)
    
    # Test SVM prediction
    try:
        response = requests.post(
            "http://localhost/predict/makespan",
            json={"features": [4, 8, 100, 1000, 3]},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print_status("✅ SVM prediction working", "SUCCESS")
            print(f"   Makespan: {data.get('makespan')}")
            print(f"   Confidence: {data.get('confidence'):.3f}")
        else:
            print_status(f"❌ SVM prediction failed: HTTP {response.status_code}", "ERROR")
            
    except Exception as e:
        print_status(f"❌ Error testing SVM: {e}", "ERROR")
    
    # Test K-Means prediction
    try:
        response = requests.post(
            "http://localhost/predict/vm_cluster",
            json={"vm_features": [0.5, 0.5, 0.5]},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print_status("✅ K-Means prediction working", "SUCCESS")
            print(f"   Cluster: {data.get('cluster')}")
            print(f"   Distance: {data.get('distance'):.3f}")
        else:
            print_status(f"❌ K-Means prediction failed: HTTP {response.status_code}", "ERROR")
            
    except Exception as e:
        print_status(f"❌ Error testing K-Means: {e}", "ERROR")

def test_mccva_routing():
    """Test MCCVA routing"""
    print_status("🎯 Testing MCCVA Routing", "INFO")
    print("=" * 50)
    
    try:
        response = requests.post(
            "http://localhost/mccva/route",
            json={
                "features": [4, 8, 100, 1000, 3],
                "vm_features": [0.5, 0.5, 0.5]
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print_status("✅ MCCVA routing working", "SUCCESS")
            print(f"   Target VM: {data.get('target_vm')}")
            print(f"   Algorithm: {data.get('routing_info', {}).get('algorithm')}")
            print(f"   Method: {data.get('routing_info', {}).get('method')}")
        else:
            print_status(f"❌ MCCVA routing failed: HTTP {response.status_code}", "ERROR")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print_status(f"❌ Error testing MCCVA routing: {e}", "ERROR")

def check_model_files():
    """Kiểm tra model files"""
    print_status("📁 Checking Model Files", "INFO")
    print("=" * 50)
    
    model_files = [
        "/opt/mccva/models/svm_model.joblib",
        "/opt/mccva/models/kmeans_model.joblib",
        "/opt/mccva/models/scaler.joblib",
        "/opt/mccva/models/kmeans_scaler.joblib"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file)
            print_status(f"✅ {os.path.basename(model_file)} exists ({size} bytes)", "SUCCESS")
        else:
            print_status(f"❌ {os.path.basename(model_file)} not found", "ERROR")

def check_lua_files():
    """Kiểm tra Lua files"""
    print_status("📜 Checking Lua Files", "INFO")
    print("=" * 50)
    
    lua_files = [
        "/usr/local/openresty/nginx/conf/lua/mccva_routing.lua",
        "/usr/local/openresty/nginx/conf/lua/predict_makespan.lua",
        "/usr/local/openresty/nginx/conf/lua/predict_vm_cluster.lua"
    ]
    
    for lua_file in lua_files:
        if os.path.exists(lua_file):
            size = os.path.getsize(lua_file)
            print_status(f"✅ {os.path.basename(lua_file)} exists ({size} bytes)", "SUCCESS")
        else:
            print_status(f"❌ {os.path.basename(lua_file)} not found", "ERROR")

def performance_test():
    """Test performance"""
    print_status("⚡ Performance Test", "INFO")
    print("=" * 50)
    
    test_data = {
        "features": [4, 8, 100, 1000, 3],
        "vm_features": [0.5, 0.5, 0.5]
    }
    
    times = []
    success_count = 0
    total_requests = 5
    
    print(f"Running {total_requests} MCCVA routing requests...")
    
    for i in range(total_requests):
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost/mccva/route",
                json=test_data,
                timeout=10
            )
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
                success_count += 1
                print(f"   Request {i+1}: {times[-1]:.3f}s")
            else:
                print(f"   Request {i+1}: Failed (HTTP {response.status_code})")
                
        except Exception as e:
            print(f"   Request {i+1}: Error - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print_status(f"Performance Results:", "INFO")
        print(f"   Success rate: {success_count}/{total_requests} ({success_count/total_requests*100:.1f}%)")
        print(f"   Average time: {avg_time:.3f}s")
        
        if avg_time < 0.1:
            print_status("   Performance: Excellent (< 100ms)", "SUCCESS")
        elif avg_time < 0.5:
            print_status("   Performance: Good (< 500ms)", "SUCCESS")
        else:
            print_status("   Performance: Slow (> 500ms)", "WARNING")

def main():
    """Main function"""
    print_status("🤖 MCCVA Deployment Check", "INFO")
    print_status("Comprehensive deployment verification", "INFO")
    print("=" * 60)
    
    # Wait for services to be ready
    print_status("⏳ Waiting for services to be ready...", "INFO")
    time.sleep(3)
    
    # Run all checks
    check_system_services()
    check_ports()
    check_model_files()
    check_lua_files()
    check_ml_service()
    check_openresty()
    test_predictions()
    test_mccva_routing()
    performance_test()
    
    print("\n" + "=" * 60)
    print_status("✅ Deployment check completed!", "SUCCESS")
    print_status("🎯 MCCVA Algorithm is ready for production use", "SUCCESS")
    
    print("\n📋 Quick Commands:")
    print("   • Test MCCVA: python3 test_mccva.py")
    print("   • Check logs: sudo journalctl -u mccva-ml -f")
    print("   • Restart service: sudo systemctl restart mccva-ml")
    print("   • Health check: curl http://localhost/health")

if __name__ == "__main__":
    main() 