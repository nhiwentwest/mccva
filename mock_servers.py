#!/usr/bin/env python3
"""
Mock Servers for MCCVA Testing
Chạy các mock servers trên các port khác nhau để test MCCVA routing
"""

import threading
import time
from flask import Flask, request, jsonify
import requests

# Tạo các Flask apps cho từng server
app_8081 = Flask(__name__)
app_8082 = Flask(__name__)
app_8083 = Flask(__name__)
app_8084 = Flask(__name__)
app_8085 = Flask(__name__)
app_8086 = Flask(__name__)
app_8087 = Flask(__name__)
app_8088 = Flask(__name__)

# Server configurations
servers = [
    {"app": app_8081, "port": 8081, "name": "VM-Low-1", "load": "low"},
    {"app": app_8082, "port": 8082, "name": "VM-Low-2", "load": "low"},
    {"app": app_8083, "port": 8083, "name": "VM-Medium-1", "load": "medium"},
    {"app": app_8084, "port": 8084, "name": "VM-Medium-2", "load": "medium"},
    {"app": app_8085, "port": 8085, "name": "VM-High-1", "load": "high"},
    {"app": app_8086, "port": 8086, "name": "VM-High-2", "load": "high"},
    {"app": app_8087, "port": 8087, "name": "VM-Balanced-1", "load": "balanced"},
    {"app": app_8088, "port": 8088, "name": "VM-Balanced-2", "load": "balanced"}
]

def create_server_routes(app, server_info):
    """Tạo routes cho từng server"""
    
    @app.route('/')
    def home():
        return jsonify({
            "server": server_info["name"],
            "load": server_info["load"],
            "port": server_info["port"],
            "status": "running",
            "timestamp": time.time()
        })
    
    @app.route('/process', methods=['POST'])
    def process_request():
        data = request.get_json() or {}
        
        # Simulate processing time based on load
        load_delays = {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.5,
            "balanced": 0.2
        }
        
        delay = load_delays.get(server_info["load"], 0.2)
        time.sleep(delay)
        
        return jsonify({
            "server": server_info["name"],
            "load": server_info["load"],
            "port": server_info["port"],
            "processed": True,
            "input_data": data,
            "processing_time": delay,
            "timestamp": time.time()
        })
    
    @app.route('/health')
    def health():
        return jsonify({
            "server": server_info["name"],
            "status": "healthy",
            "load": server_info["load"],
            "port": server_info["port"]
        })

def start_server(app, port, server_name):
    """Khởi động server trên port cụ thể"""
    try:
        print(f"Starting {server_name} on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except Exception as e:
        print(f"Error starting {server_name} on port {port}: {e}")

def main():
    """Khởi động tất cả mock servers"""
    print("Starting MCCVA Mock Servers...")
    
    # Tạo routes cho từng server
    for server in servers:
        create_server_routes(server["app"], server)
    
    # Khởi động tất cả servers trong threads riêng biệt
    threads = []
    for server in servers:
        thread = threading.Thread(
            target=start_server,
            args=(server["app"], server["port"], server["name"]),
            daemon=True
        )
        threads.append(thread)
        thread.start()
    
    print(f"Started {len(servers)} mock servers:")
    for server in servers:
        print(f"  - {server['name']}: http://localhost:{server['port']}")
    
    print("\nMock servers are running. Press Ctrl+C to stop.")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down mock servers...")

if __name__ == "__main__":
    main() 