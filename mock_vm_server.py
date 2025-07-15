#!/usr/bin/env python3
"""
Mock VM Server để test MCCVA request routing
Mô phỏng các VM pool small, medium, large để nhận request từ OpenResty
"""

import os
import json
import argparse
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

# Server configuration
SERVER_TYPE = os.environ.get('SERVER_TYPE', 'small')  # small, medium, large
SERVER_PORT = int(os.environ.get('SERVER_PORT', 8081))
SERVER_NAME = os.environ.get('SERVER_NAME', f'vm_{SERVER_TYPE}')
SERVER_STATS = {
    'received_requests': 0,
    'processed_requests': 0,
    'last_request_time': None,
    'server_type': SERVER_TYPE,
    'server_port': SERVER_PORT,
    'server_name': SERVER_NAME
}

# VM stats (simulated) based on server type
VM_STATS = {
    'small': {
        'cpu_cores': 2,
        'memory_gb': 4,
        'storage_gb': 100,
        'cpu_usage': 0.3,
        'memory_usage': 0.4,
        'storage_usage': 0.2,
        'response_time_ms': 15
    },
    'medium': {
        'cpu_cores': 4,
        'memory_gb': 16,
        'storage_gb': 200,
        'cpu_usage': 0.5,
        'memory_usage': 0.6,
        'storage_usage': 0.4,
        'response_time_ms': 25
    },
    'large': {
        'cpu_cores': 8,
        'memory_gb': 32,
        'storage_gb': 500,
        'cpu_usage': 0.7,
        'memory_usage': 0.8,
        'storage_usage': 0.6,
        'response_time_ms': 40
    }
}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'server_type': SERVER_TYPE,
        'server_name': SERVER_NAME,
        'server_port': SERVER_PORT,
        'received_requests': SERVER_STATS['received_requests'],
        'processed_requests': SERVER_STATS['processed_requests'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/process', methods=['POST'])
def process_request():
    """Process a workload request"""
    # Update stats
    SERVER_STATS['received_requests'] += 1
    SERVER_STATS['last_request_time'] = datetime.now().isoformat()
    
    try:
        # Get request data
        data = request.get_json() or {}
        
        # Extract headers
        routing_method = request.headers.get('X-MCCVA-Method', 'unknown')
        makespan = request.headers.get('X-Makespan', 'unknown')
        cluster = request.headers.get('X-Cluster', 'unknown')
        confidence = request.headers.get('X-Confidence', '0')
        algorithm = request.headers.get('X-Algorithm', 'unknown')
        
        # Process request (simulated)
        vm_stats = VM_STATS.get(SERVER_TYPE, VM_STATS['small'])
        
        # Create response
        response = {
            'processed': True,
            'server': {
                'type': SERVER_TYPE,
                'name': SERVER_NAME,
                'port': SERVER_PORT,
            },
            'request': {
                'data': data,
                'headers': {
                    'routing_method': routing_method,
                    'makespan': makespan,
                    'cluster': cluster,
                    'confidence': confidence,
                    'algorithm': algorithm
                }
            },
            'vm_stats': vm_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update stats
        SERVER_STATS['processed_requests'] += 1
        
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'server_type': SERVER_TYPE,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get server stats"""
    return jsonify({
        'server_stats': SERVER_STATS,
        'vm_stats': VM_STATS.get(SERVER_TYPE, VM_STATS['small']),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mock VM Server')
    parser.add_argument('--type', choices=['small', 'medium', 'large'], default=SERVER_TYPE,
                        help='Server type (small, medium, large)')
    parser.add_argument('--port', type=int, default=SERVER_PORT,
                        help='Server port')
    parser.add_argument('--name', default=SERVER_NAME,
                        help='Server name')
    
    args = parser.parse_args()
    
    # Update configuration
    SERVER_TYPE = args.type
    SERVER_PORT = args.port
    SERVER_NAME = args.name or f'vm_{SERVER_TYPE}'
    SERVER_STATS['server_type'] = SERVER_TYPE
    SERVER_STATS['server_port'] = SERVER_PORT
    SERVER_STATS['server_name'] = SERVER_NAME
    
    print(f"🚀 Starting Mock VM Server: {SERVER_NAME}")
    print(f"📊 Type: {SERVER_TYPE}, Port: {SERVER_PORT}")
    print(f"🌐 Endpoints: /health, /process (POST), /stats")
    
    app.run(host='0.0.0.0', port=SERVER_PORT, debug=False) 