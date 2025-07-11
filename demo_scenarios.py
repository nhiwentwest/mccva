#!/usr/bin/env python3
"""
🎯 MCCVA Stage 3 Meta-Learning Demo Scenarios
Comprehensive benchmarking script với 5 realistic workload scenarios
Author: MCCVA Team
Date: 2025-07-11
"""

import requests
import json
import time
from datetime import datetime
import sys

# Server configuration
ML_SERVICE_URL = "http://localhost:5000/predict/mccva_complete"

class MCCVADemoRunner:
    def __init__(self):
        self.scenarios = [
            {
                "name": "🖥️ Web Server - Light Load",
                "description": "Typical web server với moderate traffic, balanced resource usage",
                "workload": {
                    "cpu_cores": 2,
                    "memory_gb": 4.0,
                    "storage_gb": 100.0,
                    "network_bandwidth": 1000,
                    "priority": 2,
                    "vm_cpu_usage": 45.5,
                    "vm_memory_usage": 60.2,
                    "vm_storage_usage": 35.8
                },
                "expected": "small",
                "context": "E-commerce website với 1000-5000 users/day"
            },
            {
                "name": "🔬 Data Analytics - Heavy Compute",
                "description": "Machine learning training job với high CPU/Memory demands",
                "workload": {
                    "cpu_cores": 8,
                    "memory_gb": 32.0,
                    "storage_gb": 500.0,
                    "network_bandwidth": 10000,
                    "priority": 1,
                    "vm_cpu_usage": 95.8,
                    "vm_memory_usage": 88.5,
                    "vm_storage_usage": 75.2
                },
                "expected": "large",
                "context": "Deep learning model training trên dataset 100GB"
            },
            {
                "name": "💾 Database Server - Medium Load",
                "description": "Production database với moderate transactions",
                "workload": {
                    "cpu_cores": 4,
                    "memory_gb": 16.0,
                    "storage_gb": 1000.0,
                    "network_bandwidth": 5000,
                    "priority": 1,
                    "vm_cpu_usage": 65.3,
                    "vm_memory_usage": 78.9,
                    "vm_storage_usage": 82.1
                },
                "expected": "medium",
                "context": "PostgreSQL database với 50,000 transactions/hour"
            },
            {
                "name": "🎮 Game Server - Peak Traffic",
                "description": "Online game server trong peak hours với high network I/O",
                "workload": {
                    "cpu_cores": 6,
                    "memory_gb": 12.0,
                    "storage_gb": 200.0,
                    "network_bandwidth": 20000,
                    "priority": 1,
                    "vm_cpu_usage": 78.4,
                    "vm_memory_usage": 85.6,
                    "vm_storage_usage": 45.3
                },
                "expected": "medium",
                "context": "MMORPG server với 5000 concurrent players"
            },
            {
                "name": "📁 File Storage - Backup Operation",
                "description": "File server thực hiện daily backup với high storage I/O",
                "workload": {
                    "cpu_cores": 2,
                    "memory_gb": 8.0,
                    "storage_gb": 2000.0,
                    "network_bandwidth": 1000,
                    "priority": 3,
                    "vm_cpu_usage": 35.2,
                    "vm_memory_usage": 45.8,
                    "vm_storage_usage": 95.7
                },
                "expected": "medium",
                "context": "NAS server backup 500GB data to cloud storage"
            }
        ]
        
        self.results = []
        
    def print_header(self):
        """Print demo header"""
        print("=" * 80)
        print("🎯 MCCVA Stage 3 Meta-Learning System - Live Demo")
        print("   3-Stage Pipeline: SVM → K-Means → Meta-Learning Neural Network")
        print("=" * 80)
        print(f"⏰ Demo Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔗 Service URL: {ML_SERVICE_URL}")
        print("=" * 80)
        print()
        
    def run_scenario(self, scenario):
        """Run single scenario prediction"""
        print(f"📊 {scenario['name']}")
        print(f"   Context: {scenario['context']}")
        print(f"   Description: {scenario['description']}")
        print()
        
        # Print input parameters
        workload = scenario['workload']
        print("   🔧 Input Parameters:")
        print(f"      • CPU Cores: {workload['cpu_cores']}")
        print(f"      • Memory: {workload['memory_gb']} GB")
        print(f"      • Storage: {workload['storage_gb']} GB")
        print(f"      • Network Bandwidth: {workload['network_bandwidth']} Mbps")
        print(f"      • Priority Level: {workload['priority']}")
        print(f"      • CPU Usage: {workload['vm_cpu_usage']}%")
        print(f"      • Memory Usage: {workload['vm_memory_usage']}%")
        print(f"      • Storage Usage: {workload['vm_storage_usage']}%")
        print()
        
        try:
            # Make prediction request
            print("   🔄 Processing through MCCVA 3-Stage Pipeline...")
            start_time = time.time()
            
            response = requests.post(
                ML_SERVICE_URL,
                json=workload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract key results
                prediction = result.get('makespan', 'unknown')
                confidence = result.get('confidence', 0) * 100
                method = result.get('method', 'unknown')
                cached = result.get('cached', False)
                
                # Stage results
                stage_results = result.get('stage_results', {})
                svm_pred = stage_results.get('stage1_svm', {}).get('prediction', 'unknown')
                svm_conf = stage_results.get('stage1_svm', {}).get('confidence', 0)
                kmeans_cluster = stage_results.get('stage2_kmeans', {}).get('cluster', 0)
                kmeans_conf = stage_results.get('stage2_kmeans', {}).get('confidence', 0)
                meta_conf = stage_results.get('stage3_metalearning', {}).get('confidence', 0) * 100
                
                # Print results
                print("   ✅ MCCVA Prediction Results:")
                print(f"      🎯 Final Makespan: {prediction.upper()}")
                print(f"      📈 Overall Confidence: {confidence:.2f}%")
                print(f"      ⚡ Response Time: {response_time:.3f}s")
                print(f"      🔄 Method: {method}")
                if cached:
                    print("      💾 Result: CACHED (Optimized)")
                print()
                
                print("   📋 Stage-by-Stage Breakdown:")
                print(f"      Stage 1 (SVM): {svm_pred} (confidence: {svm_conf:.3f})")
                print(f"      Stage 2 (K-Means): Cluster {kmeans_cluster} (confidence: {kmeans_conf:.6f})")
                print(f"      Stage 3 (Meta-Learning): {prediction} (confidence: {meta_conf:.2f}%)")
                print()
                
                # Accuracy check
                expected = scenario.get('expected', 'unknown')
                accuracy = "✅ CORRECT" if prediction == expected else "❌ DIFFERENT"
                print(f"   🎯 Expected: {expected.upper()} | Predicted: {prediction.upper()} | {accuracy}")
                
                # Store result
                self.results.append({
                    'scenario': scenario['name'],
                    'expected': expected,
                    'predicted': prediction,
                    'confidence': confidence,
                    'response_time': response_time,
                    'correct': prediction == expected,
                    'cached': cached
                })
                
            else:
                print(f"   ❌ ERROR: HTTP {response.status_code}")
                print(f"      Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            
        print("-" * 80)
        print()
        
    def run_all_scenarios(self):
        """Run all demo scenarios"""
        self.print_header()
        
        for i, scenario in enumerate(self.scenarios, 1):
            print(f"🔸 Scenario {i}/{len(self.scenarios)}")
            self.run_scenario(scenario)
            
            # Pause between scenarios for demo effect
            if i < len(self.scenarios):
                time.sleep(2)
                
        self.print_summary()
        
    def print_summary(self):
        """Print demo summary"""
        print("=" * 80)
        print("📊 MCCVA Demo Summary & Performance Analysis")
        print("=" * 80)
        
        if not self.results:
            print("❌ No results to summarize")
            return
            
        # Calculate metrics
        total_scenarios = len(self.results)
        correct_predictions = sum(1 for r in self.results if r['correct'])
        accuracy_rate = (correct_predictions / total_scenarios) * 100
        avg_confidence = sum(r['confidence'] for r in self.results) / total_scenarios
        avg_response_time = sum(r['response_time'] for r in self.results) / total_scenarios
        cached_results = sum(1 for r in self.results if r['cached'])
        
        print(f"🎯 Overall Performance:")
        print(f"   • Total Scenarios: {total_scenarios}")
        print(f"   • Correct Predictions: {correct_predictions}/{total_scenarios}")
        print(f"   • Accuracy Rate: {accuracy_rate:.1f}%")
        print(f"   • Average Confidence: {avg_confidence:.1f}%")
        print(f"   • Average Response Time: {avg_response_time:.3f}s")
        print(f"   • Cached Results: {cached_results}/{total_scenarios}")
        print()
        
        # Detailed results table
        print("📋 Detailed Results:")
        print("   Scenario                          | Expected | Predicted | Confidence | Time   | Status")
        print("   " + "-" * 85)
        
        for result in self.results:
            scenario_short = result['scenario'][:30].ljust(30)
            expected = result['expected'].center(8)
            predicted = result['predicted'].center(9)
            confidence = f"{result['confidence']:.1f}%".rjust(10)
            response_time = f"{result['response_time']:.3f}s".rjust(6)
            status = "✅" if result['correct'] else "❌"
            cached_icon = "💾" if result['cached'] else "🔄"
            
            print(f"   {scenario_short} | {expected} | {predicted} | {confidence} | {response_time} | {status}{cached_icon}")
            
        print()
        
        # System health check
        print("🏥 System Health Check:")
        if accuracy_rate >= 80:
            print("   ✅ Excellent: High prediction accuracy")
        elif accuracy_rate >= 60:
            print("   ⚠️  Good: Acceptable prediction accuracy")
        else:
            print("   ❌ Warning: Low prediction accuracy - needs investigation")
            
        if avg_confidence >= 90:
            print("   ✅ Excellent: Very high confidence levels")
        elif avg_confidence >= 70:
            print("   ⚠️  Good: Acceptable confidence levels")
        else:
            print("   ❌ Warning: Low confidence - model uncertainty")
            
        if avg_response_time <= 1.0:
            print("   ✅ Excellent: Fast response times")
        elif avg_response_time <= 3.0:
            print("   ⚠️  Good: Acceptable response times")
        else:
            print("   ❌ Warning: Slow response times")
            
        print()
        print("🎉 MCCVA Stage 3 Meta-Learning Demo Complete!")
        print("   Ready for production deployment and real-world usage.")
        print("=" * 80)

def main():
    """Main demo execution"""
    demo = MCCVADemoRunner()
    
    try:
        demo.run_all_scenarios()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 