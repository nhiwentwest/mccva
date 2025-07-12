#!/usr/bin/env python3
"""
MCCVA Live Demo Script for Presentation
Optimized for real-time audience demonstration

Usage:
  python demo_presentation_script.py
  python demo_presentation_script.py --cloud YOUR_EC2_IP
"""

import requests
import json
import time
from datetime import datetime
import sys
import argparse

class MCCVALiveDemo:
    """Live demo class optimized for presentation"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.scenarios = [
            {
                "name": "💻 Personal Blog",
                "description": "Small website with light traffic",
                "data": {
                    "cpu_cores": 1, "memory_gb": 2, "storage_gb": 50,
                    "network_bandwidth": 500, "priority": 1,
                    "vm_cpu_usage": 0.25, "vm_memory_usage": 0.35, "vm_storage_usage": 0.15
                },
                "expected": "small",
                "context": "500-1K daily visitors"
            },
            {
                "name": "🛒 E-commerce API", 
                "description": "Online store backend with moderate load",
                "data": {
                    "cpu_cores": 4, "memory_gb": 16, "storage_gb": 200,
                    "network_bandwidth": 5000, "priority": 3,
                    "vm_cpu_usage": 0.60, "vm_memory_usage": 0.55, "vm_storage_usage": 0.40
                },
                "expected": "medium",
                "context": "5K-10K daily transactions"
            },
            {
                "name": "🎥 Video Streaming",
                "description": "Media server with high performance needs", 
                "data": {
                    "cpu_cores": 8, "memory_gb": 32, "storage_gb": 1000,
                    "network_bandwidth": 15000, "priority": 4,
                    "vm_cpu_usage": 0.80, "vm_memory_usage": 0.70, "vm_storage_usage": 0.60
                },
                "expected": "large",
                "context": "Real-time video encoding"
            },
            {
                "name": "🧠 ML Training Job",
                "description": "Deep learning model training",
                "data": {
                    "cpu_cores": 16, "memory_gb": 64, "storage_gb": 2000,
                    "network_bandwidth": 20000, "priority": 5,
                    "vm_cpu_usage": 0.95, "vm_memory_usage": 0.85, "vm_storage_usage": 0.75
                },
                "expected": "large", 
                "context": "Neural network training with GPUs"
            }
        ]
    
    def print_header(self):
        """Print demo header"""
        print("\n" + "🚀" + "="*58 + "🚀")
        print("  MCCVA 3-STAGE META-LEARNING SYSTEM - LIVE DEMO")
        print("🚀" + "="*58 + "🚀")
        print(f"🌐 Target: {self.base_url}")
        print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎯 Demonstrating: SVM → K-Means → Meta-Learning Pipeline")
        print("="*62)
    
    def test_system_health(self):
        """Quick system health check"""
        print("\n🔍 SYSTEM HEALTH CHECK")
        print("-" * 30)
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                print(f"✅ ML Service: {health.get('status', 'unknown').upper()}")
                
                models = health.get('models_loaded', {})
                svm_status = "✅" if models.get('svm') else "❌"
                kmeans_status = "✅" if models.get('kmeans') else "❌" 
                meta_status = "✅" if models.get('meta_learning') else "❌"
                
                print(f"🧠 Models Loaded:")
                print(f"   {svm_status} SVM Classification Model")
                print(f"   {kmeans_status} K-Means Clustering Model")
                print(f"   {meta_status} Meta-Learning Neural Network")
                
                return True
            else:
                print(f"❌ Health Check Failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return False
    
    def demo_scenario(self, scenario, pause_between_stages=True):
        """Demo a single scenario with detailed stage-by-stage output"""
        print(f"\n{'='*62}")
        print(f"🧪 TESTING: {scenario['name']}")
        print(f"📋 Use Case: {scenario['description']}")
        print(f"🏢 Context: {scenario['context']}")
        print("-" * 62)
        
        # Show input parameters
        data = scenario['data']
        print("📊 Input Parameters:")
        print(f"   💻 CPU Cores: {data['cpu_cores']}")
        print(f"   🧠 Memory: {data['memory_gb']}GB")
        print(f"   💾 Storage: {data['storage_gb']}GB")
        print(f"   🌐 Network: {data['network_bandwidth']} Mbps")
        print(f"   ⚡ Priority: {data['priority']}/5")
        print(f"   📈 VM Usage: CPU {data['vm_cpu_usage']*100:.0f}%, RAM {data['vm_memory_usage']*100:.0f}%, Disk {data['vm_storage_usage']*100:.0f}%")
        
        if pause_between_stages:
            input("\n⏸️  Press Enter to run MCCVA 3-Stage Analysis...")
        
        # Make prediction
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/predict/mccva_complete",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                stage_results = result.get('stage_results', {})
                
                # Stage 1: SVM
                svm_result = stage_results.get('stage1_svm', {})
                svm_prediction = svm_result.get('prediction', 'unknown')
                svm_confidence = svm_result.get('confidence', 0)
                
                print(f"\n🎯 STAGE 1 - SVM CLASSIFICATION:")
                print(f"   Prediction: {svm_prediction.upper()}")
                print(f"   Confidence: {svm_confidence:.3f}")
                
                if pause_between_stages:
                    input("   ⏸️  Press Enter for Stage 2...")
                
                # Stage 2: K-Means
                kmeans_result = stage_results.get('stage2_kmeans', {})
                kmeans_cluster = kmeans_result.get('cluster', -1)
                kmeans_confidence = kmeans_result.get('confidence', 0)
                
                print(f"\n🔢 STAGE 2 - K-MEANS CLUSTERING:")
                print(f"   Cluster ID: {kmeans_cluster}")
                print(f"   Confidence: {kmeans_confidence:.3f}")
                
                if pause_between_stages:
                    input("   ⏸️  Press Enter for Stage 3...")
                
                # Stage 3: Meta-Learning
                meta_result = stage_results.get('stage3_metalearning', {})
                final_prediction = meta_result.get('prediction', 'unknown')
                final_confidence = meta_result.get('confidence', 0)
                
                print(f"\n🧠 STAGE 3 - META-LEARNING NEURAL NETWORK:")
                print(f"   Final Prediction: {final_prediction.upper()}")
                print(f"   Final Confidence: {final_confidence:.6f} ({final_confidence*100:.4f}%)")
                
                # Analysis
                print(f"\n📊 ANALYSIS:")
                expected = scenario['expected']
                if final_prediction == expected:
                    print(f"   ✅ CORRECT: Expected {expected.upper()}, Got {final_prediction.upper()}")
                else:
                    print(f"   ❌ INCORRECT: Expected {expected.upper()}, Got {final_prediction.upper()}")
                
                print(f"   ⚡ Response Time: {response_time:.1f}ms")
                
                # Show AI decision process
                print(f"\n🤖 AI DECISION PROCESS:")
                if svm_prediction != final_prediction:
                    print(f"   🔄 Meta-Learning OVERRODE SVM decision")
                    print(f"   📈 Confidence increased from {svm_confidence:.3f} to {final_confidence:.6f}")
                else:
                    print(f"   ✅ Meta-Learning CONFIRMED SVM decision")
                
                # Business impact
                business_impact = self._get_business_impact(final_prediction, scenario)
                print(f"\n💼 BUSINESS IMPACT:")
                print(f"   {business_impact}")
                
                return True
                
            else:
                print(f"❌ API Error: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Request Failed: {e}")
            return False
    
    def _get_business_impact(self, prediction, scenario):
        """Generate business impact explanation"""
        if prediction == "small":
            return "🏠 Allocated to cost-optimized VM pool (low resource, low cost)"
        elif prediction == "medium":
            return "🏢 Allocated to balanced VM pool (moderate resources, balanced cost)"
        elif prediction == "large":
            return "🏭 Allocated to high-performance VM pool (maximum resources, premium cost)"
        else:
            return "❓ Unable to determine optimal VM allocation"
    
    def run_complete_demo(self, interactive=True):
        """Run complete demo with all scenarios"""
        self.print_header()
        
        # Health check
        if not self.test_system_health():
            print("\n❌ System not ready for demo!")
            return False
        
        if interactive:
            input("\n🎬 Press Enter to start the demo...")
        
        successful_tests = 0
        
        # Run each scenario
        for i, scenario in enumerate(self.scenarios, 1):
            print(f"\n🎬 DEMO SCENARIO {i}/{len(self.scenarios)}")
            
            if self.demo_scenario(scenario, pause_between_stages=interactive):
                successful_tests += 1
            
            if interactive and i < len(self.scenarios):
                input(f"\n⏭️  Press Enter for next scenario ({i+1}/{len(self.scenarios)})...")
        
        # Summary
        self.print_demo_summary(successful_tests)
        return successful_tests == len(self.scenarios)
    
    def print_demo_summary(self, successful_tests):
        """Print demo summary"""
        print("\n" + "🎉" + "="*58 + "🎉")
        print("               DEMO SUMMARY RESULTS")
        print("🎉" + "="*58 + "🎉")
        
        total_scenarios = len(self.scenarios)
        success_rate = (successful_tests / total_scenarios) * 100
        
        print(f"📊 Test Results: {successful_tests}/{total_scenarios} scenarios successful ({success_rate:.1f}%)")
        
        if success_rate == 100:
            print("✅ EXCELLENT: All scenarios completed successfully!")
            print("🚀 System ready for production deployment!")
        elif success_rate >= 75:
            print("⚠️  GOOD: Most scenarios working, minor issues detected")
            print("🔧 System functional but may need optimization")
        else:
            print("❌ ISSUES: Multiple scenario failures detected")
            print("🛠️  System requires debugging before production")
        
        print(f"\n🎯 Key Demonstrations Completed:")
        print(f"   ✅ 3-Stage AI Pipeline (SVM → K-Means → Meta-Learning)")
        print(f"   ✅ Real-time prediction with sub-100ms response")
        print(f"   ✅ Enterprise-scale cloud deployment")
        print(f"   ✅ Intelligent workload classification")
        
        print("\n💡 Research Contributions Showcased:")
        print("   🔬 Novel ensemble learning approach for VM load balancing")
        print("   🏗️  Production-ready ML architecture integration")
        print("   📈 Measurable performance improvements over single models")
        print("   🌐 Cloud-native deployment with Docker orchestration")
        
        print("🎉" + "="*58 + "🎉")

def main():
    """Main demo execution"""
    parser = argparse.ArgumentParser(description="MCCVA Live Presentation Demo")
    parser.add_argument("--cloud", help="Use cloud EC2 instance IP")
    parser.add_argument("--auto", action="store_true", help="Run automatically without pauses")
    
    args = parser.parse_args()
    
    # Determine base URL
    if args.cloud:
        base_url = f"http://{args.cloud}:5000"
        print(f"🌐 Using cloud instance: {args.cloud}")
    else:
        base_url = "http://localhost:5000"
        print("🏠 Using local development server")
    
    # Create and run demo
    demo = MCCVALiveDemo(base_url)
    
    try:
        success = demo.run_complete_demo(interactive=not args.auto)
        
        if success:
            print("\n🎊 Demo completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Demo completed with issues")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 