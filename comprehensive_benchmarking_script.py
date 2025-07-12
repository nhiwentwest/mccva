#!/usr/bin/env python3
"""
MCCVA Comprehensive Benchmarking Script
Live Demo & Performance Analysis for Presentation

This script provides:
- 5 realistic workload scenarios
- Complete 3-stage ensemble testing
- Performance metrics analysis
- Live demo script for presentation
- Issue diagnosis and reporting
"""

import requests
import json
import time
import statistics
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any
import sys
import argparse

@dataclass
class WorkloadScenario:
    """Realistic workload scenario for testing"""
    name: str
    description: str
    cpu_cores: int
    memory_gb: float
    storage_gb: float
    network_bandwidth: int
    priority: int
    vm_cpu_usage: float
    vm_memory_usage: float
    vm_storage_usage: float
    expected_makespan: str
    business_context: str

@dataclass
class TestResult:
    """Test result with detailed stage analysis"""
    scenario_name: str
    input_params: Dict[str, Any]
    svm_prediction: str
    svm_confidence: float
    kmeans_cluster: int
    kmeans_confidence: float
    meta_prediction: str
    meta_confidence: float
    response_time_ms: float
    success: bool
    expected_vs_actual: str
    issues_detected: List[str]

class MCCVABenchmarkTester:
    """Comprehensive MCCVA system tester"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        self.scenarios = self._create_realistic_scenarios()
        
    def _create_realistic_scenarios(self) -> List[WorkloadScenario]:
        """Create 5 realistic workload scenarios for demo"""
        return [
            WorkloadScenario(
                name="Personal Blog",
                description="Small WordPress site with light traffic",
                cpu_cores=1,
                memory_gb=2,
                storage_gb=50,
                network_bandwidth=500,
                priority=1,
                vm_cpu_usage=0.25,
                vm_memory_usage=0.35,
                vm_storage_usage=0.15,
                expected_makespan="small",
                business_context="500-1K daily visitors, simple content"
            ),
            
            WorkloadScenario(
                name="E-commerce API",
                description="Online store backend with moderate traffic",
                cpu_cores=4,
                memory_gb=16,
                storage_gb=200,
                network_bandwidth=5000,
                priority=3,
                vm_cpu_usage=0.60,
                vm_memory_usage=0.55,
                vm_storage_usage=0.40,
                expected_makespan="medium",
                business_context="5K-10K daily transactions, database queries"
            ),
            
            WorkloadScenario(
                name="Video Streaming",
                description="Media server for live streaming platform",
                cpu_cores=8,
                memory_gb=32,
                storage_gb=1000,
                network_bandwidth=15000,
                priority=4,
                vm_cpu_usage=0.80,
                vm_memory_usage=0.70,
                vm_storage_usage=0.60,
                expected_makespan="large",
                business_context="Real-time encoding, high bandwidth"
            ),
            
            WorkloadScenario(
                name="ML Training Job",
                description="Deep learning model training with GPU acceleration",
                cpu_cores=16,
                memory_gb=64,
                storage_gb=2000,
                network_bandwidth=20000,
                priority=5,
                vm_cpu_usage=0.95,
                vm_memory_usage=0.85,
                vm_storage_usage=0.75,
                expected_makespan="large",
                business_context="Neural network training, tensor processing"
            ),
            
            WorkloadScenario(
                name="Development Environment",
                description="Software development with IDE and testing tools",
                cpu_cores=2,
                memory_gb=8,
                storage_gb=300,
                network_bandwidth=2000,
                priority=2,
                vm_cpu_usage=0.45,
                vm_memory_usage=0.60,
                vm_storage_usage=0.30,
                expected_makespan="small",
                business_context="Code compilation, unit testing, Git operations"
            )
        ]
    
    def test_health_check(self) -> bool:
        """Test ML service health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health Check: {health_data.get('status', 'unknown')}")
                
                models = health_data.get('models_loaded', {})
                for model, loaded in models.items():
                    status = "✅" if loaded else "❌"
                    print(f"   {status} {model}")
                return True
            else:
                print(f"❌ Health Check Failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health Check Error: {e}")
            return False
    
    def test_scenario(self, scenario: WorkloadScenario) -> TestResult:
        """Test a single workload scenario"""
        print(f"\n🧪 Testing: {scenario.name}")
        print(f"   Context: {scenario.business_context}")
        
        # Prepare request
        request_data = {
            "cpu_cores": scenario.cpu_cores,
            "memory_gb": scenario.memory_gb,
            "storage_gb": scenario.storage_gb,
            "network_bandwidth": scenario.network_bandwidth,
            "priority": scenario.priority,
            "vm_cpu_usage": scenario.vm_cpu_usage,
            "vm_memory_usage": scenario.vm_memory_usage,
            "vm_storage_usage": scenario.vm_storage_usage
        }
        
        # Measure response time
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/predict/mccva_complete",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                result_data = response.json()
                
                # Extract stage results
                stage_results = result_data.get('stage_results', {})
                svm_result = stage_results.get('stage1_svm', {})
                kmeans_result = stage_results.get('stage2_kmeans', {})
                meta_result = stage_results.get('stage3_metalearning', {})
                
                svm_prediction = svm_result.get('prediction', 'unknown')
                svm_confidence = svm_result.get('confidence', 0.0)
                kmeans_cluster = kmeans_result.get('cluster', -1)
                kmeans_confidence = kmeans_result.get('confidence', 0.0)
                meta_prediction = meta_result.get('prediction', 'unknown')
                meta_confidence = meta_result.get('confidence', 0.0)
                
                # Analyze issues
                issues = self._analyze_prediction_issues(
                    scenario, svm_prediction, svm_confidence, 
                    kmeans_cluster, kmeans_confidence, 
                    meta_prediction, meta_confidence
                )
                
                # Expected vs actual
                expected_vs_actual = f"Expected: {scenario.expected_makespan}, Got: {meta_prediction}"
                if scenario.expected_makespan == meta_prediction:
                    expected_vs_actual += " ✅"
                else:
                    expected_vs_actual += " ❌"
                
                result = TestResult(
                    scenario_name=scenario.name,
                    input_params=request_data,
                    svm_prediction=svm_prediction,
                    svm_confidence=svm_confidence,
                    kmeans_cluster=kmeans_cluster,
                    kmeans_confidence=kmeans_confidence,
                    meta_prediction=meta_prediction,
                    meta_confidence=meta_confidence,
                    response_time_ms=response_time,
                    success=True,
                    expected_vs_actual=expected_vs_actual,
                    issues_detected=issues
                )
                
                print(f"   ✅ Response: {response_time:.1f}ms")
                print(f"   📊 SVM: {svm_prediction} (conf: {svm_confidence:.3f})")
                print(f"   🔢 K-Means: cluster {kmeans_cluster} (conf: {kmeans_confidence:.3f})")
                print(f"   🧠 Meta: {meta_prediction} (conf: {meta_confidence:.6f})")
                print(f"   🎯 {expected_vs_actual}")
                
                if issues:
                    print("   ⚠️  Issues detected:")
                    for issue in issues:
                        print(f"      - {issue}")
                
                return result
                
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                return self._create_error_result(scenario, f"HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Request Error: {e}")
            return self._create_error_result(scenario, str(e))
    
    def _analyze_prediction_issues(self, scenario: WorkloadScenario, 
                                 svm_pred: str, svm_conf: float,
                                 kmeans_cluster: int, kmeans_conf: float,
                                 meta_pred: str, meta_conf: float) -> List[str]:
        """Analyze prediction for potential issues"""
        issues = []
        
        # Check for identical SVM confidence (indicates broken model)
        if abs(svm_conf - 2.236) < 0.001:
            issues.append("SVM showing identical confidence 2.236 (model likely broken)")
        
        # Check for dominant cluster
        if kmeans_cluster == 5:
            issues.append("K-Means always predicting cluster 5 (possible over-clustering)")
        
        # Check for meta-learning bias
        if meta_pred == "small" and meta_conf > 0.999:
            issues.append("Meta-Learning showing extreme bias toward 'small' class")
        
        # Check logical consistency
        if scenario.cpu_cores >= 8 and scenario.memory_gb >= 32 and meta_pred == "small":
            issues.append("Logical inconsistency: Heavy workload predicted as 'small'")
        
        # Check confidence correlation
        if svm_pred != meta_pred and meta_conf > 0.99:
            issues.append("Meta-Learning overriding SVM with very high confidence")
        
        return issues
    
    def _create_error_result(self, scenario: WorkloadScenario, error: str) -> TestResult:
        """Create error result"""
        return TestResult(
            scenario_name=scenario.name,
            input_params={},
            svm_prediction="error",
            svm_confidence=0.0,
            kmeans_cluster=-1,
            kmeans_confidence=0.0,
            meta_prediction="error",
            meta_confidence=0.0,
            response_time_ms=0.0,
            success=False,
            expected_vs_actual=f"Error: {error}",
            issues_detected=[error]
        )
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all scenarios and generate comprehensive report"""
        print("🚀 MCCVA COMPREHENSIVE BENCHMARKING")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target: {self.base_url}")
        print(f"Scenarios: {len(self.scenarios)}")
        
        # Health check first
        if not self.test_health_check():
            print("❌ System not healthy - aborting tests")
            return {"status": "failed", "reason": "health_check_failed"}
        
        # Run all scenarios
        for scenario in self.scenarios:
            result = self.test_scenario(scenario)
            self.results.append(result)
            time.sleep(1)  # Brief pause between tests
        
        # Generate summary
        return self._generate_summary()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]
        
        # Performance metrics
        if successful_tests:
            response_times = [r.response_time_ms for r in successful_tests]
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
        else:
            avg_response_time = max_response_time = min_response_time = 0
        
        # Accuracy analysis
        correct_predictions = 0
        for result in successful_tests:
            scenario = next(s for s in self.scenarios if s.name == result.scenario_name)
            if scenario.expected_makespan == result.meta_prediction:
                correct_predictions += 1
        
        accuracy = (correct_predictions / len(self.scenarios)) * 100 if self.scenarios else 0
        
        # Issue analysis
        all_issues = []
        for result in self.results:
            all_issues.extend(result.issues_detected)
        
        unique_issues = list(set(all_issues))
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_scenarios": len(self.scenarios),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "accuracy_percentage": accuracy,
            "performance": {
                "avg_response_time_ms": avg_response_time,
                "min_response_time_ms": min_response_time,
                "max_response_time_ms": max_response_time
            },
            "issues_detected": unique_issues,
            "detailed_results": [
                {
                    "scenario": r.scenario_name,
                    "success": r.success,
                    "prediction": r.meta_prediction,
                    "confidence": r.meta_confidence,
                    "response_time": r.response_time_ms,
                    "expected_vs_actual": r.expected_vs_actual,
                    "issues": r.issues_detected
                }
                for r in self.results
            ]
        }
        
        self._print_summary_report(summary)
        return summary
    
    def _print_summary_report(self, summary: Dict[str, Any]):
        """Print formatted summary report"""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY REPORT")
        print("=" * 60)
        
        print(f"🎯 Overall Results:")
        print(f"   Success Rate: {summary['successful_tests']}/{summary['total_scenarios']} ({summary['successful_tests']/summary['total_scenarios']*100:.1f}%)")
        print(f"   Prediction Accuracy: {summary['accuracy_percentage']:.1f}%")
        
        print(f"\n⚡ Performance Metrics:")
        perf = summary['performance']
        print(f"   Average Response Time: {perf['avg_response_time_ms']:.1f}ms")
        print(f"   Response Time Range: {perf['min_response_time_ms']:.1f}ms - {perf['max_response_time_ms']:.1f}ms")
        
        print(f"\n🚨 Issues Detected ({len(summary['issues_detected'])}):")
        for issue in summary['issues_detected']:
            print(f"   ⚠️  {issue}")
        
        print(f"\n📋 Scenario Results:")
        for result in summary['detailed_results']:
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['scenario']}: {result['prediction']} ({result['confidence']:.3f})")
        
        print(f"\n🎭 DEMO READINESS:")
        if summary['successful_tests'] == summary['total_scenarios']:
            if summary['accuracy_percentage'] >= 80:
                print("   ✅ EXCELLENT - Ready for live demo")
            else:
                print("   ⚠️  FUNCTIONAL - Demo with caveats about accuracy")
        else:
            print("   ❌ ISSUES - Fix errors before demo")
        
        print("=" * 60)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="MCCVA Comprehensive Benchmark Testing")
    parser.add_argument("--url", default="http://localhost:5000", 
                       help="ML Service base URL (default: http://localhost:5000)")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--demo", action="store_true", 
                       help="Run in demo mode with real-time display")
    
    args = parser.parse_args()
    
    # Create tester
    tester = MCCVABenchmarkTester(args.url)
    
    if args.demo:
        print("🎭 RUNNING IN DEMO MODE")
        print("Press Enter between each test for presentation timing...")
        
        # Modified demo flow
        for scenario in tester.scenarios:
            input("\nPress Enter to test next scenario...")
            result = tester.test_scenario(scenario)
            tester.results.append(result)
    else:
        # Regular comprehensive test
        summary = tester.run_comprehensive_test()
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main() 