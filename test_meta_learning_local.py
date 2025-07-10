#!/usr/bin/env python3
"""
🧪 MCCVA META-LEARNING LOCAL TESTING SCRIPT
===========================================
Test the trained Meta-Learning ensemble (SVM + K-Means + Neural Network) 
on local machine before Docker deployment.

This script validates:
1. All model files are properly saved and loadable
2. Meta-Learning predictions work correctly
3. Ensemble logic produces consistent results
4. Performance benchmarks meet production requirements

REAL-WORLD VERSION 2.1 - Production Validation
"""

import pandas as pd
import numpy as np
import joblib
import time
import json
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

def load_meta_learning_models():
    """Load all trained Meta-Learning components"""
    print("🔄 Loading Meta-Learning Models...")
    print("=" * 50)
    
    try:
        # Load Meta-Learning components
        meta_model = joblib.load('models/meta_learning_model.joblib')
        meta_scaler = joblib.load('models/meta_learning_scaler.joblib')
        meta_encoder = joblib.load('models/meta_learning_encoder.joblib')
        meta_features = joblib.load('models/meta_learning_features.joblib')
        meta_info = joblib.load('models/meta_learning_info.joblib')
        
        # Load base models (SVM + K-Means)
        svm_model = joblib.load('models/svm_model.joblib')
        svm_scaler = joblib.load('models/svm_scaler.joblib')
        svm_encoder = joblib.load('models/svm_label_encoder.joblib')
        svm_features = joblib.load('models/svm_feature_names.joblib')
        
        kmeans_model = joblib.load('models/kmeans_model.joblib')
        kmeans_scaler = joblib.load('models/kmeans_scaler.joblib')
        kmeans_features = joblib.load('models/kmeans_feature_names.joblib')
        kmeans_info = joblib.load('models/kmeans_info.joblib')
        
        print("✅ All models loaded successfully!")
        print(f"  Meta-Learning: {meta_info['model_info']['architecture']}")
        print(f"  SVM: {len(svm_features)} features, {len(svm_encoder.classes_)} classes")
        print(f"  K-Means: {len(kmeans_features)} features, {kmeans_model.n_clusters} clusters")
        print(f"  Training Date: {meta_info['timestamp'][:10]}")
        print(f"  Test Accuracy: {meta_info['performance']['test_accuracy']:.1%}")
        
        return {
            'meta_learning': {
                'model': meta_model,
                'scaler': meta_scaler,
                'encoder': meta_encoder,
                'features': meta_features,
                'info': meta_info
            },
            'svm': {
                'model': svm_model,
                'scaler': svm_scaler,
                'encoder': svm_encoder,
                'features': svm_features
            },
            'kmeans': {
                'model': kmeans_model,
                'scaler': kmeans_scaler,
                'features': kmeans_features,
                'info': kmeans_info
            }
        }
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Make sure you've trained all models first!")
        return None

def extract_meta_features(svm_pred, svm_conf, kmeans_cluster, kmeans_conf, 
                         rule_pred, rule_conf, cpu_cores, memory, storage, priority):
    """Extract 13 meta-features exactly like training (must match train_meta_learning.py)"""
    
    # Convert SVM prediction to one-hot encoding
    svm_scores = {'small': 0, 'medium': 0, 'large': 0}
    svm_scores[svm_pred] = 1
    
    # Normalize cluster distance
    cluster_distance_norm = 1 / (1 + kmeans_conf) if kmeans_conf > 0 else 0.5
    
    # Calculate enhanced features
    compute_intensity = min((cpu_cores * memory) / 100, 1.0)
    memory_intensity = min(memory / 64, 1.0)
    storage_intensity = min(storage / 1000, 1.0)
    is_high_priority = float(priority >= 4)
    resource_balance_score = 1 - abs(compute_intensity - 0.5) - abs(memory_intensity - 0.5)
    
    # 13 meta-features for Neural Network (EXACT ORDER from training)
    meta_features = [
        # Base model confidences (3 features)
        svm_conf, kmeans_conf, rule_conf,
        
        # SVM prediction encoding (3 features)
        svm_scores['small'], svm_scores['medium'], svm_scores['large'],
        
        # K-Means outputs (2 features)
        float(kmeans_cluster), cluster_distance_norm,
        
        # Enhanced business features (5 features)
        compute_intensity, memory_intensity, storage_intensity,
        is_high_priority, resource_balance_score
    ]
    
    return meta_features

def predict_with_ensemble(models, test_case):
    """Make prediction using full MCCVA ensemble pipeline"""
    
    # Extract input features
    cpu_cores = test_case['cpu_cores']
    memory = test_case['memory_gb']
    storage = test_case.get('storage_gb', 100)
    network = test_case.get('network_mbps', 1000)
    priority = test_case.get('priority', 2)
    
    # Generate additional SVM features
    cpu_memory_ratio = cpu_cores / (memory + 1e-6)
    storage_memory_ratio = storage / (memory + 1e-6)
    network_cpu_ratio = network / (cpu_cores + 1e-6)
    resource_intensity = (cpu_cores * memory * storage) / 1000
    priority_weighted_cpu = cpu_cores * priority
    
    # SVM features (10 features)
    svm_features = [
        cpu_cores, memory, storage, network, priority,
        cpu_memory_ratio, storage_memory_ratio, network_cpu_ratio,
        resource_intensity, priority_weighted_cpu
    ]
    
    # K-Means features (5 features) 
    kmeans_features = [
        test_case.get('memory_utilization', 0.5),
        test_case.get('cpu_utilization', 0.6),
        test_case.get('storage_utilization', 0.3),
        test_case.get('network_utilization', 0.2),
        test_case.get('workload_intensity', 0.03)
    ]
    
    # Step 1: SVM Prediction
    svm_features_scaled = models['svm']['scaler'].transform([svm_features])
    svm_pred_int = models['svm']['model'].predict(svm_features_scaled)[0]
    svm_prediction = models['svm']['encoder'].inverse_transform([svm_pred_int])[0]
    svm_decision_scores = models['svm']['model'].decision_function(svm_features_scaled)
    svm_confidence = float(np.max(np.abs(svm_decision_scores[0]))) if isinstance(svm_decision_scores[0], np.ndarray) else float(np.abs(svm_decision_scores[0]))
    
    # Step 2: K-Means Prediction
    kmeans_features_scaled = models['kmeans']['scaler'].transform([kmeans_features])
    kmeans_cluster = int(models['kmeans']['model'].predict(kmeans_features_scaled)[0])
    kmeans_distances = models['kmeans']['model'].transform(kmeans_features_scaled)[0]
    kmeans_confidence = float(1 / (1 + np.min(kmeans_distances)))
    
    # Step 3: Rule-based prediction (simple heuristic)
    if cpu_cores <= 4 and memory <= 16:
        rule_prediction = 'small'
        rule_confidence = 0.8
    elif cpu_cores >= 12 or memory >= 32:
        rule_prediction = 'large'
        rule_confidence = 0.75
    else:
        rule_prediction = 'medium'
        rule_confidence = 0.7
    
    # Step 4: Extract Meta-Features
    meta_features = extract_meta_features(
        svm_prediction, svm_confidence,
        kmeans_cluster, kmeans_confidence,
        rule_prediction, rule_confidence,
        cpu_cores, memory, storage, priority
    )
    
    # Step 5: Meta-Learning Prediction
    meta_features_scaled = models['meta_learning']['scaler'].transform([meta_features])
    meta_pred_proba = models['meta_learning']['model'].predict_proba(meta_features_scaled)[0]
    meta_pred_int = np.argmax(meta_pred_proba)
    meta_prediction = models['meta_learning']['encoder'].inverse_transform([meta_pred_int])[0]
    meta_confidence = float(np.max(meta_pred_proba))
    
    return {
        'input': test_case,
        'base_predictions': {
            'svm': {'prediction': svm_prediction, 'confidence': svm_confidence},
            'kmeans': {'cluster': kmeans_cluster, 'confidence': kmeans_confidence},
            'rule': {'prediction': rule_prediction, 'confidence': rule_confidence}
        },
        'meta_learning': {
            'prediction': meta_prediction,
            'confidence': meta_confidence,
            'probabilities': dict(zip(models['meta_learning']['encoder'].classes_, meta_pred_proba))
        },
        'meta_features': meta_features
    }

def run_comprehensive_tests(models):
    """Run comprehensive validation tests"""
    print("\n🧪 RUNNING COMPREHENSIVE TESTS")
    print("=" * 50)
    
    # Test scenarios covering edge cases
    test_scenarios = [
        # Small workloads
        {
            'name': 'Small VM - Light Load',
            'cpu_cores': 2, 'memory_gb': 4, 'priority': 1,
            'cpu_utilization': 0.3, 'memory_utilization': 0.4,
            'expected': 'small'
        },
        {
            'name': 'Small VM - High Priority',
            'cpu_cores': 4, 'memory_gb': 8, 'priority': 3,
            'cpu_utilization': 0.5, 'memory_utilization': 0.6,
            'expected': 'small'
        },
        
        # Medium workloads
        {
            'name': 'Medium VM - Balanced',
            'cpu_cores': 6, 'memory_gb': 16, 'priority': 3,
            'cpu_utilization': 0.6, 'memory_utilization': 0.7,
            'expected': 'medium'
        },
        {
            'name': 'Medium VM - CPU Intensive',
            'cpu_cores': 8, 'memory_gb': 12, 'priority': 2,
            'cpu_utilization': 0.8, 'memory_utilization': 0.5,
            'expected': 'medium'
        },
        
        # Large workloads
        {
            'name': 'Large VM - High Memory',
            'cpu_cores': 12, 'memory_gb': 32, 'priority': 4,
            'cpu_utilization': 0.7, 'memory_utilization': 0.8,
            'expected': 'large'
        },
        {
            'name': 'Large VM - Enterprise',
            'cpu_cores': 16, 'memory_gb': 64, 'priority': 5,
            'cpu_utilization': 0.9, 'memory_utilization': 0.9,
            'expected': 'large'
        },
        
        # Edge cases
        {
            'name': 'Edge Case - Low Resource High Priority',
            'cpu_cores': 2, 'memory_gb': 4, 'priority': 5,
            'cpu_utilization': 0.9, 'memory_utilization': 0.9,
            'expected': None  # Let model decide
        },
        {
            'name': 'Edge Case - High Resource Low Priority',
            'cpu_cores': 16, 'memory_gb': 64, 'priority': 1,
            'cpu_utilization': 0.2, 'memory_utilization': 0.3,
            'expected': None  # Let model decide
        }
    ]
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    print(f"Testing {len(test_scenarios)} scenarios...\n")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"Test {i}: {scenario['name']}")
        
        # Make prediction
        start_time = time.time()
        result = predict_with_ensemble(models, scenario)
        prediction_time = time.time() - start_time
        
        # Check accuracy if expected result is provided
        is_correct = "N/A"
        if scenario['expected']:
            is_correct = result['meta_learning']['prediction'] == scenario['expected']
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
        
        print(f"  📊 Results:")
        print(f"    SVM: {result['base_predictions']['svm']['prediction']} (conf: {result['base_predictions']['svm']['confidence']:.2f})")
        print(f"    K-Means: Cluster {result['base_predictions']['kmeans']['cluster']} (conf: {result['base_predictions']['kmeans']['confidence']:.2f})")
        print(f"    Rule: {result['base_predictions']['rule']['prediction']} (conf: {result['base_predictions']['rule']['confidence']:.2f})")
        print(f"    🧠 Meta-Learning: {result['meta_learning']['prediction']} (conf: {result['meta_learning']['confidence']:.2f})")
        print(f"    ⏱️ Prediction Time: {prediction_time*1000:.1f}ms")
        print(f"    ✅ Correct: {is_correct}")
        print()
        
        results.append({
            'scenario': scenario['name'],
            'prediction': result['meta_learning']['prediction'],
            'confidence': result['meta_learning']['confidence'],
            'time_ms': prediction_time * 1000,
            'correct': is_correct,
            'all_predictions': result['base_predictions']
        })
    
    # Summary statistics
    print("📈 TEST SUMMARY")
    print("=" * 30)
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"✅ Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    
    avg_time = np.mean([r['time_ms'] for r in results])
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    print(f"⏱️ Average Prediction Time: {avg_time:.1f}ms")
    print(f"🎯 Average Confidence: {avg_confidence:.1%}")
    print(f"🚀 Production Ready: {'✅ YES' if avg_time < 100 and avg_confidence > 0.8 else '⚠️ NEEDS IMPROVEMENT'}")
    
    return results

def run_performance_benchmark(models):
    """Benchmark Meta-Learning performance"""
    print("\n⚡ PERFORMANCE BENCHMARK")
    print("=" * 30)
    
    # Generate random test cases for performance testing
    n_tests = 100
    print(f"Running {n_tests} random predictions...")
    
    times = []
    predictions = []
    
    start_total = time.time()
    
    for i in range(n_tests):
        # Random test case
        test_case = {
            'cpu_cores': np.random.randint(1, 16),
            'memory_gb': np.random.randint(2, 64),
            'priority': np.random.randint(1, 5),
            'cpu_utilization': np.random.uniform(0.1, 0.9),
            'memory_utilization': np.random.uniform(0.1, 0.9)
        }
        
        start_time = time.time()
        result = predict_with_ensemble(models, test_case)
        prediction_time = time.time() - start_time
        
        times.append(prediction_time * 1000)  # Convert to ms
        predictions.append(result['meta_learning']['prediction'])
    
    total_time = time.time() - start_total
    
    # Performance statistics
    print(f"📊 Performance Results:")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Average: {np.mean(times):.1f}ms")
    print(f"  Median: {np.median(times):.1f}ms")
    print(f"  Min: {np.min(times):.1f}ms")
    print(f"  Max: {np.max(times):.1f}ms")
    print(f"  95th percentile: {np.percentile(times, 95):.1f}ms")
    print(f"  Throughput: {n_tests/total_time:.0f} predictions/sec")
    
    # Prediction distribution
    pred_counts = Counter(predictions)
    print(f"\n📈 Prediction Distribution:")
    for pred, count in pred_counts.items():
        pct = (count / n_tests) * 100
        print(f"  {pred}: {count} ({pct:.1f}%)")
    
    return {
        'avg_time_ms': np.mean(times),
        'median_time_ms': np.median(times),
        'p95_time_ms': np.percentile(times, 95),
        'throughput_per_sec': n_tests / total_time,
        'prediction_distribution': dict(pred_counts)
    }

def validate_model_files():
    """Validate all required model files exist and are valid"""
    print("\n🔍 VALIDATING MODEL FILES")
    print("=" * 30)
    
    required_files = [
        'models/meta_learning_model.joblib',
        'models/meta_learning_scaler.joblib', 
        'models/meta_learning_encoder.joblib',
        'models/meta_learning_features.joblib',
        'models/meta_learning_info.joblib',
        'models/svm_model.joblib',
        'models/svm_scaler.joblib',
        'models/svm_label_encoder.joblib',
        'models/svm_feature_names.joblib',
        'models/kmeans_model.joblib',
        'models/kmeans_scaler.joblib',
        'models/kmeans_feature_names.joblib',
        'models/kmeans_info.joblib'
    ]
    
    missing_files = []
    valid_files = []
    
    for file_path in required_files:
        try:
            # Try to load the file
            data = joblib.load(file_path)
            valid_files.append(file_path)
            print(f"✅ {file_path}")
        except Exception as e:
            missing_files.append(file_path)
            print(f"❌ {file_path}: {e}")
    
    print(f"\n📊 Validation Summary:")
    print(f"  Valid Files: {len(valid_files)}/{len(required_files)}")
    print(f"  Missing Files: {len(missing_files)}")
    
    if missing_files:
        print(f"\n⚠️ Missing files:")
        for file_path in missing_files:
            print(f"    - {file_path}")
        return False
    
    print(f"✅ All model files are valid and loadable!")
    return True

def main():
    """Main testing pipeline"""
    print("🧪 MCCVA META-LEARNING LOCAL TESTING")
    print("=" * 60)
    print("Validating trained Meta-Learning ensemble before Docker deployment")
    print("REAL-WORLD VERSION 2.1 - Production Validation")
    print()
    
    start_time = datetime.now()
    
    # Step 1: Validate model files
    if not validate_model_files():
        print("❌ Model validation failed! Train models first.")
        return False
    
    # Step 2: Load models
    models = load_meta_learning_models()
    if models is None:
        print("❌ Failed to load models!")
        return False
    
    # Step 3: Run comprehensive tests
    test_results = run_comprehensive_tests(models)
    
    # Step 4: Performance benchmark
    performance_results = run_performance_benchmark(models)
    
    # Step 5: Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n" + "=" * 60)
    print("🎉 MCCVA META-LEARNING TESTING COMPLETED!")
    print("=" * 60)
    
    # Check production readiness
    is_production_ready = (
        performance_results['avg_time_ms'] < 100 and
        performance_results['throughput_per_sec'] > 50 and
        models['meta_learning']['info']['performance']['test_accuracy'] > 0.8
    )
    
    print(f"🧠 Meta-Learning Model:")
    print(f"  Architecture: {models['meta_learning']['info']['model_info']['architecture']}")
    print(f"  Test Accuracy: {models['meta_learning']['info']['performance']['test_accuracy']:.1%}")
    print(f"  Training Date: {models['meta_learning']['info']['timestamp'][:10]}")
    
    print(f"\n⚡ Performance Metrics:")
    print(f"  Average Prediction: {performance_results['avg_time_ms']:.1f}ms")
    print(f"  Throughput: {performance_results['throughput_per_sec']:.0f} predictions/sec")
    print(f"  95th Percentile: {performance_results['p95_time_ms']:.1f}ms")
    
    print(f"\n🎯 Production Status:")
    print(f"  Ready for Docker: {'✅ YES' if is_production_ready else '⚠️ NEEDS IMPROVEMENT'}")
    print(f"  API Integration: {'✅ READY' if performance_results['avg_time_ms'] < 50 else '⚠️ OPTIMIZE'}")
    print(f"  Cloud Deployment: {'✅ GO' if is_production_ready else '❌ WAIT'}")
    
    print(f"\n⏱️ Total Testing Time: {duration:.1f} seconds")
    
    # Save test results
    test_report = {
        'timestamp': start_time.isoformat(),
        'duration_seconds': duration,
        'model_info': models['meta_learning']['info'],
        'test_results': test_results,
        'performance_results': performance_results,
        'production_ready': is_production_ready
    }
    
    with open('meta_learning_test_report.json', 'w') as f:
        json.dump(test_report, f, indent=2, default=str)
    
    print(f"📄 Test report saved: meta_learning_test_report.json")
    
    return is_production_ready

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🚀 Next Steps:")
        print("1. ✅ Meta-Learning validation passed")
        print("2. 🐳 Ready for Docker deployment")
        print("3. ☁️ Deploy to cloud infrastructure")
        print("4. 📊 Monitor production performance")
    else:
        print("\n⚠️ Fix issues before deploying to cloud!") 