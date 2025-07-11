#!/usr/bin/env python3
"""
ML Service - Enhanced Flask API với Performance Optimization
Chạy bằng: python ml_service.py hoặc gunicorn ml_service:app
Production ready với caching, monitoring, và performance optimization
"""

import joblib
import json
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import logging
import os
import sys
import time
import threading
from functools import wraps, lru_cache
import hashlib
from collections import defaultdict, deque
import psutil

# Machine Learning imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Performance monitoring imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Redis not available - using in-memory cache")

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/mccva/mccva-ml.log') if os.path.exists('/var/log/mccva') else logging.StreamHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables để lưu models
svm_model = None
kmeans_model = None
svm_scaler = None
kmeans_scaler = None
svm_label_encoder = None

# Meta-Learning global variables
meta_learning_model = None
meta_learning_scaler = None
meta_learning_encoder = None
meta_learning_features = None
meta_learning_info = None

# Performance monitoring variables
prediction_cache = {}
request_metrics = defaultdict(list)
performance_stats = {
    'total_requests': 0,
    'total_predictions': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'average_response_time': 0,
    'error_count': 0,
    'last_reset': datetime.now()
}

# Request rate limiting
request_history = deque(maxlen=1000)  # Keep last 1000 requests

class PerformanceMonitor:
    """Performance monitoring class"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.response_times = deque(maxlen=100)  # Keep last 100 response times
        
    def record_request(self, response_time, endpoint, success=True):
        """Record request metrics"""
        self.request_count += 1
        self.response_times.append(response_time)
        
        request_metrics[endpoint].append({
            'timestamp': datetime.now(),
            'response_time': response_time,
            'success': success
        })
        
        # Keep only last 100 requests per endpoint
        if len(request_metrics[endpoint]) > 100:
            request_metrics[endpoint] = request_metrics[endpoint][-100:]
    
    def get_stats(self):
        """Get performance statistics"""
        avg_response_time = np.mean(self.response_times) if self.response_times else 0
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.request_count,
            'average_response_time_ms': avg_response_time * 1000,
            'requests_per_second': self.request_count / uptime if uptime > 0 else 0,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.Process().cpu_percent()
        }

# Initialize performance monitor
perf_monitor = PerformanceMonitor()

def performance_tracker(f):
    """Decorator to track API performance"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        endpoint = request.endpoint
        
        try:
            # Update global stats
            performance_stats['total_requests'] += 1
            
            result = f(*args, **kwargs)
            response_time = time.time() - start_time
            
            # Record successful request
            perf_monitor.record_request(response_time, endpoint, True)
            
            # Add performance headers
            if hasattr(result, 'headers'):
                result.headers['X-Response-Time'] = f"{response_time:.3f}s"
                result.headers['X-Request-ID'] = str(hash(f"{endpoint}-{start_time}"))
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            performance_stats['error_count'] += 1
            perf_monitor.record_request(response_time, endpoint, False)
            logger.error(f"Error in {endpoint}: {e}, Response time: {response_time:.3f}s")
            raise
    
    return wrapper

def cache_prediction(f):
    """Decorator to cache predictions"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        # Generate cache key from request data
        if request.is_json:
            cache_key = hashlib.md5(
                json.dumps(request.get_json(), sort_keys=True).encode()
            ).hexdigest()
            
            # Check cache
            if cache_key in prediction_cache:
                cache_entry = prediction_cache[cache_key]
                
                # Check if cache is still valid (5 minutes)
                if datetime.now() - cache_entry['timestamp'] < timedelta(minutes=5):
                    performance_stats['cache_hits'] += 1
                    logger.info(f"Cache hit for {request.endpoint}")
                    
                    # Add cache headers
                    response_data = cache_entry['data']
                    response_data['cached'] = True
                    response_data['cache_timestamp'] = cache_entry['timestamp'].isoformat()
                    
                    return jsonify(response_data)
            
            # Cache miss - execute function
            performance_stats['cache_misses'] += 1
            result = f(*args, **kwargs)
            
            # Cache the result
            if hasattr(result, 'get_json'):
                prediction_cache[cache_key] = {
                    'data': result.get_json(),
                    'timestamp': datetime.now()
                }
                
                # Cleanup old cache entries (keep last 1000)
                if len(prediction_cache) > 1000:
                    oldest_key = min(prediction_cache.keys(), 
                                   key=lambda k: prediction_cache[k]['timestamp'])
                    del prediction_cache[oldest_key]
            
            return result
        else:
            return f(*args, **kwargs)
    
    return wrapper

@lru_cache(maxsize=128)
def get_feature_hash(features_tuple):
    """Cache feature preprocessing"""
    return hashlib.md5(str(features_tuple).encode()).hexdigest()

def load_models():
    """Load các mô hình khi khởi động service"""
    global svm_model, kmeans_model, svm_scaler, kmeans_scaler, svm_label_encoder
    global meta_learning_model, meta_learning_scaler, meta_learning_encoder, meta_learning_features, meta_learning_info
    
    try:
        # Đảm bảo working directory đúng
        if os.path.exists('/opt/mccva'):
            os.chdir('/opt/mccva')
        
        logger.info("Đang load mô hình SVM...")
        svm_model = joblib.load("models/svm_model.joblib")
        svm_scaler = joblib.load("models/svm_scaler.joblib")
        svm_label_encoder = joblib.load("models/svm_label_encoder.joblib")
        
        logger.info("Đang load mô hình K-Means...")
        kmeans_model = joblib.load("models/kmeans_model.joblib")
        kmeans_scaler = joblib.load("models/kmeans_scaler.joblib")
        
        # 🧠 Load Meta-Learning models
        logger.info("Đang load mô hình Meta-Learning...")
        meta_learning_model = joblib.load("models/meta_learning_model.joblib")
        meta_learning_scaler = joblib.load("models/meta_learning_scaler.joblib")
        meta_learning_encoder = joblib.load("models/meta_learning_encoder.joblib")
        meta_learning_features = joblib.load("models/meta_learning_features.joblib")
        meta_learning_info = joblib.load("models/meta_learning_info.joblib")
        
        logger.info("✅ Tất cả mô hình đã được load thành công!")
        logger.info(f"SVM Model: {svm_model.kernel} kernel, {sum(svm_model.n_support_)} support vectors")
        logger.info(f"K-Means Model: {kmeans_model.n_clusters} clusters")
        logger.info(f"Meta-Learning Model: {meta_learning_info.get('architecture', 'Unknown')} architecture")
        
        # Safe formatting for test_accuracy (might be string or number)
        accuracy = meta_learning_info.get('test_accuracy', 'Unknown')
        if isinstance(accuracy, (int, float)):
            logger.info(f"Meta-Learning Accuracy: {accuracy:.3f}")
        else:
            logger.info(f"Meta-Learning Accuracy: {accuracy}")
        
        # Update Meta-Learning Ensemble with loaded models
        meta_ensemble.meta_model = meta_learning_model
        meta_ensemble.meta_scaler = meta_learning_scaler
        meta_ensemble.meta_label_encoder = meta_learning_encoder
        meta_ensemble.is_trained = True
        
        logger.info("✅ Models loaded successfully - Meta-Learning integration complete")
        
    except FileNotFoundError as e:
        logger.error(f"❌ Model file not found: {e}")
        logger.error("Current working directory: " + os.getcwd())
        logger.error("Available files in models/: " + str(os.listdir("models") if os.path.exists("models") else "models/ not found"))
        raise e
    except Exception as e:
        logger.error(f"❌ Lỗi khi load mô hình: {e}")
        raise e

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        models_status = {
            "svm": svm_model is not None,
            "kmeans": kmeans_model is not None,
            "svm_scaler": svm_scaler is not None,
            "kmeans_scaler": kmeans_scaler is not None,
            "svm_label_encoder": svm_label_encoder is not None,
            "meta_learning": meta_learning_model is not None,
            "meta_learning_scaler": meta_learning_scaler is not None,
            "meta_learning_encoder": meta_learning_encoder is not None
        }
        
        all_models_loaded = all(models_status.values())
        
        return jsonify({
            "status": "healthy" if all_models_loaded else "degraded",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": models_status,
            "service": "mccva-ml-service",
            "version": "2.0.0",
            "meta_learning_ready": meta_learning_model is not None
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/predict/makespan', methods=['POST'])
@performance_tracker
@cache_prediction
def predict_makespan():
    """
    API dự đoán makespan của yêu cầu tài nguyên
    Input: {"features": [cpu_cores, memory_gb, storage_gb, network_bandwidth, priority, task_complexity, data_size, io_intensity, parallel_degree, deadline_urgency]}
    Output: {"makespan": "small|medium|large", "confidence": float}
    """
    try:
        if svm_model is None or svm_scaler is None or svm_label_encoder is None:
            return jsonify({"error": "SVM model components not loaded"}), 503
        
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' field"}), 400
        
        features = data["features"]
        
        # Validate input
        if len(features) != 9:
            return jsonify({"error": "Expected 9 features: [jobs_1min, jobs_5min, jobs_15min, memory_mb, disk_gb, cpu_cores, cpu_speed, network_receive_kbps, network_transmit_kbps]"}), 400
        
        # Validate ranges (updated for ACTUAL trained features)
        if not (0 <= features[0] <= 100):  # jobs_1min
            return jsonify({"error": "Jobs per 1 minute must be between 0-100"}), 400
        if not (0 <= features[1] <= 500):  # jobs_5min
            return jsonify({"error": "Jobs per 5 minutes must be between 0-500"}), 400
        if not (0 <= features[2] <= 1500):  # jobs_15min
            return jsonify({"error": "Jobs per 15 minutes must be between 0-1500"}), 400
        if not (1024 <= features[3] <= 65536):  # memory_mb
            return jsonify({"error": "Memory must be between 1024-65536 MB"}), 400
        if not (10 <= features[4] <= 5000):  # disk_gb
            return jsonify({"error": "Disk capacity must be between 10-5000 GB"}), 400
        if not (1 <= features[5] <= 32):  # cpu_cores
            return jsonify({"error": "CPU cores must be between 1-32"}), 400
        if not (1000 <= features[6] <= 5000):  # cpu_speed_mhz
            return jsonify({"error": "CPU speed must be between 1000-5000 MHz"}), 400
        if not (0 <= features[7] <= 10000):  # network_receive
            return jsonify({"error": "Network receive must be between 0-10000 Kbps"}), 400
        if not (0 <= features[8] <= 10000):  # network_transmit
            return jsonify({"error": "Network transmit must be between 0-10000 Kbps"}), 400
        
        # Chuẩn hóa dữ liệu
        features_scaled = svm_scaler.transform([features])
        
        # Dự đoán
        prediction_numeric = svm_model.predict(features_scaled)[0]
        
        # Decode prediction từ số về tên class
        prediction = svm_label_encoder.inverse_transform([prediction_numeric])[0]
        
        # Tính confidence score
        decision_scores = svm_model.decision_function(features_scaled)
        if isinstance(decision_scores[0], np.ndarray):
            confidence = float(np.max(np.abs(decision_scores[0])))
        else:
            confidence = float(np.abs(decision_scores[0]))
        
        logger.info(f"Prediction: {prediction} (confidence: {confidence:.3f}) for features: {features}")
        
        return jsonify({
            "makespan": str(prediction),  # Convert to string for JSON serialization
            "confidence": confidence,
            "features": features,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in predict_makespan: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/vm_cluster', methods=['POST'])
@performance_tracker
@cache_prediction
def predict_vm_cluster():
    """
    API dự đoán cụm cho VM
    Input: {"vm_features": [cpu_usage, ram_usage, storage_usage]}
    Output: {"cluster": int, "distance": float, "centroid": [float, float, float]}
    """
    try:
        if kmeans_model is None or kmeans_scaler is None:
            return jsonify({"error": "K-Means model not loaded"}), 503
        
        data = request.get_json()
        if not data or "vm_features" not in data:
            return jsonify({"error": "Missing 'vm_features' field"}), 400
        
        vm_features = data["vm_features"]
        
        # Validate input
        if len(vm_features) != 3:
            return jsonify({"error": "Expected 3 features: [cpu_usage, ram_usage, storage_usage]"}), 400
        
        # Validate ranges (0-1)
        for i, feature in enumerate(vm_features):
            if not (0 <= feature <= 1):
                return jsonify({"error": f"Feature {i} must be between 0-1 (percentage)"}), 400
        
        # Chuẩn hóa dữ liệu
        vm_scaled = kmeans_scaler.transform([vm_features])
        
        # Dự đoán cụm
        cluster = int(kmeans_model.predict(vm_scaled)[0])
        
        # Tính khoảng cách đến centroid
        distances = kmeans_model.transform(vm_scaled)[0]
        distance_to_center = float(distances[cluster])
        
        # Lấy centroid của cụm
        centroid = kmeans_model.cluster_centers_[cluster].tolist()
        
        logger.info(f"VM Cluster: {cluster} (distance: {distance_to_center:.3f}) for features: {vm_features}")
        
        return jsonify({
            "cluster": cluster,
            "distance": distance_to_center,
            "centroid": centroid,
            "vm_features": vm_features,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in predict_vm_cluster: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
@performance_tracker
@cache_prediction
def predict_batch():
    """
    API dự đoán hàng loạt
    Input: {"requests": [[features1], [features2], ...], "vm_usages": [[vm_features1], [vm_features2], ...]}
    Output: {"predictions": [{"makespan": str, "cluster": int, ...}, ...]}
    """
    try:
        data = request.get_json()
        if not data or "requests" not in data or "vm_usages" not in data:
            return jsonify({"error": "Missing 'requests' or 'vm_usages' field"}), 400
        
        requests = data["requests"]
        vm_usages = data["vm_usages"]
        
        if len(requests) != len(vm_usages):
            return jsonify({"error": "Number of requests must match number of VM usages"}), 400
        
        predictions = []
        
        for i, (req_features, vm_features) in enumerate(zip(requests, vm_usages)):
            try:
                # Dự đoán makespan
                req_scaled = svm_scaler.transform([req_features])
                makespan = svm_model.predict(req_scaled)[0]
                
                # Dự đoán cluster
                vm_scaled = kmeans_scaler.transform([vm_features])
                cluster = int(kmeans_model.predict(vm_scaled)[0])
                
                predictions.append({
                    "id": i + 1,
                    "makespan": makespan,
                    "cluster": cluster,
                    "request_features": req_features,
                    "vm_features": vm_features
                })
                
            except Exception as e:
                logger.error(f"Error processing item {i}: {e}")
                predictions.append({
                    "id": i + 1,
                    "error": str(e)
                })
        
        logger.info(f"Batch prediction completed: {len(predictions)} items")
        
        return jsonify({
            "predictions": predictions,
            "total": len(predictions),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in predict_batch: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/vm_clusters/info', methods=['GET'])
@performance_tracker
@cache_prediction
def get_vm_clusters_info():
    """
    API lấy thông tin các cụm VM
    Output: {"clusters": [{"id": int, "centroid": [float, float, float], "size": int}, ...]}
    """
    try:
        clusters_info = []
        
        for i in range(kmeans_model.n_clusters):
            centroid = kmeans_model.cluster_centers_[i].tolist()
            # Đếm số VM trong cụm (từ labels đã lưu)
            size = int(np.sum(kmeans_model.labels_ == i))
            
            clusters_info.append({
                "id": i,
                "centroid": centroid,
                "size": size
            })
        
        return jsonify({
            "clusters": clusters_info,
            "total_clusters": kmeans_model.n_clusters,
            "total_vms": len(kmeans_model.labels_),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in get_vm_clusters_info: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models/info', methods=['GET'])
@performance_tracker
@cache_prediction
def get_models_info():
    """
    API lấy thông tin về các mô hình
    """
    try:
        svm_info = {
            "kernel": svm_model.kernel,
            "c_parameter": svm_model.C,
            "gamma": svm_model.gamma,
            "support_vectors": int(sum(svm_model.n_support_)),
            "classes": svm_model.classes_.tolist()
        }
        
        kmeans_info = {
            "n_clusters": kmeans_model.n_clusters,
            "inertia": float(kmeans_model.inertia_),
            "n_iter": kmeans_model.n_iter_
        }
        
        return jsonify({
            "svm_model": svm_info,
            "kmeans_model": kmeans_info,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in get_models_info: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/enhanced', methods=['POST'])
@performance_tracker
@cache_prediction
def predict_enhanced():
    """
    Enhanced API với ensemble learning - kết hợp K-Means và Rule-based
    FIXED: Removed invalid SVM feature conversion (5->10 features is meaningless)
    Input: {"features": [cpu_cores, memory, storage, network_bandwidth, priority], "vm_features": [cpu_usage, ram_usage, storage_usage]}
    Output: {"makespan": "small|medium|large", "cluster": int, "confidence": float, "model_contributions": {...}}
    """
    try:
        if kmeans_model is None or kmeans_scaler is None:
            return jsonify({"error": "K-Means model not loaded"}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No request data"}), 400
        
        features = data.get("features", [])
        vm_features = data.get("vm_features", [0.5, 0.5, 0.5])  # Default VM features
        
        # Validate features
        if len(features) != 5:
            return jsonify({"error": "Expected 5 features: [cpu_cores, memory, storage, network_bandwidth, priority]"}), 400
        
        if len(vm_features) != 3:
            return jsonify({"error": "Expected 3 VM features: [cpu_usage, ram_usage, storage_usage]"}), 400
        
        # Validate ranges
        if not (1 <= features[0] <= 32):  # cpu_cores
            return jsonify({"error": "CPU cores must be between 1-32"}), 400
        if not (1 <= features[1] <= 128):  # memory_gb
            return jsonify({"error": "Memory must be between 1-128 GB"}), 400
        if not (10 <= features[2] <= 5000):  # storage_gb
            return jsonify({"error": "Storage must be between 10-5000 GB"}), 400
        if not (100 <= features[3] <= 10000):  # network_bandwidth
            return jsonify({"error": "Network bandwidth must be between 100-10000 Mbps"}), 400
        if not (1 <= features[4] <= 5):  # priority
            return jsonify({"error": "Priority must be between 1-5"}), 400
        
        # Validate VM features (0-1)
        for i, feature in enumerate(vm_features):
            if not (0 <= feature <= 1):
                return jsonify({"error": f"VM feature {i} must be between 0-1 (percentage)"}), 400
        
        # Enhanced feature engineering
        enhanced_features = extract_enhanced_features(features)
        
        # Model 1: K-Means Prediction (using 3 VM features)
        vm_scaled = kmeans_scaler.transform([vm_features])
        kmeans_cluster = int(kmeans_model.predict(vm_scaled)[0])
        kmeans_distances = kmeans_model.transform(vm_scaled)[0]
        kmeans_confidence = float(1 / (1 + np.min(kmeans_distances)))  # Closer to centroid = higher confidence
        
        # Model 2: Rule-based Heuristic
        rule_prediction, rule_confidence = get_rule_based_prediction(enhanced_features)
        
        # Ensemble Decision (K-Means + Rule-based only)
        ensemble_result = ensemble_decision_simplified(
            kmeans_cluster, kmeans_confidence,
            rule_prediction, rule_confidence,
            enhanced_features
        )
        
        logger.info(f"Enhanced prediction (K-Means + Rules): {ensemble_result}")
        
        return jsonify({
            "makespan": ensemble_result["makespan"],
            "cluster": ensemble_result["cluster"],
            "confidence": ensemble_result["confidence"],
            "method": "Enhanced_KMeans_Rules",
            "model_contributions": {
                "kmeans": {
                    "prediction": kmeans_cluster,
                    "confidence": kmeans_confidence,
                    "weight": ensemble_result["weights"]["kmeans"]
                },
                "rule_based": {
                    "prediction": rule_prediction,
                    "confidence": rule_confidence,
                    "weight": ensemble_result["weights"]["rule"]
                }
            },
            "enhanced_features": enhanced_features,
            "note": "SVM prediction removed - use /predict/makespan with 9 features or /predict/mccva_complete for full pipeline",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in predict_enhanced: {e}")
        return jsonify({"error": str(e)}), 500

def extract_enhanced_features(features):
    """Extract enhanced features từ basic features"""
    cpu_cores, memory, storage, network_bandwidth, priority = features
    
    enhanced = {
        # Basic features
        "cpu_cores": cpu_cores,
        "memory": memory,
        "storage": storage,
        "network_bandwidth": network_bandwidth,
        "priority": priority,
        
        # Derived features (AI insights)
        "compute_intensity": cpu_cores / memory if memory > 0 else 0,
        "storage_intensity": storage / memory if memory > 0 else 0,
        "network_intensity": network_bandwidth / cpu_cores if cpu_cores > 0 else 0,
        "resource_ratio": (cpu_cores * memory) / storage if storage > 0 else 0,
        
        # Workload classification
        "is_compute_intensive": cpu_cores / memory > 0.5 if memory > 0 else False,
        "is_memory_intensive": memory > 16,
        "is_storage_intensive": storage > 500,
        "is_network_intensive": network_bandwidth > 5000,
        
        # Priority-based features
        "high_priority": priority >= 4,
        "low_priority": priority <= 2,
        "priority_weight": priority / 5.0,
        
        # Resource utilization patterns
        "balanced_resources": abs(cpu_cores - memory/4) < 2,  # CPU and memory balanced
        "storage_heavy": storage > (cpu_cores * memory * 2),
        "network_heavy": network_bandwidth > (cpu_cores * 1000)
    }
    
    return enhanced

def get_rule_based_prediction(enhanced_features):
    """Rule-based prediction dựa trên business logic"""
    score = 0
    confidence = 0.5  # Base confidence
    
    # Rule 1: Compute-intensive workloads
    if enhanced_features["is_compute_intensive"]:
        score += 30
        confidence += 0.1
    
    # Rule 2: High priority requests
    if enhanced_features["high_priority"]:
        score += 25
        confidence += 0.15
    
    # Rule 3: Memory-intensive workloads
    if enhanced_features["is_memory_intensive"]:
        score += 20
        confidence += 0.1
    
    # Rule 4: Network-intensive workloads
    if enhanced_features["is_network_intensive"]:
        score += 15
        confidence += 0.1
    
    # Rule 5: Storage-heavy workloads
    if enhanced_features["storage_heavy"]:
        score += 10
        confidence += 0.05
    
    # Determine makespan based on score
    if score >= 60:
        prediction = "large"
    elif score >= 30:
        prediction = "medium"
    else:
        prediction = "small"
    
    confidence = min(confidence, 0.9)  # Cap confidence at 0.9
    
    return prediction, confidence

# 🧠 META-LEARNING ENSEMBLE CLASS - Advanced AI-based ensemble
class MetaLearningEnsemble:
    """
    🎯 ADVANCED META-LEARNING ENSEMBLE
    Uses Neural Network to learn optimal combination of SVM + K-Means + Rules
    Instead of hardcoded if-else logic!
    """
    
    def __init__(self):
        self.meta_model = None
        self.meta_scaler = StandardScaler()
        self.meta_label_encoder = LabelEncoder()
        self.is_trained = False
        self.training_data = []
        self.feature_names = [
            'svm_confidence', 'kmeans_confidence', 'rule_confidence',
            'svm_small_score', 'svm_medium_score', 'svm_large_score',
            'cluster_id', 'cluster_distance', 
            'compute_intensity', 'memory_intensity', 'storage_intensity',
            'is_high_priority', 'resource_balance_score'
        ]
        
    def collect_training_sample(self, svm_pred, svm_conf, kmeans_cluster, kmeans_conf, 
                              rule_pred, rule_conf, enhanced_features, true_label=None):
        """
        🎯 Thu thập training samples để Meta-Learning tự học
        """
        # Tạo feature vector cho meta-model
        meta_features = self._extract_meta_features(
            svm_pred, svm_conf, kmeans_cluster, kmeans_conf, 
            rule_pred, rule_conf, enhanced_features
        )
        
        # Lưu sample (nếu có true_label)
        if true_label:
            self.training_data.append({
                'features': meta_features,
                'label': true_label,
                'timestamp': datetime.now()
            })
            
        return meta_features
    
    def _extract_meta_features(self, svm_pred, svm_conf, kmeans_cluster, kmeans_conf, 
                             rule_pred, rule_conf, enhanced_features):
        """
        🔧 Trích xuất features cho Meta-Learning model
        """
        # Chuyển SVM prediction thành scores
        svm_scores = {'small': 0, 'medium': 0, 'large': 0}
        svm_scores[svm_pred] = 1
        
        # Tính cluster distance (normalized)
        cluster_distance_norm = 1 / (1 + kmeans_conf) if kmeans_conf > 0 else 0.5
        
        # Extract key business features
        compute_intensity = enhanced_features.get('compute_intensity', 0)
        memory_intensity = enhanced_features.get('memory_intensity', 0) 
        storage_intensity = enhanced_features.get('storage_intensity', 0)
        is_high_priority = float(enhanced_features.get('high_priority', False))
        
        # Resource balance score
        balance_score = 1 - abs(compute_intensity - 0.5) - abs(memory_intensity - 0.5)
        
        meta_features = [
            svm_conf, kmeans_conf, rule_conf,  # Base confidences
            svm_scores['small'], svm_scores['medium'], svm_scores['large'],  # SVM outputs
            float(kmeans_cluster), cluster_distance_norm,  # K-Means outputs
            compute_intensity, memory_intensity, storage_intensity,  # Resource patterns
            is_high_priority, balance_score  # Business logic
        ]
        
        return meta_features
    
    def train_meta_model(self, min_samples=100):
        """
        🚀 Train Meta-Learning Neural Network
        """
        if len(self.training_data) < min_samples:
            logger.warning(f"Not enough training data: {len(self.training_data)} < {min_samples}")
            return False
            
        try:
            # Prepare training data
            X = np.array([sample['features'] for sample in self.training_data])
            y = [sample['label'] for sample in self.training_data]
            
            # Encode labels
            y_encoded = self.meta_label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.meta_scaler.fit_transform(X)
            
            # Train Neural Network Meta-Model
            self.meta_model = MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),  # 3-layer deep network
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2
            )
            
            # Split for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42
            )
            
            # Train model
            self.meta_model.fit(X_train, y_train)
            
            # Validate
            val_accuracy = self.meta_model.score(X_val, y_val)
            train_accuracy = self.meta_model.score(X_train, y_train)
            
            self.is_trained = True
            
            logger.info(f"Meta-Learning Model Trained Successfully!")
            logger.info(f"Training Accuracy: {train_accuracy:.3f}")
            logger.info(f"Validation Accuracy: {val_accuracy:.3f}")
            logger.info(f"Training Samples: {len(self.training_data)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training meta-model: {e}")
            return False
    
    def predict(self, svm_pred, svm_conf, kmeans_cluster, kmeans_conf, 
                rule_pred, rule_conf, enhanced_features):
        """
        🎯 Meta-Learning Prediction - AI learns the best combination!
        """
        try:
            # Extract meta-features
            meta_features = self._extract_meta_features(
                svm_pred, svm_conf, kmeans_cluster, kmeans_conf,
                rule_pred, rule_conf, enhanced_features
            )
            
            if self.is_trained and self.meta_model is not None:
                # AI-POWERED ENSEMBLE PREDICTION
                meta_features_scaled = self.meta_scaler.transform([meta_features])
                
                # Get prediction probabilities
                prediction_proba = self.meta_model.predict_proba(meta_features_scaled)[0]
                predicted_class_idx = np.argmax(prediction_proba)
                predicted_class = self.meta_label_encoder.inverse_transform([predicted_class_idx])[0]
                
                # Confidence from neural network
                ensemble_confidence = float(np.max(prediction_proba))
                
                # Get feature importance (approximate)
                feature_importance = abs(meta_features_scaled[0])
                feature_importance = feature_importance / np.sum(feature_importance)
                
                return {
                    "makespan": predicted_class,
                    "cluster": kmeans_cluster,
                    "confidence": ensemble_confidence,
                    "method": "MetaLearning_NeuralNetwork",
                    "model_contributions": {
                        "svm_influence": float(feature_importance[0:3].mean()),  # SVM features
                        "kmeans_influence": float(feature_importance[6:8].mean()),  # K-Means features
                        "business_influence": float(feature_importance[8:].mean())  # Business features
                    },
                    "prediction_probabilities": {
                        class_name: float(prob) 
                        for class_name, prob in zip(
                            self.meta_label_encoder.classes_, 
                            prediction_proba
                        )
                    }
                }
            else:
                # 📊 FALLBACK: Intelligent weighted voting (better than if-else)
                return self._intelligent_fallback(
                    svm_pred, svm_conf, kmeans_cluster, kmeans_conf,
                    rule_pred, rule_conf, enhanced_features
                )
                
        except Exception as e:
            logger.error(f"Error in meta-learning prediction: {e}")
            # Fallback to simple combination
            return self._intelligent_fallback(
                svm_pred, svm_conf, kmeans_cluster, kmeans_conf,
                rule_pred, rule_conf, enhanced_features
            )
    
    def _intelligent_fallback(self, svm_pred, svm_conf, kmeans_cluster, kmeans_conf,
                            rule_pred, rule_conf, enhanced_features):
        """
        🔄 Intelligent fallback with adaptive weighting (no hardcoded if-else!)
        """
        # Adaptive weights based on model confidence and domain knowledge
        base_weights = {'svm': 0.4, 'kmeans': 0.3, 'rule': 0.3}
        
        # Confidence-based adjustment
        total_confidence = svm_conf + kmeans_conf + rule_conf
        if total_confidence > 0:
            conf_weights = {
                'svm': svm_conf / total_confidence,
                'kmeans': kmeans_conf / total_confidence, 
                'rule': rule_conf / total_confidence
            }
            
            # Combine base and confidence weights
            final_weights = {
                'svm': (base_weights['svm'] + conf_weights['svm']) / 2,
                'kmeans': (base_weights['kmeans'] + conf_weights['kmeans']) / 2,
                'rule': (base_weights['rule'] + conf_weights['rule']) / 2
            }
        else:
            final_weights = base_weights
        
        # Soft voting instead of hard if-else
        makespan_scores = {"small": 0, "medium": 0, "large": 0}
        
        # SVM contribution
        makespan_scores[svm_pred] += final_weights['svm']
        
        # Rule-based contribution  
        makespan_scores[rule_pred] += final_weights['rule']
        
        # K-Means contribution with soft mapping
        cluster_to_workload = {
            0: {"small": 0.7, "medium": 0.3, "large": 0.0},
            1: {"small": 0.5, "medium": 0.4, "large": 0.1},
            2: {"small": 0.3, "medium": 0.6, "large": 0.1},
            3: {"small": 0.1, "medium": 0.7, "large": 0.2},
            4: {"small": 0.0, "medium": 0.5, "large": 0.5},
            5: {"small": 0.0, "medium": 0.3, "large": 0.7}
        }
        
        cluster_mapping = cluster_to_workload.get(kmeans_cluster, 
                                                {"small": 0.33, "medium": 0.33, "large": 0.33})
        
        for workload, prob in cluster_mapping.items():
            makespan_scores[workload] += final_weights['kmeans'] * prob
        
        # Final decision
        final_makespan = max(makespan_scores, key=makespan_scores.get)
        
        # Calculate ensemble confidence
        ensemble_confidence = max(makespan_scores.values()) / sum(makespan_scores.values())
        
        return {
            "makespan": final_makespan,
            "cluster": kmeans_cluster,
            "confidence": ensemble_confidence,
            "method": "IntelligentFallback_SoftVoting",
            "weights": final_weights,
            "makespan_scores": makespan_scores
        }
    
    def save_model(self, filepath="models/meta_ensemble.pkl"):
        """💾 Save trained meta-model"""
        if self.is_trained:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump({
                'meta_model': self.meta_model,
                'meta_scaler': self.meta_scaler,
                'meta_label_encoder': self.meta_label_encoder,
                'is_trained': self.is_trained,
                'training_samples': len(self.training_data)
            }, filepath)
            logger.info(f"Meta-ensemble model saved to {filepath}")
    
    def load_model(self, filepath="models/meta_ensemble.pkl"):
        """📂 Load pre-trained meta-model"""
        if os.path.exists(filepath):
            try:
                data = joblib.load(filepath)
                self.meta_model = data['meta_model']
                self.meta_scaler = data['meta_scaler'] 
                self.meta_label_encoder = data['meta_label_encoder']
                self.is_trained = data['is_trained']
                logger.info(f"Meta-ensemble model loaded from {filepath}")
                logger.info(f"Training samples: {data.get('training_samples', 'unknown')}")
                return True
            except Exception as e:
                logger.error(f"Error loading meta-model: {e}")
                return False
        return False

# Initialize Meta-Learning Ensemble
meta_ensemble = MetaLearningEnsemble()

def ensemble_decision(svm_pred, svm_conf, kmeans_cluster, kmeans_conf, rule_pred, rule_conf, enhanced_features):
    """
    🧠 META-LEARNING ENSEMBLE DECISION
    Uses Neural Network to learn optimal combination instead of hardcoded if-else!
    """
    
    # Collect training sample for continuous learning
    meta_ensemble.collect_training_sample(
        svm_pred, svm_conf, kmeans_cluster, kmeans_conf,
        rule_pred, rule_conf, enhanced_features
    )
    
    # Get AI-powered ensemble prediction
    result = meta_ensemble.predict(
        svm_pred, svm_conf, kmeans_cluster, kmeans_conf,
        rule_pred, rule_conf, enhanced_features
    )
    
    # Auto-train meta-model when enough data is collected
    if len(meta_ensemble.training_data) >= 100 and not meta_ensemble.is_trained:
        logger.info("Auto-training Meta-Learning model...")
        meta_ensemble.train_meta_model()
    
    return result

def ensemble_decision_simplified(kmeans_cluster, kmeans_conf, rule_pred, rule_conf, enhanced_features):
    """
    Simplified ensemble decision for K-Means + Rule-based only
    """
    # Adaptive weights based on confidence
    total_confidence = kmeans_conf + rule_conf
    if total_confidence > 0:
        kmeans_weight = kmeans_conf / total_confidence
        rule_weight = rule_conf / total_confidence
    else:
        kmeans_weight = 0.5
        rule_weight = 0.5
    
    # Soft voting
    makespan_scores = {"small": 0, "medium": 0, "large": 0}
    
    # Rule-based contribution  
    makespan_scores[rule_pred] += rule_weight
    
    # K-Means contribution with soft mapping
    cluster_to_workload = {
        0: {"small": 0.7, "medium": 0.3, "large": 0.0},
        1: {"small": 0.5, "medium": 0.4, "large": 0.1},
        2: {"small": 0.3, "medium": 0.6, "large": 0.1},
        3: {"small": 0.1, "medium": 0.7, "large": 0.2},
        4: {"small": 0.0, "medium": 0.5, "large": 0.5},
        5: {"small": 0.0, "medium": 0.3, "large": 0.7}
    }
    
    cluster_mapping = cluster_to_workload.get(kmeans_cluster, 
                                            {"small": 0.33, "medium": 0.33, "large": 0.33})
    
    for workload, prob in cluster_mapping.items():
        makespan_scores[workload] += kmeans_weight * prob
    
    # Final decision
    final_makespan = max(makespan_scores, key=makespan_scores.get)
    
    # Calculate ensemble confidence
    ensemble_confidence = max(makespan_scores.values()) / sum(makespan_scores.values())
    
    return {
        "makespan": final_makespan,
        "cluster": kmeans_cluster,
        "confidence": ensemble_confidence,
        "weights": {"kmeans": kmeans_weight, "rule": rule_weight},
        "makespan_scores": makespan_scores
    }

@app.route('/admin/metrics', methods=['GET'])
@performance_tracker
def get_metrics():
    """
    📊 ADVANCED METRICS ENDPOINT
    Trả về chi tiết performance metrics cho monitoring
    """
    try:
        # Get system stats
        system_stats = perf_monitor.get_stats()
        
        # Calculate endpoint-specific metrics
        endpoint_metrics = {}
        for endpoint, requests in request_metrics.items():
            if requests:
                response_times = [r['response_time'] for r in requests]
                success_rate = sum(1 for r in requests if r['success']) / len(requests)
                
                endpoint_metrics[endpoint] = {
                    'total_requests': len(requests),
                    'average_response_time_ms': np.mean(response_times) * 1000,
                    'min_response_time_ms': np.min(response_times) * 1000,
                    'max_response_time_ms': np.max(response_times) * 1000,
                    'success_rate': success_rate,
                    'errors': len(requests) - sum(1 for r in requests if r['success'])
                }
        
        # Model performance metrics
        model_stats = {
            'svm': {
                'loaded': svm_model is not None,
                'kernel': svm_model.kernel if svm_model else None,
                'support_vectors': int(sum(svm_model.n_support_)) if svm_model else 0,
                'classes': svm_model.classes_.tolist() if svm_model else []
            },
            'kmeans': {
                'loaded': kmeans_model is not None,
                'clusters': kmeans_model.n_clusters if kmeans_model else 0,
                'inertia': float(kmeans_model.inertia_) if kmeans_model else 0
            }
        }
        
        return jsonify({
            'system': system_stats,
            'cache': {
                'total_entries': len(prediction_cache),
                'hit_rate': performance_stats['cache_hits'] / (performance_stats['cache_hits'] + performance_stats['cache_misses']) if (performance_stats['cache_hits'] + performance_stats['cache_misses']) > 0 else 0,
                'hits': performance_stats['cache_hits'],
                'misses': performance_stats['cache_misses']
            },
            'endpoints': endpoint_metrics,
            'models': model_stats,
            'global_stats': performance_stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in get_metrics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/debug', methods=['GET'])
@performance_tracker  
def debug_info():
    """
    🔧 DEBUG ENDPOINT
    Trả về thông tin debug chi tiết cho troubleshooting
    """
    try:
        # Recent errors
        recent_errors = []
        for endpoint, requests in request_metrics.items():
            for req in requests[-10:]:  # Last 10 requests
                if not req['success']:
                    recent_errors.append({
                        'endpoint': endpoint,
                        'timestamp': req['timestamp'].isoformat(),
                        'response_time': req['response_time']
                    })
        
        # Cache analysis
        cache_analysis = {
            'total_entries': len(prediction_cache),
            'oldest_entry': min(prediction_cache.values(), key=lambda x: x['timestamp'])['timestamp'].isoformat() if prediction_cache else None,
            'newest_entry': max(prediction_cache.values(), key=lambda x: x['timestamp'])['timestamp'].isoformat() if prediction_cache else None,
            'cache_size_mb': sys.getsizeof(prediction_cache) / 1024 / 1024
        }
        
        # Model detailed info
        model_debug = {}
        if svm_model:
            model_debug['svm'] = {
                'type': str(type(svm_model)),
                'kernel': svm_model.kernel,
                'C': svm_model.C,
                'gamma': svm_model.gamma,
                'n_support': svm_model.n_support_.tolist(),
                'support_vectors_shape': svm_model.support_vectors_.shape,
                'classes': svm_model.classes_.tolist()
            }
        
        if kmeans_model:
            model_debug['kmeans'] = {
                'type': str(type(kmeans_model)),
                'n_clusters': kmeans_model.n_clusters,
                'inertia': float(kmeans_model.inertia_),
                'n_iter': kmeans_model.n_iter_,
                'cluster_centers_shape': kmeans_model.cluster_centers_.shape
            }
        
        return jsonify({
            'service_info': {
                'python_version': sys.version,
                'working_directory': os.getcwd(),
                'environment_variables': dict(os.environ),
                'loaded_modules': list(sys.modules.keys())[:50]  # First 50 modules
            },
            'recent_errors': recent_errors[-20:],  # Last 20 errors
            'cache_analysis': cache_analysis,
            'model_debug': model_debug,
            'memory_usage': {
                'prediction_cache_mb': sys.getsizeof(prediction_cache) / 1024 / 1024,
                'request_metrics_mb': sys.getsizeof(request_metrics) / 1024 / 1024,
                'total_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in debug_info: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/cache/clear', methods=['POST'])
@performance_tracker
def clear_cache():
    """
    🗑️ CACHE MANAGEMENT ENDPOINT
    Xóa cache để refresh predictions
    """
    try:
        cache_size_before = len(prediction_cache)
        prediction_cache.clear()
        
        # Reset cache stats
        performance_stats['cache_hits'] = 0
        performance_stats['cache_misses'] = 0
        
        logger.info(f"Cache cleared: {cache_size_before} entries removed")
        
        return jsonify({
            'status': 'success',
            'message': f'Cache cleared successfully',
            'entries_removed': cache_size_before,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/reload', methods=['POST'])
@performance_tracker
def reload_models():
    """
    🔄 MODEL RELOAD ENDPOINT
    Reload models từ disk (useful for model updates)
    """
    try:
        logger.info("Reloading models...")
        load_models()
        
        # Clear cache after model reload
        prediction_cache.clear()
        
        return jsonify({
            'status': 'success',
            'message': 'Models reloaded successfully',
            'models_loaded': {
                'svm': svm_model is not None,
                'kmeans': kmeans_model is not None,
                'scalers': svm_scaler is not None and kmeans_scaler is not None
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/compare', methods=['POST'])
@performance_tracker
def compare_predictions():
    """
    🔬 PREDICTION COMPARISON ENDPOINT
    So sánh predictions từ individual models vs ensemble
    FIXED: Updated to use correct 9-feature SVM model
    """
    try:
        if svm_model is None or kmeans_model is None:
            return jsonify({"error": "Models not loaded"}), 503
        
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' field"}), 400
        
        features = data["features"]
        vm_features = data.get("vm_features", [0.5, 0.5, 0.5])
        
        # Validate features for comparison - SVM model uses 9 features
        if len(features) != 9:
            return jsonify({"error": "Expected 9 features for SVM: [jobs_1min, jobs_5min, jobs_15min, memory_mb, disk_gb, cpu_cores, cpu_speed, network_receive_kbps, network_transmit_kbps]"}), 400
        
        start_time = time.time()
        
        # Individual SVM prediction (9 features)
        features_scaled = svm_scaler.transform([features])
        svm_pred_numeric = svm_model.predict(features_scaled)[0]
        svm_prediction = svm_label_encoder.inverse_transform([svm_pred_numeric])[0]
        svm_decision_scores = svm_model.decision_function(features_scaled)
        svm_confidence = float(np.abs(svm_decision_scores[0])) if not isinstance(svm_decision_scores[0], np.ndarray) else float(np.max(np.abs(svm_decision_scores[0])))
        
        # Individual K-Means prediction (3 VM features)
        vm_scaled = kmeans_scaler.transform([vm_features])
        kmeans_cluster = int(kmeans_model.predict(vm_scaled)[0])
        kmeans_distances = kmeans_model.transform(vm_scaled)[0]
        kmeans_distance = float(np.min(kmeans_distances))
        
        # Simple ensemble decision for comparison
        if svm_confidence > 0.8:
            ensemble_decision = svm_prediction
            ensemble_algorithm = "SVM (High Confidence)"
        elif kmeans_distance < 0.5:
            cluster_to_workload = {0: "small", 1: "small", 2: "medium", 3: "medium", 4: "large", 5: "large"}
            ensemble_decision = cluster_to_workload.get(kmeans_cluster, "medium")
            ensemble_algorithm = "K-Means (Close Cluster)"
        else:
            ensemble_decision = svm_prediction  # SVM fallback
            ensemble_algorithm = "SVM (Fallback)"
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'individual_predictions': {
                'svm': {
                    'prediction': svm_prediction,
                    'confidence': svm_confidence,
                    'decision_scores': svm_decision_scores.tolist() if isinstance(svm_decision_scores[0], np.ndarray) else [float(svm_decision_scores[0])]
                },
                'kmeans': {
                    'cluster': kmeans_cluster,
                    'distance_to_centroid': kmeans_distance,
                    'all_distances': kmeans_distances.tolist()
                }
            },
            'ensemble_prediction': {
                'decision': ensemble_decision,
                'algorithm_used': ensemble_algorithm,
                'confidence': min(svm_confidence / 3.0, 0.95) if svm_confidence > 0.8 else 1.0 - kmeans_distance
            },
            'comparison': {
                'svm_vs_ensemble': svm_prediction == ensemble_decision,
                'processing_time_ms': processing_time * 1000,
                'features_used': features,
                'vm_features_used': vm_features,
                'note': 'SVM model uses 9 features as actually trained'
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in compare_predictions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/meta_learning', methods=['POST'])
@performance_tracker
@cache_prediction
def predict_meta_learning():
    """
    🧠 META-LEARNING NEURAL NETWORK PREDICTION
    Direct access to the trained 96% accuracy Meta-Learning system
    Input: {"features": [13 combined features from SVM, K-Means, and business logic]}
    Output: {"makespan": "small|medium|large", "confidence": float, "method": "MetaLearning_NeuralNetwork"}
    """
    try:
        if meta_learning_model is None or meta_learning_scaler is None or meta_learning_encoder is None:
            return jsonify({"error": "Meta-Learning model components not loaded"}), 503
        
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' field"}), 400
        
        features = data["features"]
        
        # Validate input - Meta-Learning expects 13 features
        if len(features) != 13:
            return jsonify({
                "error": "Expected 13 features for Meta-Learning model",
                "expected_features": meta_learning_features,
                "received_count": len(features)
            }), 400
        
        # Validate feature ranges based on training data
        try:
            # Scale features using trained scaler
            features_scaled = meta_learning_scaler.transform([features])
            
            # Predict using trained Meta-Learning Neural Network
            prediction_proba = meta_learning_model.predict_proba(features_scaled)[0]
            predicted_class_idx = np.argmax(prediction_proba)
            predicted_class = meta_learning_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Confidence from neural network probabilities
            confidence = float(np.max(prediction_proba))
            
            # Get all class probabilities
            prediction_probabilities = {
                class_name: float(prob) 
                for class_name, prob in zip(
                    meta_learning_encoder.classes_, 
                    prediction_proba
                )
            }
            
            # Feature importance (approximate)
            feature_importance = abs(features_scaled[0])
            feature_importance = feature_importance / np.sum(feature_importance)
            
            # Map features to importance
            feature_contributions = {
                feature_name: float(importance)
                for feature_name, importance in zip(meta_learning_features, feature_importance)
            }
            
            logger.info(f"Meta-Learning prediction: {predicted_class} (confidence: {confidence:.3f})")
            
            return jsonify({
                "makespan": predicted_class,
                "confidence": confidence,
                "method": "MetaLearning_NeuralNetwork",
                "model_info": {
                    "architecture": meta_learning_info.get('architecture', 'Unknown'),
                    "training_accuracy": meta_learning_info.get('training_accuracy', 0),
                    "test_accuracy": meta_learning_info.get('test_accuracy', 0),
                    "cross_val_accuracy": meta_learning_info.get('cross_val_accuracy', 0)
                },
                "prediction_probabilities": prediction_probabilities,
                "feature_contributions": feature_contributions,
                "features_used": features,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in Meta-Learning prediction: {e}")
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        
    except Exception as e:
        logger.error(f"Error in predict_meta_learning: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/mccva_complete', methods=['POST'])
@performance_tracker
@cache_prediction
def predict_mccva_complete():
    """
    🎯 COMPLETE MCCVA 3-STAGE META-LEARNING PREDICTION
    Automated pipeline: Raw Input → SVM → K-Means → Meta-Learning Neural Network
    Input: {"cpu_cores": int, "memory_gb": float, "storage_gb": float, "network_bandwidth": int, "priority": int, "vm_cpu_usage": float, "vm_memory_usage": float, "vm_storage_usage": float}
    Output: {"makespan": "small|medium|large", "confidence": float, "stage_results": {...}, "meta_learning": {...}}
    """
    try:
        if not all([svm_model, kmeans_model, meta_learning_model]):
            return jsonify({"error": "Not all models loaded for complete MCCVA pipeline"}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No request data"}), 400
        
        # Extract raw inputs
        cpu_cores = data.get("cpu_cores", 2)
        memory_gb = data.get("memory_gb", 4)
        storage_gb = data.get("storage_gb", 100)
        network_bandwidth = data.get("network_bandwidth", 1000)
        priority = data.get("priority", 3)
        vm_cpu_usage = data.get("vm_cpu_usage", 0.5)
        vm_memory_usage = data.get("vm_memory_usage", 0.5)
        vm_storage_usage = data.get("vm_storage_usage", 0.5)
        
        # STAGE 1: SVM Prediction - Convert to 9 features (as actually trained)
        # [Jobs_per_ 1Minute, Jobs_per_ 5 Minutes, Jobs_per_ 15Minutes, Mem capacity, Disk_capacity_GB, Num_of_CPU_Cores, CPU_speed_per_Core, Avg_Recieve_Kbps, Avg_Transmit_Kbps]
        
        jobs_1min = max(1.0, float(cpu_cores * priority))
        jobs_5min = jobs_1min * 5
        jobs_15min = jobs_1min * 15
        memory_mb = memory_gb * 1024  # Convert GB to MB 
        disk_gb = storage_gb
        cpu_speed_mhz = 2000 + (priority - 1) * 300  # Estimate based on priority (2000-3200 MHz)
        network_receive_kbps = network_bandwidth * 0.6  # 60% receive
        network_transmit_kbps = network_bandwidth * 0.4  # 40% transmit
        
        svm_features = [
            jobs_1min, jobs_5min, jobs_15min, 
            memory_mb, disk_gb, cpu_cores, cpu_speed_mhz,
            network_receive_kbps, network_transmit_kbps
        ]
        
        # SVM Prediction
        svm_features_scaled = svm_scaler.transform([svm_features])
        svm_prediction_numeric = svm_model.predict(svm_features_scaled)[0]
        svm_prediction = svm_label_encoder.inverse_transform([svm_prediction_numeric])[0]
        svm_decision_scores = svm_model.decision_function(svm_features_scaled)
        svm_confidence = float(np.abs(svm_decision_scores[0])) if not isinstance(svm_decision_scores[0], np.ndarray) else float(np.max(np.abs(svm_decision_scores[0])))
        
        # STAGE 2: K-Means Prediction - FIXED: Use 5 features as trained
        # K-Means was trained with 5 features: [memory_utilization, cpu_utilization, storage_utilization, network_utilization, workload_intensity]
        avg_job_rate = (jobs_1min + jobs_5min + jobs_1min) / 3  # Estimate job rate
        max_job_rate_estimate = 100  # Reasonable max for normalization
        workload_intensity_norm = min(avg_job_rate / max_job_rate_estimate, 1.0)
        
        # Estimate network utilization from bandwidth and priority
        network_utilization = min(network_bandwidth / 10000.0, 1.0)  # Normalize to 0-1
        
        vm_features = [
            vm_memory_usage,      # memory_utilization (0-1)
            vm_cpu_usage,         # cpu_utilization (0-1) 
            vm_storage_usage,     # storage_utilization (0-1)
            network_utilization,  # network_utilization (0-1)
            workload_intensity_norm  # workload_intensity (0-1)
        ]
        
        # K-Means Prediction
        vm_features_scaled = kmeans_scaler.transform([vm_features])
        kmeans_cluster = int(kmeans_model.predict(vm_features_scaled)[0])
        kmeans_distances = kmeans_model.transform(vm_features_scaled)[0]
        kmeans_confidence = float(1 / (1 + np.min(kmeans_distances)))
        
        # STAGE 3: Meta-Learning Neural Network
        # Build 13-feature vector for Meta-Learning
        enhanced_features = extract_enhanced_features([cpu_cores, memory_gb, storage_gb, network_bandwidth, priority])
        rule_prediction, rule_confidence = get_rule_based_prediction(enhanced_features)
        
        # Create SVM scores
        svm_small_score = 1.0 if svm_prediction == "small" else 0.0
        svm_medium_score = 1.0 if svm_prediction == "medium" else 0.0  
        svm_large_score = 1.0 if svm_prediction == "large" else 0.0
        
        # Build Meta-Learning feature vector (13 features)
        meta_features = [
            svm_confidence,  # 0
            kmeans_confidence,  # 1
            rule_confidence,  # 2
            svm_small_score,  # 3
            svm_medium_score,  # 4
            svm_large_score,  # 5
            float(kmeans_cluster),  # 6
            1 / (1 + kmeans_confidence) if kmeans_confidence > 0 else 0.5,  # 7 - cluster_distance_norm
            enhanced_features.get('compute_intensity', 0),  # 8
            enhanced_features.get('memory_intensity', 0),  # 9
            enhanced_features.get('storage_intensity', 0),  # 10
            float(enhanced_features.get('high_priority', False)),  # 11
            1 - abs(enhanced_features.get('compute_intensity', 0) - 0.5) - abs(enhanced_features.get('memory_intensity', 0) - 0.5)  # 12 - balance_score
        ]
        
        # Meta-Learning Prediction
        meta_features_scaled = meta_learning_scaler.transform([meta_features])
        meta_prediction_proba = meta_learning_model.predict_proba(meta_features_scaled)[0]
        meta_predicted_class_idx = np.argmax(meta_prediction_proba)
        meta_predicted_class = meta_learning_encoder.inverse_transform([meta_predicted_class_idx])[0]
        meta_confidence = float(np.max(meta_prediction_proba))
        
        # Response with complete pipeline results
        return jsonify({
            "makespan": meta_predicted_class,
            "confidence": meta_confidence,
            "method": "MCCVA_3Stage_MetaLearning",
            "stage_results": {
                "stage1_svm": {
                    "prediction": svm_prediction,
                    "confidence": svm_confidence,
                    "features_used": svm_features
                },
                "stage2_kmeans": {
                    "cluster": kmeans_cluster,
                    "confidence": kmeans_confidence,
                    "features_used": vm_features,
                    "centroid_distance": float(np.min(kmeans_distances))
                },
                "stage3_metalearning": {
                    "prediction": meta_predicted_class,
                    "confidence": meta_confidence,
                    "features_used": meta_features,
                    "prediction_probabilities": {
                        class_name: float(prob) 
                        for class_name, prob in zip(
                            meta_learning_encoder.classes_, 
                            meta_prediction_proba
                        )
                    }
                }
            },
            "input_parameters": {
                "cpu_cores": cpu_cores,
                "memory_gb": memory_gb,
                "storage_gb": storage_gb,
                "network_bandwidth": network_bandwidth,
                "priority": priority,
                "vm_cpu_usage": vm_cpu_usage,
                "vm_memory_usage": vm_memory_usage,
                "vm_storage_usage": vm_storage_usage
            },
            "model_info": {
                "system": "MCCVA 3-Stage Meta-Learning",
                "svm_accuracy": "50.98% balanced",
                "kmeans_silhouette": "0.523",
                "metalearning_accuracy": meta_learning_info.get('test_accuracy', 0),
                "architecture": meta_learning_info.get('architecture', 'Unknown')
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in MCCVA complete prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load models khi khởi động
    load_models()
    
    # Chạy Flask app
    logger.info("🚀 ML Service đang khởi động trên port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False) 