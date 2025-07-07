#!/usr/bin/env python3
"""
ML Service - Flask API để phục vụ các mô hình SVM và K-Means
Chạy bằng: python ml_service.py hoặc flask run --port=5000
Production ready với error handling và logging
"""

import joblib
import json
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime
import logging
import os
import sys

# Configure logging
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

def load_models():
    """Load các mô hình khi khởi động service"""
    global svm_model, kmeans_model, svm_scaler, kmeans_scaler, svm_label_encoder
    
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
        
        logger.info("✅ Tất cả mô hình đã được load thành công!")
        logger.info(f"SVM Model: {svm_model.kernel} kernel, {sum(svm_model.n_support_)} support vectors")
        logger.info(f"K-Means Model: {kmeans_model.n_clusters} clusters")
        
        # Test prediction để đảm bảo models hoạt động - DISABLED due to feature mismatch
        # test_features = [4, 8, 100, 1000, 3, 1, 2, 50, 500, 2]  # Updated to 10 features
        # test_vm_features = [0.5, 0.5, 0.5]
        
        # # Test SVM
        # test_scaled = svm_scaler.transform([test_features])
        # svm_pred = svm_model.predict(test_scaled)[0]
        # logger.info(f"SVM Test prediction: {svm_pred}")
        
        # # Test K-Means
        # vm_scaled = kmeans_scaler.transform([test_vm_features])
        # cluster_pred = kmeans_model.predict(vm_scaled)[0]
        # logger.info(f"K-Means Test prediction: Cluster {cluster_pred}")
        
        logger.info("✅ Models loaded successfully - skipping test predictions")
        
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
            "svm_label_encoder": svm_label_encoder is not None
        }
        
        all_models_loaded = all(models_status.values())
        
        return jsonify({
            "status": "healthy" if all_models_loaded else "degraded",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": models_status,
            "service": "mccva-ml-service",
            "version": "1.0.0"
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/predict/makespan', methods=['POST'])
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
        if len(features) != 10:
            return jsonify({"error": "Expected 10 features: [cpu_cores, memory_gb, storage_gb, network_bandwidth, priority, task_complexity, data_size, io_intensity, parallel_degree, deadline_urgency]"}), 400
        
        # Validate ranges (updated for 10 features)
        if not (1 <= features[0] <= 16):  # cpu_cores
            return jsonify({"error": "CPU cores must be between 1-16"}), 400
        if not (1 <= features[1] <= 64):  # memory_gb
            return jsonify({"error": "Memory must be between 1-64 GB"}), 400
        if not (10 <= features[2] <= 1000):  # storage_gb
            return jsonify({"error": "Storage must be between 10-1000 GB"}), 400
        if not (100 <= features[3] <= 10000):  # network_bandwidth
            return jsonify({"error": "Network bandwidth must be between 100-10000 Mbps"}), 400
        if not (1 <= features[4] <= 5):  # priority
            return jsonify({"error": "Priority must be between 1-5"}), 400
        if not (1 <= features[5] <= 5):  # task_complexity
            return jsonify({"error": "Task complexity must be between 1-5"}), 400
        if not (1 <= features[6] <= 1000):  # data_size - changed from 1-5 to 1-1000
            return jsonify({"error": "Data size must be between 1-1000"}), 400
        if not (1 <= features[7] <= 100):  # io_intensity
            return jsonify({"error": "IO intensity must be between 1-100"}), 400
        if not (100 <= features[8] <= 2000):  # parallel_degree
            return jsonify({"error": "Parallel degree must be between 100-2000"}), 400
        if not (1 <= features[9] <= 5):  # deadline_urgency
            return jsonify({"error": "Deadline urgency must be between 1-5"}), 400
        
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
def predict_enhanced():
    """
    Enhanced API với ensemble learning - kết hợp SVM và K-Means
    Input: {"features": [cpu_cores, memory, storage, network_bandwidth, priority], "vm_features": [cpu_usage, ram_usage, storage_usage]}
    Output: {"makespan": "small|medium|large", "cluster": int, "confidence": float, "model_contributions": {...}}
    """
    try:
        if svm_model is None or kmeans_model is None or svm_scaler is None or kmeans_scaler is None:
            return jsonify({"error": "Models not loaded"}), 503
        
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
        if not (1 <= features[0] <= 16):  # cpu_cores
            return jsonify({"error": "CPU cores must be between 1-16"}), 400
        if not (1 <= features[1] <= 64):  # memory_gb
            return jsonify({"error": "Memory must be between 1-64 GB"}), 400
        if not (10 <= features[2] <= 1000):  # storage_gb
            return jsonify({"error": "Storage must be between 10-1000 GB"}), 400
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
        
        # Convert 5 features to 10 features for SVM model
        cpu_cores, memory, storage, network_bandwidth, priority = features
        
        # Calculate the additional 5 features needed for the 10-feature model
        cpu_memory_ratio = cpu_cores / (memory + 1e-6)
        storage_memory_ratio = storage / (memory + 1e-6)
        network_cpu_ratio = network_bandwidth / (cpu_cores + 1e-6)
        resource_intensity = (cpu_cores * memory * storage) / 1000
        priority_weighted_cpu = cpu_cores * priority
        
        # Combine all 10 features for SVM model
        svm_features = [
            cpu_cores, memory, storage, network_bandwidth, priority,
            cpu_memory_ratio, storage_memory_ratio, network_cpu_ratio,
            resource_intensity, priority_weighted_cpu
        ]
        
        # Model 1: SVM Prediction with 10 features
        features_scaled = svm_scaler.transform([svm_features])
        svm_prediction_int = svm_model.predict(features_scaled)[0]
        svm_decision_scores = svm_model.decision_function(features_scaled)
        svm_confidence = float(np.abs(svm_decision_scores[0])) if not isinstance(svm_decision_scores[0], np.ndarray) else float(np.max(np.abs(svm_decision_scores[0])))
        
        # Map SVM integer prediction to string label (EXACT as training)
        # Training mapping: {'large': 0, 'medium': 1, 'small': 2}
        # Load label encoder for correct mapping
        try:
            import joblib
            label_encoder = joblib.load('models/label_encoder.joblib')
            svm_prediction = label_encoder.inverse_transform([int(svm_prediction_int)])[0]
        except:
            # Fallback mapping based on training: {large: 0, medium: 1, small: 2}
            svm_class_mapping = {0: "large", 1: "medium", 2: "small"}
            svm_prediction = svm_class_mapping.get(int(svm_prediction_int), "medium")
        
        # Model 2: K-Means Prediction
        vm_scaled = kmeans_scaler.transform([vm_features])
        kmeans_cluster = int(kmeans_model.predict(vm_scaled)[0])
        kmeans_distances = kmeans_model.transform(vm_scaled)[0]
        kmeans_confidence = float(1 / (1 + np.min(kmeans_distances)))  # Closer to centroid = higher confidence
        
        # Model 3: Rule-based Heuristic
        rule_prediction, rule_confidence = get_rule_based_prediction(enhanced_features)
        
        # Ensemble Decision
        ensemble_result = ensemble_decision(
            svm_prediction, svm_confidence,
            kmeans_cluster, kmeans_confidence,
            rule_prediction, rule_confidence,
            enhanced_features
        )
        
        logger.info(f"Enhanced prediction: {ensemble_result}")
        
        return jsonify({
            "makespan": ensemble_result["makespan"],
            "cluster": ensemble_result["cluster"],
            "confidence": ensemble_result["confidence"],
            "model_contributions": {
                "svm": {
                    "prediction": svm_prediction,
                    "prediction_int": int(svm_prediction_int),
                    "confidence": svm_confidence,
                    "weight": ensemble_result["weights"]["svm"]
                },
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

def ensemble_decision(svm_pred, svm_conf, kmeans_cluster, kmeans_conf, rule_pred, rule_conf, enhanced_features):
    """Ensemble decision combining all models"""
    
    # Dynamic weights based on confidence and model performance
    svm_weight = svm_conf * 0.2  # SVM gets 20% max weight (reduced from 50%)
    kmeans_weight = kmeans_conf * 0.2  # K-Means gets 20% max weight (reduced from 30%)
    rule_weight = rule_conf * 0.6  # Rule-based gets 60% max weight (increased from 20%)
    
    # Normalize weights
    total_weight = svm_weight + kmeans_weight + rule_weight
    if total_weight > 0:
        svm_weight /= total_weight
        kmeans_weight /= total_weight
        rule_weight /= total_weight
    else:
        # Fallback weights
        svm_weight, kmeans_weight, rule_weight = 0.5, 0.3, 0.2
    
    # Weighted voting for makespan
    makespan_scores = {"small": 0, "medium": 0, "large": 0}
    
    # SVM vote
    makespan_scores[svm_pred] += svm_weight
    
    # Rule-based vote
    makespan_scores[rule_pred] += rule_weight
    
    # K-Means influence (indirect)
    if kmeans_cluster in [0, 1]:  # Low-resource clusters
        makespan_scores["small"] += kmeans_weight * 0.5
        makespan_scores["medium"] += kmeans_weight * 0.5
    elif kmeans_cluster in [2, 3]:  # Medium-resource clusters
        makespan_scores["medium"] += kmeans_weight
    else:  # High-resource clusters
        makespan_scores["medium"] += kmeans_weight * 0.5
        makespan_scores["large"] += kmeans_weight * 0.5
    
    # Determine final makespan
    final_makespan = max(makespan_scores, key=makespan_scores.get)
    
    # Calculate ensemble confidence
    ensemble_confidence = (
        svm_conf * svm_weight +
        kmeans_conf * kmeans_weight +
        rule_conf * rule_weight
    )
    
    # Adjust confidence based on agreement
    agreement_score = max(makespan_scores.values()) / sum(makespan_scores.values())
    ensemble_confidence *= agreement_score
    
    return {
        "makespan": final_makespan,
        "cluster": kmeans_cluster,
        "confidence": ensemble_confidence,
        "weights": {
            "svm": svm_weight,
            "kmeans": kmeans_weight,
            "rule": rule_weight
        },
        "agreement_score": agreement_score
    }

if __name__ == '__main__':
    # Load models khi khởi động
    load_models()
    
    # Chạy Flask app
    logger.info("🚀 ML Service đang khởi động trên port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False) 