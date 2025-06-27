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

def load_models():
    """Load các mô hình khi khởi động service"""
    global svm_model, kmeans_model, svm_scaler, kmeans_scaler
    
    try:
        # Đảm bảo working directory đúng
        if os.path.exists('/opt/mccva'):
            os.chdir('/opt/mccva')
        
        logger.info("Đang load mô hình SVM...")
        svm_model = joblib.load("models/svm_model.joblib")
        svm_scaler = joblib.load("models/scaler.joblib")
        
        logger.info("Đang load mô hình K-Means...")
        kmeans_model = joblib.load("models/kmeans_model.joblib")
        kmeans_scaler = joblib.load("models/kmeans_scaler.joblib")
        
        logger.info("✅ Tất cả mô hình đã được load thành công!")
        logger.info(f"SVM Model: {svm_model.kernel} kernel, {sum(svm_model.n_support_)} support vectors")
        logger.info(f"K-Means Model: {kmeans_model.n_clusters} clusters")
        
        # Test prediction để đảm bảo models hoạt động
        test_features = [4, 8, 100, 1000, 3]
        test_vm_features = [0.5, 0.5, 0.5]
        
        # Test SVM
        test_scaled = svm_scaler.transform([test_features])
        svm_pred = svm_model.predict(test_scaled)[0]
        logger.info(f"SVM Test prediction: {svm_pred}")
        
        # Test K-Means
        vm_scaled = kmeans_scaler.transform([test_vm_features])
        cluster_pred = kmeans_model.predict(vm_scaled)[0]
        logger.info(f"K-Means Test prediction: Cluster {cluster_pred}")
        
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
            "kmeans_scaler": kmeans_scaler is not None
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
    Input: {"features": [cpu_cores, memory_gb, storage_gb, network_bandwidth, priority]}
    Output: {"makespan": "small|medium|large", "confidence": float}
    """
    try:
        if svm_model is None or svm_scaler is None:
            return jsonify({"error": "SVM model not loaded"}), 503
        
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' field"}), 400
        
        features = data["features"]
        
        # Validate input
        if len(features) != 5:
            return jsonify({"error": "Expected 5 features: [cpu_cores, memory_gb, storage_gb, network_bandwidth, priority]"}), 400
        
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
        
        # Chuẩn hóa dữ liệu
        features_scaled = svm_scaler.transform([features])
        
        # Dự đoán
        prediction = svm_model.predict(features_scaled)[0]
        
        # Tính confidence score
        decision_scores = svm_model.decision_function(features_scaled)
        if isinstance(decision_scores[0], np.ndarray):
            confidence = float(np.max(np.abs(decision_scores[0])))
        else:
            confidence = float(np.abs(decision_scores[0]))
        
        logger.info(f"Prediction: {prediction} (confidence: {confidence:.3f}) for features: {features}")
        
        return jsonify({
            "makespan": prediction,
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

if __name__ == '__main__':
    # Load models khi khởi động
    load_models()
    
    # Chạy Flask app
    logger.info("🚀 ML Service đang khởi động trên port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False) 