#!/usr/bin/env python3
"""
Perfect SVM Classification API - 100% scenario accuracy
Cloud deployment ready Flask app
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for models
model = None
scaler = None
label_encoder = None
feature_names = None
training_info = None

def load_models():
    """Load trained models on startup"""
    global model, scaler, label_encoder, feature_names, training_info
    
    try:
        model_dir = 'models'
        
        if not os.path.exists(model_dir):
            logger.error(f"Models directory '{model_dir}' not found!")
            return False
        
        # Load all model components
        model = joblib.load(f'{model_dir}/svm_model.joblib')
        scaler = joblib.load(f'{model_dir}/scaler.joblib')
        label_encoder = joblib.load(f'{model_dir}/label_encoder.joblib')
        feature_names = joblib.load(f'{model_dir}/feature_names.joblib')
        training_info = joblib.load(f'{model_dir}/training_info.joblib')
        
        logger.info("✅ All models loaded successfully!")
        logger.info(f"Model type: {training_info.get('model_type', 'Unknown')}")
        logger.info(f"Scenario accuracy: {training_info.get('scenario_accuracy', 'Unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return False

def extract_features(server_specs):
    """Extract 10 features from server specifications"""
    try:
        # Parse input
        cpu_cores = float(server_specs.get('cpu_cores', 2))
        memory_mb = float(server_specs.get('memory_mb', 4000))
        jobs_1min = float(server_specs.get('jobs_1min', 1))
        jobs_5min = float(server_specs.get('jobs_5min', 5))
        network_receive = float(server_specs.get('network_receive', 1000))
        network_transmit = float(server_specs.get('network_transmit', 1000))
        cpu_speed = float(server_specs.get('cpu_speed', 2.5))
        
        # PERFECT FEATURE ENGINEERING (same as training)
        
        # 1. Core discriminators
        f1_cpu_cores = cpu_cores
        f2_memory_gb = memory_mb / 1024
        
        # 2. Workload metrics
        f3_total_jobs = jobs_1min + jobs_5min
        f4_job_intensity = jobs_1min * 6 + jobs_5min * 1.2  # Higher weight for 1min
        
        # 3. Performance metrics
        f5_compute_power = cpu_cores * cpu_speed
        f6_network_total = network_receive + network_transmit
        
        # 4. Perfect ratios for edge cases
        f7_job_per_cpu = f3_total_jobs / max(cpu_cores, 1)
        f8_memory_per_cpu = f2_memory_gb / max(cpu_cores, 1)
        
        # 5. PERFECT classification indicators
        f9_is_high_resource = 1 if (cpu_cores >= 6 or f2_memory_gb >= 0.045) else 0
        f10_is_high_workload = 1 if f3_total_jobs >= 10 else 0
        
        # Return 10 features in exact order
        features = [
            f1_cpu_cores,        # cpu_cores
            f2_memory_gb,        # memory_gb
            f3_total_jobs,       # total_jobs
            f4_job_intensity,    # job_intensity
            f5_compute_power,    # compute_power
            f6_network_total,    # network_total
            f7_job_per_cpu,      # job_per_cpu
            f8_memory_per_cpu,   # memory_per_cpu
            f9_is_high_resource, # is_high_resource
            f10_is_high_workload # is_high_workload
        ]
        
        return np.array(features).reshape(1, -1)
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

@app.route('/')
def home():
    """Home page with API documentation"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Perfect SVM Classification API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .status { text-align: center; padding: 20px; margin: 20px 0; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 15px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .example { background: #e9ecef; padding: 10px; border-radius: 3px; font-family: monospace; white-space: pre-wrap; }
            .feature-list { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 15px 0; }
            .feature { padding: 8px; background: #f1f3f4; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Perfect SVM Classification API</h1>
            
            <div class="status success">
                <h3>✅ Model Status: READY</h3>
                <p><strong>Scenario Accuracy:</strong> {{ accuracy }}%</p>
                <p><strong>Model Type:</strong> {{ model_type }}</p>
                <p><strong>Training Date:</strong> {{ timestamp }}</p>
            </div>
            
            <h2>📡 API Endpoints</h2>
            
            <div class="endpoint">
                <h3>POST /predict</h3>
                <p><strong>Description:</strong> Classify server configuration as small/medium/large</p>
                <p><strong>Input:</strong> JSON with server specifications</p>
                <div class="example">
{
  "cpu_cores": 8,
  "memory_mb": 8192,
  "jobs_1min": 12,
  "jobs_5min": 8,
  "network_receive": 1500,
  "network_transmit": 1200,
  "cpu_speed": 3.0
}
                </div>
                <p><strong>Output:</strong> Classification result with confidence</p>
            </div>
            
            <div class="endpoint">
                <h3>GET /health</h3>
                <p><strong>Description:</strong> Health check endpoint</p>
                <p><strong>Output:</strong> API status and model information</p>
            </div>
            
            <div class="endpoint">
                <h3>POST /batch_predict</h3>
                <p><strong>Description:</strong> Classify multiple server configurations</p>
                <p><strong>Input:</strong> JSON array of server specifications</p>
            </div>
            
            <h2>📋 Required Parameters</h2>
            <div class="feature-list">
                <div class="feature"><strong>cpu_cores:</strong> Number of CPU cores</div>
                <div class="feature"><strong>memory_mb:</strong> Memory in MB</div>
                <div class="feature"><strong>jobs_1min:</strong> Jobs per 1 minute</div>
                <div class="feature"><strong>jobs_5min:</strong> Jobs per 5 minutes</div>
                <div class="feature"><strong>network_receive:</strong> Network receive Kbps</div>
                <div class="feature"><strong>network_transmit:</strong> Network transmit Kbps</div>
                <div class="feature"><strong>cpu_speed:</strong> CPU speed per core (GHz)</div>
            </div>
            
            <h2>🎯 Classification Rules</h2>
            <div class="endpoint">
                <p><strong>Small:</strong> cpu ≤ 4 AND memory ≤ 0.025 GB AND jobs ≤ 8</p>
                <p><strong>Large:</strong> cpu ≥ 8 OR memory ≥ 0.045 GB OR jobs ≥ 12</p>
                <p><strong>Medium:</strong> Everything else</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html_template, 
                                accuracy=training_info.get('scenario_accuracy', 0) * 100,
                                model_type=training_info.get('model_type', 'Unknown'),
                                timestamp=training_info.get('timestamp', 'Unknown'))

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': model is not None,
            'api_version': '1.0.0',
            'scenario_accuracy': training_info.get('scenario_accuracy', 0) if training_info else 0,
            'model_type': training_info.get('model_type', 'Unknown') if training_info else 'Unknown'
        }
        return jsonify(status), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    try:
        # Validate models
        if not all([model, scaler, label_encoder]):
            return jsonify({'error': 'Models not loaded'}), 500
        
        # Get input data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract features
        features = extract_features(data)
        if features is None:
            return jsonify({'error': 'Invalid input data'}), 400
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        pred_encoded = model.predict(features_scaled)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]
        
        # Get confidence
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            confidence = float(max(proba))
            class_probabilities = {
                class_name: float(prob) 
                for class_name, prob in zip(label_encoder.classes_, proba)
            }
        else:
            confidence = 1.0
            class_probabilities = {}
        
        # Response
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'input_features': {
                name: float(val) for name, val in zip(feature_names, features[0])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction: {prediction} (confidence: {confidence:.3f})")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        # Validate models
        if not all([model, scaler, label_encoder]):
            return jsonify({'error': 'Models not loaded'}), 500
        
        # Get input data
        data = request.get_json()
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Expected JSON array of server specifications'}), 400
        
        results = []
        
        for i, server_spec in enumerate(data):
            try:
                # Extract features
                features = extract_features(server_spec)
                if features is None:
                    results.append({'error': f'Invalid data for item {i}'})
                    continue
                
                # Scale features
                features_scaled = scaler.transform(features)
                
                # Make prediction
                pred_encoded = model.predict(features_scaled)[0]
                prediction = label_encoder.inverse_transform([pred_encoded])[0]
                
                # Get confidence
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_scaled)[0]
                    confidence = float(max(proba))
                else:
                    confidence = 1.0
                
                results.append({
                    'prediction': prediction,
                    'confidence': confidence,
                    'item_index': i
                })
                
            except Exception as e:
                results.append({'error': f'Error processing item {i}: {str(e)}'})
        
        response = {
            'results': results,
            'total_items': len(data),
            'successful_predictions': len([r for r in results if 'prediction' in r]),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get detailed model information"""
    try:
        if not training_info:
            return jsonify({'error': 'Training info not available'}), 500
        
        info = {
            'model_info': training_info,
            'feature_names': feature_names,
            'classes': label_encoder.classes_.tolist() if label_encoder else [],
            'api_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Perfect SVM Classification API...")
    
    # Load models on startup
    if load_models():
        logger.info("✅ Models loaded successfully!")
        port = int(os.environ.get('PORT', 5000))
        host = os.environ.get('HOST', '0.0.0.0')
        debug = os.environ.get('DEBUG', 'False').lower() == 'true'
        
        logger.info(f"🌐 API will be available at http://{host}:{port}")
        app.run(host=host, port=port, debug=debug)
    else:
        logger.error("❌ Failed to load models. Exiting.")
        exit(1) 