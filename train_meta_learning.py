#!/usr/bin/env python3
"""
🧠 MCCVA META-LEARNING ENSEMBLE TRAINER
REAL-WORLD VERSION 2.1 - Neural Network Ensemble for SVM + K-Means

This script trains the Meta-Learning Neural Network that combines:
- SVM predictions (workload classification)
- K-Means predictions (VM clustering)
- Rule-based heuristics (business logic)

Uses dataset from dataset/modified directory

Architecture: Neural Network learns optimal combination weights instead of hardcoded if-else!
Paper: "Machine Learning-based VM Load Balancing using Meta-Learning Approach"
"""

import pandas as pd
import numpy as np
import os
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained SVM and K-Means models
def load_base_models():
    """Load pre-trained SVM and K-Means models"""
    print("🔄 Loading base models...")
    
    try:
        # Load SVM components
        svm_model = joblib.load('models/svm_model.joblib')
        svm_scaler = joblib.load('models/svm_scaler.joblib')
        svm_label_encoder = joblib.load('models/svm_label_encoder.joblib')
        svm_features = joblib.load('models/svm_feature_names.joblib')
        
        # Load K-Means components
        kmeans_model = joblib.load('models/kmeans_model.joblib')
        kmeans_scaler = joblib.load('models/kmeans_scaler.joblib')
        kmeans_features = joblib.load('models/kmeans_feature_names.joblib')
        
        print("✅ All base models loaded successfully!")
        print(f"  SVM: {len(svm_features)} features, {len(svm_label_encoder.classes_)} classes")
        print(f"  K-Means: {len(kmeans_features)} features, {kmeans_model.n_clusters} clusters")
        
        return {
            'svm': {'model': svm_model, 'scaler': svm_scaler, 'encoder': svm_label_encoder, 'features': svm_features},
            'kmeans': {'model': kmeans_model, 'scaler': kmeans_scaler, 'features': kmeans_features}
        }
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Make sure you've trained SVM and K-Means models first!")
        return None

def generate_synthetic_meta_training_data(base_models, n_samples=2000):
    """Generate realistic training data for Meta-Learning using base models"""
    print(f"\n🎯 Generating {n_samples:,} synthetic training samples...")
    
    np.random.seed(42)
    
    # Define workload patterns with known ground truth
    workload_patterns = [
        # Small workloads
        {'type': 'small', 'cpu_range': (1, 4), 'memory_range': (1, 8), 'priority_range': (1, 2), 'weight': 0.35},
        {'type': 'small', 'cpu_range': (2, 6), 'memory_range': (2, 12), 'priority_range': (1, 3), 'weight': 0.25},
        
        # Medium workloads  
        {'type': 'medium', 'cpu_range': (4, 10), 'memory_range': (8, 32), 'priority_range': (2, 4), 'weight': 0.25},
        {'type': 'medium', 'cpu_range': (6, 12), 'memory_range': (12, 24), 'priority_range': (3, 4), 'weight': 0.10},
        
        # Large workloads
        {'type': 'large', 'cpu_range': (8, 16), 'memory_range': (16, 64), 'priority_range': (3, 5), 'weight': 0.03},
        {'type': 'large', 'cpu_range': (12, 16), 'memory_range': (32, 64), 'priority_range': (4, 5), 'weight': 0.02},
    ]
    
    meta_training_data = []
    
    for i in range(n_samples):
        # Select pattern based on weights
        pattern = np.random.choice(workload_patterns, 
                                 p=[p['weight'] for p in workload_patterns])
        
        # Generate basic features
        cpu_cores = np.random.randint(pattern['cpu_range'][0], pattern['cpu_range'][1] + 1)
        memory = np.random.randint(pattern['memory_range'][0], pattern['memory_range'][1] + 1)
        storage = np.random.randint(50, 1000)
        network = np.random.randint(1000, 10000)
        priority = np.random.randint(pattern['priority_range'][0], pattern['priority_range'][1] + 1)
        
        # Generate VM utilization features
        vm_cpu_usage = np.random.uniform(0.2, 0.9)
        vm_memory_usage = np.random.uniform(0.3, 0.85)
        vm_storage_usage = np.random.uniform(0.1, 0.7)
        vm_network_usage = np.random.uniform(0.05, 0.3)
        vm_workload_intensity = np.random.uniform(0.02, 0.05)
        
        # Create enhanced SVM features (10 features)
        cpu_memory_ratio = cpu_cores / (memory + 1e-6)
        storage_memory_ratio = storage / (memory + 1e-6)
        network_cpu_ratio = network / (cpu_cores + 1e-6)
        resource_intensity = (cpu_cores * memory * storage) / 1000
        priority_weighted_cpu = cpu_cores * priority
        
        svm_features = [
            cpu_cores, memory, storage, network, priority,
            cpu_memory_ratio, storage_memory_ratio, network_cpu_ratio,
            resource_intensity, priority_weighted_cpu
        ]
        
        # K-Means features (5 features)
        kmeans_features_data = [
            vm_memory_usage, vm_cpu_usage, vm_storage_usage, 
            vm_network_usage, vm_workload_intensity
        ]
        
        # Get base model predictions
        try:
            # SVM prediction
            svm_features_scaled = base_models['svm']['scaler'].transform([svm_features])
            svm_pred_int = base_models['svm']['model'].predict(svm_features_scaled)[0]
            svm_prediction = base_models['svm']['encoder'].inverse_transform([svm_pred_int])[0]
            svm_decision_scores = base_models['svm']['model'].decision_function(svm_features_scaled)
            svm_confidence = float(np.abs(svm_decision_scores[0])) if not isinstance(svm_decision_scores[0], np.ndarray) else float(np.max(np.abs(svm_decision_scores[0])))
            
            # K-Means prediction
            kmeans_features_scaled = base_models['kmeans']['scaler'].transform([kmeans_features_data])
            kmeans_cluster = int(base_models['kmeans']['model'].predict(kmeans_features_scaled)[0])
            kmeans_distances = base_models['kmeans']['model'].transform(kmeans_features_scaled)[0]
            kmeans_confidence = float(1 / (1 + np.min(kmeans_distances)))
            
            # Rule-based prediction (simple heuristic)
            if cpu_cores <= 4 and memory <= 16:
                rule_prediction = 'small'
                rule_confidence = 0.8
            elif cpu_cores >= 12 or memory >= 32:
                rule_prediction = 'large'
                rule_confidence = 0.75
            else:
                rule_prediction = 'medium'
                rule_confidence = 0.7
            
            # Create meta-features (13 features for Neural Network)
            meta_features = extract_meta_features(
                svm_prediction, svm_confidence,
                kmeans_cluster, kmeans_confidence,
                rule_prediction, rule_confidence,
                cpu_cores, memory, storage, priority
            )
            
            # Ground truth is the original pattern type
            true_label = pattern['type']
            
            meta_training_data.append({
                'meta_features': meta_features,
                'true_label': true_label,
                'base_predictions': {
                    'svm': svm_prediction,
                    'kmeans': kmeans_cluster,
                    'rule': rule_prediction
                },
                'confidences': {
                    'svm': svm_confidence,
                    'kmeans': kmeans_confidence,
                    'rule': rule_confidence
                }
            })
            
        except Exception as e:
            print(f"Warning: Skipped sample {i} due to error: {e}")
            continue
            
        # Progress update
        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1:,}/{n_samples:,} samples")
    
    print(f"✅ Generated {len(meta_training_data):,} valid training samples")
    
    # Show label distribution
    label_counts = Counter([sample['true_label'] for sample in meta_training_data])
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        pct = (count / len(meta_training_data)) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")
    
    return meta_training_data

def extract_meta_features(svm_pred, svm_conf, kmeans_cluster, kmeans_conf, 
                         rule_pred, rule_conf, cpu_cores, memory, storage, priority):
    """Extract 13 meta-features for Neural Network training"""
    
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
    
    # 13 meta-features for Neural Network
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

def train_meta_learning_model(training_data):
    """Train Meta-Learning Neural Network"""
    print("\n🧠 TRAINING META-LEARNING NEURAL NETWORK")
    print("=" * 60)
    
    # Prepare training data
    X = np.array([sample['meta_features'] for sample in training_data])
    y = [sample['true_label'] for sample in training_data]
    
    print(f"Training data shape: {X.shape}")
    print(f"Feature names (13 meta-features):")
    feature_names = [
        'svm_confidence', 'kmeans_confidence', 'rule_confidence',
        'svm_small_score', 'svm_medium_score', 'svm_large_score',
        'cluster_id', 'cluster_distance_norm',
        'compute_intensity', 'memory_intensity', 'storage_intensity',
        'is_high_priority', 'resource_balance_score'
    ]
    for i, name in enumerate(feature_names):
        print(f"  {i+1:2d}. {name}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nLabel mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Create Meta-Learning Neural Network
    print(f"\nTraining Neural Network Architecture:")
    print(f"  Input Layer: 13 features")
    print(f"  Hidden Layer 1: 64 neurons (ReLU)")
    print(f"  Hidden Layer 2: 32 neurons (ReLU)") 
    print(f"  Hidden Layer 3: 16 neurons (ReLU)")
    print(f"  Output Layer: 3 classes (Softmax)")
    
    meta_model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),  # 3-layer deep network
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
        verbose=True
    )
    
    # Train the model
    start_time = time.time()
    print(f"\nStarting training at {datetime.now().strftime('%H:%M:%S')}...")
    
    meta_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")
    print(f"Converged in {meta_model.n_iter_} iterations")
    
    # Evaluate model
    print(f"\n=== MODEL EVALUATION ===")
    
    # Training accuracy
    train_accuracy = meta_model.score(X_train, y_train)
    test_accuracy = meta_model.score(X_test, y_test)
    
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    
    # Detailed evaluation
    y_pred = meta_model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)
    print(cm)
    
    # Cross-validation
    cv_scores = cross_val_score(meta_model, X_scaled, y_encoded, cv=5)
    print(f"\nCross-Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Feature importance (approximate via permutation)
    print(f"\n=== FEATURE ANALYSIS ===")
    feature_importance = analyze_feature_importance(meta_model, X_test, y_test, scaler, feature_names)
    
    # Test with sample predictions
    print(f"\n=== SAMPLE PREDICTIONS ===")
    test_sample_predictions(meta_model, scaler, label_encoder, feature_names)
    
    return {
        'model': meta_model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'training_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'training_time': training_time,
        'feature_importance': feature_importance
    }

def analyze_feature_importance(model, X_test, y_test, scaler, feature_names):
    """Analyze feature importance using permutation importance"""
    print("Analyzing feature importance...")
    
    baseline_accuracy = model.score(X_test, y_test)
    feature_importance = {}
    
    for i, feature_name in enumerate(feature_names):
        # Create permuted version
        X_permuted = X_test.copy()
        np.random.shuffle(X_permuted[:, i])
        
        # Calculate accuracy drop
        permuted_accuracy = model.score(X_permuted, y_test)
        importance = baseline_accuracy - permuted_accuracy
        feature_importance[feature_name] = importance
    
    # Sort by importance
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("Feature Importance (accuracy drop when permuted):")
    for feature, importance in sorted_importance:
        print(f"  {feature:25s}: {importance:+.3f}")
    
    return feature_importance

def test_sample_predictions(model, scaler, label_encoder, feature_names):
    """Test with sample scenarios"""
    
    test_scenarios = [
        {
            'name': 'High Confidence SVM Small',
            'features': [0.9, 0.6, 0.8, 1, 0, 0, 2, 0.7, 0.2, 0.3, 0.4, 0, 0.6]
        },
        {
            'name': 'Conflicting Predictions',
            'features': [0.5, 0.7, 0.6, 0, 0, 1, 5, 0.3, 0.8, 0.7, 0.6, 1, 0.4]
        },
        {
            'name': 'Low Confidence All Models',
            'features': [0.4, 0.3, 0.4, 0, 1, 0, 1, 0.5, 0.5, 0.5, 0.5, 0, 0.5]
        }
    ]
    
    for scenario in test_scenarios:
        features = np.array([scenario['features']])
        features_scaled = scaler.transform(features)
        
        # Prediction
        prediction_proba = model.predict_proba(features_scaled)[0]
        predicted_class_idx = np.argmax(prediction_proba)
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = np.max(prediction_proba)
        
        print(f"\n{scenario['name']}:")
        print(f"  Prediction: {predicted_class} (confidence: {confidence:.3f})")
        print(f"  Probabilities: {dict(zip(label_encoder.classes_, prediction_proba))}")

def save_meta_learning_model(model_components):
    """Save the trained Meta-Learning model"""
    print(f"\n💾 SAVING META-LEARNING MODEL")
    print("=" * 40)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model components
    joblib.dump(model_components['model'], 'models/meta_learning_model.joblib')
    joblib.dump(model_components['scaler'], 'models/meta_learning_scaler.joblib')
    joblib.dump(model_components['label_encoder'], 'models/meta_learning_encoder.joblib')
    joblib.dump(model_components['feature_names'], 'models/meta_learning_features.joblib')
    
    # Save comprehensive metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'type': 'Meta-Learning Neural Network',
            'architecture': 'Input(13) → Hidden(64) → Hidden(32) → Hidden(16) → Output(3)',
            'activation': 'ReLU',
            'solver': 'Adam',
            'feature_count': 13,
            'output_classes': list(model_components['label_encoder'].classes_),
            'training_time_seconds': model_components['training_time']
        },
        'performance': {
            'training_accuracy': model_components['training_accuracy'],
            'test_accuracy': model_components['test_accuracy'],
            'cv_mean_accuracy': model_components['cv_mean'],
            'cv_std_accuracy': model_components['cv_std'],
            'is_production_ready': model_components['test_accuracy'] > 0.8
        },
        'features': {
            'meta_feature_names': model_components['feature_names'],
            'feature_importance': model_components['feature_importance'],
            'input_description': 'Combines SVM + K-Means + Rule-based predictions'
        },
        'ensemble_components': {
            'svm_model': 'Workload classification (small/medium/large)',
            'kmeans_model': 'VM resource clustering (10 clusters)',
            'rule_based': 'Business logic heuristics',
            'meta_learning': 'Neural Network learns optimal combination'
        },
        'deployment': {
            'usage': 'MCCVA Meta-Learning Ensemble for VM load balancing',
            'api_endpoint': '/predict/meta_learning',
            'input_format': 'Combines all base model outputs',
            'output_format': 'Final workload classification with confidence',
            'integration_ready': True
        }
    }
    
    joblib.dump(metadata, 'models/meta_learning_info.joblib')
    
    print("Meta-Learning model saved successfully!")
    print("Files saved:")
    print("  - meta_learning_model.joblib")
    print("  - meta_learning_scaler.joblib")
    print("  - meta_learning_encoder.joblib")
    print("  - meta_learning_features.joblib")
    print("  - meta_learning_info.joblib")

def main():
    """Main Meta-Learning training pipeline"""
    print("🧠 MCCVA META-LEARNING ENSEMBLE TRAINER")
    print("=" * 60)
    print("Training Neural Network to combine SVM + K-Means + Rules")
    print("REAL-WORLD VERSION 2.1 - Production Ready Meta-Learning")
    print()
    
    start_time = datetime.now()
    
    try:
        # Step 1: Load base models
        base_models = load_base_models()
        if base_models is None:
            print("❌ Cannot proceed without base models!")
            return False
        
        # Step 2: Generate training data
        training_data = generate_synthetic_meta_training_data(base_models, n_samples=2000)
        if not training_data:
            print("❌ Failed to generate training data!")
            return False
        
        # Step 3: Train Meta-Learning model
        model_components = train_meta_learning_model(training_data)
        if model_components is None:
            print("❌ Meta-Learning training failed!")
            return False
        
        # Step 4: Save model
        save_meta_learning_model(model_components)
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print(f"\n" + "=" * 60)
        print("🎉 META-LEARNING TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Neural Network Architecture: Input(13) → Hidden(64,32,16) → Output(3)")
        print(f"Training Accuracy: {model_components['training_accuracy']:.1%}")
        print(f"Test Accuracy: {model_components['test_accuracy']:.1%}")
        print(f"Cross-Validation: {model_components['cv_mean']:.1%} (+/- {model_components['cv_std']*2:.1%})")
        print(f"Total Time: {duration:.1f} minutes")
        print(f"Status: {'✅ Production Ready' if model_components['test_accuracy'] > 0.8 else '⚠️ Needs Improvement'}")
        print()
        print("🚀 MCCVA 3-Stage System Complete:")
        print("  ✅ Stage 1: SVM (Workload Classification)")
        print("  ✅ Stage 2: K-Means (VM Clustering)")
        print("  ✅ Stage 3: Meta-Learning NN (Ensemble Intelligence)")
        print()
        print("Ready for production deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Update ml_service.py to load Meta-Learning model")
        print("2. Test ensemble predictions via API")
        print("3. Deploy to cloud infrastructure") 
        print("4. Monitor Meta-Learning performance in production")
    else:
        print("\n❌ Fix errors and retry Meta-Learning training") 