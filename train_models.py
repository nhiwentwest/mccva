#!/usr/bin/env python3
"""
Model Training Script
Tận dụng enhanced features và ensemble system có sẵn trong project
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.training_results = {}
        
    def generate_realistic_dataset(self, samples_per_class=300):
        """Tạo dataset thực tế với enhanced features"""
        print("Generating realistic dataset with enhanced features...")
        
        np.random.seed(42)
        data = []
        
        # Small tasks (low resource, high frequency)
        for _ in range(samples_per_class):
            cpu = np.random.randint(1, 4)
            memory = np.random.randint(1, 6)
            storage = np.random.randint(10, 80)
            network = np.random.randint(100, 800)
            priority = np.random.choice([1, 2], p=[0.7, 0.3])  # Mostly low priority
            
            data.append([cpu, memory, storage, network, priority, 'small'])
        
        # Medium tasks (balanced, moderate frequency)
        for _ in range(samples_per_class):
            cpu = np.random.randint(3, 7)
            memory = np.random.randint(5, 12)
            storage = np.random.randint(70, 150)
            network = np.random.randint(700, 1500)
            priority = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])  # Balanced priority
            
            data.append([cpu, memory, storage, network, priority, 'medium'])
        
        # Large tasks (high resource, low frequency)
        for _ in range(samples_per_class):
            cpu = np.random.randint(6, 17)
            memory = np.random.randint(10, 65)
            storage = np.random.randint(140, 1001)
            network = np.random.randint(1400, 10001)
            priority = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])  # Mostly high priority
            
            data.append([cpu, memory, storage, network, priority, 'large'])
        
        df = pd.DataFrame(data, columns=['cpu_cores', 'memory', 'storage', 'network_bandwidth', 'priority', 'makespan'])
        
        print("Dataset distribution:")
        print(df['makespan'].value_counts())
        print("\nFeature statistics:")
        print(df.describe())
        
        # Save dataset
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/realistic_training_data.csv', index=False)
        
        return df
    
    def extract_enhanced_features(self, df):
        """Extract enhanced features như trong ml_service.py"""
        enhanced_data = []
        
        for _, row in df.iterrows():
            cpu, memory, storage, network, priority = row[['cpu_cores', 'memory', 'storage', 'network_bandwidth', 'priority']]
            
            enhanced_features = [
                cpu, memory, storage, network, priority,  # Original features
                cpu / memory if memory > 0 else 0,  # compute_intensity
                storage / memory if memory > 0 else 0,  # storage_intensity
                network / cpu if cpu > 0 else 0,  # network_intensity
                (cpu * memory) / storage if storage > 0 else 0,  # resource_ratio
                priority / 5.0,  # priority_weight
                int(cpu / memory > 0.5 if memory > 0 else 0),  # is_compute_intensive
                int(memory > 16),  # is_memory_intensive
                int(storage > 500),  # is_storage_intensive
                int(network > 5000),  # is_network_intensive
                int(priority >= 4),  # high_priority
                int(priority <= 2),  # low_priority
                int(abs(cpu - memory/4) < 2),  # balanced_resources
                int(storage > (cpu * memory * 2)),  # storage_heavy
                int(network > (cpu * 1000))  # network_heavy
            ]
            
            enhanced_data.append(enhanced_features)
        
        enhanced_df = pd.DataFrame(enhanced_data, columns=[
            'cpu_cores', 'memory', 'storage', 'network_bandwidth', 'priority',
            'compute_intensity', 'storage_intensity', 'network_intensity', 'resource_ratio', 'priority_weight',
            'is_compute_intensive', 'is_memory_intensive', 'is_storage_intensive', 'is_network_intensive',
            'high_priority', 'low_priority', 'balanced_resources', 'storage_heavy', 'network_heavy'
        ])
        
        enhanced_df['makespan'] = df['makespan']
        return enhanced_df
    
    def train_svm_model(self, X_train, y_train, X_test, y_test):
        """Train SVM model với hyperparameter tuning"""
        print("Training SVM model with hyperparameter tuning...")
        
        # Grid search for best parameters
        param_grid = {
            'C': [1, 5, 10, 20],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'poly']
        }
        
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_svm = grid_search.best_estimator_
        train_score = best_svm.score(X_train, y_train)
        test_score = best_svm.score(X_test, y_test)
        
        print(f"Best SVM parameters: {grid_search.best_params_}")
        print(f"SVM Training accuracy: {train_score:.3f}")
        print(f"SVM Testing accuracy: {test_score:.3f}")
        
        return best_svm, test_score
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        rf.fit(X_train, y_train)
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        
        print(f"RF Training accuracy: {train_score:.3f}")
        print(f"RF Testing accuracy: {test_score:.3f}")
        
        return rf, test_score
    
    def train_ensemble_model(self, X_train, y_train, X_test, y_test, svm_model, rf_model):
        """Train ensemble model kết hợp SVM và Random Forest"""
        print("Training ensemble model...")
        
        ensemble = VotingClassifier(
            estimators=[
                ('svm', svm_model),
                ('rf', rf_model)
            ],
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        train_score = ensemble.score(X_train, y_train)
        test_score = ensemble.score(X_test, y_test)
        
        print(f"Ensemble Training accuracy: {train_score:.3f}")
        print(f"Ensemble Testing accuracy: {test_score:.3f}")
        
        return ensemble, test_score
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate tất cả models"""
        print("\n" + "="*50)
        print("Model Evaluation")
        print("="*50)
        
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            accuracy = model.score(X_test, y_test)
            
            print(f"\n{name.upper()} Model:")
            print(f"Accuracy: {accuracy:.3f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred.tolist()
            }
        
        return results
    
    def save_models(self, models, scalers):
        """Save models và scalers"""
        print("\nSaving models...")
        
        # Save SVM model (primary)
        joblib.dump(models['svm'], 'models/svm_model_improved.joblib')
        joblib.dump(scalers['standard'], 'models/scaler_improved.joblib')
        
        # Save ensemble model
        joblib.dump(models['ensemble'], 'models/ensemble_model.joblib')
        
        # Save Random Forest
        joblib.dump(models['rf'], 'models/rf_model.joblib')
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'dataset_size': len(self.training_results.get('dataset', [])),
            'feature_count': len(self.training_results.get('features', [])),
            'model_performances': self.training_results.get('performances', {}),
            'best_model': max(self.training_results.get('performances', {}).items(), key=lambda x: x[1])[0] if self.training_results.get('performances') else 'svm'
        }
        
        with open('models/training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Models saved successfully!")
    
    def create_deployment_script(self):
        """Tạo script deploy models mới"""
        deploy_script = """#!/bin/bash
# Deploy improved models

echo "Deploying improved models..."

# Backup current models
cp models/svm_model.joblib models/svm_model_backup.joblib
cp models/scaler.joblib models/scaler_backup.joblib

# Use improved models
cp models/svm_model_improved.joblib models/svm_model.joblib
cp models/scaler_improved.joblib models/scaler.joblib

# Restart ML Service
docker restart mccva-ml

echo "Models deployed successfully!"
echo "Test with: python3 test_ai_routing_host.py"
"""
        
        with open('deploy_improved_models.sh', 'w') as f:
            f.write(deploy_script)
        
        os.chmod('deploy_improved_models.sh', 0o755)
        print("Deployment script created: deploy_improved_models.sh")
    
    def run_training_pipeline(self):
        """Chạy toàn bộ training pipeline"""
        print("🚀 Starting Model Training Pipeline")
        print("="*60)
        
        # Step 1: Generate realistic dataset
        df = self.generate_realistic_dataset()
        
        # Step 2: Extract enhanced features
        enhanced_df = self.extract_enhanced_features(df)
        
        # Step 3: Prepare data
        X = enhanced_df.drop('makespan', axis=1)
        y = enhanced_df['makespan']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Step 4: Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Step 5: Train individual models
        svm_model, svm_score = self.train_svm_model(X_train_scaled, y_train, X_test_scaled, y_test)
        rf_model, rf_score = self.train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Step 6: Train ensemble model
        ensemble_model, ensemble_score = self.train_ensemble_model(
            X_train_scaled, y_train, X_test_scaled, y_test, svm_model, rf_model
        )
        
        # Step 7: Store models
        self.models = {
            'svm': svm_model,
            'rf': rf_model,
            'ensemble': ensemble_model
        }
        
        # Step 8: Evaluate all models
        performances = self.evaluate_models(self.models, X_test_scaled, y_test)
        
        # Step 9: Save results
        self.training_results = {
            'dataset': df.to_dict('records'),
            'features': X.columns.tolist(),
            'performances': {name: perf['accuracy'] for name, perf in performances.items()},
            'best_model': max(performances.items(), key=lambda x: x[1]['accuracy'])[0]
        }
        
        # Step 10: Save models
        self.save_models(self.models, self.scalers)
        
        # Step 11: Create deployment script
        self.create_deployment_script()
        
        # Summary
        print("\n" + "="*60)
        print("📊 TRAINING SUMMARY")
        print("="*60)
        print(f"Dataset size: {len(df)} samples")
        print(f"Features: {len(X.columns)} (including enhanced features)")
        print(f"Best model: {self.training_results['best_model']}")
        print(f"Best accuracy: {self.training_results['performances'][self.training_results['best_model']]:.3f}")
        
        # Save training results
        with open('training_results.json', 'w') as f:
            json.dump(self.training_results, f, indent=2)
        
        print(f"\nTraining results saved to training_results.json")
        print("✅ Training pipeline completed successfully!")
        
        return True

def main():
    trainer = ModelTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main() 