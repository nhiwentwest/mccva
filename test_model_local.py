#!/usr/bin/env python3
"""
Local Model Testing Script
Test và cải thiện AI model trước khi deploy lên cloud
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

class LocalModelTester:
    def __init__(self):
        self.svm_model = None
        self.svm_scaler = None
        self.kmeans_model = None
        self.kmeans_scaler = None
        self.test_results = {}
        
    def load_current_models(self):
        """Load models hiện tại"""
        print("Loading current models...")
        try:
            self.svm_model = joblib.load("models/svm_model.joblib")
            self.svm_scaler = joblib.load("models/scaler.joblib")
            self.kmeans_model = joblib.load("models/kmeans_model.joblib")
            self.kmeans_scaler = joblib.load("models/kmeans_scaler.joblib")
            print("✅ Current models loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def test_current_model_bias(self):
        """Test bias của model hiện tại"""
        print("\n" + "="*50)
        print("Testing Current Model Bias")
        print("="*50)
        
        test_cases = [
            # Small tasks
            {"name": "Small Task 1", "features": [2, 4, 50, 500, 1], "expected": "small"},
            {"name": "Small Task 2", "features": [1, 2, 30, 300, 1], "expected": "small"},
            
            # Medium tasks
            {"name": "Medium Task 1", "features": [4, 8, 100, 1000, 3], "expected": "medium"},
            {"name": "Medium Task 2", "features": [6, 12, 150, 1200, 3], "expected": "medium"},
            
            # Large tasks
            {"name": "Large Task 1", "features": [8, 16, 200, 2000, 5], "expected": "large"},
            {"name": "Large Task 2", "features": [12, 32, 500, 5000, 5], "expected": "large"},
            {"name": "Large Task 3", "features": [16, 64, 800, 8000, 4], "expected": "large"},
        ]
        
        correct = 0
        total = len(test_cases)
        
        for case in test_cases:
            try:
                features_scaled = self.svm_scaler.transform([case["features"]])
                prediction = self.svm_model.predict(features_scaled)[0]
                confidence = abs(self.svm_model.decision_function(features_scaled)[0])
                
                is_correct = prediction == case["expected"]
                if is_correct:
                    correct += 1
                
                status = "✅" if is_correct else "❌"
                print(f"{status} {case['name']}:")
                print(f"   Features: {case['features']}")
                print(f"   Expected: {case['expected']}")
                print(f"   Predicted: {prediction}")
                print(f"   Confidence: {confidence:.3f}")
                print()
                
            except Exception as e:
                print(f"❌ Error testing {case['name']}: {e}")
        
        accuracy = (correct / total) * 100
        print(f"Current Model Accuracy: {accuracy:.1f}% ({correct}/{total})")
        
        self.test_results["current_accuracy"] = accuracy
        return accuracy
    
    def generate_balanced_dataset(self, samples_per_class=200):
        """Tạo balanced dataset"""
        print("\n" + "="*50)
        print("Generating Balanced Dataset")
        print("="*50)
        
        np.random.seed(42)
        
        # Small tasks (low resource)
        small_data = []
        for _ in range(samples_per_class):
            small_data.append([
                np.random.randint(1, 4),      # cpu_cores: 1-3
                np.random.randint(1, 6),      # memory: 1-5 GB
                np.random.randint(10, 80),    # storage: 10-79 GB
                np.random.randint(100, 800),  # network: 100-799 Mbps
                np.random.randint(1, 3),      # priority: 1-2
                'small'
            ])
        
        # Medium tasks (balanced)
        medium_data = []
        for _ in range(samples_per_class):
            medium_data.append([
                np.random.randint(3, 7),      # cpu_cores: 3-6
                np.random.randint(5, 12),     # memory: 5-11 GB
                np.random.randint(70, 150),   # storage: 70-149 GB
                np.random.randint(700, 1500), # network: 700-1499 Mbps
                np.random.randint(2, 4),      # priority: 2-3
                'medium'
            ])
        
        # Large tasks (high resource)
        large_data = []
        for _ in range(samples_per_class):
            large_data.append([
                np.random.randint(6, 17),     # cpu_cores: 6-16
                np.random.randint(10, 65),    # memory: 10-64 GB
                np.random.randint(140, 1001), # storage: 140-1000 GB
                np.random.randint(1400, 10001), # network: 1400-10000 Mbps
                np.random.randint(3, 6),      # priority: 3-5
                'large'
            ])
        
        # Combine all data
        all_data = small_data + medium_data + large_data
        df = pd.DataFrame(all_data, columns=['cpu_cores', 'memory', 'storage', 'network_bandwidth', 'priority', 'makespan'])
        
        print("Balanced dataset created:")
        print(df['makespan'].value_counts())
        print("\nFeature ranges:")
        print(df.describe())
        
        # Save dataset
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/balanced_training_data.csv', index=False)
        print("\nBalanced dataset saved to data/balanced_training_data.csv")
        
        return df
    
    def train_improved_model(self, df):
        """Train model cải thiện"""
        print("\n" + "="*50)
        print("Training Improved Model")
        print("="*50)
        
        X = df.drop('makespan', axis=1)
        y = df['makespan']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM with optimized parameters
        svm = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        print("Training SVM model...")
        svm.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = svm.score(X_train_scaled, y_train)
        test_score = svm.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
        
        print(f"\nModel Performance:")
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Testing accuracy: {test_score:.3f}")
        print(f"Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Detailed classification report
        y_pred = svm.predict(X_test_scaled)
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save improved model
        joblib.dump(svm, 'models/svm_model_improved.joblib')
        joblib.dump(scaler, 'models/scaler_improved.joblib')
        
        print("\nImproved models saved!")
        
        self.test_results["improved_accuracy"] = test_score
        self.test_results["cv_score"] = cv_scores.mean()
        
        return svm, scaler
    
    def test_improved_model(self, svm, scaler):
        """Test model cải thiện"""
        print("\n" + "="*50)
        print("Testing Improved Model")
        print("="*50)
        
        test_cases = [
            # Small tasks
            {"name": "Small Task 1", "features": [2, 4, 50, 500, 1], "expected": "small"},
            {"name": "Small Task 2", "features": [1, 2, 30, 300, 1], "expected": "small"},
            
            # Medium tasks
            {"name": "Medium Task 1", "features": [4, 8, 100, 1000, 3], "expected": "medium"},
            {"name": "Medium Task 2", "features": [6, 12, 150, 1200, 3], "expected": "medium"},
            
            # Large tasks
            {"name": "Large Task 1", "features": [8, 16, 200, 2000, 5], "expected": "large"},
            {"name": "Large Task 2", "features": [12, 32, 500, 5000, 5], "expected": "large"},
            {"name": "Large Task 3", "features": [16, 64, 800, 8000, 4], "expected": "large"},
        ]
        
        correct = 0
        total = len(test_cases)
        
        for case in test_cases:
            try:
                features_scaled = scaler.transform([case["features"]])
                prediction = svm.predict(features_scaled)[0]
                confidence = abs(svm.decision_function(features_scaled)[0])
                
                is_correct = prediction == case["expected"]
                if is_correct:
                    correct += 1
                
                status = "✅" if is_correct else "❌"
                print(f"{status} {case['name']}:")
                print(f"   Features: {case['features']}")
                print(f"   Expected: {case['expected']}")
                print(f"   Predicted: {prediction}")
                print(f"   Confidence: {confidence:.3f}")
                print()
                
            except Exception as e:
                print(f"❌ Error testing {case['name']}: {e}")
        
        accuracy = (correct / total) * 100
        print(f"Improved Model Accuracy: {accuracy:.1f}% ({correct}/{total})")
        
        self.test_results["improved_test_accuracy"] = accuracy
        return accuracy
    
    def generate_deployment_script(self):
        """Tạo script deploy model mới"""
        print("\n" + "="*50)
        print("Generating Deployment Script")
        print("="*50)
        
        deploy_script = """#!/bin/bash
# Deploy improved models to cloud

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
    
    def run_complete_test(self):
        """Chạy test hoàn chỉnh"""
        print("🚀 Starting Local Model Testing")
        print("="*60)
        
        # Load current models
        if not self.load_current_models():
            return False
        
        # Test current model bias
        current_accuracy = self.test_current_model_bias()
        
        # Generate balanced dataset
        df = self.generate_balanced_dataset()
        
        # Train improved model
        svm, scaler = self.train_improved_model(df)
        
        # Test improved model
        improved_accuracy = self.test_improved_model(svm, scaler)
        
        # Generate deployment script
        self.generate_deployment_script()
        
        # Summary
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        print(f"Current Model Accuracy: {current_accuracy:.1f}%")
        print(f"Improved Model Accuracy: {improved_accuracy:.1f}%")
        print(f"Improvement: {improved_accuracy - current_accuracy:.1f}%")
        
        if improved_accuracy > 70:
            print("✅ Model ready for deployment!")
        else:
            print("⚠️ Model needs further improvement")
        
        # Save results
        with open('test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\nResults saved to test_results.json")
        return True

def main():
    tester = LocalModelTester()
    tester.run_complete_test()

if __name__ == "__main__":
    main() 