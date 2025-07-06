#!/usr/bin/env python3
"""
Enhanced SVM Training Script với data tốt hơn
- Phân bố đều 3 class: small, medium, large
- Sử dụng dataset từ thư mục dataset/
- Feature engineering nâng cao
- Cross-validation và hyperparameter tuning
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedSVMTrainer:
    def __init__(self):
        self.svm_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.training_history = {}
        
    def load_and_preprocess_data(self):
        """Load và preprocess data từ dataset/"""
        print("📊 Loading and preprocessing data...")
        
        all_data = []
        
        # Load tất cả file Excel trong dataset/
        dataset_files = [f for f in os.listdir('dataset') if f.endswith('.xlsx')]
        
        for file in dataset_files:
            try:
                df = pd.read_excel(f'dataset/{file}')
                print(f"  Loaded {file}: {df.shape}")
                
                # Xử lý các format khác nhau
                if len(df.columns) == 10:  # Format 1: 10 features
                    # Rename columns để chuẩn hóa
                    df.columns = ['jobs_1min', 'jobs_5min', 'jobs_15min', 'memory_gb', 
                                 'storage_gb', 'cpu_cores', 'cpu_speed', 'network_in', 
                                 'network_out', 'class_name']
                    
                    # Clean class names
                    df['class_name'] = df['class_name'].str.strip().str.replace("'", "")
                    
                elif len(df.columns) == 3:  # Format 2: 3 features
                    # Rename columns
                    df.columns = ['cpu_util', 'response_time', 'class_name']
                    df['class_name'] = df['class_name'].str.strip().str.replace("'", "")
                    
                    # Add synthetic features để match format 5 features
                    df['memory_gb'] = np.random.uniform(1, 64, len(df))
                    df['storage_gb'] = np.random.uniform(10, 1000, len(df))
                    df['network_bandwidth'] = np.random.uniform(100, 10000, len(df))
                    df['priority'] = np.random.randint(1, 6, len(df))
                    
                    # Reorder columns
                    df = df[['cpu_util', 'memory_gb', 'storage_gb', 'network_bandwidth', 'priority', 'class_name']]
                    df.columns = ['cpu_cores', 'memory_gb', 'storage_gb', 'network_bandwidth', 'priority', 'class_name']
                
                all_data.append(df)
                
            except Exception as e:
                print(f"  Warning: Could not load {file}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid data files found in dataset/")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"  Combined data shape: {combined_df.shape}")
        
        # Clean class names và map to 3 classes
        combined_df['class_name'] = combined_df['class_name'].str.strip().str.replace("'", "")
        
        # Map classes to 3 categories
        class_mapping = {
            'Very Low': 'small',
            'Low': 'small', 
            'Medium': 'medium',
            'High': 'large',
            'Very High': 'large'
        }
        
        combined_df['class_name'] = combined_df['class_name'].map(class_mapping)
        
        # Remove rows with unknown classes
        combined_df = combined_df.dropna(subset=['class_name'])
        
        print(f"  Final data shape: {combined_df.shape}")
        print(f"  Class distribution:")
        print(combined_df['class_name'].value_counts())
        
        return combined_df
    
    def balance_classes(self, df, target_col='class_name'):
        """Balance classes để có phân bố đều"""
        print("⚖️ Balancing classes...")
        
        # Get class counts
        class_counts = df[target_col].value_counts()
        min_count = class_counts.min()
        
        balanced_data = []
        
        for class_name in df[target_col].unique():
            class_data = df[df[target_col] == class_name]
            
            if len(class_data) > min_count:
                # Downsample majority class
                balanced_data.append(class_data.sample(n=min_count, random_state=42))
            else:
                # Upsample minority class
                balanced_data.append(class_data.sample(n=min_count, replace=True, random_state=42))
        
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        
        print(f"  Balanced data shape: {balanced_df.shape}")
        print(f"  Balanced class distribution:")
        print(balanced_df[target_col].value_counts())
        
        return balanced_df
    
    def feature_engineering(self, df):
        """Feature engineering nâng cao"""
        print("🔧 Feature engineering...")
        
        # Select relevant features
        feature_cols = ['cpu_cores', 'memory_gb', 'storage_gb', 'network_bandwidth', 'priority']
        
        # Ensure all features exist
        for col in feature_cols:
            if col not in df.columns:
                print(f"  Warning: {col} not found, using default values")
                df[col] = np.random.uniform(1, 100, len(df))
        
        # Create enhanced features
        df['cpu_memory_ratio'] = df['cpu_cores'] / (df['memory_gb'] + 1e-6)
        df['storage_memory_ratio'] = df['storage_gb'] / (df['memory_gb'] + 1e-6)
        df['network_cpu_ratio'] = df['network_bandwidth'] / (df['cpu_cores'] + 1e-6)
        df['resource_intensity'] = (df['cpu_cores'] * df['memory_gb'] * df['storage_gb']) / 1000
        df['priority_weighted_cpu'] = df['cpu_cores'] * df['priority']
        
        # Update feature columns
        enhanced_features = feature_cols + [
            'cpu_memory_ratio', 'storage_memory_ratio', 'network_cpu_ratio',
            'resource_intensity', 'priority_weighted_cpu'
        ]
        
        self.feature_names = enhanced_features
        print(f"  Enhanced features: {len(enhanced_features)} features")
        
        return df, enhanced_features
    
    def prepare_training_data(self, df, feature_cols, target_col='class_name'):
        """Prepare data for training"""
        print("📋 Preparing training data...")
        
        # Encode target labels
        y = self.label_encoder.fit_transform(df[target_col])
        
        # Select features
        X = df[feature_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"  Training set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")
        print(f"  Feature names: {feature_cols}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_svm_model(self, X_train, y_train):
        """Train SVM model với hyperparameter tuning"""
        print("🤖 Training SVM model...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        # Grid search với cross-validation
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
        
        self.svm_model = grid_search.best_estimator_
        
        return grid_search.best_score_
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("📈 Evaluating model...")
        
        # Predictions
        y_pred = self.svm_model.predict(X_test)
        y_pred_proba = self.svm_model.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  Test Accuracy: {accuracy:.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n  Confusion Matrix:")
        print(cm)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.svm_model, X_test, y_test, cv=5)
        print(f"\n  Cross-validation scores: {cv_scores}")
        print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return accuracy, cm
    
    def save_model(self, model_path='models/'):
        """Save trained model và scaler"""
        print("💾 Saving model...")
        
        # Create models directory if not exists
        os.makedirs(model_path, exist_ok=True)
        
        # Save SVM model
        joblib.dump(self.svm_model, f'{model_path}/svm_model.joblib')
        
        # Save scaler
        joblib.dump(self.scaler, f'{model_path}/scaler.joblib')
        
        # Save label encoder
        joblib.dump(self.label_encoder, f'{model_path}/label_encoder.joblib')
        
        # Save feature names
        joblib.dump(self.feature_names, f'{model_path}/feature_names.joblib')
        
        # Save training info
        training_info = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'SVM',
            'kernel': self.svm_model.kernel,
            'C': self.svm_model.C,
            'gamma': self.svm_model.gamma,
            'n_support_vectors': sum(self.svm_model.n_support_),
            'classes': self.label_encoder.classes_.tolist(),
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        
        joblib.dump(training_info, f'{model_path}/training_info.joblib')
        
        print(f"  Model saved to {model_path}")
        print(f"  Files saved:")
        print(f"    - svm_model.joblib")
        print(f"    - scaler.joblib") 
        print(f"    - label_encoder.joblib")
        print(f"    - feature_names.joblib")
        print(f"    - training_info.joblib")
    
    def plot_results(self, X_test, y_test, save_path='training_results/'):
        """Plot training results"""
        print("📊 Creating visualizations...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Predictions
        y_pred = self.svm_model.predict(X_test)
        y_pred_proba = self.svm_model.predict_proba(X_test)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance (using Random Forest as proxy)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_test, y_test)
        
        plt.figure(figsize=(10, 6))
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance (Random Forest Proxy)')
        plt.tight_layout()
        plt.savefig(f'{save_path}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Plots saved to {save_path}")
    
    def test_model(self):
        """Test model với sample data"""
        print("🧪 Testing model...")
        
        # Sample test cases
        test_cases = [
            {
                "name": "Web Server (Small)",
                "features": [2, 4, 50, 500, 1, 0.5, 12.5, 250, 50, 2],
                "expected": "small"
            },
            {
                "name": "Database Server (Medium)", 
                "features": [4, 8, 100, 1000, 3, 0.5, 12.5, 250, 32, 12],
                "expected": "medium"
            },
            {
                "name": "ML Training (Large)",
                "features": [12, 32, 500, 5000, 5, 0.375, 15.625, 416.67, 19200, 60],
                "expected": "large"
            }
        ]
        
        correct = 0
        total = len(test_cases)
        
        for case in test_cases:
            # Scale features
            features_scaled = self.scaler.transform([case["features"]])
            
            # Predict
            prediction = self.svm_model.predict(features_scaled)[0]
            prediction_class = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get confidence
            proba = self.svm_model.predict_proba(features_scaled)[0]
            confidence = np.max(proba)
            
            # Check accuracy
            is_correct = prediction_class == case["expected"]
            if is_correct:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"{status} {case['name']}")
            print(f"   Expected: {case['expected']}")
            print(f"   Predicted: {prediction_class}")
            print(f"   Confidence: {confidence:.3f}")
            print()
        
        accuracy = (correct / total) * 100
        print(f"📊 Test Accuracy: {accuracy:.1f}% ({correct}/{total})")
        
        return accuracy
    
    def run_training_pipeline(self):
        """Run complete training pipeline"""
        print("🚀 Starting Enhanced SVM Training Pipeline")
        print("=" * 60)
        
        try:
            # 1. Load and preprocess data
            df = self.load_and_preprocess_data()
            
            # 2. Balance classes
            balanced_df = self.balance_classes(df)
            
            # 3. Feature engineering
            enhanced_df, feature_cols = self.feature_engineering(balanced_df)
            
            # 4. Prepare training data
            X_train, X_test, y_train, y_test = self.prepare_training_data(
                enhanced_df, feature_cols
            )
            
            # 5. Train model
            cv_score = self.train_svm_model(X_train, y_train)
            
            # 6. Evaluate model
            test_accuracy, confusion_mat = self.evaluate_model(X_test, y_test)
            
            # 7. Save model
            self.save_model()
            
            # 8. Create visualizations
            self.plot_results(X_test, y_test)
            
            # 9. Test model
            test_accuracy = self.test_model()
            
            # Summary
            print("\n" + "=" * 60)
            print("🎉 Training Pipeline Completed Successfully!")
            print("=" * 60)
            print(f"📊 Final Results:")
            print(f"   - Cross-validation score: {cv_score:.4f}")
            print(f"   - Test accuracy: {test_accuracy:.4f}")
            print(f"   - Model saved to: models/")
            print(f"   - Plots saved to: training_results/")
            print(f"   - Classes: {self.label_encoder.classes_.tolist()}")
            print(f"   - Features: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    trainer = EnhancedSVMTrainer()
    success = trainer.run_training_pipeline()
    
    if success:
        print("\n✅ Training completed successfully!")
        print("🚀 Model is ready for deployment!")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 