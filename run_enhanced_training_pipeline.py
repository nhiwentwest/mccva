#!/usr/bin/env python3
"""
Enhanced Training Pipeline - Complete Workflow
1. Train SVM model với data tốt hơn
2. Phân bố đều 3 class
3. Feature engineering nâng cao
4. Deploy lên cloud
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(cmd, description, check=True):
    """Run command với error handling"""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print("Output:")
            print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Errors:")
            print(e.stderr)
        return False

def check_dependencies():
    """Check required dependencies"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'joblib', 
        'matplotlib', 'seaborn', 'openpyxl'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        print("Installing missing packages...")
        
        install_cmd = f"pip3 install {' '.join(missing_packages)}"
        if not run_command(install_cmd, "Installing missing packages"):
            print("❌ Failed to install packages")
            return False
    
    return True

def check_dataset():
    """Check dataset availability"""
    print("📊 Checking dataset...")
    
    if not os.path.exists('dataset'):
        print("❌ Dataset directory not found")
        return False
    
    dataset_files = [f for f in os.listdir('dataset') if f.endswith('.xlsx')]
    
    if not dataset_files:
        print("❌ No Excel files found in dataset/")
        return False
    
    print(f"✅ Found {len(dataset_files)} dataset files:")
    for file in dataset_files:
        print(f"  - {file}")
    
    return True

def run_training():
    """Run enhanced training"""
    print("\n🚀 Starting Enhanced Training...")
    
    # Check if training script exists
    if not os.path.exists('train_enhanced_svm.py'):
        print("❌ train_enhanced_svm.py not found")
        return False
    
    # Run training
    training_cmd = "python3 train_enhanced_svm.py"
    if not run_command(training_cmd, "Running enhanced SVM training"):
        print("❌ Training failed")
        return False
    
    # Check if models were created
    required_models = [
        'models/svm_model.joblib',
        'models/scaler.joblib',
        'models/label_encoder.joblib',
        'models/feature_names.joblib',
        'models/training_info.joblib'
    ]
    
    for model_file in required_models:
        if not os.path.exists(model_file):
            print(f"❌ Model file not found: {model_file}")
            return False
    
    print("✅ Training completed successfully!")
    return True

def test_trained_model():
    """Test trained model"""
    print("\n🧪 Testing trained model...")
    
    test_cmd = "python3 test_model_local.py"
    if not run_command(test_cmd, "Testing trained model"):
        print("❌ Model testing failed")
        return False
    
    print("✅ Model testing completed!")
    return True

def run_deployment():
    """Run cloud deployment"""
    print("\n☁️ Starting cloud deployment...")
    
    # Check if deployment script exists
    if not os.path.exists('deploy_to_cloud.py'):
        print("❌ deploy_to_cloud.py not found")
        return False
    
    # Check cloud config
    if not os.path.exists('cloud_config.json'):
        print("⚠️ cloud_config.json not found. Creating default config...")
        create_config_cmd = "python3 deploy_to_cloud.py"
        run_command(create_config_cmd, "Creating default cloud config", check=False)
        
        print("\n❌ Please configure cloud_config.json with your server details")
        print("Then run: python3 deploy_to_cloud.py")
        return False
    
    # Run deployment
    deploy_cmd = "python3 deploy_to_cloud.py"
    if not run_command(deploy_cmd, "Deploying to cloud"):
        print("❌ Deployment failed")
        return False
    
    print("✅ Deployment completed successfully!")
    return True

def create_summary_report():
    """Create summary report"""
    print("\n📋 Creating summary report...")
    
    report = f"""
# Enhanced Training Pipeline Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Pipeline Steps Completed:
✅ Dependencies checked
✅ Dataset validated  
✅ Enhanced SVM training completed
✅ Model testing completed
✅ Cloud deployment completed

## Model Information:
"""
    
    # Load training info if available
    if os.path.exists('models/training_info.joblib'):
        try:
            import joblib
            training_info = joblib.load('models/training_info.joblib')
            report += f"""
- Model Type: {training_info.get('model_type', 'Unknown')}
- Kernel: {training_info.get('kernel', 'Unknown')}
- C Parameter: {training_info.get('C', 'Unknown')}
- Gamma: {training_info.get('gamma', 'Unknown')}
- Support Vectors: {training_info.get('n_support_vectors', 'Unknown')}
- Classes: {training_info.get('classes', 'Unknown')}
- Features: {len(training_info.get('feature_names', []))}
- Training Timestamp: {training_info.get('timestamp', 'Unknown')}
"""
        except:
            report += "- Training info not available\n"
    
    report += f"""
## Files Created:
- models/svm_model.joblib
- models/scaler.joblib  
- models/label_encoder.joblib
- models/feature_names.joblib
- models/training_info.joblib
- training_results/confusion_matrix.png
- training_results/feature_importance.png

## Next Steps:
1. Monitor model performance on cloud
2. Collect feedback and retrain if needed
3. Set up automated retraining pipeline

---
Pipeline completed successfully! 🎉
"""
    
    # Save report
    with open('training_pipeline_report.md', 'w') as f:
        f.write(report)
    
    print("✅ Summary report created: training_pipeline_report.md")

def main():
    """Main pipeline function"""
    print("🚀 Enhanced Training Pipeline")
    print("=" * 50)
    print("This pipeline will:")
    print("1. Check dependencies")
    print("2. Validate dataset")
    print("3. Train enhanced SVM model")
    print("4. Test trained model")
    print("5. Deploy to cloud")
    print("6. Generate summary report")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Step 1: Check dependencies
        if not check_dependencies():
            print("❌ Dependency check failed")
            sys.exit(1)
        
        # Step 2: Check dataset
        if not check_dataset():
            print("❌ Dataset check failed")
            sys.exit(1)
        
        # Step 3: Run training
        if not run_training():
            print("❌ Training failed")
            sys.exit(1)
        
        # Step 4: Test model
        if not test_trained_model():
            print("❌ Model testing failed")
            sys.exit(1)
        
        # Step 5: Deploy to cloud
        if not run_deployment():
            print("❌ Deployment failed")
            sys.exit(1)
        
        # Step 6: Create summary
        create_summary_report()
        
        # Calculate total time
        total_time = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("🎉 Enhanced Training Pipeline Completed Successfully!")
        print("=" * 50)
        print(f"⏱️ Total time: {total_time:.2f} seconds")
        print("✅ All steps completed successfully!")
        print("🚀 Your enhanced model is now live on the cloud!")
        print("\n📋 Check training_pipeline_report.md for details")
        
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 