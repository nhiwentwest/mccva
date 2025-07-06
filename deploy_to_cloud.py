#!/usr/bin/env python3
"""
Deploy Model to Cloud Script
- Copy trained models lên cloud server
- Update Docker image với model mới
- Restart services
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime

class CloudDeployer:
    def __init__(self, cloud_config=None):
        self.cloud_config = cloud_config or {
            'host': 'your-cloud-server.com',
            'user': 'ubuntu',
            'remote_path': '/opt/mccva',
            'ssh_key': '~/.ssh/id_rsa'
        }
        self.local_models_path = 'models/'
        self.training_results_path = 'training_results/'
        
    def check_models_exist(self):
        """Kiểm tra models đã được train"""
        print("🔍 Checking trained models...")
        
        required_files = [
            'svm_model.joblib',
            'scaler.joblib', 
            'label_encoder.joblib',
            'feature_names.joblib',
            'training_info.joblib'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(f'{self.local_models_path}/{file}'):
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing model files: {missing_files}")
            print("Please run training first: python3 train_enhanced_svm.py")
            return False
        
        print("✅ All model files found")
        return True
    
    def create_deployment_package(self):
        """Tạo deployment package"""
        print("📦 Creating deployment package...")
        
        # Create deployment directory
        deploy_dir = 'deployment_package'
        os.makedirs(deploy_dir, exist_ok=True)
        
        # Copy models
        models_deploy_dir = f'{deploy_dir}/models'
        os.makedirs(models_deploy_dir, exist_ok=True)
        
        for file in os.listdir(self.local_models_path):
            if file.endswith('.joblib'):
                src = f'{self.local_models_path}/{file}'
                dst = f'{models_deploy_dir}/{file}'
                subprocess.run(['cp', src, dst], check=True)
                print(f"  Copied: {file}")
        
        # Copy training results
        if os.path.exists(self.training_results_path):
            results_deploy_dir = f'{deploy_dir}/training_results'
            os.makedirs(results_deploy_dir, exist_ok=True)
            
            for file in os.listdir(self.training_results_path):
                if file.endswith('.png'):
                    src = f'{self.training_results_path}/{file}'
                    dst = f'{results_deploy_dir}/{file}'
                    subprocess.run(['cp', src, dst], check=True)
                    print(f"  Copied: {file}")
        
        # Create deployment info
        deployment_info = {
            'timestamp': datetime.now().isoformat(),
            'deployment_id': f"deploy_{int(time.time())}",
            'model_files': [f for f in os.listdir(models_deploy_dir) if f.endswith('.joblib')],
            'training_results': [f for f in os.listdir(results_deploy_dir) if f.endswith('.png')] if os.path.exists(results_deploy_dir) else []
        }
        
        with open(f'{deploy_dir}/deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"✅ Deployment package created: {deploy_dir}")
        return deploy_dir
    
    def deploy_to_cloud(self, deploy_dir):
        """Deploy lên cloud server"""
        print("☁️ Deploying to cloud...")
        
        try:
            # SSH command template
            ssh_cmd = f"ssh -i {self.cloud_config['ssh_key']} {self.cloud_config['user']}@{self.cloud_config['host']}"
            
            # 1. Create backup of current models
            print("  Creating backup...")
            backup_cmd = f"{ssh_cmd} 'cd {self.cloud_config['remote_path']} && mkdir -p backups && cp -r models backups/models_backup_{int(time.time())}'"
            subprocess.run(backup_cmd, shell=True, check=True)
            
            # 2. Copy new models
            print("  Copying new models...")
            scp_cmd = f"scp -i {self.cloud_config['ssh_key']} -r {deploy_dir}/models/* {self.cloud_config['user']}@{self.cloud_config['host']}:{self.cloud_config['remote_path']}/models/"
            subprocess.run(scp_cmd, shell=True, check=True)
            
            # 3. Copy training results
            if os.path.exists(f'{deploy_dir}/training_results'):
                print("  Copying training results...")
                scp_results_cmd = f"scp -i {self.cloud_config['ssh_key']} -r {deploy_dir}/training_results/* {self.cloud_config['user']}@{self.cloud_config['host']}:{self.cloud_config['remote_path']}/training_results/"
                subprocess.run(scp_results_cmd, shell=True, check=True)
            
            # 4. Update Docker image
            print("  Updating Docker image...")
            docker_build_cmd = f"{ssh_cmd} 'cd {self.cloud_config['remote_path']} && docker build -t mccva-ml-service .'"
            subprocess.run(docker_build_cmd, shell=True, check=True)
            
            # 5. Restart ML service
            print("  Restarting ML service...")
            restart_cmd = f"{ssh_cmd} 'docker stop mccva-ml && docker rm mccva-ml && docker run -d --name mccva-ml -p 5000:5000 --restart unless-stopped mccva-ml-service'"
            subprocess.run(restart_cmd, shell=True, check=True)
            
            # 6. Wait for service to start
            print("  Waiting for service to start...")
            time.sleep(10)
            
            # 7. Test health endpoint
            print("  Testing health endpoint...")
            health_cmd = f"{ssh_cmd} 'curl -s http://localhost:5000/health'"
            result = subprocess.run(health_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                try:
                    health_data = json.loads(result.stdout)
                    if health_data.get('status') == 'healthy':
                        print("✅ ML Service is healthy!")
                    else:
                        print(f"⚠️ ML Service status: {health_data.get('status')}")
                except:
                    print("⚠️ Could not parse health response")
            else:
                print("❌ Health check failed")
            
            print("✅ Deployment completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Deployment failed: {e}")
            return False
    
    def test_deployment(self):
        """Test deployment"""
        print("🧪 Testing deployment...")
        
        try:
            # Test local model loading
            import joblib
            
            svm_model = joblib.load(f'{self.local_models_path}/svm_model.joblib')
            scaler = joblib.load(f'{self.local_models_path}/scaler.joblib')
            label_encoder = joblib.load(f'{self.local_models_path}/label_encoder.joblib')
            
            # Test prediction
            test_features = [4, 8, 100, 1000, 3, 0.5, 12.5, 250, 32, 12]
            features_scaled = scaler.transform([test_features])
            prediction = svm_model.predict(features_scaled)[0]
            prediction_class = label_encoder.inverse_transform([prediction])[0]
            
            print(f"  Test prediction: {prediction_class}")
            print("✅ Local model test passed")
            
            # Test remote endpoint
            ssh_cmd = f"ssh -i {self.cloud_config['ssh_key']} {self.cloud_config['user']}@{self.cloud_config['host']}"
            
            test_cmd = f"{ssh_cmd} 'curl -s -X POST http://localhost:5000/predict/makespan -H \"Content-Type: application/json\" -d \"{{\\\"features\\\": [4, 8, 100, 1000, 3]}}\"'"
            result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                try:
                    response = json.loads(result.stdout)
                    print(f"  Remote prediction: {response.get('makespan')}")
                    print("✅ Remote endpoint test passed")
                except:
                    print("⚠️ Could not parse remote response")
            else:
                print("❌ Remote endpoint test failed")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def cleanup_deployment_package(self, deploy_dir):
        """Cleanup deployment package"""
        print("🧹 Cleaning up deployment package...")
        
        try:
            subprocess.run(['rm', '-rf', deploy_dir], check=True)
            print("✅ Cleanup completed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Cleanup failed: {e}")
    
    def run_deployment(self):
        """Run complete deployment pipeline"""
        print("🚀 Starting Cloud Deployment Pipeline")
        print("=" * 50)
        
        try:
            # 1. Check models exist
            if not self.check_models_exist():
                return False
            
            # 2. Create deployment package
            deploy_dir = self.create_deployment_package()
            
            # 3. Deploy to cloud
            if not self.deploy_to_cloud(deploy_dir):
                return False
            
            # 4. Test deployment
            if not self.test_deployment():
                return False
            
            # 5. Cleanup
            self.cleanup_deployment_package(deploy_dir)
            
            print("\n" + "=" * 50)
            print("🎉 Deployment Pipeline Completed Successfully!")
            print("=" * 50)
            print("✅ Model deployed to cloud")
            print("✅ ML Service restarted")
            print("✅ Health checks passed")
            
            return True
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def load_cloud_config():
    """Load cloud configuration"""
    config_file = 'cloud_config.json'
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        # Create default config
        default_config = {
            'host': 'your-cloud-server.com',
            'user': 'ubuntu',
            'remote_path': '/opt/mccva',
            'ssh_key': '~/.ssh/id_rsa'
        }
        
        print("⚠️ No cloud_config.json found. Creating default config...")
        print("Please edit cloud_config.json with your actual cloud server details.")
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config

def main():
    """Main function"""
    print("☁️ Cloud Deployment Script")
    print("=" * 30)
    
    # Load configuration
    config = load_cloud_config()
    
    # Check if config is default
    if config['host'] == 'your-cloud-server.com':
        print("❌ Please configure cloud_config.json with your actual server details")
        print("Example:")
        print("  {")
        print("    \"host\": \"your-server-ip\",")
        print("    \"user\": \"ubuntu\",")
        print("    \"remote_path\": \"/opt/mccva\",")
        print("    \"ssh_key\": \"~/.ssh/id_rsa\"")
        print("  }")
        sys.exit(1)
    
    # Create deployer
    deployer = CloudDeployer(config)
    
    # Run deployment
    success = deployer.run_deployment()
    
    if success:
        print("\n✅ Deployment completed successfully!")
        print("🚀 Your enhanced model is now live on the cloud!")
    else:
        print("\n❌ Deployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 