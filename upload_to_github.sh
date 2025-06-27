#!/bin/bash

# MCCVA GitHub Upload Script
# Tự động upload project lên GitHub repository
# Chạy: ./upload_to_github.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# GitHub repository URL
GITHUB_REPO="https://github.com/nhiwentwest/mccva.git"

print_header "MCCVA GitHub Upload"
print_status "Uploading MCCVA Algorithm project to GitHub"

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install git first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "ml_service.py" ] || [ ! -f "deploy.sh" ]; then
    print_error "Please run this script from the MCCVA project directory"
    exit 1
fi

# Initialize git repository if not already done
if [ ! -d ".git" ]; then
    print_status "Initializing git repository..."
    git init
fi

# Add all files to git
print_status "Adding files to git..."
git add .

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    print_status "Creating .gitignore file..."
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
*.tmp
*.temp

# Model files (if too large)
# models/*.joblib
EOF
    git add .gitignore
fi

# Check if there are changes to commit
if git diff --cached --quiet; then
    print_warning "No changes to commit. Repository is up to date."
else
    # Commit changes
    print_status "Committing changes..."
    git commit -m "MCCVA Algorithm Implementation

- SVM Classification for makespan prediction
- K-Means Clustering for VM clustering  
- MCCVA Routing algorithm
- OpenResty integration
- Production-ready deployment scripts
- Comprehensive testing suite

Features:
- AI-based load balancing
- Real-time prediction endpoints
- Performance monitoring
- Error handling and logging
- Ubuntu deployment automation"
fi

# Add remote origin if not exists
if ! git remote get-url origin &> /dev/null; then
    print_status "Adding remote origin..."
    git remote add origin $GITHUB_REPO
fi

# Check if remote exists and is correct
REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
if [ "$REMOTE_URL" != "$GITHUB_REPO" ]; then
    print_status "Updating remote origin..."
    git remote set-url origin $GITHUB_REPO
fi

# Push to GitHub
print_status "Pushing to GitHub..."
if git push -u origin main 2>/dev/null; then
    print_status "✅ Successfully pushed to main branch"
elif git push -u origin master 2>/dev/null; then
    print_status "✅ Successfully pushed to master branch"
else
    # Try to create and push to main branch
    print_status "Creating main branch..."
    git checkout -b main 2>/dev/null || git checkout main
    git push -u origin main
    print_status "✅ Successfully pushed to main branch"
fi

print_header "GitHub Upload Complete!"

print_status "Repository Information:"
echo "  • GitHub URL: $GITHUB_REPO"
echo "  • Branch: main"
echo "  • Status: ✅ Uploaded successfully"

print_status "Next Steps:"
echo "  1. Visit: $GITHUB_REPO"
echo "  2. Verify all files are uploaded"
echo "  3. Run deployment script on your server"

print_status "To deploy on Amazon Cloud Ubuntu:"
echo "  git clone $GITHUB_REPO"
echo "  cd mccva"
echo "  chmod +x deploy.sh"
echo "  ./deploy.sh"

print_status "✅ MCCVA project successfully uploaded to GitHub!" 