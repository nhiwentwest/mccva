#!/bin/bash

# Fix Git Conflict Script for MCCVA
# Chạy script này để fix git conflicts trên server hiện tại

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo "================================"
    echo "$1"
    echo "================================"
}

print_header "Fix Git Conflicts for MCCVA"

PROJECT_DIR="/opt/mccva"

if [ ! -d "$PROJECT_DIR" ]; then
    print_error "Project directory not found: $PROJECT_DIR"
    exit 1
fi

cd $PROJECT_DIR

print_status "Current directory: $(pwd)"
print_status "Checking git status..."

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_error "Not a git repository"
    exit 1
fi

# Show current status
print_status "Git status:"
git status --porcelain

# Step 1: Stash local changes
print_header "Step 1: Stash Local Changes"
print_status "Stashing local changes..."

if ! git diff --quiet; then
    git stash push -m "Auto-stash before deployment update $(date)"
    print_success "✅ Local changes stashed"
else
    print_status "✅ No local changes to stash"
fi

# Step 2: Pull latest changes
print_header "Step 2: Pull Latest Changes"
print_status "Pulling latest changes from GitHub..."

git fetch origin
git reset --hard origin/main

print_success "✅ Repository updated to latest version"

# Step 3: Check if stashed changes should be applied
print_header "Step 3: Check Stashed Changes"
if git stash list | grep -q "Auto-stash before deployment update"; then
    print_warning "⚠️ Stashed changes found. You may want to review them:"
    git stash list
    
    read -p "Do you want to apply stashed changes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Applying stashed changes..."
        git stash pop
        print_success "✅ Stashed changes applied"
    else
        print_status "Stashed changes kept for manual review"
        print_status "To view stashed changes: git stash show -p"
        print_status "To apply later: git stash pop"
    fi
else
    print_status "✅ No stashed changes found"
fi

# Step 4: Verify files
print_header "Step 4: Verify Files"
print_status "Checking essential files..."

ESSENTIAL_FILES=(
    "ml_service.py"
    "mock_servers.py"
    "amazon_deploy.sh"
    "nginx.conf"
    "requirements.txt"
)

for file in "${ESSENTIAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "✅ $file exists"
    else
        print_error "❌ $file not found"
    fi
done

# Step 5: Check git status
print_header "Step 5: Final Git Status"
print_status "Current git status:"
git status

print_header "Fix Complete"
print_success "✅ Git conflicts have been resolved!"
print_status "You can now continue with deployment:"
print_status "  ./amazon_deploy.sh" 