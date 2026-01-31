#!/bin/bash

# Sanskrit Manuscript Frontend - Setup Script
# Run this script from WSL/Linux terminal only

echo "🚀 Setting up Sanskrit Manuscript Restoration Frontend..."

# Navigate to project directory
cd "$(dirname "$0")"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

echo "✓ Node.js version: $(node --version)"
echo "✓ npm version: $(npm --version)"

# Clean previous installations
echo "🧹 Cleaning previous installations..."
rm -rf node_modules package-lock.json

# Install dependencies
echo "📦 Installing dependencies..."
npm install

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
    echo ""
    echo "🎉 Setup complete! Run the following command to start:"
    echo ""
    echo "   npm run dev"
    echo ""
    echo "The application will be available at: http://localhost:3000"
else
    echo "❌ Installation failed. Please check errors above."
    echo ""
    echo "💡 If you're running from Windows via WSL, try:"
    echo "   1. Open WSL terminal (not Windows terminal)"
    echo "   2. Navigate to: cd /home/bagesh/EL-project/manuscript-frontend"
    echo "   3. Run: bash setup.sh"
fi

