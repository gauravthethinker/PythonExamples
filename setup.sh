#!/bin/bash

echo "🚀 Setting up Vector Database Loader Environment"
echo "================================================"

# Check if running as root/sudo for system packages
if [[ $EUID -eq 0 ]]; then
    echo "📦 Installing system dependencies..."
    apt update
    apt install -y python3-venv python3-pip python3-dev build-essential
else
    echo "⚠️  Run with sudo if you need to install system packages:"
    echo "   sudo ./setup.sh"
    echo ""
    echo "📦 Attempting to set up without system installation..."
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv vector_db_env

# Check if virtual environment was created successfully
if [ ! -d "vector_db_env" ]; then
    echo "❌ Failed to create virtual environment."
    echo "   You may need to run: sudo apt install python3-venv"
    exit 1
fi

# Activate virtual environment and install dependencies
echo "📚 Installing Python dependencies..."
source vector_db_env/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install dependencies
pip install chromadb sentence-transformers numpy pandas torch transformers

# Check if installation was successful
python -c "import chromadb, sentence_transformers; print('✅ Dependencies installed successfully!')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Setup completed successfully!"
    echo ""
    echo "To use the vector database loader:"
    echo "1. Activate the environment: source vector_db_env/bin/activate"
    echo "2. Run the demo: python vector_db_loader.py"
    echo "3. Or run the AI agent demo: python example_usage.py"
else
    echo "❌ Installation failed. Please check the error messages above."
    exit 1
fi