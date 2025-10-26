#!/bin/bash

# Phishing Detection Tool - Setup Script
# This script sets up the Python environment and installs all dependencies

echo "========================================="
echo "Phishing Detection Tool Setup"
echo "========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "Python version:"
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
    echo "Virtual environment created successfully!"
else
    echo ""
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements."
    exit 1
fi

# Download NLTK data
echo ""
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

if [ $? -ne 0 ]; then
    echo "Warning: Failed to download NLTK data. The script will attempt to download it during runtime."
fi

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p data
mkdir -p "saved data"
mkdir -p templates
mkdir -p test_data

echo ""
echo "========================================="
echo "Setup completed successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Download the dataset from:"
echo "   https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset"
echo ""
echo "3. Place the CSV files in the 'data/' directory"
echo ""
echo "4. Train the model:"
echo "   python main_script.py --data-dir ./data --output-dir './saved data'"
echo ""
echo "5. Run the web application:"
echo "   python app.py"
echo ""
echo "========================================="
