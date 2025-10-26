#!/usr/bin/env python3
"""
Quick validation script to test the Phishing Detection Tool setup
Tests imports, dependencies, and basic functionality
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    display_name = package_name or module_name
    try:
        importlib.import_module(module_name)
        print(f"✓ {display_name:<30} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {display_name:<30} FAILED: {e}")
        return False

def main():
    """Run validation tests"""
    print("=" * 60)
    print("Phishing Detection Tool - Setup Validation")
    print("=" * 60)
    print()
    
    # Test Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠ Warning: Python 3.8+ is recommended")
    print()
    
    print("Testing required packages...")
    print("-" * 60)
    
    required_packages = [
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "scikit-learn"),
        ("tensorflow", "TensorFlow"),
        ("keras", "Keras"),
        ("bs4", "BeautifulSoup4"),
        ("nltk", "NLTK"),
        ("textstat", "TextStat"),
        ("shap", "SHAP"),
        ("lime", "LIME"),
        ("sentence_transformers", "Sentence Transformers"),
        ("flask", "Flask"),
        ("joblib", "Joblib"),
        ("matplotlib", "Matplotlib"),
    ]
    
    success_count = 0
    total_count = len(required_packages)
    
    for module, display_name in required_packages:
        if test_import(module, display_name):
            success_count += 1
    
    print("-" * 60)
    print(f"\nResults: {success_count}/{total_count} packages available")
    
    if success_count == total_count:
        print("✓ All required packages are installed!")
    else:
        print("⚠ Some packages are missing. Run './setup.sh' or install manually.")
        return 1
    
    # Test NLTK data
    print()
    print("Testing NLTK data...")
    print("-" * 60)
    
    try:
        import nltk
        nltk_data_available = True
        
        # Test stopwords
        try:
            nltk.data.find('corpora/stopwords')
            print("✓ NLTK stopwords data available")
        except LookupError:
            print("✗ NLTK stopwords data missing")
            nltk_data_available = False
        
        # Test punkt
        try:
            nltk.data.find('tokenizers/punkt')
            print("✓ NLTK punkt tokenizer available")
        except LookupError:
            print("✗ NLTK punkt tokenizer missing")
            nltk_data_available = False
        
        if not nltk_data_available:
            print("\nDownload NLTK data with:")
            print('  python -c "import nltk; nltk.download(\'punkt\'); nltk.download(\'stopwords\')"')
    except Exception as e:
        print(f"✗ Error checking NLTK data: {e}")
    
    # Check directory structure
    print()
    print("Checking directory structure...")
    print("-" * 60)
    
    import os
    
    required_dirs = ['data', 'templates', 'test_data']
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"✓ {dir_name}/ directory exists")
        else:
            print(f"⚠ {dir_name}/ directory missing (will be created if needed)")
    
    # Check for required files
    required_files = ['main_script.py', 'app.py', 'requirements.txt', 'templates/index.html']
    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"✓ {file_name} exists")
        else:
            print(f"✗ {file_name} MISSING")
    
    print()
    print("=" * 60)
    print("Validation complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Download dataset from Kaggle and place CSV files in data/")
    print("2. Train the model: python main_script.py --data-dir ./data --output-dir './saved data'")
    print("3. Run web app: python app.py")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
