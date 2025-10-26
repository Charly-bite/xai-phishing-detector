# Phishing Email Detection with Explainable AI (XAI)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)

A Python-based tool for detecting phishing emails using machine learning and explaining the predictions with Explainable AI (XAI) techniques.

## 📖 Table of Contents
- [About the Project](#about-the-project)
- [How It Works](#how-it-works)
- [Features](#features)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Training the Model](#training-the-model)
  - [Running the Web Application](#running-the-web-application)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 📌 About the Project

This project aims to build and understand models that can automatically detect phishing attempts from emails. It uses a machine learning pipeline to preprocess text, train classifiers, and leverages XAI libraries like SHAP and LIME to interpret the model's decisions, providing transparency into why an email is flagged as phishing.

## ⚙️ How It Works

The workflow consists of:

1.  **Data Preprocessing**: The `main_script.py` takes raw email data in CSV format, cleans the text by removing HTML and special characters, and engineers features relevant for phishing detection including:
    - URL-based features
    - Urgency keywords
    - Readability scores
    - Multilingual sentence embeddings
    
2.  **Model Training**: Multiple machine learning models are trained and evaluated:
    - Logistic Regression
    - Decision Tree
    - Hybrid Neural Network (combining embeddings and structural features)
    
3.  **Model Selection**: The best performing model based on F1 score is automatically selected and saved.

4.  **Explainable AI**: SHAP and LIME generate explanations for model predictions on sample data, making results interpretable.

5.  **Web Interface**: A Flask web application (`app.py`) loads the trained model and provides a user-friendly interface for real-time phishing detection.

## ✨ Features

- **Text Preprocessing**: Cleans and tokenizes raw email text with HTML removal
- **Multilingual Support**: Uses sentence transformers for multilingual email analysis
- **Feature Engineering**: Extracts comprehensive features including:
    - URL counts and suspicious URL patterns
    - Presence of urgency-related keywords (English & Spanish)
    - Readability scores (Flesch Reading Ease)
    - Sentence embeddings
- **Multi-Model Training**: Trains and compares multiple models automatically
- **Explainable AI (XAI)**: Integrates SHAP and LIME for model interpretability
- **Web Interface**: User-friendly Flask application for real-time predictions
- **Artifact Management**: Saves trained models, preprocessors, evaluation metrics, and visualizations
- **Automatic Model Selection**: Chooses the best model based on performance metrics

## 📊 Dataset

This project uses the **"Phish No More: The Enron, Ling, CEAS, Nazario, Nigerian & SpamAssassin Datasets"**. This dataset contains approximately 82,500 emails, providing a rich source for training and evaluation.

- **Resource**: [Al-Subaiey, A., et al. (2024). Phish No More. arXiv:2405.11619.](https://arxiv.org/abs/2405.11619)
- **Download**: [Kaggle Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)

## 💻 Technology Stack

- **Data Handling**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, TensorFlow/Keras
- **NLP**: Sentence Transformers, NLTK, BeautifulSoup
- **XAI Libraries**: SHAP, LIME
- **Web Framework**: Flask
- **Utilities**: Joblib, Matplotlib, TextStat

## 🚀 Getting Started

Follow these instructions to set up the project locally.

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- 8GB+ RAM recommended for model training

### Installation

#### Option 1: Automated Setup (Recommended)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Charly-bite/Phishing-tool-XAI.git
    cd Phishing-tool-XAI
    ```

2.  **Run the setup script:**
    ```bash
    ./setup.sh
    ```
    
    This script will:
    - Create a Python virtual environment
    - Install all required dependencies
    - Download NLTK data
    - Create necessary directories

#### Option 2: Manual Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Charly-bite/Phishing-tool-XAI.git
    cd Phishing-tool-XAI
    ```

2.  **Create and activate virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/Mac
    # or
    .\venv\Scripts\activate  # On Windows
    ```

3.  **Install the required packages:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    ```python
    python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
    ```

## ▶️ Usage

### Quick Start

1.  **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)

2.  **Place CSV files** in the `data/` directory

3.  **Activate virtual environment** (if not already activated):
    ```bash
    source venv/bin/activate  # On Linux/Mac
    ```

4.  **Train the model**:
    ```bash
    python main_script.py --data-dir ./data --output-dir "./saved data"
    ```

5.  **Run the web application**:
    ```bash
    python app.py
    ```

6.  **Open your browser** and navigate to:
    ```
    http://localhost:5000
    ```

### Training the Model

The main script processes data, trains models, and generates explanations:

```bash
python main_script.py --data-dir <path-to-csv-files> --output-dir <path-to-save-results>
```

**Arguments:**
- `--data-dir`: Directory containing raw CSV data files (required)
- `--output-dir`: Directory to save models, results, and explanations (default: `phishing_results_embeddings`)

**Example:**
```bash
python main_script.py --data-dir ./data --output-dir "./saved data"
```

**What happens during training:**
1. Loads and preprocesses all CSV files from the data directory
2. Extracts features and computes embeddings
3. Trains multiple models (Logistic Regression, Decision Tree, Hybrid NN)
4. Evaluates and compares model performance
5. Selects the best model based on F1 score
6. Generates SHAP and LIME explanations
7. Saves models, preprocessors, and visualizations

**Output files:**
- `models/` - Trained model files (`.pkl` or `.keras`)
- `models/numeric_preprocessor.pkl` - Feature preprocessor
- `models/embedding_model_info.json` - Embedding model configuration
- `results.json` - Performance metrics for all models
- `explanations/` - SHAP and LIME visualizations (for best model only)
- `*.png` - Performance comparison plots

### Running the Web Application

After training, start the Flask web server:

```bash
python app.py
```

The application will:
1. Load the trained model and preprocessors
2. Start a web server on `http://localhost:5000`
3. Provide a web interface for phishing detection

**Using the web interface:**
1. Paste email content into the text area
2. Click "Analyze Email"
3. View the prediction result with confidence score
4. Use "Clear Text" to analyze another email

## 📁 Project Structure

```
Phishing-tool-XAI/
├── main_script.py           # Main training and evaluation script
├── app.py                   # Flask web application
├── requirements.txt         # Python dependencies
├── setup.sh                # Automated setup script
├── README.md               # This file
├── instructions.txt        # Original instructions (Spanish)
├── data/                   # Place your CSV datasets here
│   └── about.txt
├── saved data/             # Output directory for trained models
│   └── about.txt
├── templates/              # HTML templates for Flask
│   └── index.html
├── test_data/             # Optional test datasets
│   └── SpamAssasin.csv
└── venv/                  # Virtual environment (created by setup)
```

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🔧 Troubleshooting

**Issue: NLTK data not found**
```bash
python3 -c "import nltk; nltk.download('all')"
```

**Issue: Out of memory during training**
- Reduce batch size in `main_script.py`
- Process fewer CSV files at once
- Use a machine with more RAM

**Issue: Model not loading in web app**
- Ensure you've trained a model first using `main_script.py`
- Check that the output directory matches what `app.py` expects
- Verify all required files exist in the models directory

**Issue: Flask port already in use**
- Change the port in `app.py`: `app.run(debug=True, host='0.0.0.0', port=5001)`

## 📧 Contact

Project Link: [https://github.com/Charly-bite/Phishing-tool-XAI](https://github.com/Charly-bite/Phishing-tool-XAI)

