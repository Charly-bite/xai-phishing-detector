# Phishing Email Detection with Explainable AI (XAI)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

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
- [Contributing](#contributing)
- [License](#license)

## 📌 About the Project

This project aims to build and understand models that can automatically detect phishing attempts from emails. It uses a machine learning pipeline to preprocess text, train classifiers, and leverages XAI libraries like SHAP, LIME, and Anchor to interpret the model's decisions, providing transparency into why an email is flagged as phishing.

## ⚙️ How It Works

The workflow is divided into two main scripts:

1.  **`Preprocessing.py`**: This script takes raw email data in CSV format, cleans the text by removing HTML and special characters, and engineers features relevant for phishing detection.
2.  **`Train_model.py`**: This script uses the preprocessed data to train and evaluate several machine learning models (Logistic Regression, Decision Tree, and a hybrid Neural Network). After training, it generates explanations for model predictions on sample data.

## ✨ Features

- **Text Preprocessing**: Cleans and tokenizes raw email text.
- **Feature Engineering**: Extracts features such as:
    - URL counts and suspicious patterns.
    - Presence of urgency-related keywords.
    - Readability scores (Flesch Reading Ease).
- **Multi-Model Training**: Trains and compares Logistic Regression, Decision Trees, and a Neural Network.
- **Explainable AI (XAI)**: Integrates SHAP, LIME, and Anchor to explain model predictions, making the results interpretable.
- **Artifact Management**: Saves trained models, preprocessors, evaluation metrics, and visualizations.

## 📊 Dataset

This project uses the **"Phish No More: The Enron, Ling, CEAS, Nazario, Nigerian & SpamAssassin Datasets"**. This dataset contains approximately 82,500 emails, providing a rich source for training and evaluation.

- **Resource**: [Al-Subaiey, A., et al. (2024). Phish No More. arXiv:2405.11619.](https://arxiv.org/abs/2405.11619)

## 💻 Technology Stack

- **Data Handling**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, TensorFlow/Keras
- **XAI Libraries**: SHAP, LIME, Alibi
- **Text Processing**: NLTK, BeautifulSoup
- **Utilities**: Joblib, Matplotlib

## 🚀 Getting Started

Follow these instructions to set up the project locally.

### Prerequisites

- Python 3.8+
- Pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Charly-bite/Phishing-tool-XAI.git
    cd Phishing-tool-XAI
    ```

2.  **Install the required packages:**
    ```sh
    pip install pandas beautifulsoup4 nltk scikit-learn tensorflow shap lime alibi joblib matplotlib textstat
    ```

3.  **Download NLTK data:**
    Run the following command in a Python shell to download the 'punkt' tokenizer.
    ```python
    import nltk
    nltk.download('punkt')
    ```

## ▶️ Usage

1.  **Add Data**: Place your raw email CSV files into a directory named `Data/`.

2.  **Run Preprocessing**: Execute the script to clean the data and extract features. The output will be saved in the `Preprocessed/` directory.
    ```sh
    python Preprocessing.py
    ```

3.  **Train Models**: Run the training script to build the models and generate explanations.
    ```sh
    python train_model.py
    ```
    The trained models, results, and visualizations will be saved in the `artifacts/` directory.

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
