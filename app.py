"""
Flask Web Application for Phishing Email Detection
This app loads the trained model and provides a web interface for predictions.
"""

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os
import logging
from sentence_transformers import SentenceTransformer
import re
from bs4 import BeautifulSoup
import warnings
import json

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and preprocessing components
model = None
embedding_model = None
numeric_preprocessor = None
numeric_cols = None
embedding_dim = None
model_error = None

# Define the original numeric feature column names (should match training)
EXPECTED_NUMERIC_COLS = ['num_links', 'has_suspicious_url', 'urgency_count', 'readability_score']


def load_model_artifacts(model_dir='phishing_results_embeddings/models'):
    """Load the trained model and preprocessing components."""
    global model, embedding_model, numeric_preprocessor, numeric_cols, embedding_dim, model_error
    
    try:
        logger.info(f"Loading model artifacts from {model_dir}")
        
        # Load embedding model info
        embedding_info_path = os.path.join(model_dir, 'embedding_model_info.json')
        if os.path.exists(embedding_info_path):
            with open(embedding_info_path, 'r') as f:
                embedding_info = json.load(f)
                embedding_model_name = embedding_info.get('model_name')
                embedding_dim = embedding_info.get('embedding_dimension', 384)
                logger.info(f"Loading embedding model: {embedding_model_name}")
                embedding_model = SentenceTransformer(embedding_model_name)
        else:
            logger.warning("Embedding model info not found. Using default.")
            embedding_dim = 384
            embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Load numeric preprocessor
        preprocessor_path = os.path.join(model_dir, 'numeric_preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            numeric_preprocessor = joblib.load(preprocessor_path)
            logger.info("Numeric preprocessor loaded")
        else:
            logger.warning("Numeric preprocessor not found")
        
        # Load numeric columns info
        numeric_cols_path = os.path.join(model_dir, 'numeric_cols_info.json')
        if os.path.exists(numeric_cols_path):
            with open(numeric_cols_path, 'r') as f:
                numeric_cols_data = json.load(f)
                numeric_cols = numeric_cols_data.get('numeric_columns', EXPECTED_NUMERIC_COLS)
        else:
            numeric_cols = EXPECTED_NUMERIC_COLS
        
        # Try to load saved models (check for different formats)
        model_found = False
        
        # Check for Keras model
        keras_model_path = os.path.join(model_dir, 'HybridNN.keras')
        if os.path.exists(keras_model_path):
            from tensorflow import keras
            model = keras.models.load_model(keras_model_path)
            logger.info("Loaded HybridNN Keras model")
            model_found = True
        
        # Check for scikit-learn models
        if not model_found:
            for model_name in ['LogisticRegression.pkl', 'DecisionTree.pkl']:
                model_path = os.path.join(model_dir, model_name)
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} model")
                    model_found = True
                    break
        
        if not model_found:
            model_error = "No trained model found. Please train a model first using main_script.py"
            logger.error(model_error)
        else:
            logger.info("Model artifacts loaded successfully")
            
    except Exception as e:
        model_error = f"Error loading model: {str(e)}"
        logger.error(model_error, exc_info=True)


def clean_email(text):
    """Clean email text (same as in main_script.py)."""
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return ""
    try:
        try:
            soup = BeautifulSoup(text, 'lxml')
        except:
            soup = BeautifulSoup(text, 'html.parser')
        cleaned = soup.get_text(separator=' ')
        cleaned = re.sub(r'https?://\S+|www\.\S+', ' URL ', cleaned)
        cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?\'`áéíóúÁÉÍÓÚñÑüÜ]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return ""


def extract_url_features(text):
    """Extract URL-based features."""
    num_links, has_suspicious_url = 0, 0
    if pd.isna(text) or not isinstance(text, str):
        return 0, 0
    try:
        url_pattern = r'(https?://\S+|www\.\S+)'
        urls = re.findall(url_pattern, text[:50000])
        num_links = len(urls)
        
        if not urls:
            return num_links, has_suspicious_url
        
        suspicious_keywords = ['login', 'verify', 'account', 'secure', 'update', 'confirm', 
                              'signin', 'support', 'password', 'banking', 'activity', 'credential',
                              'iniciar sesion', 'verificar', 'cuenta', 'actualizar', 'confirmar', 
                              'contraseña', 'banco', 'actividad', 'credenciales']
        shortened_domains_pattern = r'(bit\.ly/|goo\.gl/|tinyurl\.com/|t\.co/|ow\.ly/|is\.gd/|buff\.ly/|adf\.ly/|bit\.do/|soo\.gd/)'
        
        for url in urls[:100]:
            url_lower = url.lower()
            if url_lower.startswith('http://') or re.search(shortened_domains_pattern, url_lower) or \
               any(keyword in url_lower for keyword in suspicious_keywords):
                has_suspicious_url = 1
                break
                
        return num_links, has_suspicious_url
    except Exception as e:
        logger.error(f"URL feature extraction failed: {e}")
        return 0, 0


def extract_urgency(cleaned_text):
    """Extract urgency-related features."""
    if not cleaned_text:
        return 0
    try:
        urgency_words = ['urgent', 'immediately', 'action required', 'verify', 'password', 
                        'alert', 'warning', 'limited time', 'expire', 'suspended', 'locked',
                        'urgente', 'inmediatamente', 'acción requerida', 'verifique', 
                        'contraseña', 'alerta', 'advertencia', 'tiempo limitado', 'expira']
        text_lower = cleaned_text.lower()
        count = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower)) for word in urgency_words)
        return count
    except Exception as e:
        logger.error(f"Urgency calculation failed: {e}")
        return 0


def calculate_readability(cleaned_text):
    """Calculate readability score."""
    word_count = len(cleaned_text.split())
    if not cleaned_text or word_count < 10:
        return 100.0
    try:
        import textstat
        score = textstat.flesch_reading_ease(cleaned_text)
        return max(-200, min(120, score)) if not np.isnan(score) else 50.0
    except Exception as e:
        if word_count > 5:
            logger.debug(f"Readability failed: {e}")
        return 50.0


def extract_features(text):
    """Extract all features from email text."""
    # Clean the text
    cleaned_text = clean_email(text)
    
    # Extract numeric features
    num_links, has_suspicious_url = extract_url_features(text)
    urgency_count = extract_urgency(cleaned_text)
    readability_score = calculate_readability(cleaned_text)
    
    # Create numeric features array
    numeric_features = np.array([[num_links, has_suspicious_url, urgency_count, readability_score]])
    
    # Scale numeric features if preprocessor is available
    if numeric_preprocessor is not None:
        numeric_features_scaled = numeric_preprocessor.transform(numeric_features)
    else:
        numeric_features_scaled = numeric_features
    
    # Compute embeddings
    if embedding_model is not None:
        embeddings = embedding_model.encode([cleaned_text], convert_to_numpy=True)
    else:
        embeddings = np.zeros((1, embedding_dim))
    
    return embeddings, numeric_features_scaled, cleaned_text


def predict_email(text):
    """Predict if email is phishing or legitimate."""
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Extract features
        embeddings, numeric_features, cleaned_text = extract_features(text)
        
        # Prepare input based on model type
        if hasattr(model, 'predict_proba'):  # Scikit-learn models
            # Combine embeddings and numeric features
            combined_features = np.hstack([embeddings, numeric_features])
            prediction_proba = model.predict_proba(combined_features)[0]
            prediction = 1 if prediction_proba[1] > 0.5 else 0
            confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
        else:  # Keras model
            # Pass as separate inputs
            prediction_proba = model.predict([embeddings, numeric_features], verbose=0)[0][0]
            prediction = 1 if prediction_proba > 0.5 else 0
            confidence = prediction_proba if prediction == 1 else (1 - prediction_proba)
        
        result = "Phishing" if prediction == 1 else "Legitimate"
        return result, f"{confidence * 100:.2f}%"
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return None, f"Error during prediction: {str(e)}"


@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html', model_error=model_error)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    if model is None:
        return render_template('index.html', 
                             prediction_text="Model not loaded. Please train a model first.",
                             email_text=request.form.get('email_text', ''),
                             model_error=model_error)
    
    try:
        email_text = request.form.get('email_text', '')
        if not email_text.strip():
            return render_template('index.html', 
                                 prediction_text="Please enter email content to analyze.",
                                 email_text=email_text,
                                 model_error=model_error)
        
        result, confidence = predict_email(email_text)
        
        if result:
            prediction_text = f"⚠️ This email appears to be {result} (Confidence: {confidence})"
        else:
            prediction_text = f"Error: {confidence}"
        
        return render_template('index.html', 
                             prediction_text=prediction_text,
                             email_text=email_text,
                             model_error=model_error)
    
    except Exception as e:
        logger.error(f"Error in predict route: {e}", exc_info=True)
        return render_template('index.html', 
                             prediction_text=f"An error occurred: {str(e)}",
                             email_text=request.form.get('email_text', ''),
                             model_error=model_error)


if __name__ == '__main__':
    # Load model artifacts on startup
    load_model_artifacts()
    
    # Run the Flask app
    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
