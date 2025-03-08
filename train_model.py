# File: C:\Users\acvsa\PhishingDetector\src\train_model.py

import pandas as pd
import logging
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r"C:\Users\acvsa\PhishingDetector\training.log"),  # Log file
        logging.StreamHandler(sys.stdout)  # Print to console
    ]
)

# --------------------------------------
# 1. Load Preprocessed Data
# --------------------------------------
def load_data():
    try:
        preprocessed_path = r"C:\Users\acvsa\PhishingDetector\data\preprocessed\Enron_preprocessed.csv"
        logging.info(f"Loading data from: {preprocessed_path}")
        
        if not os.path.exists(preprocessed_path):
            logging.error("Preprocessed data file not found!")
            sys.exit(1)
            
        df = pd.read_csv(preprocessed_path)
        if df.empty:
            logging.error("Loaded DataFrame is empty!")
            sys.exit(1)
            
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {str(e)}", exc_info=True)
        sys.exit(1)

# --------------------------------------
# 2-7. Train and Evaluate Model
# --------------------------------------
def train_model(df):
    try:
        logging.info("Extracting features...")
        
        # Safely convert string representation of lists to actual lists
        df['tokenized_text'] = df['tokenized_text'].apply(
            lambda x: eval(x) if isinstance(x, str) else []
        )
        
        structural_features = df[['num_links', 'has_suspicious_url', 'urgency_count', 'readability_score']]
        text_features = df['tokenized_text'].apply(lambda x: ' '.join(x))
        
        logging.info("Vectorizing text...")
        tfidf = TfidfVectorizer(max_features=1000)
        text_features_tfidf = tfidf.fit_transform(text_features)
        
        logging.info("Combining features...")
        X = pd.concat([structural_features, pd.DataFrame(text_features_tfidf.toarray())], axis=1)
        y = df['Label']
        
        logging.info("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logging.info("Training model...")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        logging.info("Evaluating model...")
        y_pred = model.predict(X_test)
        logging.info("\n" + classification_report(y_test, y_pred))
        
        return model, tfidf
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

# --------------------------------------
# 8-9. Save Model and Vectorizer
# --------------------------------------
def save_artifacts(model, tfidf):
    try:
        model_dir = r"C:\Users\acvsa\PhishingDetector\models"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "phishing_model.pkl")
        tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        
        joblib.dump(model, model_path)
        joblib.dump(tfidf, tfidf_path)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Failed to save artifacts: {str(e)}", exc_info=True)
        sys.exit(1)

# --------------------------------------
# Main Execution
# --------------------------------------
if __name__ == "__main__":
    try:
        df = load_data()
        model, tfidf = train_model(df)
        save_artifacts(model, tfidf)
        logging.info("Training completed successfully!")
    except Exception as e:
        logging.error(f"Critical error: {str(e)}", exc_info=True)
        sys.exit(1)