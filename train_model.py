# File: C:\Users\acvsa\PhishingDetector\src\train_model.py
# Add at the very top
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*The name tf.Session is deprecated.*")
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_SAVE_FORMAT'] = 'h5'  # Suppress HDF5 format warning

import pandas as pd
import logging
import sys
import json
import matplotlib.pyplot as plt
import joblib
import shap
from lime import lime_text
from alibi.explainers import AnchorText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.exceptions import NotFittedError
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate

# Custom analyzer function for pickle compatibility
def identity_analyzer(x):
    """Return input as-is for pre-tokenized data"""
    return x

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler(r"C:\Users\acvsa\PhishingDetector\training.log")
console_handler = logging.StreamHandler(sys.stdout)

# Create formatters and add to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --------------------------------------
# 1. Load Preprocessed Data
# --------------------------------------
def load_data():
    try:
        preprocessed_path = r"C:\Users\acvsa\PhishingDetector\data\preprocessed\Enron_preprocessed.csv"
        logger.info(f"Loading data from: {preprocessed_path}")
        
        if not os.path.exists(preprocessed_path):
            logger.error("Preprocessed data file not found!")
            sys.exit(1)
            
        df = pd.read_csv(preprocessed_path)
        if df.empty:
            logger.error("Loaded DataFrame is empty!")
            sys.exit(1)
            
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}", exc_info=True)
        sys.exit(1)

# --------------------------------------
# Model Development and XAI Components
# --------------------------------------
class PhishingClassifier:
    def __init__(self, model_type='logistic_regression'):
        self.model_type = model_type
        self.model = self._initialize_model()
        self.explainers = None
        self.feature_names = None
        self.tfidf = None

    def _initialize_model(self):
        """Initialize the appropriate model architecture"""
        if self.model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=2000, 
                class_weight='balanced',
                verbose=1
            )
        elif self.model_type == 'decision_tree':
            return DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'hybrid_nn':
            return self._create_hybrid_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _create_hybrid_model(self):
        """Create hybrid neural network architecture"""
        text_input = Input(shape=(1000,), name='text_features')
        structural_input = Input(shape=(4,), name='structural_features')
        
        x = Dense(128, activation='relu')(text_input)
        x = Dropout(0.5)(x)
        y = Dense(32, activation='relu')(structural_input)
        
        combined = concatenate([x, y])
        z = Dense(64, activation='relu')(combined)
        output = Dense(1, activation='sigmoid')(z)
        
        model = Model(inputs=[text_input, structural_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def init_explainers(self, X_train=None, feature_names=None):
        """Initialize explainers after model is trained"""
        try:
            self.feature_names = feature_names
            if self.model_type == 'hybrid_nn':
                return
                
            # Convert DataFrame to numpy array for SHAP
            shap_data = X_train.values if X_train is not None else None
                
            self.explainers = {
                'shap': shap.Explainer(self.model.predict, shap_data) if X_train is not None else None,
                'lime': lime_text.LimeTextExplainer(class_names=['legit', 'phishing']),
                'anchor': AnchorText(
                    predictor=lambda x: self.model.predict_proba(self._text_to_features(x)),
                    sampling_strategy='unknown'
                ) if self.model_type == 'logistic_regression' else None
            }
        except Exception as e:
            logger.error(f"Explainer init failed: {str(e)}", exc_info=True)

    def _text_to_features(self, texts):
        """Convert raw text to model input format"""
        tokenized = [text.split() for text in texts]
        return pd.concat([
            pd.DataFrame([[0]*4], columns=self.feature_names[:4]),
            pd.DataFrame(self.tfidf.transform(tokenized).toarray())
        ], axis=1)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        if self.model_type == 'hybrid_nn':
            history = self.model.fit(
                {'text_features': X_train[0], 'structural_features': X_train[1]},
                y_train,
                epochs=10,
                validation_data=(X_val, y_val) if X_val else None,
                verbose=1
            )
            return history
        else:
            self.model.fit(X_train, y_train)
        return None

    def explain(self, instance, text=None):
        """Generate explanations for predictions"""
        if self.explainers is None:
            logger.warning("Explainers not initialized")
            return {}
            
        explanations = {}
        try:
            if self.explainers['shap']:
                explanations['shap'] = self.explainers['shap'](instance)
            
            if text and self.explainers['lime']:
                exp = self.explainers['lime'].explain_instance(
                    text, lambda x: self.model.predict_proba(self._text_to_features(x)), 
                    num_features=10
                )
                explanations['lime'] = exp
                
            if text and self.explainers['anchor']:
                explanations['anchor'] = self.explainers['anchor'].explain(text)
        except Exception as e:
            logger.error(f"Explanation failed: {str(e)}")
        return explanations

# --------------------------------------
# Training and Evaluation
# --------------------------------------
def train_model(df):
    try:
        logger.info("Extracting features...")
        
        # Validate columns
        required_columns = ['label', 'tokenized_text', 'num_links', 
                           'has_suspicious_url', 'urgency_count', 'readability_score']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Convert tokenized_text from string to list
        df['tokenized_text'] = df['tokenized_text'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

        # Verify tokenization
        empty_tokens = df['tokenized_text'].apply(len) == 0
        if empty_tokens.any():
            logger.warning(f"Dropping {empty_tokens.sum()} rows with empty tokens")
            df = df[~empty_tokens]

        structural_features = df[['num_links', 'has_suspicious_url', 
                                 'urgency_count', 'readability_score']]
        
        # TF-IDF Vectorization
        logger.info("Vectorizing text...")
        tfidf = TfidfVectorizer(
            max_features=1000,
            analyzer=identity_analyzer,
            min_df=2,
            max_df=0.95
        )
        text_features = tfidf.fit_transform(df['tokenized_text'])
        
        # Handle empty vocabulary
        if len(tfidf.vocabulary_) == 0:
            raise ValueError("Empty vocabulary - check tokenization and filtering!")

        # Prepare features
        X_text = pd.DataFrame(text_features.toarray())
        X_structural = structural_features.values
        X = pd.concat([structural_features.reset_index(drop=True), X_text], axis=1)
        X.columns = X.columns.astype(str)
        y = df['label'].values

        # Split data
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train models
        logger.info("Training models...")
        results = {}
        
        # Logistic Regression
        logger.info("Training Logistic Regression...")
        lr = PhishingClassifier('logistic_regression')
        lr.tfidf = tfidf
        lr.train(X_train, y_train)
        lr.init_explainers(X_train, feature_names=X.columns.tolist())
        results['logistic_regression'] = evaluate_model(lr.model, X_test, y_test)

        # Decision Tree
        logger.info("Training Decision Tree...")
        dt = PhishingClassifier('decision_tree')
        dt.train(X_train, y_train)
        dt.init_explainers(X_train)
        results['decision_tree'] = evaluate_model(dt.model, X_test, y_test)

        # Hybrid NN
        logger.info("Training Hybrid Neural Network...")
        X_train_text = X_train.iloc[:, 4:].values
        X_test_text = X_test.iloc[:, 4:].values
        X_train_struct = X_train.iloc[:, :4].values
        X_test_struct = X_test.iloc[:, :4].values
        
        hybrid = PhishingClassifier('hybrid_nn')
        history = hybrid.train(
            (X_train_text, X_train_struct), y_train,
            (X_test_text, X_test_struct), y_test
        )
        results['hybrid_nn'] = hybrid.model.evaluate(
            [X_test_text, X_test_struct], y_test, verbose=0
        )

        # Generate explanations
        sample_idx = 0
        try:
            explanations = lr.explain(X_test.iloc[[sample_idx]], 
                                    df.iloc[X_test.index[sample_idx]].get('original_text', ''))
        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            explanations = {}

        return {
            'models': [lr, dt, hybrid],
            'results': results,
            'explanations': explanations,
            'tfidf': tfidf
        }
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

# --------------------------------------
# Visualization and Saving
# --------------------------------------
def save_artifacts(training_results):
    try:
        model_dir = r"C:\Users\acvsa\PhishingDetector\models"
        os.makedirs(model_dir, exist_ok=True)

        # Save models
        for idx, model_wrapper in enumerate(training_results['models']):
            model_path = os.path.join(model_dir, f"model_{idx}_{model_wrapper.model_type}.pkl")
            if model_wrapper.model_type == 'hybrid_nn':
                model_wrapper.model.save(model_path.replace('.pkl', '.keras'))  # Use modern format
            else:
                joblib.dump(model_wrapper.model, model_path)

        # Save TF-IDF vectorizer
        joblib.dump(training_results['tfidf'], os.path.join(model_dir, "tfidf_vectorizer.pkl"))

        # Save visualizations
        save_visualizations(training_results['explanations'], model_dir)

        # Save metrics
        with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
            json.dump(training_results['results'], f)

        logger.info("All artifacts saved successfully")
    except Exception as e:
        logger.error(f"Failed to save artifacts: {str(e)}", exc_info=True)
        sys.exit(1)

def save_visualizations(explanations, save_dir):
    """Save explanation visualizations with error handling"""
    try:
        # SHAP plot
        if explanations.get('shap') is not None:
            plt.figure()
            shap.plots.waterfall(explanations['shap'][0])
            plt.savefig(os.path.join(save_dir, 'shap_explanation.png'))
            plt.close()
        else:
            logger.warning("No SHAP explanations to save")

        # LIME plot
        if explanations.get('lime') is not None:
            explanations['lime'].save_to_file(os.path.join(save_dir, 'lime_explanation.html'))
        else:
            logger.warning("No LIME explanations to save")

        # Anchor explanations
        if explanations.get('anchor') is not None:
            with open(os.path.join(save_dir, 'anchor_explanation.txt'), 'w') as f:
                f.write(str(explanations['anchor'].anchor))
        else:
            logger.warning("No Anchor explanations to save")
    except Exception as e:
        logger.error(f"Failed to save visualizations: {str(e)}", exc_info=True)

# --------------------------------------
# Main Execution
# --------------------------------------
if __name__ == "__main__":
    try:
        logger.info("Starting training pipeline...")
        df = load_data()
        training_results = train_model(df)
        save_artifacts(training_results)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}", exc_info=True)
        sys.exit(1)
