# File: train_model.py

import warnings
import os

# Enhanced warning suppression
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="alibi|tf_keras|tensorflow")
warnings.filterwarnings("ignore", message=".*tf.Session.*")
warnings.filterwarnings("ignore", message=".*tf.losses.sparse_softmax_cross_entropy.*")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_SAVE_FORMAT'] = 'keras'  # UPDATED: Changed from 'h5' to 'keras'

# Use TF 2.x behavior
import tensorflow as tf

# TensorFlow/Keras imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate

# Other imports
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
from sklearn.metrics import classification_report, accuracy_score, f1_score  # Added f1_score import

# Import spaCy properly
import spacy

# Custom analyzer function for pickle compatibility
def identity_analyzer(x):
    """Return input as-is for pre-tokenized data"""
    return x

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler(r"C:\Users\acvsa\Desktop\Phising-tool-XAI-main\training.log")
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
        preprocessed_path = r"C:\Users\acvsa\Desktop\Phising-tool-XAI-main\data\preprocessed\Enron_preprocessed.csv"
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

# Helper function to inspect Keras model inputs
def inspect_model_inputs(model):
    """Helper function to inspect Keras model inputs"""
    # Get input layer information
    input_info = []
    for layer in model.layers:
        if isinstance(layer, Input) or layer.name in ['text_features', 'structural_features']:
            input_info.append({
                'name': layer.name,
                'shape': layer.input_shape if hasattr(layer, 'input_shape') else 'Unknown',
                'dtype': layer.dtype
            })
    
    # Get each layer's expected input shape
    layer_info = []
    for layer in model.layers:
        layer_info.append({
            'name': layer.name,
            'expected_input_shape': layer.input_shape if hasattr(layer, 'input_shape') else 'Unknown',
            'output_shape': layer.output_shape if hasattr(layer, 'output_shape') else 'Unknown',
        })
    
    return {'inputs': input_info, 'layers': layer_info}

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
        # Initialize spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("SpaCy model loaded successfully")
        except OSError:
            logger.warning("SpaCy model 'en_core_web_sm' not found. Some explainers may not work.")
            logger.warning("Try installing it with: python -m spacy download en_core_web_sm")
            self.nlp = None

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
        """Create hybrid neural network architecture with clear input/output definitions"""
        # Define inputs with correct shapes
        text_features_input = Input(shape=(1000,), name='text_features')
        structural_features_input = Input(shape=(4,), name='structural_features')
        
        # Process text features 
        text_branch = Dense(128, activation='relu', name='text_dense')(text_features_input)
        text_branch = Dropout(0.5, name='text_dropout')(text_branch)
        
        # Process structural features
        struct_branch = Dense(32, activation='relu', name='struct_dense')(structural_features_input)
        
        # Combine features
        combined = concatenate([text_branch, struct_branch], name='combined_features')
        hidden = Dense(64, activation='relu', name='hidden_layer')(combined)
        output = Dense(1, activation='sigmoid', name='output')(hidden)
        
        # Make sure to properly define the model with the correct inputs
        model = Model(
            inputs={
                'text_features': text_features_input,
                'structural_features': structural_features_input
            },
            outputs=output
        )
        
        # Debugging: print model summary
        model.summary()
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def init_explainers(self, X_train=None, feature_names=None):
        """Initialize explainers after model is trained"""
        try:
            self.feature_names = feature_names
            if self.model_type == 'hybrid_nn':
                return
            
            # FIXED: Calculate appropriate max_evals for SHAP based on feature count
            max_features = X_train.shape[1] if X_train is not None else 0
            min_evals = 2 * max_features + 1
            max_evals = max(2000, min_evals)  # Ensure enough evaluations for SHAP
                
            # Convert DataFrame to numpy array for SHAP
            shap_data = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
                
            self.explainers = {
                # FIXED: Add max_evals parameter to ensure enough evaluations
                'shap': shap.Explainer(self.model.predict, shap_data, max_evals=max_evals) if X_train is not None else None,
                'lime': lime_text.LimeTextExplainer(class_names=['legit', 'phishing'])
            }
            
            # Only add Anchor explainer if spaCy model is available
            if self.model_type == 'logistic_regression' and self.nlp is not None:
                self.explainers['anchor'] = AnchorText(
                    predictor=lambda x: self.model.predict_proba(self._text_to_features(x)),
                    sampling_strategy='unknown',
                    nlp=self.nlp
                )
            
        except Exception as e:
            logger.error(f"Explainer init failed: {str(e)}", exc_info=True)

    def _text_to_features(self, texts):
        """Convert raw text to model input format"""
        if not hasattr(self, 'tfidf') or self.tfidf is None:
            logger.error("TF-IDF vectorizer not initialized")
            return None
            
        tokenized = [text.split() for text in texts]
        
        # Create structural features with string column names
        structural_df = pd.DataFrame([[0]*4], columns=self.feature_names[:4] if self.feature_names else ["f1", "f2", "f3", "f4"])
        
        # Create TF-IDF features with string column names
        tfidf_features = self.tfidf.transform(tokenized)
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(), 
            columns=[str(i) for i in range(tfidf_features.shape[1])]
        )
        
        return pd.concat([structural_df, tfidf_df], axis=1)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model with proper input mapping"""
        if self.model_type == 'hybrid_nn':
            # Debug the input shapes
            logger.info(f"X_train[0] shape: {X_train[0].shape}")
            logger.info(f"X_train[1] shape: {X_train[1].shape}")
            
            # Create inputs for the model with explicit mapping
            train_data = {
                'text_features': X_train[0],          # Text features (1000 dims)
                'structural_features': X_train[1]     # Structural features (4 dims)
            }
            
            # Create validation data in the same format if provided
            val_data = None
            if X_val is not None:
                val_data = (
                    {
                        'text_features': X_val[0],
                        'structural_features': X_val[1]
                    },
                    y_val
                )
            
            # Train the model
            history = self.model.fit(
                train_data,
                y_train,
                epochs=10,
                batch_size=64,
                validation_data=val_data,
                verbose=1
            )
            return history
        else:
            # For non-neural network models
            if isinstance(X_train, pd.DataFrame):
                # FIXED: Store feature names before converting to numpy array
                self.feature_names = X_train.columns.tolist() 
                X_train_values = X_train.values
            else:
                X_train_values = X_train
                
            self.model.fit(X_train_values, y_train)
            return None
        
    def explain(self, instance, text=None):
        """Generate explanations for predictions"""
        if self.explainers is None:
            logger.warning("Explainers not initialized")
            return {}
            
        # Convert DataFrame to numpy array if needed
        if isinstance(instance, pd.DataFrame):
            instance_data = instance.values
        else:
            instance_data = instance

        explanations = {}
        try:
            # FIXED: Add error handling for each explainer separately
            if 'shap' in self.explainers and self.explainers['shap'] is not None:
                try:
                    explanations['shap'] = self.explainers['shap'](instance_data)
                    logger.info("SHAP explanation generated successfully")
                except Exception as e:
                    logger.error(f"SHAP explanation failed: {str(e)}")
            
            if text and 'lime' in self.explainers and self.explainers['lime'] is not None:
                try:
                    exp = self.explainers['lime'].explain_instance(
                        text, lambda x: self.model.predict_proba(self._text_to_features(x)), 
                        num_features=10
                    )
                    explanations['lime'] = exp
                    logger.info("LIME explanation generated successfully")
                except Exception as e:
                    logger.error(f"LIME explanation failed: {str(e)}")
                
            if text and 'anchor' in self.explainers and self.explainers['anchor'] is not None:
                try:
                    explanations['anchor'] = self.explainers['anchor'].explain(text)
                    logger.info("Anchor explanation generated successfully")
                except Exception as e:
                    logger.error(f"Anchor explanation failed: {str(e)}")
                    
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
        X_text = pd.DataFrame(
            text_features.toarray(), 
            columns=[str(i) for i in range(1000)]  # Ensure 1000 string columns
        )
        X_structural = structural_features.reset_index(drop=True)  # Keep as DataFrame
        
        # Combine features
        X = pd.concat([X_structural, X_text], axis=1)
        X.columns = X.columns.astype(str)
        y = df['label'].values

        # Split data more directly
        logger.info("Splitting data...")
        train_indices, test_indices = train_test_split(
            range(len(df)), test_size=0.2, random_state=42, stratify=y
        )
        
        # Select train/test data
        X_train = X.iloc[train_indices].reset_index(drop=True)
        X_test = X.iloc[test_indices].reset_index(drop=True)
        y_train = y[train_indices]
        y_test = y[test_indices]

        # Prepare data for hybrid model
        # Ensure data types are correct and shapes match expected inputs
        X_train_text = X_text.iloc[train_indices].values.astype('float32')
        X_train_struct = X_structural.iloc[train_indices].values.astype('float32')
        X_test_text = X_text.iloc[test_indices].values.astype('float32')
        X_test_struct = X_structural.iloc[test_indices].values.astype('float32')

        # Debug prints
        logger.info(f"Text features shape: {X_train_text.shape}")
        logger.info(f"Structural features shape: {X_train_struct.shape}")

        # Verify the shapes match the expected input
        assert X_train_text.shape[1] == 1000, f"Text features must have 1000 features, got {X_train_text.shape[1]}"
        assert X_train_struct.shape[1] == 4, f"Structural features must have 4 features, got {X_train_struct.shape[1]}"

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
        dt.tfidf = tfidf  # FIXED: Set tfidf for all models
        dt.train(X_train, y_train)
        dt.init_explainers(X_train, feature_names=X.columns.tolist())
        results['decision_tree'] = evaluate_model(dt.model, X_test, y_test)

        # Hybrid NN
        logger.info("Training Hybrid Neural Network...")
        hybrid = PhishingClassifier('hybrid_nn')
        hybrid.tfidf = tfidf  # FIXED: Set tfidf for all models        
                
        # Make sure the shapes are correct (text_features, structural_features)
        # X_train_text shape should be (n_samples, 1000)
        # X_train_struct shape should be (n_samples, 4)
        hybrid_train_data = (X_train_text, X_train_struct)
        hybrid_test_data = (X_test_text, X_test_struct)

        # Debug print to verify shapes
        logger.info(f"Hybrid train text shape: {X_train_text.shape}")
        logger.info(f"Hybrid train structural shape: {X_train_struct.shape}")

        history = hybrid.train(
            hybrid_train_data,  # Pass as tuple of (text_features, structural_features)
            y_train,
            hybrid_test_data,   # Pass as tuple of (text_features, structural_features)
            y_test
        )       

        # Handle history results more safely
        if history is not None and hasattr(history, 'history'):
            # Calculate F1 score for hybrid neural network on test data
            # First, get predictions (need to convert from probabilities to binary predictions)
            y_pred_proba = hybrid.model.predict({
                'text_features': X_test_text,
                'structural_features': X_test_struct
            })
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions
            
            # Calculate F1 score for hybrid neural network
            f1 = f1_score(y_test, y_pred)
            
            results['hybrid_nn'] = {
                'loss': history.history.get('loss', [0])[-1],
                'accuracy': history.history.get('accuracy', [0])[-1],
                'val_loss': history.history.get('val_loss', [0])[-1],
                'val_accuracy': history.history.get('val_accuracy', [0])[-1],
                'f1_score': float(f1)  # Added F1 score
            }
            
            # Log F1 score for hybrid model
            logger.info(f"Hybrid Neural Network F1 Score: {f1:.4f}")
        else:
            logger.warning("No training history for hybrid model")
            results['hybrid_nn'] = {
                'loss': None,
                'accuracy': None,
                'val_loss': None,
                'val_accuracy': None,
                'f1_score': None  # Added F1 score placeholder
            }

        # Generate explanations
        sample_idx = 0
        try:
            logger.info("Generating explanations for a sample instance...")
            # FIXED: Try with a smaller feature set for explanations if needed
            sample_instance = X_test.iloc[[sample_idx]]
            sample_text = df.iloc[test_indices[sample_idx]].get('original_text', '')
            
            # Try to get explanations from multiple models for better chances of success
            for model_name, model_obj in [("Logistic Regression", lr), ("Decision Tree", dt)]:
                try:
                    logger.info(f"Attempting explanation generation with {model_name}...")
                    explanations = model_obj.explain(sample_instance, sample_text)
                    if explanations and any(explanations.values()):
                        logger.info(f"Successfully generated explanations with {model_name}")
                        break
                except Exception as e:
                    logger.error(f"Explanation with {model_name} failed: {str(e)}")
            else:
                explanations = {}  # If all attempts fail
                
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
    # Ensure X_test is in the right format
    if isinstance(X_test, pd.DataFrame):
        X_test_values = X_test.values
    else:
        X_test_values = X_test
        
    y_pred = model.predict(X_test_values)
    
    # Calculate F1 score (added)
    f1 = f1_score(y_test, y_pred)
    logger.info(f"Model F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': float(f1),  # Added F1 score
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

# --------------------------------------
# Visualization and Saving
# --------------------------------------
def save_artifacts(training_results):
    try:
        model_dir = r"C:\Users\acvsa\Desktop\Phising-tool-XAI-main\models"
        os.makedirs(model_dir, exist_ok=True)

        # Save models
        for idx, model_wrapper in enumerate(training_results['models']):
            if model_wrapper.model_type == 'hybrid_nn':
                # FIXED: Use .keras extension and format
                model_path = os.path.join(model_dir, f"model_{idx}_{model_wrapper.model_type}.keras")
                model_wrapper.model.save(model_path)
                logger.info(f"Saved hybrid model to {model_path}")
            else:
                model_path = os.path.join(model_dir, f"model_{idx}_{model_wrapper.model_type}.pkl")
                joblib.dump(model_wrapper.model, model_path)
                logger.info(f"Saved {model_wrapper.model_type} model to {model_path}")

        # Save TF-IDF vectorizer
        tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        joblib.dump(training_results['tfidf'], tfidf_path)
        logger.info(f"Saved TF-IDF vectorizer to {tfidf_path}")

        # Save visualizations with improved error handling
        save_visualizations(training_results['explanations'], model_dir)

        # Save metrics
        metrics_path = os.path.join(model_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(training_results['results'], f, indent=2)
            logger.info(f"Saved metrics to {metrics_path}")

        # Save F1 scores separately for quick reference (added)
        f1_scores = {
            model_name: results.get('f1_score', 'N/A') 
            for model_name, results in training_results['results'].items()
        }
        f1_path = os.path.join(model_dir, 'f1_scores.json')
        with open(f1_path, 'w') as f:
            json.dump(f1_scores, f, indent=2)
            logger.info(f"Saved F1 scores to {f1_path}")

        logger.info("All artifacts saved successfully")
    except Exception as e:
        logger.error(f"Failed to save artifacts: {str(e)}", exc_info=True)
        sys.exit(1)

def save_visualizations(explanations, save_dir):
    """Save explanation visualizations with better error handling"""
    viz_saved = False
    
    try:
        # SHAP plot
        if explanations.get('shap') is not None:
            try:
                plt.figure(figsize=(10, 6))
                shap_path = os.path.join(save_dir, 'shap_explanation.png')
                shap.plots.waterfall(explanations['shap'][0])
                plt.tight_layout()
                plt.savefig(shap_path, dpi=300)
                plt.close()
                logger.info(f"Saved SHAP visualization to {shap_path}")
                viz_saved = True
            except Exception as e:
                logger.error(f"Failed to save SHAP visualization: {str(e)}")
        else:
            logger.warning("No SHAP explanations to save")

        # LIME plot
        if explanations.get('lime') is not None:
            try:
                lime_path = os.path.join(save_dir, 'lime_explanation.html')
                explanations['lime'].save_to_file(lime_path)
                logger.info(f"Saved LIME visualization to {lime_path}")
                viz_saved = True
            except Exception as e:
                logger.error(f"Failed to save LIME visualization: {str(e)}")
        else:
            logger.warning("No LIME explanations to save")

        # Anchor explanations
        if explanations.get('anchor') is not None:
            try:
                anchor_path = os.path.join(save_dir, 'anchor_explanation.txt')
                with open(anchor_path, 'w') as f:
                    f.write(str(explanations['anchor'].anchor))
                logger.info(f"Saved Anchor explanation to {anchor_path}")
                viz_saved = True
            except Exception as e:
                logger.error(f"Failed to save Anchor explanation: {str(e)}")
        else:
            logger.warning("No Anchor explanations to save")
            
        # If no visualizations were saved, create a placeholder
        if not viz_saved:
            placeholder_path = os.path.join(save_dir, 'no_explanations.txt')
            with open(placeholder_path, 'w') as f:
                f.write("No explanations were generated during this run. Try again with a different model or adjust parameters.")
            logger.info(f"Created placeholder explanation file at {placeholder_path}")
            
    except Exception as e:
        logger.error(f"Failed to save visualizations: {str(e)}", exc_info=True)

# --------------------------------------
# F1 Score Visualization (Added)
# --------------------------------------
def save_f1_visualization(training_results, save_dir):
    """Create and save a bar chart of F1 scores for all models"""
    try:
        # Extract F1 scores for each model
        models = []
        f1_scores = []
        
        for model_name, results in training_results['results'].items():
            if results.get('f1_score') is not None:
                models.append(model_name)
                f1_scores.append(results['f1_score'])
        
        if not models:
            logger.warning("No F1 scores available to visualize")
            return
            
        # Create bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, f1_scores, color=['blue', 'green', 'orange'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.title('F1 Score Comparison Across Models')
        plt.xlabel('Model')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1.1)  # F1 score range is 0-1
        plt.tight_layout()
        
        # Save the figure
        f1_viz_path = os.path.join(save_dir, 'f1_score_comparison.png')
        plt.savefig(f1_viz_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved F1 score visualization to {f1_viz_path}")
    except Exception as e:
        logger.error(f"Failed to create F1 score visualization: {str(e)}")

# --------------------------------------
# Main Execution
# --------------------------------------
if __name__ == "__main__":
    try:
        logger.info("Starting training pipeline...")
        df = load_data()
        training_results = train_model(df)
        save_artifacts(training_results)
        
        # Add F1 score visualization
        save_f1_visualization(training_results, r"C:\Users\acvsa\Desktop\Phising-tool-XAI-main\models")
        
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}", exc_info=True)
        sys.exit(1)
