# Combined Preprocessing, Training, and Explanation Script

import os
import re
import argparse
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings
import nltk
from nltk.corpus import stopwords
import textstat
import logging
from logging.handlers import RotatingFileHandler
import sys
import json
import joblib
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from scipy.sparse import issparse # Needed for SHAP data check

# --- Scikit-learn ---
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# --- XAI ---
import shap
from lime import lime_text
# from alibi.explainers import AnchorText # Requires spacy model, can be added back if needed

# --- TensorFlow / Keras (Optional, for Hybrid NN) ---
# Import only if needed to avoid dependency issues if TF is not installed
try:
    import tensorflow as tf
    # Explicitly check TF version if needed, though not strictly necessary for this fix
    # from packaging import version
    # if version.parse(tf.__version__) < version.parse("2.0"):
    #     raise ImportError("TensorFlow version 2.0 or higher is required.")
    from tensorflow.keras.models import Model as KerasModel
    from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    KerasModel = None # Define as None if TF not available
    print("Warning: TensorFlow not found. Hybrid Neural Network model will not be available.")


# --------------------------------------
# Global Configuration & Setup
# --------------------------------------
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure environment variables for TF if needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF logs

# --- Logging Setup Function ---
def setup_logging(log_dir):
    """Sets up console and file logging."""
    # Use Root Logger - configure it directly
    # Avoid potential issues with __name__ if script is imported elsewhere later
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if function is called again or in notebooks
    if logger.hasHandlers():
        # Remove existing handlers before adding new ones
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        os.makedirs(log_dir, exist_ok=True) # Ensure log directory exists
        log_file_path = os.path.join(log_dir, "phishing_detection_pipeline.log")
        # Use RotatingFileHandler
        file_handler = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
        # Log confirmation message to the logger itself
        logger.info(f"Logging configured. Log file: {log_file_path}")
    except Exception as e:
        # Log error using the logger (which should at least have the console handler)
        logger.error(f"Failed to set up file logger at {log_dir}: {e}", exc_info=True)

    return logger # Return the configured logger

# Initialize logger temporarily - will be reconfigured in main
# This ensures logging works even before main() is called
logger = logging.getLogger()
if not logger.handlers:
     logger.addHandler(logging.StreamHandler(sys.stdout))
     logger.setLevel(logging.INFO)


# --- NLTK Setup ---
# Ensure NLTK data path is known (optional, usually not needed unless custom path)
# nltk.data.path.append('/custom/path/to/nltk_data')
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt') # Check for punkt as well
    logger.debug("NLTK stopwords and punkt tokenizer found.")
except nltk.downloader.DownloadError as e:
    logger.warning(f"NLTK resource missing ({e}). Downloading necessary NLTK data...")
    if 'corpora/stopwords' in str(e) or not nltk.data.find('corpora/stopwords', quiet=True):
         logger.info("Downloading NLTK stopwords...")
         nltk.download('stopwords', quiet=True)
    if 'tokenizers/punkt' in str(e) or not nltk.data.find('tokenizers/punkt', quiet=True):
         logger.info("Downloading NLTK punkt tokenizer...")
         nltk.download('punkt', quiet=True) # Punkt is needed by textstat
stop_words = set(stopwords.words('english'))


# --------------------------------------
# Preprocessing Functions
# --------------------------------------
def clean_email(text):
    """Clean email text by removing HTML tags and special characters."""
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return ""
    try:
        # Use 'lxml' for potentially faster parsing if installed, fallback to html.parser
        try:
            soup = BeautifulSoup(text, 'lxml')
        except: # Catch generic exception if lxml isn't installed or fails
            soup = BeautifulSoup(text, 'html.parser')
        cleaned = soup.get_text(separator=' ')
        cleaned = re.sub(r'https?://\S+|www\.\S+', ' URL ', cleaned) # Replace URLs
        cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?\'`]', ' ', cleaned) # Keep some punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    except Exception as e:
        # Log error but return empty string to avoid downstream issues
        logger.error(f"Error cleaning text snippet: '{str(text)[:50]}...': {e}", exc_info=False)
        return ""

def extract_url_features(text):
    """Extract URL-based features from the original text."""
    num_links = 0
    has_suspicious_url = 0
    if pd.isna(text) or not isinstance(text, str):
        return 0, 0

    try:
        url_pattern = r'(https?://\S+|www\.\S+)'
        # Limit findall search string length to prevent excessive memory usage
        text_snippet_for_findall = text[:50000] # Limit search to first 50k chars
        urls = re.findall(url_pattern, text_snippet_for_findall)
        # Get full count separately if needed, handle potential long strings
        try:
             num_links = len(re.findall(url_pattern, text))
        except Exception as count_e:
             logger.warning(f"Could not get full URL count due to error: {count_e}. Using count from snippet.")
             num_links = len(urls)


        if not urls:
            return num_links, has_suspicious_url

        suspicious_keywords = ['login', 'verify', 'account', 'secure', 'update', 'confirm', 'signin', 'support', 'password', 'banking', 'activity', 'credential']
        shortened_domains_pattern = r'(bit\.ly/|goo\.gl/|tinyurl\.com/|t\.co/|ow\.ly/|is\.gd/|buff\.ly/|adf\.ly/|bit\.do/|soo\.gd/)' # Expanded list
        has_http = 0
        has_shortener = 0
        has_keywords = 0

        for url in urls[:100]: # Check only first 100 found URLs for performance
            try: # Add inner try-except for robustness with weird URLs
                url_lower = url.lower()
                if url_lower.startswith('http://'):
                    has_http = 1
                if re.search(shortened_domains_pattern, url_lower):
                    has_shortener = 1
                # Check path and query parameters for keywords more carefully
                # Find domain end robustly
                proto_end = url_lower.find('//')
                if proto_end > 0:
                     # Find first slash after "http://" or "https://" part
                     domain_part_end = url_lower.find('/', proto_end + 2)
                     path_query = url_lower[domain_part_end:] if domain_part_end > 0 else '' # Check path/query only if slash exists
                else: # Handle URLs like www.example.com without http://
                     domain_part_end = url_lower.find('/')
                     path_query = url_lower[domain_part_end:] if domain_part_end > 0 else ''

                # Check keywords in path/query AND potentially the domain itself if not just TLD
                check_string = path_query + url_lower # Check whole URL now for keywords
                if any(keyword in check_string for keyword in suspicious_keywords):
                    has_keywords = 1

                if has_http or has_shortener or has_keywords:
                    has_suspicious_url = 1
                    break # Found one suspicious URL, no need to check further
            except Exception as url_parse_e:
                 logger.debug(f"Error parsing single URL '{url[:50]}...': {url_parse_e}")
                 continue


        return num_links, has_suspicious_url

    except Exception as e:
        logger.error(f"URL feature extraction failed for text snippet '{str(text)[:50]}...': {e}", exc_info=False)
        return 0, 0

def extract_urgency(cleaned_text):
    """Count urgency-related keywords in cleaned text."""
    if not cleaned_text: return 0
    try:
        # Expanded keywords
        urgency_words = ['urgent', 'immediately', 'action required', 'verify', 'password', 'alert',
                         'warning', 'limited time', 'expire', 'suspended', 'locked', 'important',
                         'final notice', 'response required', 'security update', 'confirm account',
                         'validate', 'due date', 'restricted', 'compromised', 'unauthorized']
        text_lower = cleaned_text.lower()
        # Use regex word boundaries for more accurate counting
        count = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower)) for word in urgency_words)
        return count
    except Exception as e:
        logger.error(f"Urgency calculation failed: {e}", exc_info=False)
        return 0

def calculate_readability(cleaned_text):
    """Calculate Flesch reading ease score."""
    # Check length before calculating
    word_count = len(cleaned_text.split())
    if not cleaned_text or word_count < 10: # Increased minimum word count slightly
        return 100.0 # Assign easy score if too short to calculate reliably
    try:
        # textstat requires punkt tokenizer data from nltk
        score = textstat.flesch_reading_ease(cleaned_text)
        # Clamp score to a reasonable range
        return max(-200, min(120, score)) if not np.isnan(score) else 50.0 # Allow slightly wider range, return neutral on NaN
    except Exception as e:
        # Avoid logging warnings for very short texts if check above missed something
        if word_count > 5:
             logger.debug(f"Readability calculation failed for text snippet '{cleaned_text[:50]}...': {e}", exc_info=False)
        return 50.0 # Return neutral score on failure

# --------------------------------------
# Data Loading and Preprocessing Pipeline
# --------------------------------------
def load_and_preprocess_data(data_dir, label_col_hints=None, text_col_hints=None):
    """Loads CSVs from a directory, preprocesses, and combines them."""
    if label_col_hints is None:
        label_col_hints = ['label', 'class', 'target', 'phishing', 'type'] # Added 'type'
    if text_col_hints is None:
        text_col_hints = ['text', 'body', 'email', 'content', 'message']

    all_dfs = []
    required_columns = ['original_text', 'cleaned_text', 'label', 'num_links',
                        'has_suspicious_url', 'urgency_count', 'readability_score']

    logger.info(f"Scanning directory for CSV files: {data_dir}")
    if not os.path.isdir(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    file_list = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    logger.info(f"Found {len(file_list)} CSV files to process.")

    for filename in file_list:
        input_path = os.path.join(data_dir, filename)
        logger.info(f"--- Processing file: {filename} ---")
        try:
            # Read CSV with encoding fallback and low_memory=False for large files
            try:
                df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decoding failed for {filename}. Trying latin-1...")
                df = pd.read_csv(input_path, encoding='latin1', low_memory=False)
            # Add specific error handling for common CSV issues
            except pd.errors.ParserError as pe:
                 logger.error(f"Parser error in {filename}: {pe}. Trying to read with on_bad_lines='warn'.")
                 try:
                      df = pd.read_csv(input_path, encoding='latin1', low_memory=False, on_bad_lines='warn') # More modern
                 except Exception as parse_fallback_e:
                      logger.error(f"Could not read {filename} even with fallback parsing ({parse_fallback_e}). Skipping file.")
                      continue
            except FileNotFoundError:
                 logger.error(f"File not found during processing: {input_path}. Skipping.")
                 continue
            except Exception as e:
                logger.error(f"Failed to read {filename}: {e}")
                continue

            if df.empty:
                 logger.warning(f"File {filename} is empty or failed to load data. Skipping.")
                 continue

            logger.info(f"Raw columns: {df.columns.tolist()}")
            df.columns = df.columns.str.strip() # Strip whitespace from column names

            # --- Column Detection ---
            text_col = next((col for col in df.columns if any(hint in col.lower() for hint in text_col_hints)), None)
            label_col = next((col for col in df.columns if any(hint in col.lower() for hint in label_col_hints)), None)

            if not text_col:
                logger.error(f"No text column found in {filename} based on hints: {text_col_hints}. Skipping file.")
                continue
            if not label_col:
                logger.error(f"No label column found in {filename} based on hints: {label_col_hints}. Skipping file.")
                continue

            logger.info(f"Identified Text Column: '{text_col}', Label Column: '{label_col}'")

            # Check for sufficient data
            if len(df) < 10:
                 logger.warning(f"File {filename} has very few rows ({len(df)}). Skipping.")
                 continue

            # Filter out rows with missing text data early
            df.dropna(subset=[text_col], inplace=True)
            if df.empty:
                 logger.warning(f"No rows remaining in {filename} after dropping missing text. Skipping.")
                 continue

            # Make a copy to avoid SettingWithCopyWarning
            df_processed = pd.DataFrame()
            # Ensure label and text columns are handled correctly even if numeric
            df_processed['original_text'] = df[text_col].astype(str)
            df_processed['label'] = df[label_col] # Keep original label for mapping later

            # --- Feature Engineering ---
            logger.info("Cleaning text...")
            df_processed['cleaned_text'] = df_processed['original_text'].apply(clean_email)

            logger.info("Extracting URL features...")
            url_features = df_processed['original_text'].apply(extract_url_features)
            df_processed['num_links'] = url_features.apply(lambda x: x[0]).astype(int) # Ensure int type
            df_processed['has_suspicious_url'] = url_features.apply(lambda x: x[1]).astype(int) # Ensure int type

            logger.info("Calculating urgency scores...")
            df_processed['urgency_count'] = df_processed['cleaned_text'].apply(extract_urgency).astype(int) # Ensure int type

            logger.info("Calculating readability scores...")
            df_processed['readability_score'] = df_processed['cleaned_text'].apply(calculate_readability).astype(float) # Ensure float type

            # --- Data Cleaning & Validation ---
            # Drop rows with empty cleaned text AFTER feature extraction
            original_rows = len(df_processed)
            df_processed.dropna(subset=['cleaned_text'], inplace=True) # Drop NaN first
            df_processed = df_processed[df_processed['cleaned_text'].str.strip().astype(bool)]
            if len(df_processed) < original_rows:
                logger.warning(f"Dropped {original_rows - len(df_processed)} rows with empty/NaN cleaned text.")

            if df_processed.empty:
                 logger.warning(f"No rows remaining in {filename} after cleaning text. Skipping.")
                 continue


            # Handle non-numeric labels AFTER processing steps
            # Expanded mapping
            label_map = {'spam': 1, 'phish': 1, 'phishing': 1, 'fraud': 1, 'scam': 1, 'malicious': 1, '1': 1, 1: 1, True: 1, 'true': 1,
                         'ham': 0, 'legitimate': 0, 'safe': 0, 'normal': 0, 'benign': 0, '0': 0, 0: 0, False: 0, 'false': 0}


            # Apply mapping robustly
            original_label_type = df_processed['label'].dtype
            try:
                 # Convert boolean-like explicitly first
                 if pd.api.types.is_bool_dtype(original_label_type):
                     df_processed['label'] = df_processed['label'].astype(int) # True->1, False->0
                 # Try converting to numeric directly if possible (handles '1', '0', 1.0, 0.0)
                 df_processed['label'] = pd.to_numeric(df_processed['label'], errors='coerce')
                 # Apply map only if direct numeric conversion wasn't enough or for string types
                 if df_processed['label'].isnull().any() or pd.api.types.is_object_dtype(original_label_type):
                      # Use original column again for mapping if direct conversion failed
                      df_processed['label'] = df[label_col].astype(str).str.strip().str.lower().map(label_map)

            except Exception as label_conv_e:
                 logger.error(f"Error converting label column in {filename}: {label_conv_e}. Attempting string map.")
                 try:
                      # Fallback to string mapping
                      df_processed['label'] = df[label_col].astype(str).str.strip().str.lower().map(label_map)
                 except Exception as fallback_e:
                      logger.error(f"Fallback label mapping failed for {filename}: {fallback_e}. Skipping file.")
                      continue


            # Drop rows where label conversion failed (became NaN) or labels are missing
            original_rows = len(df_processed)
            df_processed.dropna(subset=['label'], inplace=True)
            if len(df_processed) < original_rows:
                logger.warning(f"Dropped {original_rows - len(df_processed)} rows with missing/unmappable labels.")

            if df_processed.empty:
                 logger.warning(f"No rows remaining in {filename} after label processing. Skipping.")
                 continue

            # Ensure label column is integer
            df_processed['label'] = df_processed['label'].astype(int)

            # Final check for NaNs in feature columns before appending
            feature_cols_check = ['num_links', 'has_suspicious_url', 'urgency_count', 'readability_score']
            nan_in_features = df_processed[feature_cols_check].isnull().any().any()
            if nan_in_features:
                 nan_feature_rows = df_processed[feature_cols_check].isnull().any(axis=1).sum()
                 logger.warning(f"Found {nan_feature_rows} rows with NaN in feature columns in {filename}. Imputation will handle this during training.")
                 # Optional: Fill NaNs here if preferred over pipeline imputation
                 # df_processed[feature_cols_check] = df_processed[feature_cols_check].fillna(0)


            # Select and reorder columns
            present_required_columns = [col for col in required_columns if col in df_processed.columns]
            df_processed = df_processed[present_required_columns]

            logger.info(f"Finished processing {filename}. Valid rows added: {len(df_processed)}")
            all_dfs.append(df_processed)

        except Exception as e:
            logger.error(f"Critical error processing {filename}: {e}", exc_info=True)
            continue

    if not all_dfs:
        logger.error("No data could be loaded and processed from the specified directory.")
        sys.exit(1)

    # Combine all processed dataframes
    try:
        final_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"--- Combined all files. Total rows: {len(final_df)} ---")
        if final_df.empty:
             logger.error("Concatenated DataFrame is empty. Exiting.")
             sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to concatenate processed DataFrames: {e}", exc_info=True)
        sys.exit(1)


    # Final check for NaNs introduced by concatenation or processing errors
    numeric_cols_to_check = ['num_links', 'has_suspicious_url', 'urgency_count', 'readability_score']
    nan_rows = final_df[numeric_cols_to_check].isnull().any(axis=1).sum()
    if nan_rows > 0:
        logger.warning(f"Found {nan_rows} total rows with NaN in numeric columns after concatenation. Imputation will handle this.")

    # Check label counts after concatenation
    if 'label' in final_df.columns:
         logger.info(f"Final combined dataset shape: {final_df.shape}")
         # Calculate label distribution safely, handle potential non-numeric labels if any slipped through
         try:
             label_counts = final_df['label'].value_counts(normalize=True, dropna=False)
             logger.info(f"Label distribution:\n{label_counts}")
             if label_counts.get(0, 0) == 0 or label_counts.get(1, 0) == 0:
                  logger.warning("Final dataset contains only one class label. Model training might fail or be meaningless.")
         except Exception as label_count_e:
              logger.error(f"Could not calculate final label distribution: {label_count_e}")
    else:
         logger.error("Label column missing in final concatenated DataFrame. Exiting.")
         sys.exit(1)


    return final_df

# --------------------------------------
# Model Definition (Keras Hybrid NN)
# --------------------------------------
def build_hybrid_nn(text_input_dim, struct_input_dim, learning_rate=0.001):
    """Builds the Keras Hybrid Neural Network model."""
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available, cannot build Hybrid NN.")
        return None

    text_input = Input(shape=(text_input_dim,), name='text_features_input')
    struct_input = Input(shape=(struct_input_dim,), name='structural_features_input')

    # Text processing branch
    text_dense1 = Dense(128, activation='relu')(text_input)
    text_dropout = Dropout(0.4)(text_dense1) # Increased dropout slightly
    text_dense2 = Dense(64, activation='relu')(text_dropout)

    # Structural features branch
    struct_dense = Dense(32, activation='relu')(struct_input)

    # Combine branches
    combined = Concatenate()([text_dense2, struct_dense])

    # Output layers
    dense_combined = Dense(64, activation='relu')(combined)
    dropout_combined = Dropout(0.4)(dense_combined) # Increased dropout slightly
    output = Dense(1, activation='sigmoid', name='output')(dropout_combined)

    # Build model
    model = KerasModel(inputs=[text_input, struct_input], outputs=output)

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')] # Add more metrics
    )
    logger.info("Hybrid Neural Network Architecture:")
    model.summary(print_fn=logger.info)
    return model

# --------------------------------------
# XAI Helper Functions
# --------------------------------------
def predict_proba_for_lime(texts, pipeline):
    """
    Wrapper function for LIME. Takes raw text, runs through the *full* pipeline,
    and returns probabilities.
    """
    try:
        # LIME expects a list/array of texts
        text_col_name = None
        numeric_cols = []
        # Extract names from fitted transformer (safer)
        fitted_preprocessor = pipeline.named_steps['preprocessor']
        for name, _, cols in fitted_preprocessor.transformers_:
            if name == 'text':
                text_col_name = cols
            elif name == 'numeric':
                numeric_cols = cols

        if not text_col_name or not numeric_cols:
             # Try getting names from original definition if fitted fails (less ideal)
             preprocessor_def = pipeline.steps[0][1]
             for name, _, cols in preprocessor_def.transformers:
                 if name == 'text': text_col_name = cols
                 elif name == 'numeric': numeric_cols = cols
             if not text_col_name or not numeric_cols:
                  raise ValueError("Could not extract text/numeric column names from pipeline preprocessor.")

        data_for_pipeline = pd.DataFrame({
            text_col_name: texts,
            **{col: [0] * len(texts) for col in numeric_cols}
        })

        # Use the pipeline's predict_proba method
        probabilities = pipeline.predict_proba(data_for_pipeline)
        return probabilities
    except Exception as e:
        logger.error(f"Error in LIME predict_proba wrapper: {e}", exc_info=True)
        return np.array([[0.5, 0.5]] * len(texts))

def get_feature_names(column_transformer):
    """Gets feature names from a fitted ColumnTransformer. More robust version."""
    feature_names = []
    if column_transformer is None:
         logger.error("Preprocessor object is None, cannot get feature names.")
         return ["feature_error"]

    try:
        # Check if using verbose_feature_names_out=False (newer sklearn)
        # In this case, get_feature_names_out() on ColumnTransformer directly gives clean names
        if hasattr(column_transformer, 'get_feature_names_out') and \
           getattr(column_transformer, 'verbose_feature_names_out', True) is False:
             return list(column_transformer.get_feature_names_out())

        # Manual fallback for older sklearn or if above fails
        for name, transformer_obj, columns in column_transformer.transformers_:
            if name == 'remainder' and transformer_obj == 'drop':
                continue
            if transformer_obj == 'passthrough':
                 feature_names.extend(columns)
                 continue

            # Get names from the fitted transformer object
            current_feature_names = None
            if isinstance(transformer_obj, Pipeline):
                 # Try last step of pipeline
                 try:
                      last_step = transformer_obj.steps[-1][1]
                      if hasattr(last_step, 'get_feature_names_out'):
                           # Pass original column names 'columns' to inner transformer's get_feature_names_out
                           current_feature_names = last_step.get_feature_names_out(columns)
                      else: # Fallback to original pipeline input column names
                           current_feature_names = columns
                 except Exception:
                      current_feature_names = columns # Fallback
            elif hasattr(transformer_obj, 'get_feature_names_out'):
                 try:
                      # Pass original column names 'columns'
                      current_feature_names = transformer_obj.get_feature_names_out(columns)
                 except TypeError: # Some older versions might not accept 'columns'
                      try:
                           current_feature_names = transformer_obj.get_feature_names_out()
                      except Exception as e_inner:
                           logger.warning(f"get_feature_names_out failed for {name}: {e_inner}. Using input columns.")
                           current_feature_names = columns # Fallback
                 except Exception as e_outer:
                     logger.warning(f"get_feature_names_out failed for {name}: {e_outer}. Using input columns.")
                     current_feature_names = columns # Fallback
            else: # Transformer doesn't have get_feature_names_out
                 current_feature_names = columns

            # Add prefix manually if verbose_feature_names_out is not False (or unknown)
            # This mimics the older behavior somewhat, but without double underscores usually
            is_verbose = getattr(column_transformer, 'verbose_feature_names_out', True)
            if is_verbose and isinstance(current_feature_names, (list, np.ndarray)):
                 feature_names.extend([f"{name}__{fname}" for fname in current_feature_names])
            elif isinstance(current_feature_names, (list, np.ndarray)):
                 feature_names.extend(current_feature_names)
            elif isinstance(current_feature_names, str): # Handle single column case
                 feature_names.append(f"{name}__{current_feature_names}" if is_verbose else current_feature_names)


    except Exception as e:
        logger.error(f"Error getting feature names: {e}. Returning basic names.", exc_info=True)
        # Fallback if error occurs during name extraction
        try: # Attempt to estimate total features
             n_features = column_transformer.transform(pd.DataFrame(columns=column_transformer.feature_names_in_)).shape[1]
             return [f"feature_{i}" for i in range(n_features)]
        except:
             logger.warning("Could not estimate feature count for fallback names.")
             return ["feature_unknown"]

    # Ensure list contains only strings
    feature_names = [str(fn) for fn in feature_names]
    return feature_names


# --------------------------------------
# Training, Evaluation, and Explanation
# --------------------------------------
def train_evaluate_explain(df, text_col='cleaned_text', numeric_cols=None, label_col='label', output_dir='results'):
    """Trains models, evaluates, generates explanations, and saves artifacts."""

    if numeric_cols is None:
        numeric_cols = ['num_links', 'has_suspicious_url', 'urgency_count', 'readability_score']

    logger.info("--- Starting Model Training and Evaluation ---")
    logger.info(f"Using Text Column: '{text_col}', Numeric Columns: {numeric_cols}, Label Column: '{label_col}'")

    # Ensure required columns exist
    required_input_cols = [text_col, label_col] + numeric_cols
    # Add original_text if used later
    if 'original_text' not in df.columns and any('original_text' in col for col in required_input_cols):
         required_input_cols.append('original_text') # Make sure it exists if needed

    missing_cols = [col for col in required_input_cols if col not in df.columns]
    if missing_cols:
         logger.error(f"Input DataFrame is missing required columns: {missing_cols}. Exiting.")
         sys.exit(1)

    # Drop rows with NaNs in feature columns before splitting if not handled by SimpleImputer strategy effectively
    # df.dropna(subset=[text_col] + numeric_cols, inplace=True) # Optional stricter cleaning

    X = df[[text_col] + numeric_cols]
    y = df[label_col]
    # Ensure original_text is present before trying to access it
    original_texts = df['original_text'] if 'original_text' in df.columns else None

    if X.empty or y.empty:
         logger.error("Feature matrix X or target vector y is empty before splitting. Exiting.")
         sys.exit(1)


    # --- Train/Test Split ---
    logger.info("Splitting data into training and test sets (80/20)...")
    # Prepare arrays to split
    arrays_to_split = [X, y]
    if original_texts is not None:
         arrays_to_split.append(original_texts)
    arrays_to_split.append(df.index) # Always split the index

    try:
        # Check for sufficient samples per class for stratification
        min_class_count = y.value_counts().min()
        use_stratify = True
        if min_class_count < 2:
             logger.warning(f"The minority class has only {min_class_count} samples. Stratification might fail or be unreliable. Splitting without stratify.")
             use_stratify = False

        split_results = train_test_split(
                 *arrays_to_split, # Unpack arrays
                 test_size=0.2,
                 random_state=42,
                 stratify=y if use_stratify else None # Conditionally stratify
        )

        # Unpack results carefully based on whether original_texts was included
        num_outputs_per_set = (len(arrays_to_split))
        X_train = split_results[0]
        X_test = split_results[1]
        y_train = split_results[2]
        y_test = split_results[3]
        current_index = 4
        if original_texts is not None:
            original_texts_train = split_results[current_index]
            original_texts_test = split_results[current_index + 1]
            current_index += 2
        else:
            original_texts_train = None
            original_texts_test = None
        train_indices = split_results[current_index]
        test_indices = split_results[current_index + 1]


    except ValueError as e:
         if 'stratify' in str(e):
              logger.warning(f"Stratify failed despite checks: {e}. Performing split without stratification.")
              split_results = train_test_split(
                  *arrays_to_split, test_size=0.2, random_state=42
              )
              # Unpack again
              X_train, X_test, y_train, y_test = split_results[0], split_results[1], split_results[2], split_results[3]
              current_index = 4
              if original_texts is not None:
                  original_texts_train, original_texts_test = split_results[current_index], split_results[current_index + 1]
                  current_index += 2
              else: original_texts_train, original_texts_test = None, None
              train_indices, test_indices = split_results[current_index], split_results[current_index + 1]

         else:
              logger.error(f"Error during train/test split: {e}", exc_info=True)
              raise e # Reraise other errors

    if X_train.empty or X_test.empty:
         logger.error("Training or testing set is empty after split. Check data.")
         sys.exit(1)

    logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    logger.info(f"Train labels distribution:\n{y_train.value_counts(normalize=True, dropna=False)}")
    logger.info(f"Test labels distribution:\n{y_test.value_counts(normalize=True, dropna=False)}")


    # --- Define Preprocessing ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(stop_words=list(stop_words), max_features=2000, min_df=3, max_df=0.9), text_col),
            ('numeric', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=False # Avoids prefixes like 'text__', 'numeric__' in newer sklearn
    )
    logger.info("Preprocessor defined.")


    # --- Model Training and Evaluation Results ---
    models = {}
    results = {}
    explanations = {}
    fitted_preprocessor = None # Initialize

    # == 1. Logistic Regression ==
    model_name = "LogisticRegression"
    logger.info(f"\n--- Training {model_name} ---")
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', random_state=42, C=1.0)) # Added C
    ])

    try:
        logger.info(f"Fitting {model_name} pipeline...")
        lr_pipeline.fit(X_train, y_train)
        logger.info(f"Evaluating {model_name}...")
        y_pred_lr = lr_pipeline.predict(X_test)
        # Handle potential issues if only one class predicted
        report_lr = classification_report(y_test, y_pred_lr, output_dict=True, zero_division=0)
        results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred_lr),
            'f1_score': f1_score(y_test, y_pred_lr, average='binary', zero_division=0),
            'precision': precision_score(y_test, y_pred_lr, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred_lr, average='binary', zero_division=0),
            'report': report_lr
        }
        models[model_name] = lr_pipeline
        logger.info(f"{model_name} Training Complete. F1 Score: {results[model_name]['f1_score']:.4f}")
        if fitted_preprocessor is None:
            fitted_preprocessor = lr_pipeline.named_steps['preprocessor'] # Get fitted preprocessor
    except Exception as e:
        logger.error(f"Failed to train or evaluate {model_name}: {e}", exc_info=True)
        results[model_name] = {'error': str(e)}


    # == 2. Decision Tree ==
    model_name = "DecisionTree"
    logger.info(f"\n--- Training {model_name} ---")
    dt_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42, min_samples_split=10, min_samples_leaf=5)) # Added pruning params
    ])

    try:
        logger.info(f"Fitting {model_name} pipeline...")
        dt_pipeline.fit(X_train, y_train)
        logger.info(f"Evaluating {model_name}...")
        y_pred_dt = dt_pipeline.predict(X_test)
        report_dt = classification_report(y_test, y_pred_dt, output_dict=True, zero_division=0)
        results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred_dt),
            'f1_score': f1_score(y_test, y_pred_dt, average='binary', zero_division=0),
            'precision': precision_score(y_test, y_pred_dt, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred_dt, average='binary', zero_division=0),
            'report': report_dt
        }
        models[model_name] = dt_pipeline
        logger.info(f"{model_name} Training Complete. F1 Score: {results[model_name]['f1_score']:.4f}")
        if fitted_preprocessor is None: # Get fitted preprocessor if LR failed
            fitted_preprocessor = dt_pipeline.named_steps['preprocessor']
    except Exception as e:
        logger.error(f"Failed to train or evaluate {model_name}: {e}", exc_info=True)
        results[model_name] = {'error': str(e)}

    # == 3. Hybrid Neural Network (Optional) ==
    model_name = "HybridNN"
    if TF_AVAILABLE:
        logger.info(f"\n--- Training {model_name} ---")
        try:
            if fitted_preprocessor is None:
                 logger.info("Fitting preprocessor definition for Hybrid NN...")
                 # Need a standalone preprocessor instance to fit if others failed
                 temp_preprocessor = ColumnTransformer(
                    transformers=[
                        ('text', TfidfVectorizer(stop_words=list(stop_words), max_features=2000, min_df=3, max_df=0.9), text_col),
                        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols)
                    ],
                    remainder='drop', verbose_feature_names_out=False
                 )
                 # Fit on training data
                 try:
                      fitted_preprocessor = temp_preprocessor.fit(X_train)
                      logger.info("Temporarily fitted preprocessor for NN.")
                 except Exception as fit_e:
                      logger.error(f"Failed to fit temporary preprocessor: {fit_e}", exc_info=True)
                      raise # Reraise to stop NN training

            logger.info("Transforming data for Hybrid NN...")
            X_train_transformed = fitted_preprocessor.transform(X_train)
            X_test_transformed = fitted_preprocessor.transform(X_test)

            # Determine input dimensions after transformation
            try:
                 text_transformer = fitted_preprocessor.named_transformers_['text']
                 text_input_dim = len(text_transformer.get_feature_names_out())
                 # Get original numeric cols list length from transformer tuple
                 numeric_cols_list = fitted_preprocessor.transformers_[1][2]
                 struct_input_dim = len(numeric_cols_list)

                 logger.info(f"NN Input Dims - Text: {text_input_dim}, Structural: {struct_input_dim}")
                 # Check consistency
                 if X_train_transformed.shape[1] != (text_input_dim + struct_input_dim):
                      logger.warning(f"Transformed shape ({X_train_transformed.shape[1]}) doesn't match sum of calculated dims ({text_input_dim + struct_input_dim}). Check preprocessor definition vs. fitting.")
                      # Attempt to infer dims from shape if inconsistent
                      struct_input_dim = X_train_transformed.shape[1] - text_input_dim
                      logger.warning(f"Adjusted structural input dim to: {struct_input_dim}")


                 # Split transformed data for Keras model inputs
                 if issparse(X_train_transformed):
                     X_train_transformed_dense = X_train_transformed.toarray()
                     X_test_transformed_dense = X_test_transformed.toarray()
                 else:
                     X_train_transformed_dense = X_train_transformed
                     X_test_transformed_dense = X_test_transformed

                 # Ensure correct slicing
                 X_train_text_nn = X_train_transformed_dense[:, :text_input_dim].astype(np.float32)
                 X_train_struct_nn = X_train_transformed_dense[:, text_input_dim:].astype(np.float32)
                 X_test_text_nn = X_test_transformed_dense[:, :text_input_dim].astype(np.float32)
                 X_test_struct_nn = X_test_transformed_dense[:, text_input_dim:].astype(np.float32)

                 # Check shapes after slicing
                 if X_train_struct_nn.shape[1] != struct_input_dim:
                      logger.error(f"Structural feature slice incorrect: expected {struct_input_dim} features, got {X_train_struct_nn.shape[1]}.")
                      raise ValueError("Structural feature slicing mismatch.")


                 # Build and Train
                 keras_model = build_hybrid_nn(text_input_dim, struct_input_dim)
                 if keras_model:
                     logger.info("Starting Keras model training...")
                     history = keras_model.fit(
                         [X_train_text_nn, X_train_struct_nn], y_train,
                         epochs=15, batch_size=64, validation_split=0.15,
                         callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
                         verbose=1 # Set to 0 for less output, 1 for progress bar, 2 for one line per epoch
                     )
                     logger.info("Keras model training finished.")

                     logger.info(f"Evaluating {model_name}...")
                     loss, accuracy, precision_keras, recall_keras = keras_model.evaluate([X_test_text_nn, X_test_struct_nn], y_test, verbose=0)
                     y_pred_proba_nn = keras_model.predict([X_test_text_nn, X_test_struct_nn])
                     y_pred_nn = (y_pred_proba_nn > 0.5).astype(int).flatten()

                     # Use sklearn metrics on numpy arrays for consistency report
                     report_nn = classification_report(y_test, y_pred_nn, output_dict=True, zero_division=0)
                     f1_nn = f1_score(y_test, y_pred_nn, average='binary', zero_division=0)

                     results[model_name] = {
                         'accuracy': accuracy_score(y_test, y_pred_nn), # Sklearn accuracy
                         'f1_score': f1_nn,                             # Sklearn F1
                         'precision': precision_score(y_test, y_pred_nn, average='binary', zero_division=0), # Sklearn precision
                         'recall': recall_score(y_test, y_pred_nn, average='binary', zero_division=0),       # Sklearn recall
                         'report': report_nn,
                         'keras_test_loss': loss,
                         'keras_test_accuracy': accuracy,        # Keras eval accuracy
                         'keras_test_precision': precision_keras,# Keras eval precision
                         'keras_test_recall': recall_keras,      # Keras eval recall
                         'history': {k: [float(val) for val in v] for k, v in history.history.items()}
                     }
                     models[model_name] = keras_model
                     logger.info(f"{model_name} Training Complete. F1 Score: {results[model_name]['f1_score']:.4f}")

                 else:
                      logger.warning("Skipping Hybrid NN training as model building failed.")
                      results[model_name] = {'error': 'TensorFlow/Keras model building failed.'}
            except Exception as nn_dim_error:
                 logger.error(f"Error setting up dimensions or data for Hybrid NN: {nn_dim_error}", exc_info=True)
                 results[model_name] = {'error': f'NN data preparation error: {nn_dim_error}'}
        except Exception as e:
            logger.error(f"Failed to train or evaluate {model_name}: {e}", exc_info=True)
            results[model_name] = {'error': str(e)}
    else:
         logger.info("Skipping Hybrid NN training as TensorFlow is not available.")
         results[model_name] = {'error': 'TensorFlow not available.'}


    # --- XAI Explanation Generation ---
    logger.info("\n--- Generating Explanations for a Sample Instance ---")
    # Check conditions before proceeding
    if not any(isinstance(m, Pipeline) for m in models.values()):
        logger.warning("No successfully trained scikit-learn Pipeline models found. Skipping explanation generation.")
        return models, results, explanations, fitted_preprocessor
    if fitted_preprocessor is None:
        logger.error("Preprocessor was not fitted. Cannot generate explanations.")
        return models, results, explanations, fitted_preprocessor
    if X_test.empty:
         logger.error("Test set is empty, cannot generate explanations.")
         return models, results, explanations, fitted_preprocessor

    sample_idx_in_test = 0
    if sample_idx_in_test >= len(X_test):
        logger.warning("Test set smaller than sample index. Choosing index 0.")
        sample_idx_in_test = 0

    try:
        # FIX: Access Index object using integer position, not .iloc
        original_sample_index = test_indices[sample_idx_in_test]
        sample_raw_data = X_test.iloc[[sample_idx_in_test]]
        # Handle case where original_texts might be None
        sample_original_text = original_texts_test.iloc[sample_idx_in_test] if original_texts_test is not None else "Original text not available"
        sample_cleaned_text = sample_raw_data[text_col].iloc[0]
    except IndexError:
        logger.error(f"Cannot access sample index {sample_idx_in_test}. Test set size: {len(X_test)}")
        return models, results, explanations, fitted_preprocessor
    except Exception as sample_e:
         logger.error(f"Error getting sample data: {sample_e}", exc_info=True)
         return models, results, explanations, fitted_preprocessor


    logger.info(f"Explaining instance with original index: {original_sample_index}")
    logger.info(f"Sample Cleaned Text: '{sample_cleaned_text[:200]}...'")

    # Prepare background data for SHAP
    num_shap_background_samples = 100
    shap_background_data_transformed = None # Initialize
    if len(X_train) == 0:
         logger.warning("Training data is empty, cannot create SHAP background data.")
    else:
        if len(X_train) < num_shap_background_samples:
            num_shap_background_samples = len(X_train)
        try:
            shap_background_indices = np.random.choice(X_train.index, size=num_shap_background_samples, replace=False)
            shap_background_data = X_train.loc[shap_background_indices]
            shap_background_data_transformed = fitted_preprocessor.transform(shap_background_data)
            logger.info(f"Prepared SHAP background data with shape: {shap_background_data_transformed.shape}")
        except Exception as e:
            logger.error(f"Failed to transform SHAP background data: {e}", exc_info=True)
            # Keep shap_background_data_transformed as None


    # Iterate through models that are sklearn Pipelines
    for model_name, model_obj in models.items():
        # Skip non-pipeline models or failed models
        if not isinstance(model_obj, Pipeline) or model_name not in results or 'error' in results[model_name]:
            if model_name != "HybridNN": # Don't warn for NN if skipping is expected
                 logger.debug(f"Skipping explanations for {model_name} (not a valid Pipeline or failed training).")
            continue

        logger.info(f"--- Explaining with {model_name} ---")
        model_explanations = {}
        pipeline = model_obj # It's a pipeline

        # == SHAP ==
        if shap_background_data_transformed is None:
             logger.warning(f"Skipping SHAP for {model_name} (background data unavailable).")
             model_explanations['shap'] = {'error': 'Background data unavailable'}
        else:
             try:
                 logger.info("Initializing SHAP Explainer...")
                 classifier = pipeline.named_steps['classifier']
                 predict_proba_wrapper = lambda x: classifier.predict_proba(x) # Assumes x is transformed

                 transformed_instance = fitted_preprocessor.transform(sample_raw_data)

                 # Ensure background and instance are dense numpy for KernelExplainer
                 if issparse(shap_background_data_transformed):
                     shap_background_dense = shap_background_data_transformed.toarray()
                 else:
                     shap_background_dense = shap_background_data_transformed
                 if issparse(transformed_instance):
                      transformed_instance_dense = transformed_instance.toarray()
                 else:
                      transformed_instance_dense = transformed_instance

                 # Check for NaN/infs which can cause issues
                 if np.any(np.isnan(shap_background_dense)) or np.any(np.isinf(shap_background_dense)):
                     logger.warning("NaN or Inf found in SHAP background data. Attempting imputation...")
                     imputer_temp = SimpleImputer(strategy='median')
                     shap_background_dense = imputer_temp.fit_transform(shap_background_dense)

                 if np.any(np.isnan(transformed_instance_dense)) or np.any(np.isinf(transformed_instance_dense)):
                     logger.warning("NaN or Inf found in SHAP instance data. Attempting imputation...")
                     # Use previously fitted imputer if possible, or fit new one
                     try:
                          numeric_imputer = fitted_preprocessor.named_transformers_['numeric'].named_steps['imputer']
                          # Apply only to numeric columns if possible - complex here, applying to all as fallback
                          transformed_instance_dense = numeric_imputer.transform(transformed_instance_dense) # Might error if shapes mismatch
                     except:
                           imputer_temp_inst = SimpleImputer(strategy='median')
                           transformed_instance_dense = imputer_temp_inst.fit_transform(transformed_instance_dense)


                 explainer_shap = shap.KernelExplainer(predict_proba_wrapper, shap_background_dense)

                 logger.info("Calculating SHAP values...")
                 shap_values = explainer_shap.shap_values(transformed_instance_dense, nsamples=100) # Use dense instance

                 try:
                      feature_names = get_feature_names(fitted_preprocessor)
                 except Exception as fn_e:
                      logger.error(f"Failed to get feature names for SHAP: {fn_e}")
                      feature_names = [f"feature_{i}" for i in range(transformed_instance_dense.shape[1])]

                 # Handle shap_values potentially being a single array for binary classification
                 positive_class_shap_values = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values

                 # Handle expected_value format
                 base_value = explainer_shap.expected_value
                 if isinstance(base_value, (list, np.ndarray)):
                      if len(base_value) > 1: # Binary classification output?
                           base_value = base_value[1] # Take positive class base value
                      elif len(base_value) == 1:
                           base_value = base_value[0] # Single output

                 # Check if shap_values or base_value contain NaN/inf
                 if np.any(np.isnan(positive_class_shap_values)) or np.any(np.isnan(base_value)):
                      logger.warning(f"NaN found in SHAP values or base value for {model_name}. Explanation might be unreliable.")

                 shap_exp = shap.Explanation(
                     values=positive_class_shap_values,
                     base_values=base_value,
                     data=transformed_instance_dense,
                     feature_names=feature_names
                 )
                 model_explanations['shap'] = shap_exp
                 logger.info("SHAP explanation generated.")

             except Exception as e:
                 logger.error(f"SHAP explanation failed for {model_name}: {e}", exc_info=True)
                 model_explanations['shap'] = {'error': str(e)}

        # == LIME ==
        try:
            logger.info("Initializing LIME Explainer...")
            lime_explainer = lime_text.LimeTextExplainer(class_names=['Legitimate', 'Phishing'])
            logger.info("Calculating LIME explanation...")
            lime_exp = lime_explainer.explain_instance(
                sample_cleaned_text,
                lambda texts: predict_proba_for_lime(texts, pipeline),
                num_features=15, num_samples=1000 # Adjust num_samples if too slow
            )
            model_explanations['lime'] = lime_exp
            logger.info("LIME explanation generated.")
        except Exception as e:
            logger.error(f"LIME explanation failed for {model_name}: {e}", exc_info=True)
            model_explanations['lime'] = {'error': str(e)}

        # Add Anchor section here if re-enabled

        explanations[model_name] = model_explanations

    return models, results, explanations, fitted_preprocessor


# --------------------------------------
# Saving Artifacts and Visualizations
# --------------------------------------
def save_artifacts(output_dir, models, results, explanations, preprocessor):
    """Saves models, preprocessor, results, and visualizations."""
    logger.info(f"\n--- Saving Artifacts to: {output_dir} ---")
    os.makedirs(output_dir, exist_ok=True)

    # --- Save Models ---
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    for name, model in models.items():
        try:
            if name == "HybridNN" and TF_AVAILABLE and model is not None and isinstance(model, KerasModel):
                model_path = os.path.join(models_dir, f"{name}.keras")
                model.save(model_path)
                logger.info(f"Saved {name} model to {model_path}")
            elif isinstance(model, Pipeline):
                model_path = os.path.join(models_dir, f"{name}_pipeline.pkl")
                joblib.dump(model, model_path, compress=3) # Add compression
                logger.info(f"Saved {name} pipeline to {model_path}")
            else:
                 logger.warning(f"Model {name} is not a recognized type for saving (Type: {type(model)}).")
        except Exception as e:
            logger.error(f"Failed to save model {name}: {e}", exc_info=True)


    # --- Save Preprocessor ---
    if preprocessor:
        try:
            preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
            joblib.dump(preprocessor, preprocessor_path, compress=3) # Add compression
            logger.info(f"Saved fitted preprocessor to {preprocessor_path}")
        except Exception as e:
            logger.error(f"Failed to save preprocessor: {e}", exc_info=True)

    # --- Save Results ---
    try:
        results_path = os.path.join(output_dir, "results.json")
        # Improved default serializer
        def default_serializer(obj):
            if isinstance(obj, (np.integer, int)): return int(obj)
            elif isinstance(obj, (np.floating, float)):
                # Handle NaN/Inf for JSON
                if np.isnan(obj): return 'NaN'
                if np.isinf(obj): return 'Infinity' if obj > 0 else '-Infinity'
                return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)): return bool(obj)
            elif isinstance(obj, (bytes, bytearray)): return obj.decode('utf-8', errors='replace') # Decode bytes
            try: return json.JSONEncoder().encode(obj)
            except TypeError: return str(obj) # Fallback

        # Use default=default_serializer for robust JSON conversion
        # Need deep conversion for nested dicts/lists
        def convert_dict(item):
             if isinstance(item, dict):
                  return {k: convert_dict(v) for k, v in item.items()}
             elif isinstance(item, list):
                  return [convert_dict(elem) for elem in item]
             else:
                  # Apply serializer to individual values
                  try:
                      # Attempt direct serialization first
                      json.dumps(item)
                      return item
                  except TypeError:
                      # Use custom serializer if direct fails
                      return default_serializer(item)

        serializable_results = convert_dict(results)

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Saved evaluation results to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}", exc_info=True)


    # --- Save Explanations & Visualizations ---
    explanations_dir = os.path.join(output_dir, "explanations")
    os.makedirs(explanations_dir, exist_ok=True)
    if not explanations:
         logger.warning("No explanations generated to save.")
    else:
        for model_name, model_explanations in explanations.items():
            # Skip if explanations for this model failed or were skipped
            if not model_explanations or model_explanations.get('status') == 'Skipped':
                continue

            model_expl_dir = os.path.join(explanations_dir, model_name.replace(" ", "_"))
            os.makedirs(model_expl_dir, exist_ok=True)
            logger.info(f"Saving explanations for {model_name}...")

            # Save SHAP plots
            shap_exp = model_explanations.get('shap')
            if shap_exp and isinstance(shap_exp, shap.Explanation):
                try:
                    # Check if explanation object actually has data
                    if shap_exp.values is None or len(shap_exp.values) == 0 or shap_exp.data is None:
                         logger.warning(f"SHAP Explanation object for {model_name} appears empty. Skipping plot saving.")
                         continue

                    # Ensure we are plotting for the first instance if multiple were passed (should be 1 here)
                    exp_to_plot = shap_exp[0]

                    # Waterfall plot
                    fig_wf = plt.figure() # Create new figure for waterfall
                    try:
                         shap.plots.waterfall(exp_to_plot, max_display=20, show=False)
                         plt.tight_layout()
                         plt.savefig(os.path.join(model_expl_dir, "shap_waterfall.png"), dpi=150, bbox_inches='tight')
                    except Exception as wf_e:
                         logger.warning(f"SHAP waterfall plot failed for {model_name}: {wf_e}. Trying summary plot.")
                         plt.close(fig_wf) # Close figure if waterfall failed
                         fig_sum = plt.figure() # New figure for summary
                         try:
                             shap.summary_plot(shap_exp.values, features=exp_to_plot.data, feature_names=exp_to_plot.feature_names, max_display=20, show=False)
                             plt.tight_layout()
                             plt.savefig(os.path.join(model_expl_dir, "shap_summary_fallback.png"), dpi=150, bbox_inches='tight')
                         except Exception as sum_e:
                              logger.error(f"Fallback SHAP summary plot also failed for {model_name}: {sum_e}")
                         plt.close(fig_sum) # Close summary figure

                    # Ensure waterfall figure is closed if it was created and didn't error before summary plot
                    if 'fig_wf' in locals() and plt.fignum_exists(fig_wf.number):
                        plt.close(fig_wf)


                    # Force plot requires JS - save as HTML
                    try:
                        # Ensure base_values is correctly accessed
                        base_value_for_plot = exp_to_plot.base_values
                        # Handle scalar vs array base value
                        if isinstance(base_value_for_plot, (list, np.ndarray)): base_value_for_plot = base_value_for_plot[0]

                        # Check values before plotting
                        if np.any(np.isnan(exp_to_plot.values)) or np.any(np.isnan(base_value_for_plot)):
                             logger.warning(f"NaN found in SHAP values/base for force plot ({model_name}). Skipping force plot.")
                        else:
                             force_plot_html = shap.force_plot(
                                 base_value_for_plot,
                                 exp_to_plot.values, # Values for the first instance
                                 features=exp_to_plot.data,   # Data for the first instance
                                 feature_names=exp_to_plot.feature_names,
                                 show=False, matplotlib=False # Generate HTML
                             )
                             if force_plot_html:
                                 shap.save_html(os.path.join(model_expl_dir, "shap_force_plot.html"), force_plot_html)
                             else: logger.warning("SHAP force_plot did not return an object.")

                    except ImportError: logger.warning("IPython not available. Cannot save SHAP force plot HTML via save_html.")
                    except Exception as force_err: logger.warning(f"Could not save SHAP force plot: {force_err}", exc_info=False) # Less verbose log

                    logger.info(f"Saved SHAP plots for {model_name}.")
                except IndexError: logger.error(f"IndexError saving SHAP plots for {model_name}.")
                except Exception as e: logger.error(f"Failed to save SHAP plots for {model_name}: {e}", exc_info=True)
            elif shap_exp and isinstance(shap_exp, dict) and 'error' in shap_exp:
                 logger.warning(f"Skipping SHAP plots for {model_name} due to error: {shap_exp['error']}")


            # Save LIME plot (HTML)
            lime_exp = model_explanations.get('lime')
            if lime_exp and hasattr(lime_exp, 'save_to_file'):
                try:
                    lime_html_path = os.path.join(model_expl_dir, 'lime_explanation.html')
                    lime_exp.save_to_file(lime_html_path)
                    logger.info(f"Saved LIME HTML for {model_name}.")
                except Exception as e: logger.error(f"Failed to save LIME HTML for {model_name}: {e}", exc_info=True)
            elif lime_exp and isinstance(lime_exp, dict) and 'error' in lime_exp:
                 logger.warning(f"Skipping LIME plot saving for {model_name} due to error: {lime_exp['error']}")

            # Save Anchor explanation (Text) - Add back if used
            # ...

# (Keep save_f1_visualization as is)
def save_f1_visualization(results, output_dir):
    """Create and save a bar chart of F1 scores for all models."""
    logger.info("Creating F1 score comparison visualization...")
    models_perf = []
    for model_name, result_data in results.items():
        if isinstance(result_data, dict) and 'f1_score' in result_data and 'error' not in result_data:
            # Check if f1_score is valid number
            f1_val = result_data['f1_score']
            if isinstance(f1_val, (int, float)) and not np.isnan(f1_val):
                models_perf.append((model_name, f1_val))
            else:
                 logger.warning(f"Invalid F1 score type ({type(f1_val)}) or value ({f1_val}) for model {model_name}. Skipping from F1 plot.")

    if not models_perf:
        logger.warning("No valid F1 scores available to visualize.")
        return

    models_perf.sort(key=lambda item: item[1], reverse=True)
    models = [item[0] for item in models_perf]
    f1_scores = [item[1] for item in models_perf]

    try:
        fig_f1 = plt.figure(figsize=(max(6, len(models)*1.5), 5)) # Create figure specifically for F1 plot
        bars = plt.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(models)])
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        plt.title('Model F1 Score Comparison')
        plt.xlabel('Model')
        plt.ylabel('F1 Score (Binary)')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=15, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        f1_viz_path = os.path.join(output_dir, 'f1_score_comparison.png')
        plt.savefig(f1_viz_path, dpi=150)
        plt.close(fig_f1) # Close the specific figure
        logger.info(f"Saved F1 score visualization to {f1_viz_path}")
    except Exception as e:
        logger.error(f"Failed to create F1 score visualization: {e}", exc_info=True)


# --------------------------------------
# Main Execution
# --------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing Detection Training and Explanation Pipeline")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing raw CSV data files.")
    parser.add_argument("--output-dir", type=str, default="phishing_results", help="Directory to save models, results, and explanations.")
    # Example adding hints:
    # parser.add_argument("--label-hints", nargs='+', default=['label', 'class', 'target', 'phishing', 'type'], help="Keywords to identify label column.")
    # parser.add_argument("--text-hints", nargs='+', default=['text', 'body', 'email', 'content', 'message'], help="Keywords to identify text column.")

    args = parser.parse_args()

    # --- Setup Logging using output directory ---
    logger = setup_logging(args.output_dir) # Configure root logger

    try:
        logger.info("🚀 Starting Phishing Detection Pipeline 🚀")
        logger.info(f"Script Arguments: {args}")
        logger.info(f"Data Input Directory: {args.data_dir}")
        logger.info(f"Output Directory: {args.output_dir}")

        # 1. Load and Preprocess Data
        df_processed = load_and_preprocess_data(args.data_dir) # Add hints if using args: , args.label_hints, args.text_hints)

        # Optional: Save combined preprocessed data (useful for debugging)
        # try:
        #     os.makedirs(args.output_dir, exist_ok=True)
        #     save_path = os.path.join(args.output_dir, "combined_preprocessed_data.csv.gz")
        #     df_processed.to_csv(save_path, index=False, compression='gzip')
        #     logger.info(f"Saved combined preprocessed data to {save_path}")
        # except Exception as save_e:
        #      logger.error(f"Failed to save combined preprocessed data: {save_e}")


        # 2. Train, Evaluate, Explain
        models, results, explanations, preprocessor = train_evaluate_explain(
            df_processed,
            output_dir=args.output_dir # Pass output_dir if needed inside
        )

        # 3. Save Artifacts
        # Ensure output dir exists before saving final results/plots
        os.makedirs(args.output_dir, exist_ok=True)
        if models or results or explanations or preprocessor:
             # Save main artifacts like models, results json, preprocessor
             save_artifacts(args.output_dir, models, results, explanations, preprocessor)
             # Save summary visualization like F1 scores
             save_f1_visualization(results, args.output_dir)
        else:
             logger.warning("No artifacts generated to save.")


        logger.info("✅ Pipeline Completed Successfully! ✅")

    # (Keep exception handling as is)
    except FileNotFoundError as e:
        logger.critical(f"File Not Found Error: {e}. Please check input paths.")
        sys.exit(1)
    except pd.errors.EmptyDataError as e:
        logger.critical(f"Empty Data Error: {e}. One of the input CSVs might be empty or unreadable.")
        sys.exit(1)
    except KeyError as e:
         logger.critical(f"Column Key Error: {e}. Check if required columns exist in CSVs or hints are correct.")
         sys.exit(1)
    except MemoryError as e:
         logger.critical(f"Memory Error occurred: {e}. The dataset might be too large for available RAM. Consider processing in chunks or using a machine with more memory.")
         sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred in main execution: {e}", exc_info=True)
        sys.exit(1)
