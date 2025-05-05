# Combined Preprocessing, Training, and Explanation Script

import os
import re
import argparse
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings
# --- NLTK Imports ---
import nltk
from nltk.corpus import stopwords
# --- End NLTK Imports ---

import textstat
import logging
from logging.handlers import RotatingFileHandler
import sys
import json
import joblib
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from scipy.sparse import issparse

# --- Scikit-learn ---
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # Still needed for feature names in SHAP fallback potentially
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
from lime.lime_text import LimeTextExplainer

# --- TensorFlow / Keras Configuration ---
# Set TF environment variables BEFORE importing TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress INFO and WARNING messages

# --- TensorFlow / Keras Import (Now MANDATORY) ---
import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
print(f"--- TensorFlow Loaded (Version: {tf.__version__}) ---") # Confirmation

# --- Multilingual Embeddings ---
from sentence_transformers import SentenceTransformer

# --------------------------------------
# Global Configuration & Setup
# --------------------------------------
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logger (keep as is)
def setup_logging(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        for handler in logger.handlers[:]: handler.close(); logger.removeHandler(handler)
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
    console_handler = logging.StreamHandler(sys.stdout); console_handler.setFormatter(log_formatter); logger.addHandler(console_handler)
    try:
        os.makedirs(log_dir, exist_ok=True); log_file_path = os.path.join(log_dir, "phishing_detection_pipeline.log")
        file_handler = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'); file_handler.setFormatter(log_formatter); logger.addHandler(file_handler)
        logger.info(f"Logging configured. Log file: {log_file_path}")
    except Exception as e: logger.error(f"Failed file logger setup: {e}", exc_info=True)
    return logger


# --- NLTK Setup (FIXED Exception Handling) ---
try:
    # Try to find resources - this raises LookupError if not found
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    logging.debug("NLTK data found. OK.")
except LookupError as e:
    # Catch the correct LookupError
    logging.warning(f"NLTK resource missing ({e}). Attempting programmatic download...");
    # Attempt to download the specific missing resources
    try:
        nltk.data.find('corpora/stopwords', quiet=True) # Check quietly first
    except LookupError:
        logging.info("Downloading NLTK stopwords...");
        try:
            nltk.download('stopwords', quiet=True)
            logging.info("NLTK stopwords downloaded.")
        except Exception as download_e:
            logging.error(f"Failed to download stopwords: {download_e}", exc_info=False)

    try:
        nltk.data.find('tokenizers/punkt', quiet=True) # Check quietly first
    except LookupError:
        logging.info("Downloading NLTK punkt tokenizer...");
        try:
            nltk.download('punkt', quiet=True)
            logging.info("NLTK punkt tokenizer downloaded.")
        except Exception as download_e:
            logging.error(f"Failed to download punkt: {download_e}", exc_info=False)

except Exception as e:
    # Catch any other unexpected errors during NLTK setup
    logging.error(f"An unexpected error occurred during NLTK setup: {e}", exc_info=True)


# Keep English stopwords, as the current numeric features were designed with them in mind
# This line needs to be OUTSIDE the try...except block, as it will fail if stopwords wasn't downloaded
# If download failed above, this will raise LookupError, which is expected behavior if NLTK setup fails
try:
    stop_words_en = set(stopwords.words('english'))
except LookupError:
     logging.critical("NLTK stopwords resource is not available even after attempted download. Cannot proceed.");
     sys.exit(1) # Exit if stopwords are still missing

# --- End NLTK Setup ---


# Sentence Transformer Model Name
# Using a multilingual model optimized for sentence similarity
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
embedding_model = None # Will be loaded later

# --------------------------------------
# Preprocessing Functions (Adjusted)
# --------------------------------------
# clean_email: Keep as is (mostly language agnostic)
def clean_email(text):
    logger = logging.getLogger()
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "": return ""
    try:
        try: soup = BeautifulSoup(text, 'lxml')
        except: soup = BeautifulSoup(text, 'html.parser')
        cleaned = soup.get_text(separator=' ')
        # Keep URLs identifiable
        cleaned = re.sub(r'https?://\S+|www\.\S+', ' URL ', cleaned)
        # Allow broader range of characters including common accented letters
        cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?\'`áéíóúÁÉÍÓÚñÑüÜ]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    except Exception as e: logger.error(f"Error cleaning text: '{str(text)[:50]}...': {e}",exc_info=False); return ""


# extract_url_features: Keep patterns language-independent, update keywords
def extract_url_features(text):
    num_links, has_suspicious_url = 0, 0; logger = logging.getLogger()
    if pd.isna(text) or not isinstance(text, str): return 0, 0
    try:
        url_pattern = r'(https?://\S+|www\.\S+)'
        text_snippet_for_findall = text[:50000]
        urls = re.findall(url_pattern, text_snippet_for_findall)
        try: num_links = len(re.findall(url_pattern, text))
        except Exception as count_e: logger.warning(f"URL count error: {count_e}. Using snippet count."); num_links = len(urls)

        if not urls: return num_links, has_suspicious_url

        # Combine English and Spanish keywords (can be expanded)
        suspicious_keywords_en = ['login', 'verify', 'account', 'secure', 'update', 'confirm', 'signin', 'support', 'password', 'banking', 'activity', 'credential']
        suspicious_keywords_es = ['iniciar sesion', 'verificar', 'cuenta', 'actualizar', 'confirmar', 'contraseña', 'banco', 'actividad', 'credenciales']
        suspicious_keywords = suspicious_keywords_en + suspicious_keywords_es # Use both

        shortened_domains_pattern = r'(bit\.ly/|goo\.gl/|tinyurl\.com/|t\.co/|ow\.ly/|is\.gd/|buff\.ly/|adf\.ly/|bit\.do/|soo\.gd/)'

        has_http, has_shortener, has_keywords = 0, 0, 0
        for url in urls[:100]:
            try:
                url_lower = url.lower()
                if url_lower.startswith('http://'): has_http = 1
                if re.search(shortened_domains_pattern, url_lower): has_shortener = 1

                proto_end = url_lower.find('//')
                path_query = ''
                if proto_end > 0:
                    domain_part_end = url_lower.find('/', proto_end + 2)
                    path_query = url_lower[domain_part_end:] if domain_part_end > 0 else ''
                else:
                    domain_part_end = url_lower.find('/')
                    path_query = url_lower[domain_part_end:] if domain_part_end > 0 else ''

                check_string = path_query + url_lower
                # Check for *any* of the combined keywords
                if any(keyword in check_string for keyword in suspicious_keywords): has_keywords = 1

                if has_http or has_shortener or has_keywords:
                    has_suspicious_url = 1
                    break
            except Exception as url_parse_e: logger.debug(f"URL parse error '{url[:50]}...': {url_parse_e}"); continue
        return num_links, has_suspicious_url
    except Exception as e: logger.error(f"URL feature extraction failed: '{str(text)[:50]}...': {e}",exc_info=False); return 0, 0

# extract_urgency: Update keywords for Spanish
def extract_urgency(cleaned_text):
    logger = logging.getLogger();
    if not cleaned_text: return 0
    try:
        # Combine English and Spanish urgency keywords (can be expanded)
        urgency_words_en = ['urgent', 'immediately', 'action required', 'verify', 'password', 'alert', 'warning', 'limited time', 'expire', 'suspended', 'locked', 'important', 'final notice', 'response required', 'security update', 'confirm account', 'validate', 'due date', 'restricted', 'compromised', 'unauthorized']
        urgency_words_es = ['urgente', 'inmediatamente', 'acción requerida', 'verifique', 'contraseña', 'alerta', 'advertencia', 'tiempo limitado', 'expira', 'suspendida', 'bloqueada', 'importante', 'último aviso', 'requiere respuesta', 'actualización de seguridad', 'confirmar cuenta', 'validar', 'fecha límite', 'restringido', 'comprometida', 'no autorizado']
        urgency_words = urgency_words_en + urgency_words_es # Use both

        text_lower = cleaned_text.lower();
        # Use word boundaries \b
        count = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower)) for word in urgency_words)
        return count
    except Exception as e: logger.error(f"Urgency calculation failed: {e}", exc_info=False); return 0

# calculate_readability: Acknowledge limitation or remove. Keep for now.
# The Flesch score is less reliable for Spanish. We keep it, but its importance might decrease.
def calculate_readability(cleaned_text):
    logger = logging.getLogger(); word_count = len(cleaned_text.split())
    if not cleaned_text or word_count < 10: return 100.0
    try:
        # Flesch Reading Ease is language-tuned (primarily English). Less reliable for Spanish.
        score = textstat.flesch_reading_ease(cleaned_text);
        return max(-200, min(120, score)) if not np.isnan(score) else 50.0
    except Exception as e:
        if word_count > 5: logger.debug(f"Readability failed: '{cleaned_text[:50]}...': {e}", exc_info=False)
        return 50.0


# New function to compute embeddings
def compute_embeddings(texts, model, batch_size=32):
    logger = logging.getLogger()
    if not texts or len(texts) == 0:
        logger.warning("No texts provided for embedding computation.")
        # Return an array with 0 samples but correct dimension if model is loaded
        embedding_dim = model.get_sentence_embedding_dimension() if model and hasattr(model, 'get_sentence_embedding_dimension') else 384
        return np.empty((0, embedding_dim), dtype=np.float32)

    logger.info(f"Computing embeddings using {EMBEDDING_MODEL_NAME} for {len(texts)} texts...")
    # Ensure inputs are strings and handle potential NaNs
    texts = [str(t) if pd.notna(t) else "" for t in texts]
    try:
        # SentenceTransformer handles batching internally
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
        logger.info(f"Embedding computation complete. Shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to compute embeddings: {e}", exc_info=True)
        # Return zeros or raise error depending on desired behavior
        # Returning zeros allows pipeline to continue but might impact training
        embedding_dim = model.get_sentence_embedding_dimension() if model and hasattr(model, 'get_sentence_embedding_dimension') else 384 # Assume default dim if needed
        logger.warning(f"Returning array of zeros with shape ({len(texts)}, {embedding_dim}) due to embedding failure.")
        return np.zeros((len(texts), embedding_dim), dtype=np.float32)


# Data Loading and Preprocessing Pipeline (Adjusted)
def load_and_preprocess_data(data_dir, embedding_model, label_col_hints=None, text_col_hints=None):
    logger = logging.getLogger();
    if label_col_hints is None: label_col_hints = ['label', 'class', 'target', 'phishing', 'type']
    if text_col_hints is None: text_col_hints = ['text', 'body', 'email', 'content', 'message']
    all_dfs = [];
    # Define the standard numeric columns that will be extracted
    original_numeric_cols = ['num_links', 'has_suspicious_url', 'urgency_count', 'readability_score']

    # Determine expected embedding dimension *before* processing files, if model loaded
    expected_embedding_dim = embedding_model.get_sentence_embedding_dimension() if embedding_model and hasattr(embedding_model, 'get_sentence_embedding_dimension') else 384


    logger.info(f"Scanning directory for CSV files: {data_dir}")
    if not os.path.isdir(data_dir): logger.error(f"Data directory not found: {data_dir}"); sys.exit(1)
    file_list = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    logger.info(f"Found {len(file_list)} CSV files to process.")

    for filename in file_list:
        input_path = os.path.join(data_dir, filename); logger.info(f"--- Processing file: {filename} ---")
        try:
            try: df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)
            except UnicodeDecodeError: logger.warning(f"UTF-8 failed for {filename}. Trying latin-1..."); df = pd.read_csv(input_path, encoding='latin1', low_memory=False)
            except pd.errors.ParserError as pe:
                 logger.error(f"Parser error in {filename}: {pe}. Trying on_bad_lines='warn'.")
                 try: df = pd.read_csv(input_path, encoding='latin1', low_memory=False, on_bad_lines='warn')
                 except Exception as parse_fallback_e: logger.error(f"Read fallback failed for {filename}: {parse_fallback_e}. Skipping."); continue
            except FileNotFoundError: logger.error(f"File not found: {input_path}. Skipping."); continue
            except Exception as e: logger.error(f"Failed to read {filename}: {e}"); continue

            if df.empty: logger.warning(f"{filename} is empty. Skipping."); continue

            logger.info(f"Raw columns: {df.columns.tolist()}");
            df.columns = df.columns.str.strip() # Clean column names

            text_col = next((col for col in df.columns if any(hint in col.lower() for hint in text_col_hints)), None)
            label_col = next((col for col in df.columns if any(hint in col.lower() for hint in label_col_hints)), None)

            if not text_col: logger.error(f"No text column found using hints {text_col_hints} in {filename}. Skipping."); continue
            if not label_col: logger.error(f"No label column found using hints {label_col_hints} in {filename}. Skipping."); continue
            logger.info(f"Identified Text column: '{text_col}', Label column: '{label_col}'")

            if len(df) < 10: logger.warning(f"{filename} has < 10 rows after identifying columns. Skipping."); continue

            # Select only relevant columns early to save memory
            df = df[[text_col, label_col]].copy() # Use .copy() to avoid SettingWithCopyWarning

            df.dropna(subset=[text_col], inplace=True)
            if df.empty: logger.warning(f"No rows after dropping missing text in {filename}. Skipping."); continue

            df_processed = pd.DataFrame();
            df_processed['original_text'] = df[text_col].astype(str);

            # Robust label mapping (keep as is)
            label_map = {'spam': 1, 'phish': 1, 'phishing': 1, 'fraud': 1, 'scam': 1, 'malicious': 1, '1': 1, 1: 1, True: 1, 'true': 1, 'ham': 0, 'legitimate': 0, 'safe': 0, 'normal': 0, 'benign': 0, '0': 0, 0: 0, False: 0, 'false': 0}
            original_label_type = df[label_col].dtype
            try:
                 # Attempt direct conversion first if applicable
                 # Access label column from original df before dropping NaNs later
                 labels_series = df[label_col]
                 if pd.api.types.is_bool_dtype(original_label_type): labels_series = labels_series.astype(int)
                 labels_series = pd.to_numeric(labels_series, errors='coerce')
                 # Fallback to string mapping for objects or if coercion failed
                 if labels_series.isnull().any() or pd.api.types.is_object_dtype(original_label_type):
                      labels_series = df[label_col].astype(str).str.strip().str.lower().map(label_map)
                 df_processed['label'] = labels_series # Assign processed label to df_processed
            except Exception as label_conv_e:
                 logger.error(f"Label conversion error in {filename}: {label_conv_e}. Trying string map as final fallback.")
                 try:
                      labels_series = df[label_col].astype(str).str.strip().str.lower().map(label_map)
                      df_processed['label'] = labels_series
                 except Exception as fallback_e: logger.error(f"Final fallback label map failed for {filename}: {fallback_e}. Skipping file."); continue

            original_rows = len(df_processed);
            df_processed.dropna(subset=['label'], inplace=True)
            if len(df_processed) < original_rows: logger.warning(f"Dropped {original_rows - len(df_processed)} rows with missing/unmappable labels in {filename}.")
            if df_processed.empty: logger.warning(f"No rows after label processing in {filename}. Skipping."); continue
            df_processed['label'] = df_processed['label'].astype(int)


            logger.info("Cleaning text...");
            df_processed['cleaned_text'] = df_processed['original_text'].apply(clean_email)

            # Drop rows where cleaned text is empty or NaN
            original_rows = len(df_processed)
            df_processed.dropna(subset=['cleaned_text'], inplace=True)
            df_processed = df_processed[df_processed['cleaned_text'].str.strip().astype(bool)]
            if len(df_processed) < original_rows: logger.warning(f"Dropped {original_rows - len(df_processed)} rows with empty/NaN cleaned text after cleaning in {filename}.")
            if df_processed.empty: logger.warning(f"No rows remaining after text cleaning in {filename}. Skipping."); continue


            logger.info("Extracting numeric features...");
            # Use original_text for URL extraction before cleaning potentialy removed URLs
            url_features = df_processed['original_text'].apply(extract_url_features)
            df_processed['num_links'] = url_features.apply(lambda x: x[0]).astype(int)
            df_processed['has_suspicious_url'] = url_features.apply(lambda x: x[1]).astype(int)
            # Use cleaned_text for urgency and readability
            df_processed['urgency_count'] = df_processed['cleaned_text'].apply(extract_urgency).astype(int)
            df_processed['readability_score'] = df_processed['cleaned_text'].apply(calculate_readability).astype(float)

            # --- Compute Embeddings ---
            if embedding_model: # Only compute embeddings if model was loaded successfully
                 embeddings = compute_embeddings(df_processed['cleaned_text'].tolist(), embedding_model)
                 # Add embedding dimensions as new columns
                 embedding_dim = embeddings.shape[1] if embeddings.ndim > 1 else 0
                 if embedding_dim > 0:
                     embedding_cols = [f'embedding_{i}' for i in range(embedding_dim)]
                     df_embeddings = pd.DataFrame(embeddings, index=df_processed.index, columns=embedding_cols)
                     df_processed = pd.concat([df_processed, df_embeddings], axis=1)
                     logger.info(f"Added {embedding_dim} embedding columns.")
                 else:
                     logger.error("Embedding computation resulted in invalid shape (0 dimensions). Skipping adding embedding columns from this file.")
                     # Add NaN columns with expected dimension if computation failed but model was loaded
                     embedding_cols = [f'embedding_{i}' for i in range(expected_embedding_dim)]
                     for col in embedding_cols: df_processed[col] = np.nan # Add NaN columns to keep structure consistent
            else:
                 logger.warning("Embedding model not loaded. Skipping embedding computation for this file.")
                 # Add NaN columns with expected dimension if model was not loaded
                 embedding_cols = [f'embedding_{i}' for i in range(expected_embedding_dim)]
                 for col in embedding_cols: df_processed[col] = np.nan # Add NaN columns to keep structure consistent


            # Select and reorder columns to ensure consistent structure
            embedding_cols_present_in_file = [col for col in df_processed.columns if col.startswith('embedding_')]
            all_cols_for_file_df = ['original_text', 'cleaned_text', 'label'] + original_numeric_cols + embedding_cols_present_in_file
            df_processed = df_processed[[col for col in all_cols_for_file_df if col in df_processed.columns]]


            # Check for NaN in numeric features and embeddings before appending
            embedding_cols_present_check = [col for col in df_processed.columns if col.startswith('embedding_')]
            numeric_and_embedding_cols_check = original_numeric_cols + embedding_cols_present_check
            nan_rows_features = df_processed[numeric_and_embedding_cols_check].isnull().any(axis=1).sum()
            if nan_rows_features > 0: logger.warning(f"{nan_rows_features} rows with NaN in numeric or embedding columns found in {filename}. Imputation will handle.")


            logger.info(f"Finished processing {filename}. Valid rows added: {len(df_processed)}");
            all_dfs.append(df_processed)

        except Exception as e: logger.error(f"Critical error processing {filename}: {e}", exc_info=True); continue

    if not all_dfs: logger.error("No data loaded or processed successfully from any file."); raise RuntimeError("No data loaded or processed successfully from source directory.")

    try:
        final_df = pd.concat(all_dfs, ignore_index=True); logger.info(f"--- Combined all files. Total rows: {len(final_df)} ---")
        if final_df.empty: logger.error("Concatenated DataFrame empty."); raise RuntimeError("Concatenated DataFrame empty.")
    except Exception as e: logger.error(f"Failed to concat DataFrames: {e}", exc_df=True); raise e # Changed exc_info to exc_df typo fix? No, exc_info is correct

    # Final checks post-concat
    embedding_cols_present_final = [col for col in final_df.columns if col.startswith('embedding_')]
    numeric_cols_final = original_numeric_cols # Use the defined list
    all_feature_cols_final = numeric_cols_final + embedding_cols_present_final

    # Check if expected numeric columns are actually present after concat
    missing_numeric_cols_final = [col for col in numeric_cols_final if col not in final_df.columns]
    if missing_numeric_cols_final:
         logger.warning(f"Expected numeric columns {missing_numeric_cols_final} are missing in the final DataFrame.")
         # Adjust the list of features actually present
         numeric_cols_final = [col for col in numeric_cols_final if col in final_df.columns]
         all_feature_cols_final = numeric_cols_final + embedding_cols_present_final


    if not numeric_cols_final and not embedding_cols_present_final:
         logger.critical("No numeric or embedding features present in the final DataFrame. Cannot train model."); sys.exit(1)
    elif not embedding_cols_present_final and embedding_model:
         logger.warning("Embedding model was loaded but no embedding columns were added to the final DataFrame.")
    elif not numeric_cols_final:
         logger.warning("No original numeric columns were added to the final DataFrame.")


    nan_rows = final_df[all_feature_cols_final].isnull().any(axis=1).sum()
    if nan_rows > 0: logger.warning(f"{nan_rows} total rows with NaN in feature columns post-concat. Imputation during training will handle.")

    if 'label' in final_df.columns:
         logger.info(f"Final shape: {final_df.shape}")
         try: label_counts = final_df['label'].value_counts(normalize=True, dropna=False); logger.info(f"Label distribution:\n{label_counts}")
         except Exception as label_count_e: logger.error(f"Could not get final label distribution: {label_count_e}")
    else: logger.error("Label column missing post-concat."); raise RuntimeError("Label column missing post-concat.")

    # Return DataFrame including original text, cleaned text, label, numeric features, and embedding features
    return final_df


# --- Data Saving and Loading ---
def save_processed_data(df, file_path):
    """Saves the processed DataFrame to a Parquet file."""
    logger = logging.getLogger()
    try:
        # Ensure directory exists
        output_dir = os.path.dirname(file_path)
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(file_path)
        logger.info(f"Saved processed DataFrame to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save processed DataFrame to {file_path}: {e}", exc_info=True)

def load_processed_data(file_path):
    """Loads a processed DataFrame from a Parquet file."""
    logger = logging.getLogger()
    if not os.path.exists(file_path):
        logger.info(f"Processed data file not found at {file_path}. Will process from raw data.")
        return None
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded processed DataFrame from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load processed DataFrame from {file_path}: {e}", exc_info=True)
        logger.warning("Will attempt to process from raw data instead.")
        return None


# Model Definition (Hybrid NN - Adjusted for Embedding Input)
def build_hybrid_nn(embedding_dim, struct_input_dim, learning_rate=0.001):
    logger = logging.getLogger();
    if embedding_dim <= 0 and struct_input_dim <= 0:
        logger.error(f"Invalid input dimensions for NN: embedding={embedding_dim}, struct={struct_input_dim}. Cannot build.")
        return None # Cannot build with zero total dimensions
    # Build even if one dimension is zero, but the other is > 0
    # Check if we only have one input type and adjust model structure if needed
    # For simplicity with the current HybridNN structure which expects 2 inputs,
    # we will only build if both dimensions are > 0. If only one type is present,
    # the training loop will skip HybridNN (as implemented below).

    # Define inputs based on available dimensions
    inputs = []
    if embedding_dim > 0: inputs.append(Input(shape=(embedding_dim,), name='embedding_features_input'))
    if struct_input_dim > 0: inputs.append(Input(shape=(struct_input_dim,), name='structural_features_input'))

    if not inputs: return None # Should be caught by the check above, but safety

    # Process inputs - handle single or multiple inputs
    if len(inputs) == 1:
        # Only one input type (either embeddings or structural)
        processed = Dense(128, activation='relu')(inputs[0])
        processed = Dropout(0.4)(processed)
        processed = Dense(64, activation='relu')(processed)
    elif len(inputs) == 2:
        # Both input types (embeddings and structural)
        # Assuming inputs[0] is embeddings, inputs[1] is structural
        embed_dense1 = Dense(128, activation='relu')(inputs[0]);
        embed_dropout = Dropout(0.4)(embed_dense1)
        embed_dense2 = Dense(64, activation='relu')(embed_dropout)

        struct_dense = Dense(32, activation='relu')(inputs[1])

        combined = Concatenate()([embed_dense2, struct_dense])

        processed = Dense(64, activation='relu')(combined);
        processed = Dropout(0.4)(processed)
    else:
         logger.error(f"Unexpected number of NN inputs ({len(inputs)}). Cannot build.")
         return None


    # Output layer
    output = Dense(1, activation='sigmoid', name='output')(processed)

    # Define model inputs based on the list
    if len(inputs) == 1:
         model = KerasModel(inputs=inputs[0], outputs=output) # Single input
    else: # len(inputs) == 2
         model = KerasModel(inputs=inputs, outputs=output) # List of inputs


    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

    logger.info("Hybrid NN Architecture (Embeddings + Structural):"); model.summary(print_fn=logger.info);
    return model

# XAI Helper Function (Adjusted for Embeddings)
class PredictProbaWrapper:
    """
    Wrapper class for predict_proba tailored for LIME when using embeddings and numeric features.
    LIME perturbs raw text; this wrapper computes embeddings for perturbed text,
    combines them with fixed numeric features for the specific instance, and predicts.
    """
    def __init__(self, model, embedding_model, numeric_preprocessor, original_numeric_features_scaled, embedding_dim, original_numeric_cols):
        self.model = model # The trained classifier (LR, DT, or Keras model)
        self.embedding_model = embedding_model # The Sentence Transformer model
        self.numeric_preprocessor = numeric_preprocessor # The fitted numeric ColumnTransformer (or None)
        self.original_numeric_features_scaled = original_numeric_features_scaled # The scaled numeric features of the instance being explained (shape 1, n_numeric_features)
        self.embedding_dim = embedding_dim # Dimension of embeddings
        self.original_numeric_cols = original_numeric_cols # List of names of original numeric columns
        self.logger = logging.getLogger()

        # Determine model type and prediction method
        if isinstance(model, (LogisticRegression, DecisionTreeClassifier)):
             # sklearn models trained on combined features (handled by __call__)
             self._predict_fn = lambda x: model.predict_proba(x)
        elif isinstance(model, KerasModel):
             # Keras needs inputs split correctly (handled by __call__)
             self._predict_fn = lambda x: model.predict(x) # Keras predict returns probabilities (shape n, 1)
        else:
            raise TypeError(f"Unsupported model type for PredictProbaWrapper: {type(model)}")

    def __call__(self, raw_texts):
        """
        Processes raw text strings from LIME, computes embeddings, combines with
        numeric features, and returns prediction probabilities.
        """
        n_samples = len(raw_texts)

        # 1. Compute embeddings for the perturbed texts
        # Use a try-except block just in case embedding computation fails during perturbation
        try:
             if not self.embedding_model or self.embedding_dim <= 0:
                  # If embedding model or dimension is missing, return zeros for embeddings
                  self.logger.warning("Embedding model not available or embedding dim is zero in LIME wrapper. Using zeros for embeddings.")
                  embeddings = np.zeros((n_samples, self.embedding_dim), dtype=np.float32)
             else:
                 embeddings = compute_embeddings(raw_texts, self.embedding_model, batch_size=min(n_samples, 64)) # Batching embeddings
                 if embeddings.shape[0] != n_samples or embeddings.shape[1] != self.embedding_dim:
                      self.logger.error(f"Embedding computation returned wrong number of samples or dimension: Expected ({n_samples}, {self.embedding_dim}), got {embeddings.shape}. Using zeros.")
                      embeddings = np.zeros((n_samples, self.embedding_dim), dtype=np.float32)

        except Exception as e:
             self.logger.error(f"Error computing embeddings for LIME perturbations: {e}", exc_info=False)
             # Return zeros for embeddings on critical failure
             embeddings = np.zeros((n_samples, self.embedding_dim), dtype=np.float32)


        # 2. Get the fixed numeric features for the sample being explained
        # original_numeric_features_scaled is already scaled and has shape (1, n_numeric_features)
        # We need to replicate it for each perturbed text sample
        n_numeric_features = self.original_numeric_features_scaled.shape[1] if self.original_numeric_features_scaled is not None else len(self.original_numeric_cols) # Get dim from array if exists, fallback to expected
        if self.original_numeric_features_scaled is None or self.original_numeric_features_scaled.shape[0] == 0 or n_numeric_features == 0:
             self.logger.debug("Original numeric features not available for LIME wrapper or dim is zero. Using zeros.")
             replicated_numeric_features = np.zeros((n_samples, n_numeric_features), dtype=np.float32)
        else:
             replicated_numeric_features = np.tile(self.original_numeric_features_scaled, (n_samples, 1)).astype(np.float32)


        # 3. Combine embeddings and numeric features according to model input
        inputs_for_model = None
        if isinstance(self.model, (LogisticRegression, DecisionTreeClassifier)):
             # Models expecting a single combined array (embeddings | numeric)
             # Ensure correct order based on how the training data was combined (embeddings first, then numeric)
             if embeddings.shape[1] > 0 and replicated_numeric_features.shape[1] > 0:
                  inputs_for_model = np.hstack([embeddings, replicated_numeric_features]).astype(np.float32)
             elif embeddings.shape[1] > 0: # Only embeddings
                  inputs_for_model = embeddings.astype(np.float32)
             elif replicated_numeric_features.shape[1] > 0: # Only numeric
                  inputs_for_model = replicated_numeric_features.astype(np.float32)
             else:
                  self.logger.error("No features (embeddings or numeric) available for prediction in LIME wrapper.")
                  return np.array([[0.5, 0.5]] * n_samples) # Return neutral

        elif isinstance(self.model, KerasModel):
             # Keras model expects inputs split according to build_hybrid_nn (embeddings first, then structural if both exist)
             keras_inputs_list = []
             if self.embedding_dim > 0: keras_inputs_list.append(embeddings)
             if n_numeric_features > 0: keras_inputs_list.append(replicated_numeric_features)

             if not keras_inputs_list:
                 self.logger.error("No features (embeddings or numeric) available for Keras prediction in LIME wrapper.")
                 return np.array([[0.5, 0.5]] * n_samples) # Return neutral

             inputs_for_model = keras_inputs_list # List of input arrays


        # 4. Predict probabilities using the wrapped model
        try:
            # Sklearn models return (n_samples, 2) probabilities
            # Keras predict returns (n_samples, 1) probabilities for sigmoid output
            probabilities = self._predict_fn(inputs_for_model)

            # Ensure output shape is (n_samples, 2) for LIME
            if isinstance(self.model, KerasModel):
                 # Convert Keras (n_samples, 1) output to (n_samples, 2)
                 if probabilities.shape == (n_samples, 1):
                      probabilities = np.hstack([1 - probabilities, probabilities])
                 else:
                      self.logger.warning(f"Keras prediction function returned unexpected shape {probabilities.shape}. Expected ({n_samples}, 1) or ({n_samples}, 2).")
                      # Attempt to fix if shape is wrong (e.g., just (n_samples,) for some reason)
                      if probabilities.shape == (n_samples,):
                          probabilities = probabilities.reshape(-1, 1)
                          probabilities = np.hstack([1 - probabilities, probabilities])
                      else:
                          self.logger.error("Cannot fix unexpected Keras prediction shape.")
                          return np.array([[0.5, 0.5]] * n_samples) # Fallback if shape cannot be fixed
            # Sklearn models should already return (n_samples, 2)

            if probabilities.shape != (n_samples, 2): # Final shape check
                 self.logger.error(f"Final prediction shape is incorrect {probabilities.shape}. Expected ({n_samples}, 2).")
                 return np.array([[0.5, 0.5]] * n_samples)

            if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
                 self.logger.warning("NaN/Inf in prediction probabilities. Returning neutral.")
                 return np.array([[0.5, 0.5]] * n_samples)

            return probabilities
        except Exception as e:
            self.logger.error(f"Error during model prediction in LIME wrapper: {e}", exc_info=True)
            # Return a neutral probability distribution if prediction fails
            return np.array([[0.5, 0.5]] * n_samples)


# Helper to get feature names for SHAP (Adjusted for Embeddings)
def get_feature_names(numeric_cols_present, embedding_dim):
    """Generates feature names for combined numeric and embedding features based on what's present."""
    numeric_feature_names = list(numeric_cols_present) # Use names of numeric features actually present
    embedding_feature_names = [f'embedding_{i}' for i in range(embedding_dim)] if embedding_dim > 0 else []
    return numeric_feature_names + embedding_feature_names


# Plotting Functions (Keep as is, F1 plot now shows best model)
def save_training_history_plot(history, output_dir):
    logger = logging.getLogger();
    if not history or not hasattr(history, 'history') or not history.history: logger.warning("No training history found."); return
    history_dict = history.history; epochs = range(1, len(history_dict.get('loss', [])) + 1)
    has_loss = 'loss' in history_dict and 'val_loss' in history_dict; has_acc = 'accuracy' in history_dict and 'val_accuracy' in history_dict
    if not has_loss and not has_acc: logger.warning("History missing loss/acc keys."); return
    try:
        fig = plt.figure(figsize=(12, 5))
        if has_loss:
            plt.subplot(1, 2, 1); plt.plot(epochs, history_dict['loss'], 'bo-', label='Training Loss'); plt.plot(epochs, history_dict['val_loss'], 'ro-', label='Validation Loss')
            plt.title('Training and Validation Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        if has_acc:
            plot_index = 2 if has_loss else 1; plt.subplot(1, plot_index, plot_index)
            plt.plot(epochs, history_dict['accuracy'], 'bo-', label='Training Accuracy'); plt.plot(epochs, history_dict['val_accuracy'], 'ro-', label='Validation Accuracy')
            plt.title('Training and Validation Accuracy'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plot_path = os.path.join(output_dir, 'hybrid_nn_training_history.png')
        plt.savefig(plot_path, dpi=150); plt.close(fig); logger.info(f"Saved training history plot: {plot_path}")
    except Exception as e: logger.error(f"Failed history plot: {e}", exc_info=True)


def save_f1_visualization(results, output_dir):
    logger = logging.getLogger(); logger.info("Creating F1 score comparison visualization...")
    models_perf = []
    for model_name, result_data in results.items():
        if isinstance(result_data, dict) and 'f1_score' in result_data and 'error' not in result_data:
            f1_val = result_data['f1_score']
            if isinstance(f1_val, (int, float)) and not np.isnan(f1_val): models_perf.append((model_name, f1_val))
            else: logger.warning(f"Invalid F1 {type(f1_val)}/{f1_val} for {model_name}. Skipping.")
    if not models_perf: logger.warning("No valid F1 scores to visualize."); return
    models_perf.sort(key=lambda item: item[1], reverse=True); models = [item[0] for item in models_perf]; f1_scores = [item[1] for item in models_perf]
    try:
        fig_f1 = plt.figure(figsize=(max(6, len(models)*1.5), 5))
        bars = plt.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(models)])
        for bar in bars: height = bar.get_height(); plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        plt.title('Model F1 Score Comparison'); plt.xlabel('Model'); plt.ylabel('F1 Score (Binary)'); plt.ylim(0, 1.05); plt.xticks(rotation=15, ha='right'); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
        f1_viz_path = os.path.join(output_dir, 'f1_score_comparison.png'); plt.savefig(f1_viz_path, dpi=150); plt.close(fig_f1); logger.info(f"Saved F1 score visualization: {f1_viz_path}")
    except Exception as e: logger.error(f"Failed F1 plot: {e}", exc_info=True)


# Model Selection Function (Keep as is)
def select_best_model(results):
    logger = logging.getLogger()
    best_model_name = None
    best_f1 = -1

    logger.info("Selecting best model based on F1 score...")

    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'f1_score' in metrics and 'error' not in metrics:
            current_f1 = metrics['f1_score']
            # Prefer models that trained successfully (not just error)
            if current_f1 > best_f1 or (best_model_name is None and current_f1 >= 0):
                best_f1 = current_f1
                best_model_name = model_name
                logger.debug(f"Found new best model: {model_name} with F1 = {best_f1:.4f}")
        else:
            logger.debug(f"Skipping model {model_name} for selection due to missing F1 or error.")

    if best_model_name:
        logger.info(f"Best model selected: {best_model_name} (F1 = {best_f1:.4f})")
    else:
        logger.warning("Could not select a best model (no successful training results found).")

    return best_model_name

# Training/Evaluation/Explanation Function (Major Refactoring)
def train_evaluate_explain(df, text_col='cleaned_text', original_numeric_cols=None, label_col='label', output_dir='results', embedding_model=None):
    logger = logging.getLogger();
    if original_numeric_cols is None: original_numeric_cols = ['num_links', 'has_suspicious_url', 'urgency_count', 'readability_score']
    logger.info("--- Starting Model Training and Evaluation with Embeddings ---")
    logger.info(f"Using Text: '{text_col}', Original Numeric: {original_numeric_cols}, Label: '{label_col}'")

    # Identify actual embedding columns present in the DataFrame
    embedding_cols_present = [col for col in df.columns if col.startswith('embedding_')]
    embedding_dim = len(embedding_cols_present)

    # Identify actual numeric columns present that match the original_numeric_cols list
    present_numeric_cols = [col for col in original_numeric_cols if col in df.columns]
    struct_input_dim = len(present_numeric_cols)


    all_feature_cols = present_numeric_cols + embedding_cols_present
    required_input_cols = [text_col, label_col, 'original_text'] + all_feature_cols
    missing_cols = [col for col in required_input_cols if col not in df.columns];
    if missing_cols: logger.error(f"Missing required columns in DataFrame before split: {missing_cols}. Exiting."); sys.exit(1)

    # Features X now contains *only* the columns that will be used for training
    X = df[all_feature_cols];
    y = df[label_col];
    original_texts = df['original_text'];
    cleaned_texts = df[text_col]; # Keep cleaned texts for LIME

    if X.empty or y.empty: logger.error("X or y empty before split."); sys.exit(1)
    logger.info(f"Dataset size before split: {len(X)} rows.")
    logger.info(f"Feature set shape before split: {X.shape}")

    logger.info("Splitting data (80/20)...")
    # Need to stratify based on the label
    arrays_to_split = [X, y, original_texts, cleaned_texts, df.index] # Include cleaned_texts and original index
    try:
        min_class_count = y.value_counts().min();
        use_stratify = min_class_count >= 2
        if not use_stratify:
            logger.warning(f"Minority class count ({min_class_count}) is less than 2. Cannot use stratify for splitting.")

        split_results = train_test_split(*arrays_to_split, test_size=0.2, random_state=42, stratify=y if use_stratify else None)
        X_train, X_test, y_train, y_test, original_texts_train, original_texts_test, cleaned_texts_train, cleaned_texts_test, train_indices, test_indices = split_results

    except ValueError as e:
         if 'stratify' in str(e):
              logger.warning(f"Stratify split failed: {e}. Trying without stratify.");
              split_results = train_test_split(*arrays_to_split, test_size=0.2, random_state=42)
              X_train, X_test, y_train, y_test, original_texts_train, original_texts_test, cleaned_texts_train, cleaned_texts_test, train_indices, test_indices = split_results
         else:
             logger.error(f"Unexpected split error: {e}", exc_info=True);
             raise e

    if X_train.empty or X_test.empty: logger.error("Train/Test sets are empty after split."); sys.exit(1)
    logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    logger.info(f"Train feature shape: {X_train.shape}, Test feature shape: {X_test.shape}")
    logger.info(f"Train labels distribution:\n{y_train.value_counts(normalize=True, dropna=False)}")
    logger.info(f"Test labels distribution:\n{y_test.value_counts(normalize=True, dropna=False)}")

    # Define and fit the preprocessor ONLY for numeric features if they exist
    fitted_numeric_preprocessor = None
    if struct_input_dim > 0:
         numeric_preprocessor = Pipeline([
             ('imputer', SimpleImputer(strategy='median')),
             ('scaler', StandardScaler())
         ])
         logger.info("Fitting numeric preprocessor...");
         # Fit on the original numeric columns from the training data
         X_train_numeric_scaled = numeric_preprocessor.fit_transform(X_train[present_numeric_cols])
         X_test_numeric_scaled = numeric_preprocessor.transform(X_test[present_numeric_cols])
         logger.info("Numeric preprocessor fitted and data transformed.")
         fitted_numeric_preprocessor = numeric_preprocessor # This is our fitted preprocessor now
    else:
         logger.warning("No original numeric columns found or struct_input_dim is zero. Skipping numeric preprocessor.")
         # Create empty arrays for numeric features to avoid errors later
         X_train_numeric_scaled = np.empty((len(X_train), 0))
         X_test_numeric_scaled = np.empty((len(X_test), 0))


    # Get embedding features (already computed and in X_train/X_test)
    if embedding_dim > 0:
         X_train_embeddings = X_train[embedding_cols_present].values.astype(np.float32)
         X_test_embeddings = X_test[embedding_cols_present].values.astype(np.float32) # Use test columns found
         logger.info(f"Embedding features shape: Train={X_train_embeddings.shape}, Test={X_test_embeddings.shape}")
    else:
         logger.warning("No embedding columns found or embedding_dim is zero. Skipping models that require embeddings.")
         X_train_embeddings = np.empty((len(X_train), 0))
         X_test_embeddings = np.empty((len(X_test), 0))


    # Combine features for models that expect a single input (LR, DT)
    # Ensure we handle cases where one or both feature types are missing
    X_train_combined = np.hstack([X_train_embeddings, X_train_numeric_scaled]).astype(np.float32) if X_train_embeddings.shape[1] > 0 or X_train_numeric_scaled.shape[1] > 0 else np.empty((len(X_train), 0))
    X_test_combined = np.hstack([X_test_embeddings, X_test_numeric_scaled]).astype(np.float32) if X_test_embeddings.shape[1] > 0 or X_test_numeric_scaled.shape[1] > 0 else np.empty((len(X_test), 0))

    if X_train_combined.shape[1] == 0:
         logger.critical("No features (numeric or embedding) available for training any model.");
         return {}, {}, {}, fitted_numeric_preprocessor, None, EMBEDDING_MODEL_NAME # Return empty results if no features


    logger.info(f"Combined features shape: Train={X_train_combined.shape}, Test={X_test_combined.shape}")

    # Determine which models to try based on available features
    models_to_try = []
    if X_train_combined.shape[1] > 0: # Can train LR/DT if any features are present
         models_to_try.extend(["LogisticRegression", "DecisionTree"])
    # Can train HybridNN only if both embeddings and structural features are present AND > 0 dim
    if embedding_dim > 0 and struct_input_dim > 0:
         models_to_try.append("HybridNN")
    elif embedding_dim > 0:
         logger.warning("HybridNN requires both embedding and structural features. Skipping HybridNN as structural features are missing.")
    elif struct_input_dim > 0:
         logger.warning("HybridNN requires both embedding and structural features. Skipping HybridNN as embedding features are missing.")


    # Adjust feature names for SHAP/LIME based on what's actually present
    final_feature_names = get_feature_names(present_numeric_cols, embedding_dim)
    # Cross-check calculated feature names length with actual combined feature shape
    if len(final_feature_names) != X_train_combined.shape[1]:
         logger.error(f"Mismatch between generated feature names ({len(final_feature_names)}) and combined feature shape ({X_train_combined.shape[1]}). Using generic names for XAI.")
         final_feature_names = [f"feature_{i}" for i in range(X_train_combined.shape[1])]


    models = {}
    results = {}
    explanations = {} # Will only store explanations for the best model


    # == Training Loop ==
    for model_name in models_to_try:
         logger.info(f"\n--- Training {model_name} ---")
         try:
             if model_name == "LogisticRegression":
                 # Train LR on combined features
                 lr_model = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', random_state=42, C=1.0)
                 logger.info(f"Fitting {model_name}...");
                 lr_model.fit(X_train_combined, y_train);
                 models[model_name] = lr_model;
                 logger.info(f"Evaluating {model_name}...");
                 y_pred = lr_model.predict(X_test_combined)

             elif model_name == "DecisionTree":
                 # Train DT on combined features
                 dt_model = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42, min_samples_split=10, min_samples_leaf=5)
                 logger.info(f"Fitting {model_name}...");
                 dt_model.fit(X_train_combined, y_train);
                 models[model_name] = dt_model;
                 logger.info(f"Evaluating {model_name}...");
                 y_pred = dt_model.predict(X_test_combined)

             elif model_name == "HybridNN":
                 # Train Hybrid NN on separate inputs
                 # This block is only reached if embedding_dim > 0 and struct_input_dim > 0 due to models_to_try list
                 keras_model = build_hybrid_nn(embedding_dim, struct_input_dim)
                 if keras_model:
                     logger.info("Starting Keras training...")
                     # Feed embeddings and scaled numeric features as separate inputs
                     nn_train_inputs = [X_train_embeddings, X_train_numeric_scaled]
                     nn_test_inputs = [X_test_embeddings, X_test_numeric_scaled]

                     nn_history = keras_model.fit(
                         nn_train_inputs, y_train,
                         epochs=15,
                         batch_size=64,
                         validation_split=0.15,
                         callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
                         verbose=1
                     )
                     logger.info("Keras training finished.")
                     models[model_name] = keras_model; # Store Keras model

                     logger.info(f"Evaluating {model_name} on test set...")
                     loss, accuracy, precision_keras, recall_keras = keras_model.evaluate(nn_test_inputs, y_test, verbose=0)

                     y_pred_proba = keras_model.predict(nn_test_inputs)
                     y_pred = (y_pred_proba > 0.5).astype(int).flatten()

                     # Store Keras evaluation metrics alongside sklearn metrics
                     report = classification_report(y_test, y_pred, output_dict=True, zero_division=0);
                     f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

                     results[model_name] = {
                         'accuracy': accuracy_score(y_test, y_pred), # Sklearn accuracy
                         'f1_score': f1, # Sklearn F1
                         'precision': precision_score(y_test, y_pred, average='binary', zero_division=0), # Sklearn Precision
                         'recall': recall_score(y_test, y_pred, average='binary', zero_division=0), # Sklearn Recall
                         'report': report,
                         'keras_test_loss': loss, # Keras evaluation metrics
                         'keras_test_accuracy': accuracy,
                         'keras_test_precision': precision_keras,
                         'keras_test_recall': recall_keras,
                         'history': {k: [float(val) for val in v] for k, v in nn_history.history.items()} # Store history
                     }
                     logger.info(f"{model_name} Training and Evaluation OK. F1 Score: {results[model_name]['f1_score']:.4f}")
                 else:
                     logger.warning("Skip NN: build failed.");
                     results[model_name] = {'error': 'Keras build failed.'}
                 continue # Continue outer loop after handling NN


             # For sklearn models (LR, DT) that were trained above
             report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
             results[model_name] = {
                 'accuracy': accuracy_score(y_test, y_pred),
                 'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0),
                 'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                 'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                 'report': report
             }
             logger.info(f"{model_name} Training and Evaluation OK. F1 Score: {results[model_name]['f1_score']:.4f}");


         except Exception as e:
             logger.error(f"Failed to train or evaluate {model_name}: {e}", exc_info=True);
             results[model_name] = {'error': str(e)}


    # --- Plot NN Training History ---
    # Plot history only if NN was trained and history exists
    if "HybridNN" in models and "HybridNN" in results and 'history' in results["HybridNN"]:
         # Need to reconstruct a dummy history object or just pass the dict
         class DummyHistory: # Helper class for plotting function
             def __init__(self, history_dict):
                  self.history = history_dict
         save_training_history_plot(DummyHistory(results["HybridNN"]["history"]), output_dir)


    # --- Model Selection ---
    best_model_name = select_best_model(results)
    best_model_obj = models.get(best_model_name) # Get the actual model object

    # --- XAI Explanation Generation (Only for the best model if one was selected) ---
    logger.info("\n--- Generating Explanations for a Sample Instance (Best Model Only) ---")

    if not best_model_name:
         logger.warning("No best model selected. Skipping XAI explanations.")
         # Return the features used so save_artifacts knows what was present
         return models, results, explanations, fitted_numeric_preprocessor, best_model_name, EMBEDDING_MODEL_NAME, present_numeric_cols, embedding_dim

    if best_model_name not in models or 'error' in results.get(best_model_name, {}):
        logger.warning(f"Best model '{best_model_name}' is not available or had errors. Skipping XAI.")
        return models, results, explanations, fitted_numeric_preprocessor, best_model_name, EMBEDDING_MODEL_NAME, present_numeric_cols, embedding_dim

    # Check if necessary components for XAI are available
    # SHAP needs features, LIME text needs text, embedding model, numeric preprocessor, sample data
    if X_test.empty:
        logger.error("Test set is empty. Cannot generate XAI explanations.");
        return models, results, explanations, fitted_numeric_preprocessor, best_model_name, EMBEDDING_MODEL_NAME, present_numeric_cols, embedding_dim

    # Select a sample instance from the test set for explanation
    sample_idx_in_test = 0
    if sample_idx_in_test >= len(X_test):
        logger.warning(f"Test set has only {len(X_test)} samples. Using index 0 for explanation.")
        sample_idx_in_test = 0

    try:
        # Get data for the sample
        original_sample_index = test_indices[sample_idx_in_test]
        sample_original_text = original_texts_test.iloc[sample_idx_in_test] if original_texts_test is not None else "Original text not available"
        sample_cleaned_text = cleaned_texts_test.iloc[sample_idx_in_test]

        # Get sample features based on what was actually used (same slicing as test sets)
        # Need to get the raw numeric features for the sample first, then scale them
        sample_raw_data_row = df.loc[[original_sample_index]] # Get original row from full df using original index

        sample_numeric_features_scaled = None
        if struct_input_dim > 0 and fitted_numeric_preprocessor:
             # Select original numeric columns from the sample row and transform
             sample_numeric_features_raw = sample_raw_data_row[present_numeric_cols]
             sample_numeric_features_scaled = fitted_numeric_preprocessor.transform(sample_numeric_features_raw) # Shape (1, n_numeric)
        else:
             sample_numeric_features_scaled = np.empty((1, 0)) # Shape (1, 0) if no numeric feats

        sample_embedding_features = X_test_embeddings[[sample_idx_in_test], :] if embedding_dim > 0 else np.empty((1, 0)) # Shape (1, n_embedding)
        sample_combined_features = np.hstack([sample_embedding_features, sample_numeric_features_scaled]).astype(np.float32) # Shape (1, n_total)

    except IndexError:
        logger.error(f"Cannot access sample index {sample_idx_in_test} from the test set indices. Test size: {len(X_test)}. Skipping XAI.");
        return models, results, explanations, fitted_numeric_preprocessor, best_model_name, EMBEDDING_MODEL_NAME, present_numeric_cols, embedding_dim
    except Exception as sample_e:
        logger.error(f"Error getting sample data for explanation: {sample_e}", exc_info=True);
        return models, results, explanations, fitted_numeric_preprocessor, best_model_name, EMBEDDING_MODEL_NAME, present_numeric_cols, embedding_dim

    logger.info(f"Explaining instance with original index: {original_sample_index}")
    logger.info(f"Sample Cleaned Text: '{sample_cleaned_text[:500]}...'")

    # --- Generate Explanations for the Best Model ---
    model_name = best_model_name
    model_obj = models[model_name] # Use the model object obtained earlier

    logger.info(f"--- Generating XAI Explanations for the best model: {model_name} ---")

    model_explanations = {}

    # == SHAP ==
    # SHAP explanation on the combined feature space (embeddings + scaled numeric)
    shap_exp = None
    try:
        logger.info(f"Generating SHAP explanation for {model_name} on combined features...")

        # Prepare background data for SHAP
        # Use X_train_combined which contains the features used for training LR/DT/HybridNN (if both existed)
        shap_background_data = X_train_combined
        num_shap_background_samples = min(500, shap_background_data.shape[0])

        if num_shap_background_samples == 0 or shap_background_data.shape[1] == 0:
             logger.warning("Training set is empty or has no features. Cannot create SHAP background data or explain. Skipping SHAP.")
             model_explanations['shap'] = {'error': 'SHAP background data unavailable or no features.'}
             # Do not raise, just mark as error and continue to LIME

        else: # Proceed with SHAP if background data is available
             try:
                 # Select random indices from background data
                 shap_background_indices = np.random.choice(shap_background_data.shape[0], size=num_shap_background_samples, replace=False)
                 shap_background_subset = shap_background_data[shap_background_indices, :].astype(np.float32)
                 logger.info(f"Prepared SHAP background subset shape: {shap_background_subset.shape}")

                 # Check for NaN/Inf in background subset
                 if np.any(np.isnan(shap_background_subset)) or np.any(np.isinf(shap_background_subset)):
                      logger.warning("NaN/Inf found in SHAP background subset. Imputing NaNs/Infs...")
                      try:
                          imputer_temp = SimpleImputer(strategy='median')
                          shap_background_subset = imputer_temp.fit_transform(shap_background_subset)
                          logger.warning("NaN/Inf imputation applied to SHAP background subset.")
                      except Exception as imp_e:
                          logger.error(f"Failed to impute NaN/Inf in SHAP background subset: {imp_e}", exc_info=True)
                          shap_background_subset = None # Set to None if imputation fails

                 if shap_background_subset is None or shap_background_subset.shape[0] == 0 or shap_background_subset.shape[1] == 0:
                      logger.warning("SHAP background subset is None/empty after handling NaN/Inf. Cannot explain.")
                      model_explanations['shap'] = {'error': 'SHAP background data unavailable after cleaning.'}
                 else: # Background subset is ready
                      # Check for NaN/Inf in sample instance data
                      sample_combined_features_clean = sample_combined_features.copy() # Work on a copy
                      if np.any(np.isnan(sample_combined_features_clean)) or np.any(np.isinf(sample_combined_features_clean)):
                           logger.warning("NaN/Inf found in sample instance features. Imputing NaNs/Infs...")
                           try:
                               imputer_temp_inst = SimpleImputer(strategy='median')
                               sample_combined_features_clean = imputer_temp_inst.fit_transform(sample_combined_features_clean)
                               logger.warning("NaN/Inf imputation applied to sample instance features.")
                           except Exception as imp_e:
                               logger.error(f"Failed to impute NaN/Inf in sample instance features: {imp_e}", exc_info=True)
                               sample_combined_features_clean = None # Set to None if imputation fails

                      if sample_combined_features_clean is None or sample_combined_features_clean.shape[0] == 0 or sample_combined_features_clean.shape[1] == 0:
                           logger.warning("Sample instance features are None/empty after handling NaN/Inf. Cannot explain.")
                           model_explanations['shap'] = {'error': 'Sample instance features unavailable after cleaning.'}

                      else: # Sample instance data is ready
                           shap_values = None; explainer_shap = None; base_value = None; feature_names = final_feature_names # Use the determined feature names

                           # Determine explainer based on model type
                           if isinstance(model_obj, (LogisticRegression, DecisionTreeClassifier)):
                                logger.info(f"Using shap.LinearExplainer or TreeExplainer for {model_name}...")
                                # These explainers work directly on the combined feature array
                                if isinstance(model_obj, LogisticRegression):
                                     explainer_shap = shap.LinearExplainer(model_obj, shap_background_subset)
                                elif isinstance(model_obj, DecisionTreeClassifier):
                                     # Use background data for TreeExplainer as well
                                     explainer_shap = shap.TreeExplainer(model_obj, shap_background_subset, feature_perturbation="tree_path_dependent")
                                shap_values = explainer_shap.shap_values(sample_combined_features_clean)
                                base_value = explainer_shap.expected_value if hasattr(explainer_shap, 'expected_value') else None

                                # Process SHAP values for binary classification [shap_for_class_0, shap_for_class_1]
                                positive_class_index = 1
                                if isinstance(shap_values, list) and len(shap_values) > positive_class_index:
                                     positive_class_shap_values = shap_values[positive_class_index]
                                     if isinstance(base_value, (list, np.ndarray)) and len(base_value) > positive_class_index:
                                         base_value = base_value[positive_class_index]
                                     elif isinstance(base_value, (list, np.ndarray)) and len(base_value) > 0:
                                         base_value = base_value[0] # Fallback if list length doesn't match
                                else:
                                     positive_class_shap_values = shap_values # Use as is if not a list

                           elif isinstance(model_obj, KerasModel):
                                logger.info(f"Using shap.DeepExplainer for Keras {model_name}...")
                                # DeepExplainer needs background and instance inputs matching model inputs
                                # Split background data and sample instance data into embeddings and numeric parts
                                # Handle zero dimensions
                                background_inputs = []
                                if embedding_dim > 0: background_inputs.append(shap_background_subset[:, :embedding_dim].astype(np.float32))
                                if struct_input_dim > 0: background_inputs.append(shap_background_subset[:, embedding_dim:].astype(np.float32))

                                instance_inputs = []
                                if embedding_dim > 0: instance_inputs.append(sample_combined_features_clean[:, :embedding_dim].astype(np.float32))
                                if struct_input_dim > 0: instance_inputs.append(sample_combined_features_clean[:, embedding_dim:].astype(np.float32))

                                if not background_inputs or not instance_inputs: # Should not happen if dim > 0 checks pass, but safety
                                     logger.warning("Skip Keras SHAP: Input data preparation failed.")
                                     model_explanations['shap'] = {'error': 'Keras input data preparation failed.'}
                                     raise ValueError("Keras input data preparation failed for SHAP.") # Flag to skip

                                explainer_shap = shap.DeepExplainer(model_obj, background_inputs)
                                shap_values = explainer_shap.shap_values(instance_inputs) # Shape [n_classes, n_samples, n_features]

                                base_value = explainer_shap.expected_value # Shape [n_classes]

                                # Access values/base value for the positive class (index 1) and the single sample (index 0)
                                positive_class_shap_values = shap_values[1][0] if isinstance(shap_values, list) and len(shap_values) > 1 else (shap_values[0] if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1 else shap_values) # Handle potential (1, n_features) case
                                base_value = base_value[1] if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1 else (base_value[0] if isinstance(base_value, (list, np.ndarray)) else base_value) # Handle potential list/array case

                           else:
                                logger.warning(f"Unsupported SHAP explainer type for {model_name}: {type(model_obj)}. Skipping SHAP.")
                                model_explanations['shap'] = {'error': 'Unsupported model type for SHAP'};
                                raise ValueError("Unsupported model type for SHAP") # Flag to skip

                           # Ensure values, data, base_value are in correct format for Explanation object
                           # sample_combined_features_clean should be (1, n_features), need (n_features,) for Explanation data
                           data_for_explanation = sample_combined_features_clean.flatten() if sample_combined_features_clean.ndim > 1 else sample_combined_features_clean
                           # Ensure positive_class_shap_values is 1D
                           if isinstance(positive_class_shap_values, np.ndarray) and positive_class_shap_values.ndim > 1 and positive_class_shap_values.shape[0] == 1:
                                positive_class_shap_values = positive_class_shap_values.flatten()
                           elif not isinstance(positive_class_shap_values, np.ndarray): # Convert list/etc to array
                                positive_class_shap_values = np.asarray(positive_class_shap_values)

                           if isinstance(base_value, np.ndarray) and base_value.size == 1: base_value = base_value.item() # Convert numpy scalar to python scalar

                           # Create the SHAP Explanation object
                           if positive_class_shap_values is not None and data_for_explanation is not None and feature_names is not None:
                                if len(positive_class_shap_values) != len(feature_names) or len(data_for_explanation) != len(feature_names):
                                     logger.error("SHAP values/data/feature_names length mismatch. Cannot create Explanation object.")
                                     model_explanations['shap'] = {'error': 'SHAP data dimension mismatch'}
                                else:
                                     # Handle potential object types in base_value from Keras Explainer
                                     if not isinstance(base_value, (int, float, type(None))):
                                         logger.warning(f"Unexpected base_value type for SHAP ({type(base_value)}). Setting to None.")
                                         base_value = None

                                     shap_exp = shap.Explanation(values=positive_class_shap_values, base_values=base_value, data=data_for_explanation, feature_names=feature_names)
                                     model_explanations['shap'] = shap_exp; logger.info("SHAP OK.")
                           else:
                                logger.warning(f"SHAP values, data, or feature names are None for {model_name}. Cannot create Explanation object.")
                                model_explanations['shap'] = {'error': 'Incomplete SHAP data'}

             except ValueError as ve:
                  if "Keras input data preparation failed for SHAP" in str(ve) or "Unsupported model type for SHAP" in str(ve): pass # Already logged
                  else: logger.error(f"SHAP failed background/sample prep for {model_name} (ValueError): {ve}", exc_info=True); model_explanations['shap'] = {'error': str(ve)}
             except TypeError as te:
                  logger.error(f"SHAP failed background/sample prep for {model_name} (TypeError): {te}", exc_info=True); model_explanations['shap'] = {'error': str(te)}
             except Exception as e:
                  logger.error(f"SHAP failed background/sample prep for {model_name}: {e}", exc_info=True);
                  model_explanations['shap'] = {'error': str(e)}


    except ValueError as ve:
        if "SHAP background data unavailable" in str(ve): pass # Already logged
        else: logger.error(f"SHAP failed {model_name} (ValueError - Outer): {ve}", exc_info=True); model_explanations['shap'] = {'error': str(ve)}
    except Exception as e:
        logger.error(f"SHAP failed {model_name} (Outer block error): {e}", exc_info=True);
        model_explanations['shap'] = {'error': str(e)}


    # == LIME ==
    # LIME text explanation on raw text, but using the model trained on embeddings + features
    lime_exp = None
    try:
        logger.info(f"Initializing LIME text explainer for {model_name}...")
        # Need to use the cleaned text for LIME
        # Create the wrapper function/object that LIME will call for prediction
        # Pass the best model, embedding model, numeric preprocessor, and the sample's scaled numeric features
        if model_obj is None or fitted_numeric_preprocessor is None or embedding_model is None:
             logger.warning(f"Cannot initialize LIME for {model_name}: Model, numeric preprocessor, or embedding model is None. Skipping LIME.")
             model_explanations['lime'] = {'error': 'Required components for LIME are missing.'}
             # Do not raise, just mark as error
        elif embedding_dim <= 0:
             logger.warning("Embedding dimension is 0. Cannot use LIME text explainer with this setup. Skipping LIME.")
             model_explanations['lime'] = {'error': 'Embedding dimension is zero.'}
             # Do not raise, just mark as error
        else: # All required components seem present for LIME
             # Ensure sample_numeric_features_scaled is correctly shaped (1, n_features) for the wrapper
             sample_numeric_features_scaled_for_wrapper = sample_numeric_features_scaled.copy()
             if sample_numeric_features_scaled_for_wrapper.ndim == 1: sample_numeric_features_scaled_for_wrapper = sample_numeric_features_scaled_for_wrapper.reshape(1, -1)

             predict_proba_wrapper = PredictProbaWrapper(
                 model=model_obj,
                 embedding_model=embedding_model, # Pass the actual model object
                 numeric_preprocessor=fitted_numeric_preprocessor,
                 original_numeric_features_scaled=sample_numeric_features_scaled_for_wrapper, # Pass the scaled numeric features for the specific instance
                 embedding_dim=embedding_dim, # Pass embedding dimension
                 original_numeric_cols=present_numeric_cols # Pass the list of numeric columns used
             )

             logger.info(f"Calculating LIME explanation for {model_name} on raw text...");
             # LIME explainer takes the prediction function wrapper
             lime_explainer = LimeTextExplainer(class_names=['Legitimate', 'Phishing'])
             lime_exp = lime_explainer.explain_instance(
                 sample_cleaned_text, # Explain the cleaned text
                 predict_proba_wrapper, # Use our custom wrapper for prediction
                 num_features=15, # Max features to show
                 num_samples=1000 # Increased samples for better stability
             )
             model_explanations['lime'] = lime_exp;
             logger.info(f"LIME explanation calculated for {model_name}.")


    except Exception as e:
        logger.error(f"LIME failed for {model_name}: {e}", exc_info=True);
        model_explanations['lime'] = {'error': str(e)}

    explanations[model_name] = model_explanations # Store explanations for the best model

    # Return the models, results, explanations (only for best), numeric preprocessor, and the best model name
    # Also return embedding model name or path and the list of present numeric columns
    return models, results, explanations, fitted_numeric_preprocessor, best_model_name, EMBEDDING_MODEL_NAME, present_numeric_cols, embedding_dim


# Saving Artifacts Function (Adjusted)
def save_artifacts(output_dir, models, results, explanations, numeric_preprocessor, best_model_name, embedding_model_name, present_numeric_cols, embedding_dim):
    logger = logging.getLogger();
    logger.info(f"\n--- Saving Artifacts to: {output_dir} ---")
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "models"); os.makedirs(models_dir, exist_ok=True)

    # Save the NUMERIC PREPROCESSOR if available
    if numeric_preprocessor:
        try:
            preprocessor_path = os.path.join(models_dir, "numeric_preprocessor.pkl");
            joblib.dump(numeric_preprocessor, preprocessor_path, compress=3);
            logger.info(f"Saved numeric preprocessor: {preprocessor_path}")
        except Exception as e: logger.error(f"Failed saving numeric preprocessor: {e}", exc_info=True)
    else:
        logger.warning("Numeric preprocessor was not fitted/available. Skipping save.")

    # Save the list of numeric columns that were actually used
    try:
        numeric_cols_info_path = os.path.join(models_dir, "numeric_cols_info.json");
        with open(numeric_cols_info_path, 'w') as f:
            json.dump({"numeric_columns": present_numeric_cols}, f, indent=2)
        logger.info(f"Saved numeric columns info: {numeric_cols_info_path}")
    except Exception as e: logger.error(f"Failed saving numeric columns info: {e}", exc_info=True)


    # Save the EMBEDDING MODEL IDENTIFIER and DIMENSION if available
    if embedding_model_name or embedding_dim > 0:
        try:
            embedding_info_path = os.path.join(models_dir, "embedding_model_info.json");
            with open(embedding_info_path, 'w') as f:
                json.dump({"model_name": embedding_model_name if embedding_model_name else "unknown", "embedding_dimension": embedding_dim}, f, indent=2)
            logger.info(f"Saved embedding model info: {embedding_info_path}")
        except Exception as e: logger.error(f"Failed saving embedding model info: {e}", exc_info=True)
    else:
         logger.warning("Embedding model name or dimension not available. Skipping embedding model info save.")


    # Save ONLY the BEST MODEL
    if best_model_name and best_model_name in models:
        model_to_save = models[best_model_name]
        try:
            if best_model_name == "HybridNN" and isinstance(model_to_save, KerasModel):
                 model_path = os.path.join(models_dir, f"{best_model_name}.keras");
                 # Save Keras model directly in the Keras format
                 model_to_save.save(model_path);
                 logger.info(f"Saved best model ({best_model_name}): {model_path}")
            # Scikit-learn models are saved directly (no pipeline needed as preproc is separate)
            elif isinstance(model_to_save, (LogisticRegression, DecisionTreeClassifier)):
                 model_path = os.path.join(models_dir, f"{best_model_name}.pkl");
                 joblib.dump(model_to_save, model_path, compress=3);
                 logger.info(f"Saved best model ({best_model_name}): {model_path}")
            else:
                 logger.warning(f"Best model '{best_model_name}' type {type(model_to_save)} not saved (not Keras, LR, or DT).")
        except Exception as e: logger.error(f"Failed saving best model ({best_model_name}): {e}", exc_info=True)
    elif best_model_name:
         logger.warning(f"Best model '{best_model_name}' not found in models dictionary. Cannot save it.")
    else:
         logger.warning("No best model name provided. No model saved.")


    # Save ALL Results (Performance of all models)
    try:
        results_path = os.path.join(output_dir, "results.json")
        # Define a custom serializer to handle non-JSON serializable objects
        def default_serializer(obj):
            if isinstance(obj, (np.integer, int)): return int(obj)
            elif isinstance(obj, (np.floating, float)):
                 if np.isnan(obj): return 'NaN'
                 if np.isposinf(obj): return 'Infinity'
                 if np.isneginf(obj): return '-Infinity'
                 return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)): return bool(obj)
            elif isinstance(obj, (bytes, bytearray)): return obj.decode('utf-8', errors='replace')
            # Handle Keras History object if present in results (should be in 'history' key)
            if isinstance(obj, tf.keras.callbacks.History):
                 # Serialize history dictionary instead of the object
                 return {k: [float(val) for val in v] for k, v in obj.history.items()}
            try:
                 # Attempt default JSON encoding
                 return json.JSONEncoder.default(json.JSONEncoder(), obj)
            except TypeError:
                 # Fallback to string representation for unknown types
                 return str(obj)

        # Recursive function to apply the serializer to nested dictionaries and lists
        def convert_dict(item):
             if isinstance(item, dict):
                  return {k: convert_dict(v) for k, v in item.items()}
             elif isinstance(item, list):
                  return [convert_dict(elem) for elem in item]
             else:
                  return default_serializer(item)

        # Apply the conversion to the results dictionary
        serializable_results = convert_dict(results)

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Saved full results (performance of all models): {results_path}")
    except Exception as e: logger.error(f"Failed saving results: {e}", exc_info=True)


    # Save Explanations (Only for the BEST model)
    explanations_dir = os.path.join(output_dir, "explanations"); os.makedirs(explanations_dir, exist_ok=True)

    if best_model_name and best_model_name in explanations:
        model_explanations = explanations[best_model_name]
        if not model_explanations:
             logger.warning(f"No explanations generated for the best model '{best_model_name}'. Skipping saving explanations.")
        else:
            model_expl_dir = os.path.join(explanations_dir, best_model_name.replace(" ", "_"));
            os.makedirs(model_expl_dir, exist_ok=True)
            logger.info(f"Saving explanations for best model ({best_model_name})...")

            # SHAP Save (Plots use the Explanation object directly)
            shap_exp = model_explanations.get('shap')
            if shap_exp and isinstance(shap_exp, shap.Explanation):
                try:
                    # Ensure SHAP values and data are in the correct shape for plotting (often 1D or (1, n_features))
                    if shap_exp.values is None or (isinstance(shap_exp.values, np.ndarray) and shap_exp.values.size == 0) or shap_exp.data is None or shap_exp.feature_names is None:
                         logger.warning(f"SHAP explanation data is incomplete/invalid for {best_model_name}. Skipping SHAP plots.")
                    else:
                         # Prepare data for plotting (ensure 1D for plots that expect it)
                         vals_for_plot = shap_exp.values.flatten() if isinstance(shap_exp.values, np.ndarray) and shap_exp.values.ndim > 1 else np.asarray(shap_exp.values)
                         data_for_plot = shap_exp.data.flatten() if isinstance(shap_exp.data, np.ndarray) and shap_exp.data.ndim > 1 else np.asarray(shap_exp.data)
                         base_values_for_plot = shap_exp.base_values
                         if isinstance(base_values_for_plot, (list, np.ndarray)) and len(base_values_for_plot) > 0:
                             base_values_for_plot = base_values_for_plot[0] # Use the first base value for binary classification/scalar

                         if vals_for_plot.shape != data_for_plot.shape or len(vals_for_plot) != len(shap_exp.feature_names):
                             logger.warning(f"SHAP values/data/feature_names shape mismatch for {best_model_name}. Skipping SHAP plots.")
                         else:
                             # Create a plot-compatible Explanation object if needed (flattened data/values)
                             plot_exp_data = shap.Explanation(values=vals_for_plot, base_values=base_values_for_plot, data=data_for_plot, feature_names=shap_exp.feature_names)

                             # Waterfall Plot
                             fig_wf = plt.figure()
                             try:
                                 shap.plots.waterfall(plot_exp_data, max_display=20, show=False)
                                 plt.tight_layout();
                                 plt.savefig(os.path.join(model_expl_dir, "shap_waterfall.png"), dpi=150, bbox_inches='tight')
                                 logger.info(f"Saved SHAP waterfall plot for {best_model_name}.")
                             except Exception as wf_e:
                                 logger.warning(f"SHAP waterfall failed for {best_model_name}: {wf_e}. Trying summary plot as fallback.", exc_info=False)
                                 plt.close(fig_wf);
                                 # Summary Plot (Fallback)
                                 fig_sum = plt.figure()
                                 try:
                                     # Summary plot expects values as (n_samples, n_features), data as (n_samples, n_features)
                                     # Need to reshape our single instance data/values to (1, n_features)
                                     shap.summary_plot(vals_for_plot.reshape(1, -1), features=data_for_plot.reshape(1, -1), feature_names=shap_exp.feature_names, max_display=20, show=False)
                                     plt.tight_layout();
                                     plt.savefig(os.path.join(model_expl_dir, "shap_summary_fallback.png"), dpi=150, bbox_inches='tight')
                                     logger.info(f"Saved SHAP summary fallback plot for {best_model_name}.")
                                 except Exception as sum_e:
                                     logger.error(f"SHAP summary fallback failed for {best_model_name}: {sum_e}", exc_info=True)
                                 finally:
                                     plt.close(fig_sum)

                             finally:
                                 if 'fig_wf' in locals() and plt.fignum_exists(fig_wf.number): plt.close(fig_wf)

                             # Force plot (Requires IPython display, saves to HTML)
                             try:
                                 force_plot_html = shap.force_plot(base_values_for_plot, vals_for_plot, features=data_for_plot, feature_names=shap_exp.feature_names, show=False, matplotlib=False)
                                 if force_plot_html:
                                     shap.save_html(os.path.join(model_expl_dir, "shap_force_plot.html"), force_plot_html)
                                     logger.info(f"Saved SHAP force plot HTML for {best_model_name}.")
                                 else:
                                     logger.warning(f"SHAP force_plot for {best_model_name} did not return an HTML object.")
                             except ImportError:
                                 logger.warning("IPython display backend not found. Skipping SHAP force plot HTML generation.")
                             except Exception as force_err:
                                 logger.warning(f"SHAP force plot failed for {best_model_name}: {force_err}", exc_info=False)

                except Exception as e:
                    logger.error(f"Failed saving SHAP plots for {best_model_name}: {e}", exc_info=True)
            elif shap_exp and isinstance(shap_exp, dict) and 'error' in shap_exp:
                logger.warning(f"Skip SHAP plots for {best_model_name} due to error: {shap_exp['error']}")
            else:
                 logger.warning(f"SHAP explanation object is not valid for saving for {best_model_name}.")


            # LIME Save
            lime_exp = model_explanations.get('lime')
            if lime_exp and hasattr(lime_exp, 'save_to_file'):
                try:
                    lime_html_path = os.path.join(model_expl_dir, 'lime_explanation.html');
                    lime_exp.save_to_file(lime_html_path);
                    logger.info(f"Saved LIME HTML for {best_model_name}.")
                except Exception as e: logger.error(f"Failed saving LIME HTML for {best_model_name}: {e}", exc_info=True)
            elif lime_exp and isinstance(lime_exp, dict) and 'error' in lime_exp:
                logger.warning(f"Skip LIME saving for {best_model_name} due to error: {lime_exp['error']}")
            else:
                 logger.warning(f"LIME explanation object is not valid for saving for {best_model_name}.")

    elif best_model_name:
         logger.warning(f"Best model '{best_model_name}' not found in explanations dictionary. Cannot save explanations.")
    else:
         logger.warning("No best model name provided. No explanations saved.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing Detection Training and Explanation Pipeline with Embeddings")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing raw CSV data files.")
    parser.add_argument("--output-dir", type=str, default="phishing_results_embeddings", help="Directory to save models, results, and explanations.")
    args = parser.parse_args()

    # Configure root logger
    logger = setup_logging(args.output_dir)

    # Define the path for the saved processed data file
    processed_data_path = os.path.join(args.output_dir, "processed_data.parquet")

    try:
        logger.info("🚀 Starting Phishing Detection Pipeline with Embeddings 🚀")
        logger.info(f"Script Arguments: {args}")

        # Log TensorFlow status explicitly
        logger.info(f"TensorFlow Loaded: True (Version: {tf.__version__})")

        logger.info(f"Data Input Directory: {args.data_dir}")
        logger.info(f"Output Directory: {args.output_dir}")

        # --- Attempt to load processed data ---
        df_processed = load_processed_data(processed_data_path)

        if df_processed is None: # If loading failed or file didn't exist
             # --- Load Multilingual Embedding Model ---
             logger.info(f"Loading Sentence Transformer model: {EMBEDDING_MODEL_NAME}...")
             try:
                 embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                 logger.info("Sentence Transformer model loaded successfully.")
             except Exception as e:
                 logger.critical(f"Failed to load Sentence Transformer model {EMBEDDING_MODEL_NAME}: {e}", exc_info=True)
                 logger.critical("Please ensure you have an internet connection or the model files downloaded.")
                 # Do not exit here, continue without embeddings but log severe warning
                 embedding_model = None
                 pass # Continue execution

             # Load and preprocess data (now includes embedding computation if model loaded)
             logger.info("Processing data from raw CSV files...")
             df_processed = load_and_preprocess_data(args.data_dir, embedding_model)

             # --- Save processed data ---
             if df_processed is not None and not df_processed.empty:
                 save_processed_data(df_processed, processed_data_path)
             else:
                  logger.error("Processed DataFrame is empty. Skipping save.")
        else:
             # Embedding model needs to be loaded even if data is loaded from file, for LIME XAI
             logger.info(f"Processed data loaded from {processed_data_path}. Loading embedding model for XAI...")
             try:
                 embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                 logger.info("Sentence Transformer model loaded successfully for XAI.")
             except Exception as e:
                 logger.error(f"Failed to load Sentence Transformer model {EMBEDDING_MODEL_NAME} for XAI: {e}", exc_info=True)
                 logger.warning("LIME explanations will be skipped if needed.")
                 embedding_model = None
                 pass # Continue execution


        # Proceed with training, evaluation, and XAI if data was loaded/processed
        if df_processed is None or df_processed.empty:
             logger.critical("No processed data available to train models. Exiting.");
             sys.exit(1)

        # Define original_numeric_cols here, used by train_evaluate_explain and save_artifacts
        original_numeric_cols = ['num_links', 'has_suspicious_url', 'urgency_count', 'readability_score']


        # Train, evaluate, select best model, and generate explanations for the best model
        # Pass the global embedding_model and original_numeric_cols
        models, results, explanations, numeric_preprocessor, best_model_name, embedding_model_name_saved, present_numeric_cols, embedding_dim = train_evaluate_explain(
            df_processed, output_dir=args.output_dir, original_numeric_cols=original_numeric_cols, embedding_model=embedding_model
        )

        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Save artifacts: numeric preprocessor, embedding model info, BEST model, ALL results, Explanations for BEST model, Plots
        # Pass the global embedding_model_name_saved and the list of present_numeric_cols and embedding_dim
        save_artifacts(args.output_dir, models, results, explanations, numeric_preprocessor, best_model_name, embedding_model_name_saved, present_numeric_cols, embedding_dim)

        # Save the F1 comparison plot for all models (uses results, not dependent on models dict content after saving)
        save_f1_visualization(results, args.output_dir)
        # Training history plot for NN (if it was trained) is handled within train_evaluate_explain


        logger.info("✅ Pipeline Completed Successfully! ✅")

    except ImportError as e:
         logger.critical(f"ImportError: {e}. Ensure all required libraries (pandas, numpy, sklearn, tensorflow, shap, lime, transformers, sentence-transformers, beautifulsoup4, lxml, textstat, nltk, pyarrow, fastparquet) are installed.")
         logger.critical("For Parquet support, install 'pyarrow' and 'fastparquet': pip install pyarrow fastparquet")
         sys.exit(1)
    except LookupError as e: # Catch NLTK LookupError if it happens despite internal handling
         logger.critical(f"NLTK LookupError: {e}. NLTK data resources are missing and automatic download failed. Please try manual download: import nltk; nltk.download('all')");
         sys.exit(1)
    except FileNotFoundError as e: logger.critical(f"File Not Found Error: {e}. Check --data-dir exists and contains CSVs."); sys.exit(1)
    except pd.errors.EmptyDataError as e: logger.critical(f"Empty Data Error: {e}. Data files might be empty."); sys.exit(1)
    except KeyError as e: logger.critical(f"Column Key Error: {e}. Check column names in your CSV data or if expected embedding columns were created."); sys.exit(1)
    except MemoryError as e: logger.critical(f"Memory Error: {e}. The dataset, embedding model, or features might be too large for available RAM. Consider smaller batch sizes or models."); sys.exit(1)
    except RuntimeError as e: logger.critical(f"Runtime Error: {e}"); sys.exit(1) # Catch specific runtime errors raised by the script
    except Exception as e: logger.critical(f"Unexpected critical error: {e}", exc_info=True); sys.exit(1)
