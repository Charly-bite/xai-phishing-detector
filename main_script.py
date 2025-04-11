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
from scipy.sparse import issparse

# --- Scikit-learn ---
# (Keep all sklearn imports)
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
# (Keep XAI imports)
import shap
from lime import lime_text

# --- TensorFlow / Keras Configuration ---
# Set TF environment variables BEFORE importing TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress INFO and WARNING messages

# --- TensorFlow / Keras Import (Now MANDATORY) ---
# REMOVED the try...except block. Script will fail here if TF is not installed.
import tensorflow as tf
# If using TF<2.0, some Keras paths might be different
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
print(f"--- TensorFlow Loaded (Version: {tf.__version__}) ---") # Confirmation

# --------------------------------------
# Global Configuration & Setup
# --------------------------------------
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Logging Setup Function ---
# (Keep setup_logging function as is)
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
logger = logging.getLogger();
if not logger.handlers: logger.addHandler(logging.StreamHandler(sys.stdout)); logger.setLevel(logging.INFO)

# --- NLTK Setup ---
# (Keep NLTK setup as is)
try:
    nltk.data.find('corpora/stopwords'); nltk.data.find('tokenizers/punkt'); logger.debug("NLTK OK.")
except nltk.downloader.DownloadError as e:
    logger.warning(f"NLTK resource missing ({e}). Downloading...");
    if 'stopwords' in str(e) or not nltk.data.find('corpora/stopwords', quiet=True): logger.info("Downloading stopwords..."); nltk.download('stopwords', quiet=True)
    if 'punkt' in str(e) or not nltk.data.find('tokenizers/punkt', quiet=True): logger.info("Downloading punkt..."); nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))


# --------------------------------------
# Preprocessing Functions
# (Keep clean_email, extract_url_features, extract_urgency, calculate_readability as they are)
# --------------------------------------
def clean_email(text):
    logger = logging.getLogger();
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "": return ""
    try:
        try: soup = BeautifulSoup(text, 'lxml')
        except: soup = BeautifulSoup(text, 'html.parser')
        cleaned = soup.get_text(separator=' ')
        cleaned = re.sub(r'https?://\S+|www\.\S+', ' URL ', cleaned)
        cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?\'`]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    except Exception as e: logger.error(f"Error cleaning text: '{str(text)[:50]}...': {e}",exc_info=False); return ""
def extract_url_features(text):
    num_links, has_suspicious_url = 0, 0; logger = logging.getLogger() # Get logger
    if pd.isna(text) or not isinstance(text, str): return 0, 0
    try:
        url_pattern = r'(https?://\S+|www\.\S+)'
        text_snippet_for_findall = text[:50000]
        urls = re.findall(url_pattern, text_snippet_for_findall)
        try: num_links = len(re.findall(url_pattern, text))
        except Exception as count_e: logger.warning(f"URL count error: {count_e}. Using snippet count."); num_links = len(urls)
        if not urls: return num_links, has_suspicious_url
        suspicious_keywords = ['login', 'verify', 'account', 'secure', 'update', 'confirm', 'signin', 'support', 'password', 'banking', 'activity', 'credential']
        shortened_domains_pattern = r'(bit\.ly/|goo\.gl/|tinyurl\.com/|t\.co/|ow\.ly/|is\.gd/|buff\.ly/|adf\.ly/|bit\.do/|soo\.gd/)'
        has_http, has_shortener, has_keywords = 0, 0, 0
        for url in urls[:100]:
            try:
                url_lower = url.lower()
                if url_lower.startswith('http://'): has_http = 1
                if re.search(shortened_domains_pattern, url_lower): has_shortener = 1
                proto_end = url_lower.find('//'); path_query = ''
                if proto_end > 0: domain_part_end = url_lower.find('/', proto_end + 2); path_query = url_lower[domain_part_end:] if domain_part_end > 0 else ''
                else: domain_part_end = url_lower.find('/'); path_query = url_lower[domain_part_end:] if domain_part_end > 0 else ''
                check_string = path_query + url_lower
                if any(keyword in check_string for keyword in suspicious_keywords): has_keywords = 1
                if has_http or has_shortener or has_keywords: has_suspicious_url = 1; break
            except Exception as url_parse_e: logger.debug(f"URL parse error '{url[:50]}...': {url_parse_e}"); continue
        return num_links, has_suspicious_url
    except Exception as e: logger.error(f"URL feature extraction failed: '{str(text)[:50]}...': {e}",exc_info=False); return 0, 0
def extract_urgency(cleaned_text):
    logger = logging.getLogger();
    if not cleaned_text: return 0
    try:
        urgency_words = ['urgent', 'immediately', 'action required', 'verify', 'password', 'alert', 'warning', 'limited time', 'expire', 'suspended', 'locked', 'important', 'final notice', 'response required', 'security update', 'confirm account', 'validate', 'due date', 'restricted', 'compromised', 'unauthorized']
        text_lower = cleaned_text.lower(); count = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower)) for word in urgency_words)
        return count
    except Exception as e: logger.error(f"Urgency calculation failed: {e}", exc_info=False); return 0
def calculate_readability(cleaned_text):
    logger = logging.getLogger(); word_count = len(cleaned_text.split())
    if not cleaned_text or word_count < 10: return 100.0
    try:
        score = textstat.flesch_reading_ease(cleaned_text); return max(-200, min(120, score)) if not np.isnan(score) else 50.0
    except Exception as e:
        if word_count > 5: logger.debug(f"Readability failed: '{cleaned_text[:50]}...': {e}", exc_info=False)
        return 50.0


# --------------------------------------
# Data Loading and Preprocessing Pipeline
# (Keep load_and_preprocess_data as is)
# --------------------------------------
def load_and_preprocess_data(data_dir, label_col_hints=None, text_col_hints=None):
    logger = logging.getLogger();
    if label_col_hints is None: label_col_hints = ['label', 'class', 'target', 'phishing', 'type']
    if text_col_hints is None: text_col_hints = ['text', 'body', 'email', 'content', 'message']
    all_dfs = []; required_columns = ['original_text', 'cleaned_text', 'label', 'num_links', 'has_suspicious_url', 'urgency_count', 'readability_score']
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
            logger.info(f"Raw columns: {df.columns.tolist()}"); df.columns = df.columns.str.strip()
            text_col = next((col for col in df.columns if any(hint in col.lower() for hint in text_col_hints)), None)
            label_col = next((col for col in df.columns if any(hint in col.lower() for hint in label_col_hints)), None)
            if not text_col: logger.error(f"No text column in {filename}. Skipping."); continue
            if not label_col: logger.error(f"No label column in {filename}. Skipping."); continue
            logger.info(f"Identified Text: '{text_col}', Label: '{label_col}'")
            if len(df) < 10: logger.warning(f"{filename} has < 10 rows. Skipping."); continue
            df.dropna(subset=[text_col], inplace=True)
            if df.empty: logger.warning(f"No rows after dropping missing text in {filename}. Skipping."); continue
            df_processed = pd.DataFrame(); df_processed['original_text'] = df[text_col].astype(str); df_processed['label'] = df[label_col]
            logger.info("Cleaning text..."); df_processed['cleaned_text'] = df_processed['original_text'].apply(clean_email)
            logger.info("Extracting URL features..."); url_features = df_processed['original_text'].apply(extract_url_features)
            df_processed['num_links'] = url_features.apply(lambda x: x[0]).astype(int)
            df_processed['has_suspicious_url'] = url_features.apply(lambda x: x[1]).astype(int)
            logger.info("Calculating urgency scores..."); df_processed['urgency_count'] = df_processed['cleaned_text'].apply(extract_urgency).astype(int)
            logger.info("Calculating readability scores..."); df_processed['readability_score'] = df_processed['cleaned_text'].apply(calculate_readability).astype(float)
            original_rows = len(df_processed); df_processed.dropna(subset=['cleaned_text'], inplace=True)
            df_processed = df_processed[df_processed['cleaned_text'].str.strip().astype(bool)]
            if len(df_processed) < original_rows: logger.warning(f"Dropped {original_rows - len(df_processed)} rows with empty/NaN cleaned text.")
            if df_processed.empty: logger.warning(f"No rows after cleaning text in {filename}. Skipping."); continue
            label_map = {'spam': 1, 'phish': 1, 'phishing': 1, 'fraud': 1, 'scam': 1, 'malicious': 1, '1': 1, 1: 1, True: 1, 'true': 1, 'ham': 0, 'legitimate': 0, 'safe': 0, 'normal': 0, 'benign': 0, '0': 0, 0: 0, False: 0, 'false': 0}
            original_label_type = df_processed['label'].dtype
            try:
                 if pd.api.types.is_bool_dtype(original_label_type): df_processed['label'] = df_processed['label'].astype(int)
                 df_processed['label'] = pd.to_numeric(df_processed['label'], errors='coerce')
                 if df_processed['label'].isnull().any() or pd.api.types.is_object_dtype(original_label_type):
                      df_processed['label'] = df[label_col].astype(str).str.strip().str.lower().map(label_map)
            except Exception as label_conv_e:
                 logger.error(f"Label conversion error in {filename}: {label_conv_e}. Trying string map.")
                 try: df_processed['label'] = df[label_col].astype(str).str.strip().str.lower().map(label_map)
                 except Exception as fallback_e: logger.error(f"Fallback label map failed for {filename}: {fallback_e}. Skipping."); continue
            original_rows = len(df_processed); df_processed.dropna(subset=['label'], inplace=True)
            if len(df_processed) < original_rows: logger.warning(f"Dropped {original_rows - len(df_processed)} rows with missing/unmappable labels.")
            if df_processed.empty: logger.warning(f"No rows after label processing in {filename}. Skipping."); continue
            df_processed['label'] = df_processed['label'].astype(int)
            feature_cols_check = ['num_links', 'has_suspicious_url', 'urgency_count', 'readability_score']
            if df_processed[feature_cols_check].isnull().any().any(): logger.warning(f"NaN found in features in {filename}. Imputation will handle.")
            present_required_columns = [col for col in required_columns if col in df_processed.columns]; df_processed = df_processed[present_required_columns]
            logger.info(f"Finished processing {filename}. Valid rows added: {len(df_processed)}"); all_dfs.append(df_processed)
        except Exception as e: logger.error(f"Critical error processing {filename}: {e}", exc_info=True); continue
    if not all_dfs: logger.error("No data loaded."); raise RuntimeError("No data loaded from source directory.")
    try:
        final_df = pd.concat(all_dfs, ignore_index=True); logger.info(f"--- Combined all files. Total rows: {len(final_df)} ---")
        if final_df.empty: logger.error("Concatenated DataFrame empty."); raise RuntimeError("Concatenated DataFrame empty.")
    except Exception as e: logger.error(f"Failed to concat DataFrames: {e}", exc_info=True); raise e
    numeric_cols_to_check = ['num_links', 'has_suspicious_url', 'urgency_count', 'readability_score']
    nan_rows = final_df[numeric_cols_to_check].isnull().any(axis=1).sum()
    if nan_rows > 0: logger.warning(f"{nan_rows} total rows with NaN in numeric columns. Imputation will handle.")
    if 'label' in final_df.columns:
         logger.info(f"Final shape: {final_df.shape}")
         try: label_counts = final_df['label'].value_counts(normalize=True, dropna=False); logger.info(f"Label distribution:\n{label_counts}")
         except Exception as label_count_e: logger.error(f"Could not get final label distribution: {label_count_e}")
    else: logger.error("Label column missing post-concat."); raise RuntimeError("Label column missing post-concat.")
    return final_df


# --------------------------------------
# Model Definition (Keras Hybrid NN)
# --------------------------------------
def build_hybrid_nn(text_input_dim, struct_input_dim, learning_rate=0.001):
    logger = logging.getLogger();
    # Removed TF_AVAILABLE check - assume it's installed
    text_input = Input(shape=(text_input_dim,), name='text_features_input')
    struct_input = Input(shape=(struct_input_dim,), name='structural_features_input')
    text_dense1 = Dense(128, activation='relu')(text_input); text_dropout = Dropout(0.4)(text_dense1)
    text_dense2 = Dense(64, activation='relu')(text_dropout)
    struct_dense = Dense(32, activation='relu')(struct_input)
    combined = Concatenate()([text_dense2, struct_dense])
    dense_combined = Dense(64, activation='relu')(combined); dropout_combined = Dropout(0.4)(dense_combined)
    output = Dense(1, activation='sigmoid', name='output')(dropout_combined)
    model = KerasModel(inputs=[text_input, struct_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    logger.info("Hybrid NN Architecture:"); model.summary(print_fn=logger.info); return model

# --------------------------------------
# XAI Helper Functions
# (Keep predict_proba_for_lime and get_feature_names as they are)
# --------------------------------------
def predict_proba_for_lime(texts, pipeline):
    logger = logging.getLogger();
    try:
        text_col_name = None; numeric_cols = []
        fitted_preprocessor = pipeline.named_steps['preprocessor']
        for name, _, cols in fitted_preprocessor.transformers_:
            if name == 'text': text_col_name = cols
            elif name == 'numeric': numeric_cols = cols
        if not text_col_name or not numeric_cols:
             preprocessor_def = pipeline.steps[0][1]
             for name, _, cols in preprocessor_def.transformers:
                 if name == 'text': text_col_name = cols
                 elif name == 'numeric': numeric_cols = cols
             if not text_col_name or not numeric_cols: raise ValueError("Could not extract col names.")
        data_for_pipeline = pd.DataFrame({text_col_name: texts, **{col: [0]*len(texts) for col in numeric_cols}})
        probabilities = pipeline.predict_proba(data_for_pipeline); return probabilities
    except Exception as e: logger.error(f"LIME wrapper error: {e}", exc_info=True); return np.array([[0.5, 0.5]]*len(texts))
def get_feature_names(column_transformer):
    logger = logging.getLogger(); feature_names = []
    if column_transformer is None: logger.error("Preprocessor None."); return ["feature_error"]
    try:
        if hasattr(column_transformer, 'get_feature_names_out') and getattr(column_transformer, 'verbose_feature_names_out', True) is False:
             return list(column_transformer.get_feature_names_out())
        for name, transformer_obj, columns in column_transformer.transformers_:
            if name == 'remainder' and transformer_obj == 'drop': continue
            if transformer_obj == 'passthrough': feature_names.extend(columns); continue
            current_feature_names = None
            if isinstance(transformer_obj, Pipeline):
                 try:
                      last_step = transformer_obj.steps[-1][1]
                      if hasattr(last_step, 'get_feature_names_out'): current_feature_names = last_step.get_feature_names_out(columns)
                      else: current_feature_names = columns
                 except Exception: current_feature_names = columns
            elif hasattr(transformer_obj, 'get_feature_names_out'):
                 try: current_feature_names = transformer_obj.get_feature_names_out(columns)
                 except TypeError:
                      try: current_feature_names = transformer_obj.get_feature_names_out()
                      except Exception as e_inner: logger.warning(f"get_feature_names_out failed {name}: {e_inner}. Using inputs."); current_feature_names = columns
                 except Exception as e_outer: logger.warning(f"get_feature_names_out failed {name}: {e_outer}. Using inputs."); current_feature_names = columns
            else: current_feature_names = columns
            is_verbose = getattr(column_transformer, 'verbose_feature_names_out', True)
            if is_verbose and isinstance(current_feature_names, (list, np.ndarray)): feature_names.extend([f"{name}__{fname}" for fname in current_feature_names])
            elif isinstance(current_feature_names, (list, np.ndarray)): feature_names.extend(current_feature_names)
            elif isinstance(current_feature_names, str): feature_names.append(f"{name}__{current_feature_names}" if is_verbose else current_feature_names)
    except Exception as e:
        logger.error(f"Error getting feature names: {e}. Returning basic.", exc_info=True)
        try: n_features = column_transformer.transform(pd.DataFrame(columns=column_transformer.feature_names_in_)).shape[1]; return [f"feature_{i}" for i in range(n_features)]
        except: logger.warning("Could not estimate feature count."); return ["feature_unknown"]
    feature_names = [str(fn) for fn in feature_names]; return feature_names

# --------------------------------------
# Plotting Functions
# (Keep save_training_history_plot and save_f1_visualization as they are)
# --------------------------------------
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

# --- Training/Evaluation/Explanation Function ---
def train_evaluate_explain(df, text_col='cleaned_text', numeric_cols=None, label_col='label', output_dir='results'):
    # (Exact function definition copied from previous corrected script, including iloc fix and SHAP explainer fix)
    logger = logging.getLogger();
    if numeric_cols is None: numeric_cols = ['num_links', 'has_suspicious_url', 'urgency_count', 'readability_score']
    logger.info("--- Starting Model Training and Evaluation ---"); logger.info(f"Using Text: '{text_col}', Numeric: {numeric_cols}, Label: '{label_col}'")
    required_input_cols = [text_col, label_col, 'original_text'] + numeric_cols
    missing_cols = [col for col in required_input_cols if col not in df.columns];
    if missing_cols: logger.error(f"Missing columns: {missing_cols}. Exiting."); sys.exit(1)
    X = df[[text_col] + numeric_cols]; y = df[label_col]; original_texts = df['original_text']
    if X.empty or y.empty: logger.error("X or y empty before split."); sys.exit(1)
    logger.info("Splitting data (80/20)..."); arrays_to_split = [X, y, original_texts, df.index]
    try:
        min_class_count = y.value_counts().min(); use_stratify = min_class_count >= 2
        if not use_stratify: logger.warning(f"Minority class ({min_class_count}) < 2. No stratify.")
        split_results = train_test_split(*arrays_to_split, test_size=0.2, random_state=42, stratify=y if use_stratify else None)
        X_train, X_test, y_train, y_test, original_texts_train, original_texts_test, train_indices, test_indices = split_results
    except ValueError as e:
         if 'stratify' in str(e):
              logger.warning(f"Stratify failed: {e}. No stratify."); split_results = train_test_split(*arrays_to_split, test_size=0.2, random_state=42)
              X_train, X_test, y_train, y_test, original_texts_train, original_texts_test, train_indices, test_indices = split_results
         else: logger.error(f"Split error: {e}", exc_info=True); raise e
    if X_train.empty or X_test.empty: logger.error("Train/Test empty post-split."); sys.exit(1)
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}"); logger.info(f"Train labels:\n{y_train.value_counts(normalize=True, dropna=False)}"); logger.info(f"Test labels:\n{y_test.value_counts(normalize=True, dropna=False)}")
    preprocessor = ColumnTransformer(transformers=[('text', TfidfVectorizer(stop_words=list(stop_words), max_features=2000, min_df=3, max_df=0.9), text_col), ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols)], remainder='drop', verbose_feature_names_out=False)
    logger.info("Preprocessor defined."); models, results, explanations, fitted_preprocessor = {}, {}, {}, None
    # == LR ==
    model_name = "LogisticRegression"; logger.info(f"\n--- Training {model_name} ---"); lr_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', random_state=42, C=1.0))])
    try:
        logger.info(f"Fitting {model_name}..."); lr_pipeline.fit(X_train, y_train); logger.info(f"Evaluating {model_name}..."); y_pred_lr = lr_pipeline.predict(X_test)
        report_lr = classification_report(y_test, y_pred_lr, output_dict=True, zero_division=0)
        results[model_name] = {'accuracy': accuracy_score(y_test, y_pred_lr), 'f1_score': f1_score(y_test, y_pred_lr, average='binary', zero_division=0),'precision': precision_score(y_test, y_pred_lr, average='binary', zero_division=0), 'recall': recall_score(y_test, y_pred_lr, average='binary', zero_division=0), 'report': report_lr}
        models[model_name] = lr_pipeline; logger.info(f"{model_name} OK. F1: {results[model_name]['f1_score']:.4f}");
        if fitted_preprocessor is None: fitted_preprocessor = lr_pipeline.named_steps['preprocessor']
    except Exception as e: logger.error(f"Failed {model_name}: {e}", exc_info=True); results[model_name] = {'error': str(e)}
    # == DT ==
    model_name = "DecisionTree"; logger.info(f"\n--- Training {model_name} ---"); dt_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42, min_samples_split=10, min_samples_leaf=5))])
    try:
        logger.info(f"Fitting {model_name}..."); dt_pipeline.fit(X_train, y_train); logger.info(f"Evaluating {model_name}..."); y_pred_dt = dt_pipeline.predict(X_test)
        report_dt = classification_report(y_test, y_pred_dt, output_dict=True, zero_division=0)
        results[model_name] = {'accuracy': accuracy_score(y_test, y_pred_dt), 'f1_score': f1_score(y_test, y_pred_dt, average='binary', zero_division=0), 'precision': precision_score(y_test, y_pred_dt, average='binary', zero_division=0), 'recall': recall_score(y_test, y_pred_dt, average='binary', zero_division=0), 'report': report_dt}
        models[model_name] = dt_pipeline; logger.info(f"{model_name} OK. F1: {results[model_name]['f1_score']:.4f}")
        if fitted_preprocessor is None: fitted_preprocessor = dt_pipeline.named_steps['preprocessor']
    except Exception as e: logger.error(f"Failed {model_name}: {e}", exc_info=True); results[model_name] = {'error': str(e)}
    # == NN ==
    model_name = "HybridNN"; nn_history = None
    # REMOVED: if TF_AVAILABLE: check - Now always attempts NN
    logger.info(f"\n--- Training {model_name} ---")
    try:
        if fitted_preprocessor is None:
                logger.info("Fitting preprocessor definition for Hybrid NN...")
                temp_preprocessor = ColumnTransformer(transformers=[ ('text', TfidfVectorizer(stop_words=list(stop_words), max_features=2000, min_df=3, max_df=0.9), text_col), ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols)], remainder='drop', verbose_feature_names_out=False)
                try: fitted_preprocessor = temp_preprocessor.fit(X_train); logger.info("Temp fitted preprocessor for NN.")
                except Exception as fit_e: logger.error(f"Failed to fit temp preprocessor: {fit_e}", exc_info=True); raise
        logger.info("Transforming data for NN..."); X_train_transformed = fitted_preprocessor.transform(X_train); X_test_transformed = fitted_preprocessor.transform(X_test)
        try:
                text_transformer = fitted_preprocessor.named_transformers_['text']; text_input_dim = len(text_transformer.get_feature_names_out())
                numeric_cols_list = fitted_preprocessor.transformers_[1][2]; struct_input_dim = len(numeric_cols_list)
                logger.info(f"NN Dims - Text: {text_input_dim}, Struct: {struct_input_dim}")
                if X_train_transformed.shape[1] != (text_input_dim + struct_input_dim): logger.warning(f"Shape mismatch: {X_train_transformed.shape[1]} vs {text_input_dim + struct_input_dim}. Adjusting."); struct_input_dim = X_train_transformed.shape[1] - text_input_dim; logger.warning(f"Adjusted struct dim: {struct_input_dim}")
                if issparse(X_train_transformed): X_train_transformed_dense, X_test_transformed_dense = X_train_transformed.toarray(), X_test_transformed.toarray()
                else: X_train_transformed_dense, X_test_transformed_dense = X_train_transformed, X_test_transformed
                X_train_text_nn = X_train_transformed_dense[:, :text_input_dim].astype(np.float32); X_train_struct_nn = X_train_transformed_dense[:, text_input_dim:].astype(np.float32)
                X_test_text_nn = X_test_transformed_dense[:, :text_input_dim].astype(np.float32); X_test_struct_nn = X_test_transformed_dense[:, text_input_dim:].astype(np.float32)
                if X_train_struct_nn.shape[1] != struct_input_dim: logger.error(f"Struct slice error: expected {struct_input_dim}, got {X_train_struct_nn.shape[1]}."); raise ValueError("Struct slice mismatch.")
                keras_model = build_hybrid_nn(text_input_dim, struct_input_dim)
                if keras_model:
                    logger.info("Starting Keras training..."); nn_history = keras_model.fit([X_train_text_nn, X_train_struct_nn], y_train, epochs=15, batch_size=64, validation_split=0.15, callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)], verbose=1)
                    logger.info("Keras training finished.")
                    logger.info(f"Evaluating {model_name}..."); loss, accuracy, precision_keras, recall_keras = keras_model.evaluate([X_test_text_nn, X_test_struct_nn], y_test, verbose=0)
                    y_pred_proba_nn = keras_model.predict([X_test_text_nn, X_test_struct_nn]); y_pred_nn = (y_pred_proba_nn > 0.5).astype(int).flatten()
                    report_nn = classification_report(y_test, y_pred_nn, output_dict=True, zero_division=0); f1_nn = f1_score(y_test, y_pred_nn, average='binary', zero_division=0)
                    results[model_name] = {'accuracy': accuracy_score(y_test, y_pred_nn), 'f1_score': f1_nn, 'precision': precision_score(y_test, y_pred_nn, average='binary', zero_division=0), 'recall': recall_score(y_test, y_pred_nn, average='binary', zero_division=0), 'report': report_nn, 'keras_test_loss': loss, 'keras_test_accuracy': accuracy, 'keras_test_precision': precision_keras, 'keras_test_recall': recall_keras, 'history': {k: [float(val) for val in v] for k, v in nn_history.history.items()} }
                    models[model_name] = keras_model; logger.info(f"{model_name} OK. F1: {results[model_name]['f1_score']:.4f}")
                else: logger.warning("Skip NN: build failed."); results[model_name] = {'error': 'Keras build failed.'}
        except Exception as nn_dim_error: logger.error(f"NN data prep error: {nn_dim_error}", exc_info=True); results[model_name] = {'error': f'NN data prep error: {nn_dim_error}'}
    except Exception as e: logger.error(f"Failed {model_name}: {e}", exc_info=True); results[model_name] = {'error': str(e)}
    # REMOVED: else block for TF unavailable

    # --- Plot NN Training History ---
    if nn_history: save_training_history_plot(nn_history, output_dir)

    # --- XAI Explanation Generation ---
    # (Keep XAI block with the test_indices[idx] fix and TreeExplainer fix)
    logger.info("\n--- Generating Explanations for a Sample Instance ---")
    if not any(isinstance(m, Pipeline) for m in models.values()): logger.warning("No Pipeline models OK. Skipping XAI."); return models, results, explanations, fitted_preprocessor
    if fitted_preprocessor is None: logger.error("Preprocessor not fitted. Skipping XAI."); return models, results, explanations, fitted_preprocessor
    if X_test.empty: logger.error("Test set empty. Skipping XAI."); return models, results, explanations, fitted_preprocessor
    sample_idx_in_test = 0
    if sample_idx_in_test >= len(X_test): logger.warning("Test set small. Using index 0."); sample_idx_in_test = 0
    try:
        original_sample_index = test_indices[sample_idx_in_test] # Access Index using integer position
        sample_raw_data = X_test.iloc[[sample_idx_in_test]]
        sample_original_text = original_texts_test.iloc[sample_idx_in_test] if original_texts_test is not None else "Original text not available"
        sample_cleaned_text = sample_raw_data[text_col].iloc[0]
    except IndexError: logger.error(f"Cannot access sample index {sample_idx_in_test}. Test size: {len(X_test)}"); return models, results, explanations, fitted_preprocessor
    except Exception as sample_e: logger.error(f"Error getting sample data: {sample_e}", exc_info=True); return models, results, explanations, fitted_preprocessor
    logger.info(f"Explaining instance with original index: {original_sample_index}")
    logger.info(f"Sample Cleaned Text: '{sample_cleaned_text[:200]}...'")
    num_shap_background_samples = 100; shap_background_data_transformed = None
    if len(X_train) == 0: logger.warning("Train empty. Cannot create SHAP background.")
    else:
        if len(X_train) < num_shap_background_samples: num_shap_background_samples = len(X_train)
        try:
            shap_background_indices = np.random.choice(X_train.index, size=num_shap_background_samples, replace=False)
            shap_background_data = X_train.loc[shap_background_indices]
            shap_background_data_transformed = fitted_preprocessor.transform(shap_background_data)
            logger.info(f"Prepared SHAP background shape: {shap_background_data_transformed.shape}")
        except Exception as e: logger.error(f"Failed SHAP BG transform: {e}", exc_info=True)

    for model_name, model_obj in models.items():
        # Skip non-pipeline models or failed models
        if not isinstance(model_obj, Pipeline) or model_name not in results or 'error' in results[model_name]:
            # REMOVED: Check for HybridNN here, explanations attempted only for Pipelines now
            logger.debug(f"Skip explanations for {model_name} (not valid Pipeline or failed training).")
            continue
        logger.info(f"--- Explaining with {model_name} ---"); model_explanations = {}
        pipeline = model_obj; classifier = pipeline.named_steps['classifier']
        # == SHAP ==
        shap_exp = None # Initialize shap_exp
        shap_background_dense = None
        if shap_background_data_transformed is not None:
            if issparse(shap_background_data_transformed): shap_background_dense = shap_background_data_transformed.toarray()
            else: shap_background_dense = shap_background_data_transformed
            if np.any(np.isnan(shap_background_dense)) or np.any(np.isinf(shap_background_dense)):
                logger.warning("NaN/Inf in SHAP BG. Imputing...");
                try: imputer_temp = SimpleImputer(strategy='median'); shap_background_dense = imputer_temp.fit_transform(shap_background_dense)
                except Exception as imp_e: logger.error(f"Failed to impute BG data: {imp_e}"); shap_background_dense = None

        if shap_background_dense is None and isinstance(classifier, LogisticRegression):
             logger.warning(f"Skip SHAP: {model_name} (BG data unavailable/unusable for LinearExplainer).")
             model_explanations['shap'] = {'error': 'BG data unavailable/unusable'}
        else:
             try:
                 logger.info("Transforming instance data for SHAP...");
                 transformed_instance = fitted_preprocessor.transform(sample_raw_data)
                 if issparse(transformed_instance): transformed_instance_dense = transformed_instance.toarray()
                 else: transformed_instance_dense = transformed_instance
                 if np.any(np.isnan(transformed_instance_dense)) or np.any(np.isinf(transformed_instance_dense)):
                     logger.warning("NaN/Inf in SHAP instance. Imputing...");
                     try: numeric_imputer = fitted_preprocessor.named_transformers_['numeric'].named_steps['imputer']; transformed_instance_dense = numeric_imputer.transform(transformed_instance_dense)
                     except: imputer_temp_inst = SimpleImputer(strategy='median'); transformed_instance_dense = imputer_temp_inst.fit_transform(transformed_instance_dense)

                 shap_values = None; explainer_shap = None; base_value = None
                 if isinstance(classifier, LogisticRegression):
                     logger.info("Using shap.LinearExplainer...");
                     explainer_shap = shap.LinearExplainer(classifier, shap_background_dense);
                     shap_values = explainer_shap.shap_values(transformed_instance_dense)
                     base_value = explainer_shap.expected_value if hasattr(explainer_shap, 'expected_value') else None
                 elif isinstance(classifier, DecisionTreeClassifier):
                     logger.info("Using shap.TreeExplainer with tree_path_dependent...");
                     # Use "tree_path_dependent" and omit data for TreeExplainer init
                     explainer_shap = shap.TreeExplainer(classifier, feature_perturbation="tree_path_dependent")
                     shap_values = explainer_shap.shap_values(transformed_instance_dense)
                     base_value = explainer_shap.expected_value if hasattr(explainer_shap, 'expected_value') else None
                 else:
                     logger.warning(f"Unsupported SHAP type: {type(classifier)}. Skipping."); model_explanations['shap'] = {'error': 'Unsupported model type'}; raise TypeError("Unsupported")

                 logger.info("Processing SHAP values..."); positive_class_index = 1
                 if isinstance(shap_values, list) and len(shap_values) > positive_class_index:
                      positive_class_shap_values = shap_values[positive_class_index]
                      if isinstance(base_value, (list, np.ndarray)) and len(base_value) > positive_class_index: base_value = base_value[positive_class_index]
                      elif isinstance(base_value, (list, np.ndarray)): base_value = base_value[0]
                 else: positive_class_shap_values = shap_values
                 if np.any(np.isnan(positive_class_shap_values)) or (base_value is not None and np.any(np.isnan(base_value))): logger.warning(f"NaN in SHAP values/base {model_name}.")
                 try: feature_names = get_feature_names(fitted_preprocessor)
                 except Exception as fn_e: logger.error(f"SHAP get_feature_names failed: {fn_e}"); feature_names = [f"feature_{i}" for i in range(transformed_instance_dense.shape[1])]

                 # Ensure base_value is serializable or None
                 if base_value is not None and not isinstance(base_value, (int, float, type(None))):
                      if isinstance(base_value, np.ndarray) and base_value.size == 1:
                           base_value = base_value.item() # Convert numpy scalar to python type
                      else:
                           logger.warning(f"Unexpected base_value type ({type(base_value)}) for SHAP Explanation. Setting to None.")
                           base_value = None

                 shap_exp = shap.Explanation(values=positive_class_shap_values, base_values=base_value, data=transformed_instance_dense, feature_names=feature_names)
                 model_explanations['shap'] = shap_exp; logger.info("SHAP OK.")
             except TypeError as te:
                  if "Unsupported" in str(te): pass # Already logged
                  else: logger.error(f"SHAP failed {model_name} (TypeError): {te}", exc_info=True); model_explanations['shap'] = {'error': str(te)}
             except Exception as e: logger.error(f"SHAP failed {model_name}: {e}", exc_info=True); model_explanations['shap'] = {'error': str(e)}
        # == LIME ==
        try:
            logger.info("Init LIME..."); lime_explainer = lime_text.LimeTextExplainer(class_names=['Legitimate', 'Phishing'])
            logger.info("Calculating LIME..."); lime_exp = lime_explainer.explain_instance(sample_cleaned_text, lambda texts: predict_proba_for_lime(texts, pipeline), num_features=15, num_samples=1000)
            model_explanations['lime'] = lime_exp; logger.info("LIME OK.")
        except Exception as e: logger.error(f"LIME failed {model_name}: {e}", exc_info=True); model_explanations['lime'] = {'error': str(e)}
        explanations[model_name] = model_explanations

    return models, results, explanations, fitted_preprocessor


# --- Saving Artifacts Function ---
def save_artifacts(output_dir, models, results, explanations, preprocessor):
    # (Exact function definition copied from previous corrected script)
    logger = logging.getLogger(); logger.info(f"\n--- Saving Artifacts to: {output_dir} ---"); os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "models"); os.makedirs(models_dir, exist_ok=True)
    for name, model in models.items():
        try:
            # Use isinstance check for Keras model directly
            if name == "HybridNN" and isinstance(model, KerasModel):
                 model_path = os.path.join(models_dir, f"{name}.keras"); model.save(model_path); logger.info(f"Saved {name} model: {model_path}")
            elif isinstance(model, Pipeline):
                 model_path = os.path.join(models_dir, f"{name}_pipeline.pkl"); joblib.dump(model, model_path, compress=3); logger.info(f"Saved {name} pipeline: {model_path}")
            else: logger.warning(f"Model {name} type {type(model)} not saved.")
        except Exception as e: logger.error(f"Failed saving model {name}: {e}", exc_info=True)
    if preprocessor:
        try: preprocessor_path = os.path.join(models_dir, "preprocessor.pkl"); joblib.dump(preprocessor, preprocessor_path, compress=3); logger.info(f"Saved preprocessor: {preprocessor_path}")
        except Exception as e: logger.error(f"Failed saving preprocessor: {e}", exc_info=True)
    try: # Save Results
        results_path = os.path.join(output_dir, "results.json")
        def default_serializer(obj):
            if isinstance(obj, (np.integer, int)): return int(obj)
            elif isinstance(obj, (np.floating, float)): return 'NaN' if np.isnan(obj) else ('Infinity' if np.isinf(obj) else float(obj))
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)): return bool(obj)
            elif isinstance(obj, (bytes, bytearray)): return obj.decode('utf-8', errors='replace')
            try: return json.JSONEncoder.default(json.JSONEncoder(), obj)
            except TypeError: return str(obj)
        def convert_dict(item):
             if isinstance(item, dict): return {k: convert_dict(v) for k, v in item.items()}
             elif isinstance(item, list): return [convert_dict(elem) for elem in item]
             else: return default_serializer(item)
        serializable_results = convert_dict(results)
        with open(results_path, 'w') as f: json.dump(serializable_results, f, indent=2); logger.info(f"Saved results: {results_path}")
    except Exception as e: logger.error(f"Failed saving results: {e}", exc_info=True)
    # Save Explanations & Visualizations
    explanations_dir = os.path.join(output_dir, "explanations"); os.makedirs(explanations_dir, exist_ok=True)
    if not explanations: logger.warning("No explanations generated to save.")
    else:
        for model_name, model_explanations in explanations.items():
            if not model_explanations or model_explanations.get('status') == 'Skipped': continue
            model_expl_dir = os.path.join(explanations_dir, model_name.replace(" ", "_")); os.makedirs(model_expl_dir, exist_ok=True)
            logger.info(f"Saving explanations for {model_name}..."); shap_exp = model_explanations.get('shap')
            if shap_exp and isinstance(shap_exp, shap.Explanation):
                try:
                    if shap_exp.values is None or len(shap_exp.values) == 0 or shap_exp.data is None: logger.warning(f"SHAP empty for {model_name}. Skipping plots."); continue
                    # Handle cases where shap_exp might be multi-dimensional incorrectly after LinearExplainer
                    if shap_exp.values.ndim > 1 and shap_exp.values.shape[0] == 1:
                         exp_to_plot = shap_exp[0] # Select the first (only) sample's explanation data
                    else:
                         exp_to_plot = shap_exp # Assume it's already correctly shaped (1D for values/data)

                    fig_wf = plt.figure()
                    try:
                        vals_for_wf = exp_to_plot.values; base_values_for_wf = exp_to_plot.base_values
                        # Ensure base_values is scalar for waterfall if it's an array/list
                        if isinstance(base_values_for_wf, (list, np.ndarray)): base_values_for_wf = base_values_for_wf[0]
                        if vals_for_wf.ndim > 1: vals_for_wf = vals_for_wf[:,0]
                        wf_exp = shap.Explanation(values=vals_for_wf, base_values=base_values_for_wf, data=exp_to_plot.data, feature_names=exp_to_plot.feature_names)
                        shap.plots.waterfall(wf_exp, max_display=20, show=False)
                        plt.tight_layout(); plt.savefig(os.path.join(model_expl_dir, "shap_waterfall.png"), dpi=150, bbox_inches='tight')
                    except Exception as wf_e:
                         logger.warning(f"SHAP waterfall failed: {wf_e}. Trying summary.")
                         plt.close(fig_wf); fig_sum = plt.figure()
                         try: shap.summary_plot(shap_exp.values, features=exp_to_plot.data, feature_names=exp_to_plot.feature_names, max_display=20, show=False); plt.tight_layout(); plt.savefig(os.path.join(model_expl_dir, "shap_summary_fallback.png"), dpi=150, bbox_inches='tight')
                         except Exception as sum_e: logger.error(f"SHAP summary fallback failed: {sum_e}")
                         plt.close(fig_sum)
                    if 'fig_wf' in locals() and plt.fignum_exists(fig_wf.number): plt.close(fig_wf)
                    # Force plot
                    try:
                        base_value_for_plot = exp_to_plot.base_values
                        if isinstance(base_value_for_plot, (list, np.ndarray)): base_value_for_plot = base_value_for_plot[0]
                        vals_for_force = exp_to_plot.values
                        if vals_for_force.ndim > 1: vals_for_force = vals_for_force[:,0]
                        data_for_force = exp_to_plot.data
                        if data_for_force.ndim > 1: data_for_force = data_for_force[0] # Use first row if data is 2D

                        if np.any(np.isnan(vals_for_force)) or (base_value_for_plot is not None and np.any(np.isnan(base_value_for_plot))): logger.warning(f"NaN in SHAP values/base force plot ({model_name}). Skipping.")
                        else:
                             force_plot_html = shap.force_plot(base_value_for_plot, vals_for_force, features=data_for_force, feature_names=exp_to_plot.feature_names, show=False, matplotlib=False)
                             if force_plot_html: shap.save_html(os.path.join(model_expl_dir, "shap_force_plot.html"), force_plot_html)
                             else: logger.warning("SHAP force_plot did not return object.")
                    except ImportError: logger.warning("IPython missing for SHAP force plot.")
                    except Exception as force_err: logger.warning(f"SHAP force plot failed: {force_err}", exc_info=False)
                    logger.info(f"Saved SHAP plots for {model_name}.")
                except IndexError: logger.error(f"IndexError saving SHAP plots for {model_name}.")
                except Exception as e: logger.error(f"Failed SHAP plots save {model_name}: {e}", exc_info=True)
            elif shap_exp and isinstance(shap_exp, dict) and 'error' in shap_exp: logger.warning(f"Skip SHAP plots {model_name}: {shap_exp['error']}")
            lime_exp = model_explanations.get('lime')
            if lime_exp and hasattr(lime_exp, 'save_to_file'):
                try: lime_html_path = os.path.join(model_expl_dir, 'lime_explanation.html'); lime_exp.save_to_file(lime_html_path); logger.info(f"Saved LIME HTML for {model_name}.")
                except Exception as e: logger.error(f"Failed saving LIME HTML {model_name}: {e}", exc_info=True)
            elif lime_exp and isinstance(lime_exp, dict) and 'error' in lime_exp: logger.warning(f"Skip LIME saving {model_name}: {lime_exp['error']}")


# (Keep save_f1_visualization as is)
# ... (Definition of save_f1_visualization) ...


# --------------------------------------
# Main Execution
# --------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing Detection Training and Explanation Pipeline")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing raw CSV data files.")
    parser.add_argument("--output-dir", type=str, default="phishing_results", help="Directory to save models, results, and explanations.")
    args = parser.parse_args()

    logger = setup_logging(args.output_dir) # Configure root logger

    try:
        logger.info("🚀 Starting Phishing Detection Pipeline 🚀")
        logger.info(f"Script Arguments: {args}")
        # Log TensorFlow status explicitly
        # REMOVED: if TF_AVAILABLE check, now assumed true
        logger.info(f"TensorFlow Loaded: True (Version: {tf.__version__})")
        logger.info(f"Data Input Directory: {args.data_dir}")
        logger.info(f"Output Directory: {args.output_dir}")

        df_processed = load_and_preprocess_data(args.data_dir)

        models, results, explanations, preprocessor = train_evaluate_explain(
            df_processed, output_dir=args.output_dir
        )

        os.makedirs(args.output_dir, exist_ok=True)
        if models or results or explanations or preprocessor:
             save_artifacts(args.output_dir, models, results, explanations, preprocessor)
             save_f1_visualization(results, args.output_dir)
             # Training history plot saved inside train_evaluate_explain
        else:
             logger.warning("No artifacts generated to save.")

        logger.info("✅ Pipeline Completed Successfully! ✅")

    except ImportError as e:
         # Specific handling if TF import itself fails
         logger.critical(f"ImportError: {e}. TensorFlow is required but not installed or accessible.")
         logger.critical("Please install TensorFlow: pip install tensorflow")
         sys.exit(1)
    except FileNotFoundError as e: logger.critical(f"File Not Found Error: {e}."); sys.exit(1)
    except pd.errors.EmptyDataError as e: logger.critical(f"Empty Data Error: {e}."); sys.exit(1)
    except KeyError as e: logger.critical(f"Column Key Error: {e}."); sys.exit(1)
    except MemoryError as e: logger.critical(f"Memory Error: {e}. Dataset too large?"); sys.exit(1)
    except Exception as e: logger.critical(f"Unexpected critical error: {e}", exc_info=True); sys.exit(1)
