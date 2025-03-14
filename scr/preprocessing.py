import os
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning  # Updated import
import warnings  # New import
import nltk
from nltk.corpus import stopwords
import textstat
import logging
from logging.handlers import RotatingFileHandler

# Suppress BeautifulSoup URL warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)  # Added here

# --------------------------------------
# Configure Logging
# --------------------------------------
LOG_DIR = r"#RUTA#"
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize logger FIRST
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "preprocessing.log"),
    maxBytes=1024*1024,  # 1MB
    backupCount=3,
    encoding='utf-8'
)
console_handler = logging.StreamHandler()

# Formatting
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --------------------------------------
# Initialize NLTK
# --------------------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# --------------------------------------
# Preprocessing Functions (Fixed Logging)
# --------------------------------------
def clean_email(text):
    """Clean email text by removing HTML tags and special characters."""
    try:
        if pd.isna(text) or text.strip() == "":
            logger.debug("Empty text encountered during cleaning")
            return ""
        soup = BeautifulSoup(text, 'html.parser')
        cleaned = soup.get_text(separator=' ')
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}", exc_info=True)
        return ""

def extract_url_features(urls):
    """Extract URL-based features (suspicious flags, count)."""
    try:
        suspicious_keywords = ['login', 'verify', 'account', 'secure', 'update']
        shortened_domains = r'(bit\.ly|goo\.gl|tinyurl|t\.co|ow\.ly)'
        has_suspicious_url = 0import os
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning  # Updated import
import warnings  # New import
import nltk
from nltk.corpus import stopwords
import textstat
import logging
from logging.handlers import RotatingFileHandler

# Suppress BeautifulSoup URL warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)  # Added here

# --------------------------------------
# Configure Logging
# --------------------------------------
LOG_DIR = r"#RUTA#\logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize logger FIRST
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "preprocessing.log"),
    maxBytes=1024*1024,  # 1MB
    backupCount=3,
    encoding='utf-8'
)
console_handler = logging.StreamHandler()

# Formatting
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --------------------------------------
# Initialize NLTK
# --------------------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# --------------------------------------
# Preprocessing Functions (Fixed Logging)
# --------------------------------------
def clean_email(text):
    """Clean email text by removing HTML tags and special characters."""
    try:
        if pd.isna(text) or text.strip() == "":
            logger.debug("Empty text encountered during cleaning")
            return ""
        soup = BeautifulSoup(text, 'html.parser')
        cleaned = soup.get_text(separator=' ')
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}", exc_info=True)
        return ""

# In preprocessing.py, add after clean_email()

def extract_url_features(urls):
    """Extract URL-based features."""
    try:
        suspicious_keywords = ['login', 'verify', 'account', 'secure', 'update']
        shortened_domains = r'(bit\.ly|goo\.gl|tinyurl|t\.co|ow\.ly)'
        has_suspicious_url = 0
        for url in urls:
            if re.match(r'http://', url):
                has_suspicious_url = 1
                break
            if re.search(shortened_domains, url):
                has_suspicious_url = 1
                break
            if any(keyword in url.lower() for keyword in suspicious_keywords):
                has_suspicious_url = 1
                break
        return len(urls), has_suspicious_url
    except Exception as e:
        logger.error(f"URL feature extraction failed: {str(e)}", exc_info=True)
        return 0, 0

def extract_urgency(text):
    """Count urgency-related keywords."""
    try:
        urgency_words = ['urgent', 'immediately', 'action required', 'verify', 'password', 'alert']
        return sum(text.lower().count(word) for word in urgency_words)
    except Exception as e:
        logger.error(f"Urgency calculation failed: {str(e)}", exc_info=True)
        return 0

# --------------------------------------
# Main Pipeline (Critical Fixes)
# --------------------------------------
def preprocess_dataset(input_path, output_path):
    """Process a dataset and save the cleaned version."""
    try:
        logger.info(f"Starting preprocessing for: {os.path.basename(input_path)}")
        
        # Read CSV with encoding fallback
        try:
            df = pd.read_csv(input_path, encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning("UTF-8 decoding failed. Trying latin-1...")
            df = pd.read_csv(input_path, encoding='latin1')
        
        logger.info(f"Columns in raw data: {df.columns.tolist()}")
        logger.info(f"Initial rows: {len(df)}")
        
        # Detect text/label columns (case-insensitive)
        text_col = next((col for col in df.columns if 'text' in col.lower() or 'body' in col.lower()), None)
        label_col = next((col for col in df.columns if 'label' in col.lower() or 'class' in col.lower()), None)
        
        if not text_col:
            logger.error("No text column found!")
            return
        if not label_col:
            logger.error("No label column found!")
            return
        
        # Clean text and validate
        df['cleaned_text'] = df[text_col].apply(clean_email)
        df['cleaned_text'] = df['cleaned_text'].fillna('')  # Handle NaN
        df = df[df['cleaned_text'].str.strip().astype(bool)]
        if df.empty:
            logger.warning("No valid text after cleaning!")
            return


        # ======== ADDED FEATURE EXTRACTION ========
        logger.info("🔗 Extracting URL features...")
        url_pattern = r'https?://\S+|www\.\S+'
        df['urls'] = df[text_col].apply(lambda x: re.findall(url_pattern, str(x)))
        
        # Extract URL-based features
        df[['num_links', 'has_suspicious_url']] = df['urls'].apply(
            lambda x: extract_url_features(x)
        ).apply(pd.Series)
  # Clean text and validate
        df['cleaned_text'] = df[text_col].apply(clean_email)
        df = df[df['cleaned_text'].str.strip().astype(bool)]
        if df.empty:
            logger.warning("No valid text after cleaning!")
            return

        # ------------------------------------------------------------
        # Extract URLs and Features
        # ------------------------------------------------------------
        logger.info("🔗 Extracting URL features...")
        url_pattern = r'https?://\S+|www\.\S+'
        df['urls'] = df[text_col].apply(lambda x: re.findall(url_pattern, str(x)))
        
        # Extract URL-based features (num_links, has_suspicious_url)
        df[['num_links', 'has_suspicious_url']] = df['urls'].apply(
            lambda x: extract_url_features(x)
        ).apply(pd.Series)

        # ------------------------------------------------------------
        # Urgency Keywords
        # ------------------------------------------------------------
        logger.info("⏳ Calculating urgency scores...")
        df['urgency_count'] = df['cleaned_text'].apply(extract_urgency)  # <-- Fixed indentation

        # ------------------------------------------------------------
        # Readability Score
        # ------------------------------------------------------------
        logger.info("📖 Calculating readability scores...")
        df['readability_score'] = df['cleaned_text'].apply(
            lambda x: textstat.flesch_reading_ease(x) if isinstance(x, str) else 0
        )

        # ------------------------------------------------------------
        # Tokenization
        # ------------------------------------------------------------
        logger.info("✂️ Tokenizing text...")
        df['tokenized_text'] = df['cleaned_text'].apply(
            lambda x: nltk.word_tokenize(x.lower()) 
            if isinstance(x, str) and x.strip() != "" 
            else []
        )

        # Filter out empty token lists
        df = df[df['tokenized_text'].apply(len) > 0]
        if df.empty:
            logger.error("No valid tokens after tokenization!")
            sys.exit(1)
        # Save preprocessed data
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")
        
    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)

# --------------------------------------
# Run Preprocessing (Guarantee Logging)
# --------------------------------------
if __name__ == "__main__":
    try:
        logger.info("🚀 Starting preprocessing pipeline")
        
        DATA_DIR = r"C:\Users\acvsa\PhishingDetector\data"
        OUTPUT_DIR = os.path.join(DATA_DIR, "preprocessed")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        datasets = {
            "Enron": os.path.join(DATA_DIR, "Enron.csv"),
            "Ling": os.path.join(DATA_DIR, "Ling.csv"),
            "CEAS_08": os.path.join(DATA_DIR, "CEAS_08.csv"),
	    "Nazario": os.path.join(DATA_DIR, "Nazario.csv"),
	    "Nigerian_Fraud": os.path.join(DATA_DIR, "Nigerian_Fraud.csv"),
	    "phishing_email": os.path.join(DATA_DIR, "phishing_email.csv"),
	    "SpamAssasin": os.path.join(DATA_DIR, "SpamAssasin.csv"),
        }
        
        for dataset_name, csv_path in datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")
            if not os.path.isfile(csv_path):
                logger.warning(f"File not found: {csv_path}")
                continue
            output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_preprocessed.csv")
            preprocess_dataset(csv_path, output_path)
        
        logger.info("Preprocessing completed!")
    except Exception as e:
        logger.critical(f"Fatal error in main: {str(e)}", exc_info=True)
