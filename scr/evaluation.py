import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import sys

# Import evaluation metrics
from evaluation_metrics import evaluate_model, calculate_cross_dataset_metrics

# Get the logger from preprocessing
logger = logging.getLogger('__main__')

# --------------------------------------
# Feature Engineering & Model Training
# --------------------------------------
def train_phishing_model(preprocessed_datasets, output_dir):
    """
    Train a phishing detection model using preprocessed datasets.
    
    Parameters:
    -----------
    preprocessed_datasets : dict
        Dictionary mapping dataset names to file paths
    output_dir : str
        Directory to save model and evaluation results
    
    Returns:
    --------
    model : trained model
        The trained phishing detection model
    """
    try:
        logger.info("Starting model training...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data from multiple sources
        all_data = []
        dataset_splits = {}  # Store test splits for cross-dataset evaluation
        
        for name, filepath in preprocessed_datasets.items():
            logger.info(f"Loading preprocessed data from {name}...")
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                continue
                
            df = pd.read_csv(filepath)
            
            # Identify label column
            label_col = next((col for col in df.columns if 'label' in col.lower() or 'class' in col.lower()), None)
            if not label_col:
                logger.warning(f"No label column found in {name} dataset")
                continue
            
            # Make sure we have binary labels (0/1)
            if df[label_col].nunique() != 2:
                logger.warning(f"Non-binary labels in {name} dataset, attempting to convert...")
                # Convert to binary (assuming phishing is positive class)
                unique_values = df[label_col].unique()
                mapping = {unique_values[0]: 0, unique_values[1]: 1}
                df[label_col] = df[label_col].map(mapping)
            
            # Create a binary split for each dataset for later cross-evaluation
