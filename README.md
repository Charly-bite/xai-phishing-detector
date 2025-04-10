In this project we try to make a defensive tool to discover phishing attack when emails are used as vector attacks, we collect raw data from the Phishing Email Dataset courtesy of *Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman, S. A. U. (2024, May 19). Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection.

Resource: https://arxiv.org/abs/2405.11619

Dataset name is: Phish No More: The Enron, Ling, CEAS, Nazario, Nigerian & SpamAssassin Datasets.
It contains approximately 82,500 emails email adresses, 42,891 spam emails and 39,595 legitimate email.
We preprocess all this data fisrt to then train the XAI model. 

How it works? 
We have two main scripts:
1. Preprocessing.py: Preprocessing Script: Takes raw email data (CSV files) and prepares it for machine learning by cleaning text, extracting features (like URL counts, urgency words, readability), and tokenizing the text.
2. Train_model.py: Training & Explanation Script: Takes the preprocessed data, trains different machine learning models (Logistic Regression, Decision Tree, potentially a Neural Network), evaluates them, and uses Explainable AI (XAI) techniques (SHAP, LIME, Anchor) to understand why the models make certain predictions.

Overall Goal: To build and understand models that can automatically detect phishing emails.
Analysis and Explanation of preprocessing.py (Preprocessing)
What it Does:

1.Setup: Imports necessary libraries (pandas for data, BeautifulSoup for HTML, nltk for text processing, re for regular expressions, logging). Sets up logging to record progress and errors. Downloads NLTK data (punkt for tokenization, stopwords for common English words).

2.Clean_email(text):
Takes raw email text.
Uses BeautifulSoup to remove HTML tags (common in emails).
Uses re (regular expressions) to remove characters that aren't letters, numbers, or whitespace.
Removes extra spaces.
Returns the cleaned text.

3.Extract_url_features(urls):
Takes a list of URLs found in an email.
Counts the total number of URLs (num_links).
Checks if any URL looks suspicious: uses http:// (less secure), is a known URL shortener (like bit.ly), or contains keywords like 'login', 'verify'. Sets has_suspicious_url to 1 if found, otherwise 0.
Returns the count and the suspicious flag.

4.Extract_urgency(text):
Takes the cleaned text.
Counts how many times specific "urgency" keywords (like 'urgent', 'immediately', 'action required') appear. Phishing often uses urgency.
Returns the count.

5.Preprocess_dataset(input_path"Data", output_path"Preprocessed"):
This is the main function for processing one dataset file.
Reads the CSV file (trying different encodings like 'utf-8' and 'latin1' which is good practice).
Crucially, it tries to automatically find the 'text' and 'label' columns based on keywords in the column names. This is clever but can be fragile if column names vary unexpectedly.
Applies clean_email to the text column.
Finds all URLs in the original text using a regular expression (url_pattern).
Applies extract_url_features to the found URLs.
Applies extract_urgency to the cleaned text.
Calculates a readability score (textstat.flesch_reading_ease) on the cleaned text. Lower scores generally mean harder-to-read text.
Tokenizes the cleaned text using nltk.word_tokenize, converting words to lowercase. Tokenization breaks text into individual words or symbols.
Removes rows where tokenization resulted in no tokens.
Saves the processed data (with new feature columns) to a new CSV file.

6.if __name__ == "__main__": block:
This code runs when the script is executed directly.
Defines input data directories and output directories.
Specifies a dictionary datasets containing the names and paths of the CSV files to process.
Loops through each dataset, checks if the file exists, constructs the output path, and calls preprocess_dataset.

Strengths:
Modular functions for cleaning and feature extraction.
Handles potential encoding issues in CSVs.
Uses logging effectively.
Extracts relevant features for phishing detection (URLs, urgency, readability).
Applies standard text preprocessing (cleaning, tokenization).

Areas for Potential Optimization/Improvement:
Hardcoded Paths: Paths like C:\Users\...\ make the script unusable on other computers or if directories change. These should be relative paths or command-line arguments.
Automatic Column Detection: While convenient, relying on if 'text' in col.lower() can fail if columns are named differently (e.g., 'Email Body', 'content'). It might be safer to require specific column names or pass them as arguments.
URL Regex: The URL pattern https?://\S+|www\.\S+ is decent but might miss some URLs or capture too much in edge cases. More robust URL parsing could be used if needed.
Error Handling: The try...except blocks are good, but some errors in helper functions return default values (like 0). Depending on the analysis, explicitly flagging rows with errors might be better.
NLTK Downloads: Downloading NLTK data every time the script runs is unnecessary if it's already downloaded. A check could be added.


Analysis and Explanation of train_model.py (Training & Explanation)

1.Setup: Imports a lot more libraries:
ML: sklearn (TfidfVectorizer, models, metrics, pipelines, preprocessing), potentially tensorflow/keras (though commented out/partially used).
XAI: shap, lime, alibi (for Anchor).
Other: numpy, pandas, matplotlib (for plotting), joblib (for saving models), json, os, logging, warnings, sys.
Sets up logging similar to the first script.

2.Preprocessing Definition (Near Top): Defines a ColumnTransformer and Pipeline.
ColumnTransformer: A powerful sklearn tool to apply different transformations to different columns. Here, it applies:
TfidfVectorizer to the tokenized_text column. TF-IDF converts text into numerical vectors, weighting words by importance (common words get lower weight). The analyzer=lambda x: process_tokens(x) part is crucial – it tells TF-IDF how to handle the pre-tokenized text (which might be stored as a string like ['word1', 'word2']).
StandardScaler to the numeric columns (num_links, etc.). This scales numeric features to have zero mean and unit variance, which helps many ML models perform better.
Pipeline: Chains the ColumnTransformer (preprocessing) and a LogisticRegression classifier together. This ensures preprocessing is applied consistently during training and prediction.

3-}.main() Function (First one):
Seems like a standalone script to train just a Logistic Regression model using the pipeline defined earlier.
Loads one specific preprocessed dataset (Enron_preprocessed.csv).
Identifies the target column ('label').
Defines text and numeric columns explicitly (good!).
Builds another ColumnTransformer and Pipeline (redundant with the one at the top). This one uses SimpleImputer for numeric columns (fills missing values with the mean) instead of StandardScaler. This inconsistency should be fixed. //Fix in process
Splits data into training and testing sets.
Trains the Logistic Regression pipeline.
Evaluates the model (accuracy, F1 score, classification report).
Saves the trained pipeline using joblib.
Attempts to generate and plot feature importance based on the Logistic Regression coefficients (only works for linear models).

4.load_data(): A function to load the preprocessed data, with good error handling (checks existence, size, empty dataframe). Again, uses a hardcoded default path.//Fix in process

5.Helper Functions (inspect_model_inputs, save_model_inspection, prepare_model_inputs): These seem geared towards inspecting Keras (Neural Network) models, but the main training logic doesn't fully utilize them later. They might be remnants of previous development.//Fix in process

6.PhishingClassifier Class:
Intention: To wrap different model types (LR, DT, NN) and provide a common interface for training, prediction, and explanation. This is a good design idea.
__init__: Initializes the model type.
train: Contains logic to train the specific model type. For the 'hybrid_nn', it defines and trains a Keras model. For others, it expects self.model to be already set (e.g., a scikit-learn pipeline) and calls .fit().
_train_hybrid_nn: Defines a simple neural network using Keras. It takes separate text and structural features, processes them through Dense layers, concatenates them, and outputs a prediction. Crucially, this expects pre-processed numerical inputs.
predict: Calls the underlying model's prediction method. Tries to return probabilities if possible.
init_explainers: Sets up SHAP, LIME, and potentially Anchor explainers after a model is trained.
SHAP: Uses shap.Explainer. Needs the model's prediction function and some background data (X_train).
LIME: Uses lime_text.LimeTextExplainer. Designed for text data.
Anchor: Uses alibi.explainers.AnchorText. Also for text, provides rule-based explanations.
_text_to_features: This is a potential major issue. This function seems to re-implement TF-IDF transformation inside the class, likely differently from the main ColumnTransformer used for training. This inconsistency will lead to incorrect explanations for LIME/Anchor. Explainers like LIME need a function that takes raw (or near-raw) text and performs the entire original preprocessing and prediction pipeline. //Fix in process
explain: Calls the initialized explainers (SHAP, LIME, Anchor) on a given data instance.

7.train_model(df) (Second major function):
Intended main training workflow, training multiple models.
Loads data (implicitly assumes df is passed in).
Validates required columns.
Handles tokenized_text conversion from string-lists to actual lists (good!).
Defines the correct ColumnTransformer (TF-IDF for text, StandardScaler for numeric).
Splits data.
Fits and transforms the data using the preprocessor once (efficient!).
Splits the transformed data into text (X_train_text) and structural (X_train_struct) parts, likely for the hybrid NN.
Logistic Regression: Trains a Pipeline containing just the LogisticRegression classifier (since preprocessing is already done). Evaluates it. Creates a PhishingClassifier instance to hold the trained pipeline and initializes explainers for it.
Decision Tree: Similar process to Logistic Regression.
Hybrid Neural Network: Creates a PhishingClassifier instance ('hybrid_nn'). Trains it using the _train_hybrid_nn method, passing the separated text and structural features. Evaluates based on Keras history. //Note: Explainers are not initialized for the NN in this flow, which makes sense as explaining NNs directly with these specific libraries is harder.
Explanation Generation: Calls generate_sample_explanations using the transformed test data and the LR/DT models.
Returns a dictionary containing trained models, results, explanations, and the preprocessor.

8.Evaluate_model: A helper to calculate metrics (seems slightly redundant given sklearn's classification_report).

9.save_artifacts, save_visualizations, save_f1_visualization: Functions to save the trained models (using joblib for sklearn, .keras for TF), the preprocessor, evaluation metrics (JSON), and visualizations of explanations (SHAP plots, LIME HTML) and F1 scores. Handles saving for multiple models and includes good error handling.

10.generate_sample_explanations: Takes test data, original indices, the original dataframe, and the trained models. Selects a sample instance. Tries to get the original text for LIME/Anchor (important!). Calls the explain method of the LR and DT PhishingClassifier objects.

11.if __name__ == "__main__": block (Second one):
This is the main entry point for the training script.
Calls load_data().
Calls train_model() to perform the main workflow.
Calls save_artifacts() and save_f1_visualization().

Strengths:
Attempts to train and compare multiple model types.
Uses standard ML practices (pipelines, train/test split, scaling).
Implements multiple XAI techniques (SHAP, LIME, Anchor).
Good structure for saving results and artifacts.
Includes visualization of results.
Handles potential issues with tokenized text format.

Areas for Optimization and Correction:
Hardcoded Paths: Same issue as the first script. //Needs fixing.
Redundancy and Structure:
Two main blocks and multiple definitions/attempts at preprocessing pipelines (ColumnTransformer). Consolidate into the main train_model flow.
The PhishingClassifier class is a good idea but its role needs clarification. It mixes model holding, training logic (for NN), and explainer initialization.
The _text_to_features method inside PhishingClassifier is highly problematic and needs to be removed or corrected. Explanations should use the original preprocessing pipeline.
XAI Input Correction:
SHAP: Should explain predictions on the transformed data (X_test_transformed) using feature names derived from the ColumnTransformer. The current code seems to do this mostly correctly.
LIME/Anchor: Need a prediction function that takes raw text (or tokenized text before TF-IDF), applies the full preprocessor pipeline (TF-IDF + Scaling) and then the classifier, and returns prediction probabilities. This requires passing the preprocessor and the final classifier to the explainer setup.
Hybrid NN Input: Ensure the separate text/structural features passed to the Keras model (_train_hybrid_nn) exactly match the output slices of the ColumnTransformer. The current slicing looks okay but needs verification.
Clarity: The code is complex. Adding more comments, especially around the XAI setup and the ColumnTransformer, would help.
Error Handling in XAI: While visualizations have error handling, the core explain method could benefit from more granular try...except blocks around each explainer type.
Model Saving: Saving the entire Pipeline (preprocessor + classifier) for LR and DT is generally better than saving just the classifier, as it bundles the necessary preprocessing. The code saves the pipeline in the first main but seems to save individual models later. It should save the full pipeline for LR/DT and the Keras model separately. The preprocessor should also be saved separately for potential reuse or inspection.

