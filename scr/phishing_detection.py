import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Use the existing logger from preprocessing.py
logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, y_prob=None, output_dir=None, model_name="model"):
    """
    Evaluate model performance with multiple metrics.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like, optional
        Probability estimates for positive class
    output_dir : str, optional
        Directory to save visualizations
    model_name : str, optional
        Name of the model for labeling output files
        
    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics
    """
    try:
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Additional metrics commonly used in security context
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Calculate MCC (Matthews Correlation Coefficient)
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0
        
        # Gather all metrics in a dictionary
        metrics = {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'specificity': round(specificity, 4),
            'false_positive_rate': round(fpr, 4),
            'false_negative_rate': round(fnr, 4),
            'matthews_corr_coef': round(mcc, 4),
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        # Log all metrics
        logger.info(f"Evaluation metrics for {model_name}:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value}")
            
        # If probabilities are provided, calculate AUC-ROC
        if y_prob is not None:
            try:
                # ROC AUC Score
                roc_auc = roc_auc_score(y_true, y_prob)
                metrics['roc_auc'] = round(roc_auc, 4)
                
                # Calculate precision-recall AUC
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
                pr_auc = auc(recall_curve, precision_curve)
                metrics['pr_auc'] = round(pr_auc, 4)
                
                logger.info(f"  ROC AUC: {metrics['roc_auc']}")
                logger.info(f"  PR AUC: {metrics['pr_auc']}")
                
                # If output directory is provided, save plots
                if output_dir:
                    create_evaluation_plots(y_true, y_pred, y_prob, metrics, output_dir, model_name)
            
            except Exception as e:
                logger.error(f"Error calculating AUC metrics: {str(e)}", exc_info=True)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}", exc_info=True)
        return None

def create_evaluation_plots(y_true, y_pred, y_prob, metrics, output_dir, model_name):
    """
    Create and save evaluation plots.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like
        Probability estimates for positive class
    metrics : dict
        Dictionary of metrics from evaluate_model
    output_dir : str
        Directory to save plots
    model_name : str
        Name of the model for labeling output files
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legitimate', 'Phishing'],
                    yticklabels=['Legitimate', 'Phishing'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
        plt.close()
        
        # Key metrics visualization
        plt.figure(figsize=(10, 6))
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
        values = [metrics[m] for m in key_metrics]
        
        bars = plt.bar(key_metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.ylim(0, 1.1)
        plt.ylabel('Score')
        plt.title(f'Performance Metrics - {model_name}')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_metrics.png'))
        plt.close()
        
        # Create summary text file with all metrics
        with open(os.path.join(output_dir, f'{model_name}_metrics_summary.txt'), 'w') as f:
            f.write(f"EVALUATION RESULTS FOR: {model_name}\n")
            f.write("="*50 + "\n\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
        
        logger.info(f"Evaluation plots saved to {output_dir}")
    
    except Exception as e:
        logger.error(f"Error creating evaluation plots: {str(e)}", exc_info=True)

def calculate_cross_dataset_metrics(model, datasets, dataset_names, output_dir):
    """
    Evaluate model performance across multiple datasets to check for generalization.
    
    Parameters:
    -----------
    model : trained model object
        The model to evaluate (must have predict and predict_proba methods)
    datasets : list of tuples
        List of (X_test, y_test) pairs for each dataset
    dataset_names : list of str
        Names corresponding to each dataset
    output_dir : str
        Directory to save evaluation results
    
    Returns:
    --------
    dict
        Dictionary containing metrics for each dataset
    """
    try:
        results = {}
        
        # Create directory for cross-dataset evaluation
        cross_eval_dir = os.path.join(output_dir, "cross_dataset_evaluation")
        os.makedirs(cross_eval_dir, exist_ok=True)
        
        # Evaluate on each dataset
        for (X_test, y_test), dataset_name in zip(datasets, dataset_names):
            logger.info(f"Evaluating on {dataset_name} dataset...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics for this dataset
            metrics = evaluate_model(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                output_dir=os.path.join(cross_eval_dir, dataset_name),
                model_name=f"model_on_{dataset_name}"
            )
            
            results[dataset_name] = metrics
        
        # Create comparison chart across datasets
        compare_datasets(results, dataset_names, cross_eval_dir)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in cross-dataset evaluation: {str(e)}", exc_info=True)
        return None

def compare_datasets(results, dataset_names, output_dir):
    """
    Create comparison visualizations across datasets.
    
    Parameters:
    -----------
    results : dict
        Dictionary with dataset names as keys and metric dictionaries as values
    dataset_names : list of str
        Names of the datasets to compare
    output_dir : str
        Directory to save the comparison visualizations
    """
    try:
        # Key metrics to compare
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Prepare data for plotting
        data = {metric: [results[dataset].get(metric, 0) for dataset in dataset_names] 
                for metric in key_metrics}
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        x = np.arange(len(dataset_names))
        width = 0.2
        
        # Plot bars for each metric
        for i, metric in enumerate(key_metrics):
            plt.bar(x + i*width - width*1.5, data[metric], width, label=metric)
        
        plt.xlabel('Dataset')
        plt.ylabel('Score')
        plt.title('Cross-Dataset Performance Comparison')
        plt.xticks(x, dataset_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, "cross_dataset_comparison.png"))
        plt.close()
        
        # Save comparison data to CSV
        df = pd.DataFrame({dataset: {metric: results[dataset].get(metric, 0) 
                                    for metric in key_metrics} 
                        for dataset in dataset_names})
        df.to_csv(os.path.join(output_dir, "cross_dataset_metrics.csv"))
        
        logger.info(f"Cross-dataset comparison saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating dataset comparison: {str(e)}", exc_info=True)

# Example integration with your existing code
# This can be called in your model training script after preprocessing
"""
# After model training and prediction:
metrics = evaluate_model(
    y_true=y_test,
    y_pred=y_pred,
    y_prob=y_prob,
    output_dir="path/to/evaluation/results",
    model_name="phishing_detector_v1"
)

# To evaluate across multiple datasets:
calculate_cross_dataset_metrics(
    model=trained_model,
    datasets=[(X_test1, y_test1), (X_test2, y_test2)],
    dataset_names=["Enron", "Nazario"],
    output_dir="path/to/evaluation/results"
)
"""
