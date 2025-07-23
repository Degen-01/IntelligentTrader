"""
Model Evaluation and Monitoring for AI/ML trading models.
Provides tools for comprehensive offline evaluation and continuous online monitoring.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Union, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from datetime import datetime

# Assuming Prometheus metrics are integrated for live monitoring
from intelligent_trader.core.metrics import TRADING_MODEL_PREDICTIONS_TOTAL, TRADING_MODEL_ACCURACY, \
    TRADING_MODEL_F1_SCORE, TRADING_MODEL_PRECISION, TRADING_MODEL_RECALL # (assuming these are defined in core/metrics.py)

logger = logging.getLogger(__name__)

class ModelEvaluationError(Exception):
    """Custom exception for model evaluation failures."""
    pass

class ModelEvaluator:
    def __init__(self, model_name: str, model_type: str = "classification"):
        self.model_name = model_name
        self.model_type = model_type.lower()
        if self.model_type not in ["classification", "regression"]:
            raise ValueError("model_type must be 'classification' or 'regression'.")
        logger.info(f"Initialized ModelEvaluator for {model_name} ({model_type}).")

    def evaluate_classification_model(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluates a classification model and returns a dictionary of metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (for AUC-ROC).
        
        Returns:
            A dictionary containing classification metrics.
        """
        if len(y_true) != len(y_pred):
            raise ModelEvaluationError("y_true and y_pred must have the same length.")

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(), # Convert to list for JSON serialization
            "classification_report": classification_report(y_true, y_pred, output_dict=True)
        }
        
        if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] > 1:
            try:
                # For multi-class AUC, need to handle each class
                # This is a simplification; for proper multi-class AUC, consider OVR or other strategies
                metrics["roc_auc_score"] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except ValueError as e:
                logger.warning(f"Could not calculate ROC AUC score: {e}. Ensure y_proba is suitable.")
                metrics["roc_auc_score"] = None
        else:
            metrics["roc_auc_score"] = None # Not applicable or insufficient data

        logger.info(f"Classification metrics for {self.model_name}: {metrics}")
        return metrics

    def evaluate_regression_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates a regression model and returns a dictionary of metrics.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
        
        Returns:
            A dictionary containing regression metrics.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        if len(y_true) != len(y_pred):
            raise ModelEvaluationError("y_true and y_pred must have the same length.")

        metrics = {
            "mean_absolute_error": mean_absolute_error(y_true, y_pred),
            "mean_squared_error": mean_squared_error(y_true, y_pred),
            "root_mean_squared_error": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2_score": r2_score(y_true, y_pred)
        }
        logger.info(f"Regression metrics for {self.model_name}: {metrics}")
        return metrics

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generic evaluation method that dispatches to classification or regression.
        
        Args:
            y_true: True labels/values.
            y_pred: Predicted labels/values.
            y_proba: Predicted probabilities (only for classification).
        
        Returns:
            A dictionary containing relevant evaluation metrics.
        """
        if self.model_type == "classification":
            return self.evaluate_classification_model(y_true, y_pred, y_proba)
        elif self.model_type == "regression":
            return self.evaluate_regression_model(y_true, y_pred)
        else:
            raise ModelEvaluationError(f"Unsupported model type: {self.model_type}")

    def log_metrics_to_prometheus(self, metrics: Dict[str, Union[float, int]]):
        """
        Logs key performance metrics to Prometheus for continuous monitoring.
        This method assumes that Prometheus metric objects are globally accessible or passed.
        """
        if self.model_type == "classification":
            if "accuracy" in metrics:
                TRADING_MODEL_ACCURACY.labels(model_name=self.model_name).set(metrics["accuracy"])
            if "f1_score" in metrics:
                TRADING_MODEL_F1_SCORE.labels(model_name=self.model_name).set(metrics["f1_score"])
            if "precision" in metrics:
                TRADING_MODEL_PRECISION.labels(model_name=self.model_name).set(metrics["precision"])
            if "recall" in metrics:
                TRADING_MODEL_RECALL.labels(model_name=self.model_name).set(metrics["recall"])
            logger.info(f"Logged classification metrics for {self.model_name} to Prometheus.")
        # Add similar logic for regression metrics if needed
        else:
            logger.warning(f"Prometheus logging not fully implemented for model type: {self.model_type}")

    def monitor_live_predictions(self, 
                                 prediction_timestamp: datetime, 
                                 symbol: str, 
                                 predicted_signal: int, 
                                 true_signal: Optional[int] = None, 
                                 confidence: Optional[float] = None):
        """
        Monitors live predictions, including logging and updating Prometheus counters.
        
        Args:
            prediction_timestamp: The timestamp of the prediction.
            symbol: The asset symbol.
            predicted_signal: The predicted trading signal (e.g., 0 for HOLD, 1 for BUY, 2 for SELL).
            true_signal: The actual signal that occurred (for accuracy tracking, optional).
            confidence: The confidence score of the prediction (optional).
        """
        TRADING_MODEL_PREDICTIONS_TOTAL.labels(
            model_name=self.model_name, 
            symbol=symbol, 
            predicted_signal=predicted_signal
        ).inc()

        log_msg = (f"Live prediction for {self.model_name} on {symbol} at {prediction_timestamp}: "
                   f"Signal={predicted_signal}, Confidence={confidence:.4f}" if confidence is not None else "")
        if true_signal is not None:
            is_correct = (predicted_signal == true_signal)
            log_msg += f", True_Signal={true_signal}, Correct={is_correct}"
            if is_correct:
                TRADING_MODEL_ACCURACY.labels(model_name=self.model_name).inc() # This is a counter, not a gauge of ratio
                # For ratio, would need total correct predictions and total predictions in a given window.
                # A more robust approach uses Pushgateway or PromQL to calculate ratio.
        
        logger.info(log_msg)

