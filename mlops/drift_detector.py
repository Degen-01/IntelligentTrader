"""
Data Drift and Model Drift Detection for AI trading models.
Identifies changes in data distributions or model performance degradation over time.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Libraries for drift detection (e.g., Alibi-Detect, Evidently AI, or custom implementations)
# For this example, we'll implement simple statistical checks.
from scipy.stats import ks_2samp, chisquare
from sklearn.metrics import accuracy_score

from intelligent_trader.monitoring.alerts import AlertManager
from intelligent_trader.core.config import Config

logger = logging.getLogger(__name__)

class DriftDetectionError(Exception):
    """Custom exception for drift detection failures."""
    pass

class DriftDetector:
    def __init__(self, config: Config, alert_manager: AlertManager):
        self.config = config
        self.alert_manager = alert_manager
        
        self.data_drift_threshold = config.get("DATA_DRIFT_THRESHOLD", 0.05) # p-value threshold for KS test
        self.model_drift_accuracy_drop_threshold = config.get("MODEL_DRIFT_ACCURACY_DROP_THRESHOLD", 0.05) # % drop
        self.model_drift_min_samples = config.get("MODEL_DRIFT_MIN_SAMPLES", 100) # Minimum samples for evaluation

        logger.info(f"Initialized DriftDetector with data drift threshold={self.data_drift_threshold}, "
                    f"model drift accuracy drop threshold={self.model_drift_accuracy_drop_threshold}.")

    def detect_data_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects data drift by comparing feature distributions between two datasets.
        Uses Kolmogorov-Smirnov test for numerical features and Chi-squared for categorical.
        
        Args:
            reference_data: DataFrame representing the baseline data distribution (e.g., training data).
            current_data: DataFrame representing the current data distribution (e.g., recent production data).
            
        Returns:
            A dictionary detailing drift status per feature.
        """
        drift_results = {"data_drift_detected": False, "features_drifted": {}}
        
        common_features = list(set(reference_data.columns) & set(current_data.columns))
        if not common_features:
            raise DriftDetectionError("No common features found between reference and current data.")

        for feature in common_features:
            if feature not in reference_data.columns or feature not in current_data.columns:
                continue

            ref_series = reference_data[feature].dropna()
            curr_series = current_data[feature].dropna()

            if ref_series.empty or curr_series.empty:
                logger.warning(f"Skipping drift detection for feature '{feature}' due to empty series after dropping NaNs.")
                continue

            if pd.api.types.is_numeric_dtype(ref_series) and pd.api.types.is_numeric_dtype(curr_series):
                # Kolmogorov-Smirnov test for numerical data
                statistic, p_value = ks_2samp(ref_series, curr_series)
                if p_value < self.data_drift_threshold:
                    drift_results["data_drift_detected"] = True
                    drift_results["features_drifted"][feature] = {
                        "type": "numerical",
                        "metric": "KS_p_value",
                        "value": p_value,
                        "threshold": self.data_drift_threshold,
                        "drift_detected": True
                    }
                    logger.warning(f"Data drift detected in numerical feature '{feature}': KS p-value={p_value:.4f} < {self.data_drift_threshold}")
                    self.alert_manager.send_alert("WARNING", f"Data drift detected in feature '{feature}'.")
            elif pd.api.types.is_categorical_dtype(ref_series) or pd.api.types.is_object_dtype(ref_series):
                # Chi-squared test for categorical data
                ref_counts = ref_series.value_counts()
                curr_counts = curr_series.value_counts()
                
                # Align indices and fill missing categories with 0
                all_categories = pd.Index(ref_counts.index.tolist() + curr_counts.index.tolist()).unique()
                expected_freq = ref_counts.reindex(all_categories, fill_value=0)
                observed_freq = curr_counts.reindex(all_categories, fill_value=0)

                # Chi-squared requires non-zero expected frequencies for calculation.
                # Filter out categories that are zero in both or very small in expected.
                # A common heuristic is to add a small value (e.g., 1) to all counts if any are zero.
                expected_freq_nz = expected_freq[expected_freq > 0]
                observed_freq_aligned = observed_freq.reindex(expected_freq_nz.index, fill_value=0)
                
                if len(expected_freq_nz) > 1 and observed_freq_aligned.sum() > 0:
                    try:
                        statistic, p_value = chisquare(f_obs=observed_freq_aligned, f_exp=expected_freq_nz)
                        if p_value < self.data_drift_threshold:
                            drift_results["data_drift_detected"] = True
                            drift_results["features_drifted"][feature] = {
                                "type": "categorical",
                                "metric": "Chi2_p_value",
                                "value": p_value,
                                "threshold": self.data_drift_threshold,
                                "drift_detected": True
                            }
                            logger.warning(f"Data drift detected in categorical feature '{feature}': Chi2 p-value={p_value:.4f} < {self.data_drift_threshold}")
                            self.alert_manager.send_alert("WARNING", f"Data drift detected in feature '{feature}'.")
                    except ValueError as e:
                        logger.warning(f"Could not perform Chi-squared test for feature '{feature}': {e}. Possibly not enough categories or zero counts.")
                else:
                    logger.debug(f"Skipping Chi-squared test for feature '{feature}': Not enough categories or zero observed counts.")
            else:
                logger.debug(f"Skipping drift detection for feature '{feature}': Unsupported data type.")
        
        if drift_results["data_drift_detected"]:
            logger.critical("Overall data drift detected! This may impact model performance. Consider retraining.")
            self.alert_manager.send_alert("CRITICAL", "Significant data drift detected. Model retraining may be required.")
        else:
            logger.info("No significant data drift detected.")
        
        return drift_results

    def detect_model_drift(self, 
                           reference_model_performance: Dict[str, float], 
                           current_predictions: np.ndarray, 
                           current_true_labels: np.ndarray) -> Dict[str, Any]:
        """
        Detects model drift by comparing current live performance against reference performance.
        Primarily checks for a significant drop in accuracy for classification models.
        
        Args:
            reference_model_performance: Dictionary of reference metrics (e.g., {'accuracy': 0.9}).
            current_predictions: Array of predictions from the model on recent data.
            current_true_labels: Array of true labels corresponding to current_predictions.
            
        Returns:
            A dictionary detailing model drift status.
        """
        model_drift_results = {"model_drift_detected": False, "details": {}}

        if len(current_predictions) < self.model_drift_min_samples:
            logger.info(f"Not enough samples ({len(current_predictions)}) to confidently detect model drift. Minimum required: {self.model_drift_min_samples}")
            model_drift_results["details"]["message"] = "Insufficient samples for robust model drift detection."
            return model_drift_results

        if "accuracy" in reference_model_performance:
            reference_accuracy = reference_model_performance["accuracy"]
            current_accuracy = accuracy_score(current_true_labels, current_predictions)
            
            accuracy_drop = reference_accuracy - current_accuracy
            relative_accuracy_drop = accuracy_drop / reference_accuracy if reference_accuracy > 0 else 0

            model_drift_results["details"]["reference_accuracy"] = reference_accuracy
            model_drift_results["details"]["current_accuracy"] = current_accuracy
            model_drift_results["details"]["accuracy_drop"] = accuracy_drop
            model_drift_results["details"]["relative_accuracy_drop"] = relative_accuracy_drop

            if relative_accuracy_drop > self.model_drift_accuracy_drop_threshold:
                model_drift_results["model_drift_detected"] = True
                model_drift_results["details"]["drift_detected"] = True
                logger.critical(f"Model drift detected! Accuracy dropped by {relative_accuracy_drop:.2%} (from {reference_accuracy:.2f} to {current_accuracy:.2f}). Consider retraining.")
                self.alert_manager.send_alert("CRITICAL", f"Model drift detected! Accuracy dropped by {relative_accuracy_drop:.2%}. Retraining recommended.")
            else:
                logger.info(f"No significant model drift detected. Accuracy: {current_accuracy:.2f} (Reference: {reference_accuracy:.2f}).")
        else:
            logger.warning("Reference model performance does not contain 'accuracy' metric for drift detection.")
            model_drift_results["details"]["message"] = "Accuracy metric not available in reference performance."
        
        return model_drift_results

