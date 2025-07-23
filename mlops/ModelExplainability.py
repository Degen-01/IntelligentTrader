import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Callable, Optional

# Try to import SHAP and LIME
try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False
    logging.warning("SHAP library not found. Install 'shap' for full explainability features.")

try:
    import lime
    import lime.lime_tabular
    lime_available = True
except ImportError:
    lime_available = False
    logging.warning("LIME library not found. Install 'lime' for full explainability features.")

logger = logging.getLogger(__name__)

class ModelExplainabilityError(Exception):
    """Custom exception for model explainability failures."""
    pass

class ModelExplainability:
    def __init__(self, model: Any, feature_names: List[str], class_names: Optional[List[str]] = None, model_type: str = "classification"):
        """
        Initializes the ModelExplainability class.
        
        Args:
            model: The trained AI model (e.g., sklearn model, Keras model).
            feature_names: List of feature names used by the model.
            class_names: List of class names for classification models (e.g., ['HOLD', 'BUY', 'SELL']).
            model_type: Type of the model ('classification' or 'regression').
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.model_type = model_type.lower()

        if self.model_type not in ["classification", "regression"]:
            raise ValueError("model_type must be 'classification' or 'regression'.")
        
        logger.info(f"Initialized ModelExplainability for model with {len(feature_names)} features.")

    def _get_model_predict_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Returns a callable prediction function compatible with SHAP/LIME.
        Adapts the model's prediction method based on model type.
        """
        if self.model_type == "classification":
            # For classification, LIME/SHAP often expect predict_proba
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba
            else:
                # Fallback for models without predict_proba (e.g., some SVMs)
                # This might require calibration or won't work perfectly with all explainers
                logger.warning("Model does not have 'predict_proba'. Using 'predict' which might not be ideal for some explainers.")
                # For LIME/SHAP, if predict_proba is not available, try to create a dummy one
                # For binary classification, this might be `lambda x: np.array([[1-p, p] for p in self.model.predict(x)])`
                # For multi-class, it's more complex without actual probabilities.
                # It's generally recommended to use models that output probabilities.
                raise ModelExplainabilityError("Classification model must have a 'predict_proba' method for explainability.")
        elif self.model_type == "regression":
            if hasattr(self.model, 'predict'):
                return self.model.predict
            else:
                raise ModelExplainabilityError("Regression model must have a 'predict' method.")
        else:
            raise ValueError("Unsupported model type.")

    def get_shap_values(self, X: pd.DataFrame, background_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Computes SHAP (SHapley Additive exPlanations) values for model predictions.
        Requires the 'shap' library.
        
        Args:
            X: A Pandas DataFrame or NumPy array of input features for which to calculate SHAP values.
            background_data: A representative background dataset (e.g., training data subset) for KernelExplainer.
                             Crucial for accurate explanations, especially for model-agnostic explainers.
            
        Returns:
            A dictionary containing SHAP values and related information.
        """
        if not shap_available:
            raise ModelExplainabilityError("SHAP library not installed. Cannot compute SHAP values.")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)

        if background_data is None:
            logger.warning("No background_data provided for SHAP. This may lead to less accurate explanations, especially for KernelExplainer.")
            # If no background data, SHAP might fall back to a less ideal default or raise error.
            # For tree models, `shap.TreeExplainer` can often work without explicit background data.
            # For KernelExplainer, a small sample of training data is usually required.
            if hasattr(self.model, 'tree_model'): # Heuristic for tree-based models like XGBoost, LightGBM
                explainer = shap.TreeExplainer(self.model)
            else:
                # Fallback for other models, this might fail without background data
                try:
                    explainer = shap.KernelExplainer(self._get_model_predict_fn(), X) 
                except Exception as e:
                    logger.error(f"Failed to initialize KernelExplainer without background data: {e}")
                    raise ModelExplainabilityError(f"KernelExplainer requires background_data for robust explanations. Error: {e}")
        else:
            # For KernelExplainer, background_data is usually a sample of the training data
            explainer = shap.KernelExplainer(self._get_model_predict_fn(), background_data)
        
        try:
            shap_values = explainer(X)
            
            logger.info(f"Computed SHAP values for {X.shape[0]} instances.")
            
            # SHAP values can be multi-dimensional for multi-output models (e.g., multi-class classification)
            # We return them in a structure that can be easily serialized.
            if hasattr(shap_values, 'values') and shap_values.values.ndim > 1:
                # For multi-class, shap_values.values will be (N_instances, N_features, N_classes)
                # Or (N_instances, N_features) if binary/regression
                # Convert to list of lists for serialization
                raw_values = shap_values.values.tolist()
            else: # Single output (regression or binary classification)
                raw_values = shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values

            expected_value = explainer.expected_value.tolist() if hasattr(explainer.expected_value, 'tolist') else explainer.expected_value

            return {
                "method": "SHAP",
                "shap_values": raw_values,
                "expected_value": expected_value,
                "feature_names": self.feature_names,
                "class_names": self.class_names,
                "data_points_explained": X.index.tolist() if isinstance(X, pd.DataFrame) else list(range(X.shape[0]))
            }
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}", exc_info=True)
            raise ModelExplainabilityError(f"Failed to compute SHAP values: {e}")

    def get_lime_explanation(self, instance: Union[np.ndarray, pd.Series], training_data_raw: pd.DataFrame, num_features: int = 10) -> Dict[str, Any]:
        """
        Generates a LIME (Local Interpretable Model-agnostic Explanations) explanation for a single prediction.
        Requires the 'lime' library.
        
        Args:
            instance: A single data instance (NumPy array or Pandas Series) to explain.
            training_data_raw: A Pandas DataFrame of the original training data. LIME uses this to learn feature distributions.
            num_features: Number of features to include in the explanation.
            
        Returns:
            A dictionary containing LIME explanation details.
        """
        if not lime_available:
            raise ModelExplainabilityError("LIME library not installed. Cannot compute LIME explanation.")

        if not isinstance(training_data_raw, pd.DataFrame):
            raise TypeError("training_data_raw must be a Pandas DataFrame.")

        # LIME expects a numpy array for feature data
        if isinstance(instance, pd.Series):
            instance_array = instance.values
        elif isinstance(instance, np.ndarray):
            instance_array = instance
        else:
            raise TypeError("Instance must be a Pandas Series or NumPy array.")
        
        if instance_array.shape[0] != len(self.feature_names):
            raise ValueError(f"Instance dimensions ({instance_array.shape[0]}) do not match feature names count ({len(self.feature_names)}).")

        try:
            # Initialize LIME TabularExplainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data_raw.values, # LIME expects numpy array for training data
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode=self.model_type, # 'classification' or 'regression'
                discretize_continuous=True # Useful for numerical features
            )

            # Get explanation for the instance
            explanation = explainer.explain_instance(
                data_row=instance_array,
                predict_fn=self._get_model_predict_fn(),
                num_features=num_features,
                num_samples=1000 # Number of perturbed samples for explanation
            )
            
            logger.info(f"Computed LIME explanation for instance.")
            
            # Extract explanation as a list of (feature, weight) tuples
            explanation_list = explanation.as_list()

            return {
                "method": "LIME",
                "explanation_features": explanation_list,
                "predicted_value": explanation.predicted_value,
                "true_value": explanation.true_value if hasattr(explanation, 'true_value') else None,
                "prediction_probabilities": explanation.predict_proba.tolist() if hasattr(explanation, 'predict_proba') else None,
                "feature_names": self.feature_names,
                "class_names": self.class_names,
            }
        except Exception as e:
            logger.error(f"Error computing LIME explanation: {e}", exc_info=True)
            raise ModelExplainabilityError(f"Failed to compute LIME explanation: {e}")

    def explain_prediction(self, X_instance: Union[pd.DataFrame, pd.Series, np.ndarray],
                           method: str = "SHAP", **kwargs) -> Dict[str, Any]:
        """
        Provides an explanation for a single prediction using the specified method (SHAP or LIME).
        
        Args:
            X_instance: A single data instance (DataFrame row, Series, or NumPy array).
            method: The explanation method to use ('SHAP' or 'LIME').
            **kwargs: Additional arguments for the chosen explanation method (e.g., background_data for SHAP, training_data_raw for LIME).
            
        Returns:
            A dictionary containing the explanation details.
        """
        if method.lower() == "shap":
            # SHAP expects a DataFrame for multiple instances, or can handle a single row (2D array)
            if isinstance(X_instance, (pd.Series, np.ndarray)):
                X_instance = pd.DataFrame(X_instance.reshape(1, -1), columns=self.feature_names)
            elif isinstance(X_instance, pd.DataFrame) and len(X_instance) > 1:
                logger.warning("SHAP explanation requested for multiple instances. Returning SHAP values for all.")
            
            return self.get_shap_values(X_instance, **kwargs)
        elif method.lower() == "lime":
            if isinstance(X_instance, pd.DataFrame) and len(X_instance) > 1:
                logger.warning("LIME explanation requested for multiple instances, but LIME explains one by one. Explaining the first instance.")
                instance = X_instance.iloc[0]
            elif isinstance(X_instance, pd.DataFrame):
                instance = X_instance.iloc[0] if len(X_instance) == 1 else pd.Series(X_instance.values.flatten(), index=self.feature_names)
            else:
                instance = X_instance
            
            return self.get_lime_explanation(instance, **kwargs)
        else:
            raise ValueError(f"Unsupported explanation method: {method}. Choose 'SHAP' or 'LIME'.")

