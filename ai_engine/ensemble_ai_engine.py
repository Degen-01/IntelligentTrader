"""
Implements an Ensemble AI Engine that combines predictions from multiple individual models.
This enhances robustness and accuracy by leveraging collective intelligence.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import joblib
import os

from intelligent_trader.ai_engine import AIModel, AIEngine
from intelligent_trader.core.config import Config
from intelligent_trader.monitoring.alerts import AlertManager
from intelligent_trader.mlops.model_store import ModelStore
from intelligent_trader.mlops.model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

class EnsembleAIEngine(AIEngine):
    """
    Extends the AIEngine to manage and utilize an ensemble of AI models.
    Supports training multiple base models and combining their predictions.
    """

    def __init__(self, config: Config, alert_manager: AlertManager, model_store: ModelStore, evaluator: ModelEvaluator):
        super().__init__(config, alert_manager, model_store, evaluator)
        self.base_models_config = config.get("ENSEMBLE_BASE_MODELS", [])
        self.meta_model_config = config.get("ENSEMBLE_META_MODEL", {"type": "LogisticRegression"})
        self.ensemble_models: Dict[str, Dict[str, AIModel]] = {} # {symbol: {model_name: model_instance}}
        self.meta_models: Dict[str, Any] = {} # {symbol: meta_model_instance}
        self.load_ensemble_models()
        logger.info(f"Initialized EnsembleAIEngine with {len(self.base_models_config)} base models.")

    def load_ensemble_models(self):
        """Loads all base models and meta-models from the model store."""
        for symbol in self.model_store.list_symbols():
            self.ensemble_models[symbol] = {}
            for model_name in self.base_models_config:
                # Load the latest deployed version of each base model
                try:
                    latest_version = self.model_store.get_deployed_model_version(symbol, model_name['name'])
                    if latest_version:
                        model_instance = self._initialize_model(model_name['type'])
                        model_instance.load(symbol, latest_version)
                        self.ensemble_models[symbol][model_name['name']] = model_instance
                        logger.info(f"Loaded base model {model_name['name']} v{latest_version} for {symbol}.")
                    else:
                        logger.warning(f"No deployed version found for base model {model_name['name']} for {symbol}.")
                except Exception as e:
                    logger.error(f"Failed to load base model {model_name['name']} for {symbol}: {e}")
            
            # Load meta-model
            try:
                meta_model_name = "EnsembleMetaModel"
                latest_meta_version = self.model_store.get_deployed_model_version(symbol, meta_model_name)
                if latest_meta_version:
                    meta_model_path = self.model_store._get_model_path(symbol, meta_model_name, latest_meta_version)
                    self.meta_models[symbol] = joblib.load(os.path.join(meta_model_path, f"{meta_model_name}_v{latest_meta_version}.joblib"))
                    logger.info(f"Loaded meta-model for {symbol}.")
                else:
                    logger.warning(f"No deployed meta-model found for {symbol}.")
            except Exception as e:
                logger.error(f"Failed to load meta-model for {symbol}: {e}")


    async def train_ensemble_model(self, 
                                   symbol: str, 
                                   data: pd.DataFrame, 
                                   target_column: str = 'target',
                                   model_type: str = 'ensemble'):
        """
        Orchestrates the training of all base models and the meta-model for the ensemble.
        """
        logger.info(f"Starting ensemble model training for {symbol}.")
        
        features = [col for col in data.columns if col not in [target_column, 'future_return']]
        X, y = data[features], data[target_column]
        
        if X.empty or y.empty:
            raise ValueError("Input data or target is empty for ensemble training.")
            
        # Ensure sufficient data for time series split
        if len(X) < self.config.get("TRAINING_MIN_SAMPLES", 100):
            logger.warning(f"Insufficient data for training {symbol} (need at least 100, got {len(X)}).")
            return
            
        # 1. Train Base Models
        trained_base_models = {}
        predictions_for_meta_training = pd.DataFrame(index=X.index)

        with ThreadPoolExecutor(max_workers=len(self.base_models_config)) as executor:
            futures = {executor.submit(self._train_base_model_sync, model_conf, X, y, symbol): model_conf for model_conf in self.base_models_config}
            
            for future in as_completed(futures):
                model_conf = futures[future]
                model_name = model_conf['name']
                try:
                    base_model, val_preds_proba, y_val = future.result()
                    if base_model:
                        trained_base_models[model_name] = base_model
                        if val_preds_proba is not None and len(val_preds_proba) == len(y_val):
                            predictions_for_meta_training[model_name] = val_preds_proba[:, 1] # Assuming binary classification, take proba of positive class
                            logger.info(f"Base model '{model_name}' trained successfully and validation predictions collected.")
                        else:
                            logger.warning(f"Validation predictions from base model '{model_name}' were invalid or empty.")
                except Exception as e:
                    logger.error(f"Error training base model '{model_name}': {e}", exc_info=True)
                    self.alert_manager.send_alert("ERROR", f"Ensemble base model training failed: {model_name} for {symbol} - {e}")
        
        if not trained_base_models:
            logger.error(f"No base models trained successfully for ensemble for {symbol}. Aborting ensemble training.")
            self.alert_manager.send_alert("CRITICAL", f"Ensemble training aborted for {symbol}: No base models trained.")
            return

        # Ensure that `y_val` from the last base model training (or a consistent validation set) is used for meta-model target
        # For simplicity, we'll re-align y based on the index of predictions_for_meta_training
        meta_X = predictions_for_meta_training.dropna()
        if meta_X.empty:
            logger.warning(f"No valid predictions from base models to train meta-model for {symbol}.")
            return

        meta_y = y.loc[meta_X.index] # Align target with ensemble predictions

        # 2. Train Meta-Model
        logger.info(f"Training meta-model for {symbol} on base model predictions.")
        meta_model = self._train_meta_model(meta_X, meta_y)
        
        if meta_model:
            # Save and deploy trained base models and meta-model
            self.ensemble_models[symbol] = trained_base_models
            self.meta_models[symbol] = meta_model

            for model_name, model_instance in trained_base_models.items():
                version = self._get_new_model_version(symbol, model_name)
                model_instance.save(symbol, version)
                self.model_store.deploy_model_version(symbol, model_name, version)
                logger.info(f"Deployed base model {model_name} v{version} for {symbol}.")
            
            # Save meta-model directly (it's not an AIModel instance)
            meta_model_version = self._get_new_model_version(symbol, "EnsembleMetaModel")
            meta_model_path = self.model_store._get_model_path(symbol, "EnsembleMetaModel", meta_model_version)
            os.makedirs(meta_model_path, exist_ok=True)
            joblib.dump(meta_model, os.path.join(meta_model_path, f"EnsembleMetaModel_v{meta_model_version}.joblib"))
            self.model_store.deploy_model_version(symbol, "EnsembleMetaModel", meta_model_version)
            logger.info(f"Deployed meta-model v{meta_model_version} for {symbol}.")
            
            logger.info(f"Ensemble model training and deployment completed for {symbol}.")
        else:
            logger.error(f"Meta-model training failed for {symbol}.")


    def _train_base_model_sync(self, model_conf: Dict[str, Any], X: pd.DataFrame, y: pd.Series, symbol: str) -> Tuple[Optional[AIModel], Optional[np.ndarray], Optional[pd.Series]]:
        """Synchronous wrapper for training a single base model."""
        try:
            model_instance = self._initialize_model(model_conf['type'])
            
            # Use TimeSeriesSplit for training and getting validation predictions
            # This logic mimics the cross_validate_model but for a single train/val split
            # For simplicity, using a fixed split (e.g., last 20% for validation)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

            if X_train.empty or X_val.empty or y_train.empty or y_val.empty:
                logger.warning(f"Insufficient data for base model {model_conf['name']} train/val split for {symbol}.")
                return None, None, None

            model_instance.train(X_train, y_train, model_conf.get('params', {}))
            val_preds_proba = model_instance.predict_proba(X_val)
            
            logger.info(f"Base model '{model_conf['name']}' trained and validated.")
            return model_instance, val_preds_proba, y_val
        except Exception as e:
            logger.error(f"Error training base model {model_conf['name']} for {symbol}: {e}")
            return None, None, None


    def _train_meta_model(self, X_meta: pd.DataFrame, y_meta: pd.Series) -> Any:
        """
        Trains the meta-model (blender) on the predictions of the base models.
        """
        meta_model_type = self.meta_model_config['type']
        meta_model_params = self.meta_model_config.get('params', {})
        
        if meta_model_type == "LogisticRegression":
            meta_model = LogisticRegression(solver='liblinear', random_state=42, **meta_model_params)
        elif meta_model_type == "XGBoostClassifier":
            from xgboost import XGBClassifier
            meta_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **meta_model_params)
        else:
            raise ValueError(f"Unsupported meta-model type: {meta_model_type}")
            
        try:
            meta_model.fit(X_meta, y_meta)
            logger.info(f"Meta-model '{meta_model_type}' trained successfully.")
            return meta_model
        except Exception as e:
            logger.error(f"Error training meta-model '{meta_model_type}': {e}", exc_info=True)
            return None


    def generate_ensemble_signals(self, symbol: str, latest_data: pd.DataFrame) -> Tuple[int, float]:
        """
        Generates a trading signal by combining predictions from all base models
        using the trained meta-model.
        
        Returns:
            Tuple of (signal: int, confidence: float)
        """
        if symbol not in self.ensemble_models or not self.ensemble_models[symbol]:
            logger.warning(f"No base models loaded for ensemble for {symbol}. Cannot generate ensemble signal.")
            return 0, 0.5 # Default to hold
        
        if symbol not in self.meta_models or not self.meta_models[symbol]:
            logger.warning(f"No meta-model loaded for {symbol}. Cannot generate ensemble signal.")
            return 0, 0.5 # Default to hold

        features = [col for col in latest_data.columns if col not in ['target', 'future_return']]
        X_live = latest_data[features]
        
        if X_live.empty:
            logger.warning(f"Live data is empty for {symbol}. Cannot generate ensemble signal.")
            return 0, 0.5

        # 1. Get predictions from all base models
        base_model_live_preds = pd.DataFrame(index=X_live.index)
        for model_name, model_instance in self.ensemble_models[symbol].items():
            try:
                # Need to handle feature consistency for each base model
                # Ensure latest_data has the features the model was trained on
                preds_proba = model_instance.predict_proba(X_live)
                base_model_live_preds[model_name] = preds_proba[:, 1]
            except Exception as e:
                logger.warning(f"Could not get prediction from base model '{model_name}': {e}")
                # Potentially fill with default or drop model for this prediction
                base_model_live_preds[model_name] = 0.5 # Neutral if prediction fails
        
        # 2. Use meta-model to combine base model predictions
        X_meta_live = base_model_live_preds.dropna() # Drop rows where any base model failed
        
        if X_meta_live.empty:
            logger.warning(f"No valid base model predictions to feed meta-model for {symbol}.")
            return 0, 0.5

        # Ensure meta-model features match its training features
        # X_meta_live should have columns corresponding to model_names used in _train_meta_model
        
        final_prediction_proba = self.meta_models[symbol].predict_proba(X_meta_live)
        
        # Assuming binary classification (0 or 1)
        signal = 1 if final_prediction_proba[:, 1] > 0.5 else (-1 if final_prediction_proba[:, 0] > 0.5 else 0)
        confidence = np.max(final_prediction_proba) # Confidence in the predicted class
        
        logger.info(f"Ensemble signal for {symbol}: {signal}, Confidence: {confidence:.4f}")
        return signal, confidence


# Example Ensemble Config (to be placed in intelligent_trader/configs/config.py or settings)
"""
ENSEMBLE_BASE_MODELS = [
    {"name": "logistic_reg_model", "type": "LogisticRegressionModel", "params": {"C": 0.1}},
    {"name": "xgb_model", "type": "XGBoostModel", "params": {"n_estimators": 50}},
    {"name": "lstm_model", "type":
