"""
Comprehensive model evaluation system for trading models.
Computes accuracy, precision, recall, F1-score, and financial metrics like Sharpe ratio.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

from intelligent_trader.core.config import Config
from intelligent_trader.monitoring.alerts import AlertManager

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    """Data class for storing comprehensive model performance metrics."""
    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    
    # Regression metrics (for continuous predictions)
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    
    # Financial metrics
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_return: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    
    # Trading-specific metrics
    total_trades: int = 0
    profitable_trades: int = 0
    avg_trade_return: Optional[float] = None
    avg_holding_period: Optional[float] = None
    
    # Model confidence metrics
    avg_prediction_confidence: Optional[float] = None
    confidence_accuracy_correlation: Optional[float] = None
    
    # Temporal metrics
    evaluation_date: datetime = None
    evaluation_period_days: Optional[int] = None

class ModelEvaluator:
    """
    Comprehensive model evaluation system for trading models.
    Supports both classification and regression evaluation with financial metrics.
    """
    
    def __init__(self, config: Config, alert_manager: AlertManager):
        self.config = config
        self.alert_manager = alert_manager
        
        # Performance thresholds from config
        self.min_accuracy = config.get("MODEL_MIN_ACCURACY", 0.60)
        self.min_precision = config.get("MODEL_MIN_PRECISION", 0.55)
        self.min_recall = config.get("MODEL_MIN_RECALL", 0.55)
        self.min_f1_score = config.get("MODEL_MIN_F1_SCORE", 0.55)
        self.min_sharpe_ratio = config.get("MODEL_MIN_SHARPE_RATIO", 0.5)
        
        # Risk-free rate for Sharpe calculation
        self.risk_free_rate = config.get("RISK_FREE_RATE", 0.02)  # 2% annual
        
        logger.info("Initialized ModelEvaluator with performance thresholds.")

    def evaluate_classification_model(self, 
                                    y_true: np.ndarray, 
                                    y_pred: np.ndarray, 
                                    y_pred_proba: Optional[np.ndarray] = None,
                                    class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluates classification model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (for AUC calculation)
            class_names: Names of classes for detailed reporting
            
        Returns:
            Dictionary containing classification metrics
        """
        try:
            # Basic classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # AUC-ROC for binary/multiclass
            auc_roc = None
            if y_pred_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:  # Binary classification
                        auc_roc = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:  # Multiclass
                        auc_roc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                except Exception as e:
                    logger.warning(f"Could not calculate AUC-ROC: {e}")
            
            # Detailed classification report
            class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            # Average prediction confidence
            avg_confidence = None
            confidence_accuracy_corr = None
            if y_pred_proba is not None:
                avg_confidence = np.mean(np.max(y_pred_proba, axis=1))
                # Correlation between confidence and correctness
                correct_predictions = (y_true == y_pred).astype(int)
                max_confidences = np.max(y_pred_proba, axis=1)
                confidence_accuracy_corr = np.corrcoef(max_confidences, correct_predictions)[0][1]
            
            results = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc_roc": auc_roc,
                "classification_report": class_report,
                "confusion_matrix": conf_matrix.tolist(),
                "avg_prediction_confidence": avg_confidence,
                "confidence_accuracy_correlation": confidence_accuracy_corr,
                "total_predictions": len(y_true)
            }
            
            # Check if model meets minimum thresholds
            performance_check = self._check_classification_thresholds(results)
            results["performance_check"] = performance_check
            
            logger.info(f"Classification evaluation completed. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in classification evaluation: {e}", exc_info=True)
            raise

    def evaluate_regression_model(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates regression model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing regression metrics
        """
        try:
            # Basic regression metrics
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            # Additional metrics
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
            
            # Directional accuracy (for financial predictions)
            directional_accuracy = None
            if len(y_true) > 1:
                true_direction = np.sign(np.diff(y_true))
                pred_direction = np.sign(np.diff(y_pred))
                directional_accuracy = np.mean(true_direction == pred_direction)
            
            results = {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "r2_score": r2,
                "mape": mape,
                "directional_accuracy": directional_accuracy,
                "total_predictions": len(y_true)
            }
            
            logger.info(f"Regression evaluation completed. R²: {r2:.4f}, RMSE: {rmse:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in regression evaluation: {e}", exc_info=True)
            raise

    def calculate_financial_metrics(self, 
                                  returns: pd.Series, 
                                  predictions: np.ndarray,
                                  actual_prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Calculates financial performance metrics for trading strategies.
        
        Args:
            returns: Series of strategy returns
            predictions: Model predictions that generated the returns
            actual_prices: Actual price series for additional calculations
            
        Returns:
            Dictionary containing financial metrics
        """
        try:
            if len(returns) == 0:
                logger.warning("Empty returns series provided for financial metrics calculation.")
                return {}
            
            # Convert to numpy for calculations
            returns_array = returns.values
            
            # Basic return metrics
            total_return = np.prod(1 + returns_array) - 1
            annualized_return = (1 + total_return) ** (252 / len(returns_array)) - 1
            
            # Risk metrics
            volatility = np.std(returns_array) * np.sqrt(252)  # Annualized volatility
            
            # Sharpe Ratio
            excess_returns = returns_array - (self.risk_free_rate / 252)
            sharpe_ratio = np.mean(excess_returns) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
            
            # Sortino Ratio (downside deviation)
            downside_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
            
            # Maximum Drawdown
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Win Rate and Profit Factor
            winning_trades = returns_array[returns_array > 0]
            losing_trades = returns_array[returns_array < 0]
            
            win_rate = len(winning_trades) / len(returns_array) if len(returns_array) > 0 else 0
            
            avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
            avg_loss = np.mean(np.abs(losing_trades)) if len(losing_trades) > 0 else 0
            profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if avg_loss > 0 else float('inf')
            
            # Calmar Ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Information Ratio (if benchmark is available, using risk-free rate as proxy)
            benchmark_return = self.risk_free_rate / 252
            excess_return_vs_benchmark = returns_array - benchmark_return
            tracking_error = np.std(excess_return_vs_benchmark) * np.sqrt(252)
            information_ratio = np.mean(excess_return_vs_benchmark) / tracking_error * np.sqrt(252) if tracking_error > 0 else 0
            
            results = {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "calmar_ratio": calmar_ratio,
                "information_ratio": information_ratio,
                "total_trades": len(returns_array),
                "profitable_trades": len(winning_trades),
                "avg_trade_return": np.mean(returns_array),
                "avg_win": avg_win,
                "avg_loss": avg_loss
            }
            
            # Check if financial performance meets thresholds
            financial_check = self._check_financial_thresholds(results)
            results["financial_performance_check"] = financial_check
            
            logger.info(f"Financial metrics calculated. Sharpe: {sharpe_ratio:.4f}, Max DD: {max_drawdown:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating financial metrics: {e}", exc_info=True)
            raise

    def cross_validate_model(self, 
                           model, 
                           X: pd.DataFrame, 
                           y: pd.Series, 
                           cv_folds: int = 5,
                           time_series_split: bool = True) -> Dict[str, Any]:
        """
        Performs cross-validation evaluation for time series data.
        
        Args:
            model: The model to evaluate
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds
            time_series_split: Whether to use TimeSeriesSplit (recommended for financial data)
            
        Returns:
            Dictionary containing cross-validation results
        """
        try:
            if time_series_split:
                cv_splitter = TimeSeriesSplit(n_splits=cv_folds)
            else:
                from sklearn.model_selection import KFold
                cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            cv_scores = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': []
            }
            
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model on fold
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_val)
                
                # Evaluate fold
                fold_metrics = self.evaluate_classification_model(
                    y_val.values, y_pred, y_pred_proba
                )
                
                # Store scores
                cv_scores['accuracy'].append(fold_metrics['accuracy'])
                cv_scores['precision'].append(fold_metrics['precision'])
                cv_scores['recall'].append(fold_metrics['recall'])
                cv_scores['f1_score'].append(fold_metrics['f1_score'])
                
                fold_results.append({
                    'fold': fold,
                    'train_size': len(train_idx),
                    'val_size': len(val_idx),
                    'metrics': fold_metrics
                })
                
                logger.info(f"Fold {fold}: Accuracy={fold_metrics['accuracy']:.4f}, F1={fold_metrics['f1_score']:.4f}")
            
            # Calculate mean and std of CV scores
            cv_summary = {}
            for metric, scores in cv_scores.items():
                cv_summary[f'{metric}_mean'] = np.mean(scores)
                cv_summary[f'{metric}_std'] = np.std(scores)
            
            results = {
                'cv_summary': cv_summary,
                'fold_results': fold_results,
                'cv_method': 'TimeSeriesSplit' if time_series_split else 'KFold',
                'n_folds': cv_folds
            }
            
            logger.info(f"Cross-validation completed. Mean accuracy: {cv_summary['accuracy_mean']:.4f} ± {cv_summary['accuracy_std']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}", exc_info=True)
            raise

    def _check_classification_thresholds(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Checks if classification metrics meet minimum thresholds."""
        return {
            "accuracy_pass": metrics["accuracy"] >= self.min_accuracy,
            "precision_pass": metrics["precision"] >= self.min_precision,
            "recall_pass": metrics["recall"] >= self.min_recall,
            "f1_score_pass": metrics["f1_score"] >= self.min_f1_score,
            "overall_pass": all([
                metrics["accuracy"] >= self.min_accuracy,
                metrics["precision"] >= self.min_precision,
                metrics["recall"] >= self.min_recall,
                metrics["f1_score"] >= self.min_f1_score
            ])
        }

    def _check_financial_thresholds(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Checks if financial metrics meet minimum thresholds."""
        return {
            "sharpe_ratio_pass": metrics.get("sharpe_ratio", 0) >= self.min_sharpe_ratio,
            "positive_return": metrics.get("total_return", 0) > 0,
            "win_rate_acceptable": metrics.get("win_rate", 0) > 0.4,  # At least 40% win rate
            "max_drawdown_acceptable": metrics.get("max_drawdown", 0) > -0.2,  # Max 20% drawdown
            "overall_financial_pass": all([
                metrics.get("sharpe_ratio", 0) >= self.min_sharpe_ratio,
                metrics.get("total_return", 0) > 0,
                metrics.get("max_drawdown", 0) > -0.2
            ])
        }

    def generate_evaluation_report(self, 
                                 model_name: str,
                                 evaluation_results: Dict[str, Any],
                                 save_path: Optional[str] = None) -> str:
        """
        Generates a comprehensive evaluation report.
        
        Args:
            model_name: Name of the evaluated model
            evaluation_results: Results from evaluation methods
            save_path: Optional path to save the report
            
        Returns:
            String containing the formatted report
        """
        try:
            report_lines = [
                f"Model Evaluation Report: {model_name}",
                "=" * 50,
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ""
            ]
            
            # Classification metrics
            if 'accuracy' in evaluation_results:
                report_lines.extend([
                    "Classification Metrics:",
                    "-" * 25,
                    f"Accuracy: {evaluation_results['accuracy']:.4f}",
                    f"Precision: {evaluation_results['precision']:.4f}",
                    f"Recall: {evaluation_results['recall']:.4f}",
                    f"F1-Score: {evaluation_results['f1_score']:.4f}",
                ])
                
                if evaluation_results.get('auc_roc'):
                    report_lines.append(f"AUC-ROC: {evaluation_results['auc_roc']:.4f}")
                
                report_lines.append("")
            
            # Financial metrics
            if 'sharpe_ratio' in evaluation_results:
                report_lines.extend([
                    "Financial Metrics:",
                    "-" * 20,
                    f"Sharpe Ratio: {evaluation_results['sharpe_ratio']:.4f}",
                    f"Total Return: {evaluation_results['total_return']:.4f}",
                    f"Max Drawdown: {evaluation_results['max_drawdown']:.4f}",
                    f"Win Rate: {evaluation_results['win_rate']:.4f}",
                    f"Profit Factor: {evaluation_results['profit_factor']:.4f}",
                    ""
                ])
            
            # Performance checks
            if 'performance_check' in evaluation_results:
                check = evaluation_results['performance_check']
                report_lines.extend([
                    "Performance Threshold Check:",
                    "-" * 30,
                    f"Overall Pass: {'✓' if check['overall_pass'] else '✗'}",
                    f"Accuracy Pass: {'✓' if check['accuracy_pass'] else '✗'}",
                    f"Precision Pass: {'✓' if check['precision_pass'] else '✗'}",
                    f"Recall Pass: {'✓' if check['recall_pass'] else '✗'}",
                    f"F1-Score Pass: {'✓' if check['f1_score_pass'] else '✗'}",
                    ""
                ])
            
            report = "\n".join(report_lines)
            
            # Save report if path provided
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report)
                logger.info(f"Evaluation report saved to {save_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {e}", exc_info=True)
            raise

