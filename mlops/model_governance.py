"""
Model Governance policies and frameworks for ML trading models.
Ensures compliance, auditability, and responsible AI practices.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os

from intelligent_trader.mlops.model_store import ModelStore, ModelStoreError
from intelligent_trader.monitoring.alerts import AlertManager # Assuming AlertManager is available
from intelligent_trader.core.config import Config # For accessing global configurations

logger = logging.getLogger(__name__)

class ModelGovernanceError(Exception):
    """Custom exception for model governance failures."""
    pass

class ModelGovernance:
    def __init__(self, config: Config, model_store: ModelStore, alert_manager: AlertManager):
        self.config = config
        self.model_store = model_store
        self.alert_manager = alert_manager
        
        self.governance_policies = self._load_governance_policies()
        logger.info("Initialized ModelGovernance with loaded policies.")

    def _load_governance_policies(self) -> Dict[str, Any]:
        """
        Loads governance policies from a configuration or file.
        Policies define rules for model deployment, performance thresholds, etc.
        """
        policies_path = self.config.get("MODEL_GOVERNANCE_POLICIES_PATH", "configs/governance_policies.json")
        if not os.path.exists(policies_path):
            logger.warning(f"Model governance policies file not found at {policies_path}. Using default empty policies.")
            return {}
        try:
            with open(policies_path, 'r') as f:
                policies = json.load(f)
            logger.info(f"Loaded governance policies from {policies_path}.")
            return policies
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding governance policies JSON from {policies_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load governance policies from {policies_path}: {e}")
            return {}

    def enforce_deployment_policy(self, model_name: str, version: str, performance_metrics: Dict[str, float]) -> bool:
        """
        Enforces deployment policies based on predefined rules.
        
        Args:
            model_name: The name of the model.
            version: The version string of the model.
            performance_metrics: Dictionary of performance metrics (e.g., accuracy, f1_score).
            
        Returns:
            True if the model meets deployment criteria, False otherwise.
        """
        policy = self.governance_policies.get("deployment_policy", {})
        required_metrics = policy.get("required_metrics", {})
        
        for metric, threshold in required_metrics.items():
            if metric not in performance_metrics:
                self.alert_manager.send_alert(
                    "WARNING", 
                    f"Deployment check for {model_name} version {version} failed: Missing required metric '{metric}'."
                )
                logger.warning(f"Deployment check failed for {model_name} {version}: Missing metric {metric}")
                return False
            
            if performance_metrics[metric] < threshold:
                self.alert_manager.send_alert(
                    "CRITICAL", 
                    f"Deployment check for {model_name} version {version} failed: {metric} ({performance_metrics[metric]:.4f}) below threshold ({threshold:.4f})."
                )
                logger.error(f"Deployment check failed for {model_name} {version}: {metric} {performance_metrics[metric]:.4f} < {threshold:.4f}")
                return False
        
        logger.info(f"Model {model_name} version {version} meets all deployment policy criteria.")
        return True

    def conduct_model_audit(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Conducts an audit for a model, retrieving its metadata and performance history.
        If no version is specified, audits the currently deployed version.
        
        Returns:
            A dictionary containing audit details.
        """
        audit_version = version
        if audit_version is None:
            audit_version = self.model_store.get_deployed_model_version(model_name)
            if audit_version is None:
                raise ModelGovernanceError(f"No version specified and no deployed version found for {model_name}.")

        try:
            metadata = self.model_store.load_model_metadata(model_name, audit_version)
            audit_report = {
                "model_name": model_name,
                "version": audit_version,
                "status": "Audited",
                "audit_timestamp": datetime.now().isoformat(),
                "metadata": metadata,
                "performance_history": "TODO: Link to a performance logging system/database",
                "deployed_status": (audit_version == self.model_store.get_deployed_model_version(model_name))
            }
            logger.info(f"Audit conducted for {model_name} version {audit_version}.")
            return audit_report
        except ModelStoreError as e:
            logger.error(f"Audit failed for {model_name} version {audit_version}: {e}")
            raise ModelGovernanceError(f"Audit failed: {e}")
        except Exception as e:
            logger.critical(f"An unexpected error occurred during audit for {model_name} version {audit_version}: {e}", exc_info=True)
            raise ModelGovernanceError(f"Unexpected error during audit: {e}")

    def review_model_performance(self, model_name: str, version: str) -> bool:
        """
        Reviews a model's live performance against predefined thresholds.
        This would typically pull from monitoring metrics, not static metadata.
        """
        # Placeholder: In a real system, this would query Prometheus or a dedicated monitoring database
        # For simplicity, we'll check if metrics exist in metadata, but real-time is preferred.
        try:
            metadata = self.model_store.load_model_metadata(model_name, version)
            latest_live_metrics = metadata.get("live_performance_metrics", {}) # This should come from a live system
            
            monitoring_policy = self.governance_policies.get("monitoring_policy", {})
            performance_thresholds = monitoring_policy.get("performance_thresholds", {})

            for metric, threshold in performance_thresholds.items():
                if metric not in latest_live_metrics:
                    logger.warning(f"Live performance review for {model_name} version {version}: Missing live metric '{metric}'.")
                    continue # Cannot evaluate if metric is missing
                
                if latest_live_metrics[metric] < threshold:
                    self.alert_manager.send_alert(
                        "CRITICAL", 
                        f"Live performance degradation for {model_name} version {version}: {metric} ({latest_live_metrics[metric]:.4f}) fell below threshold ({threshold:.4f})."
                    )
                    logger.warning(f"Live performance for {model_name} {version}: {metric} {latest_live_metrics[metric]:.4f} below threshold {threshold:.4f}")
                    return False
            
            logger.info(f"Model {model_name} version {version} live performance is within acceptable bounds.")
            return True

        except ModelStoreError as e:
            logger.error(f"Could not review live performance for {model_name} version {version}: {e}")
            return False
        except Exception as e:
            logger.critical(f"An unexpected error occurred during live performance review for {model_name} version {version}: {e}", exc_info=True)
            return False

    def get_model_lineage(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Retrieves the lineage information for a specific model version,
        including data sources, preprocessing steps, training parameters, etc.
        """
        try:
            metadata = self.model_store.load_model_metadata(model_name, version)
            lineage_info = {
                "training_data_sources": metadata.get("training_data_sources", "N/A"),
                "preprocessing_steps": metadata.get("preprocessing_steps", "N/A"),
                "training_parameters": metadata.get("training_parameters", "N/A"),
                "code_version": metadata.get("code_version", "N/A"), # Git commit hash or similar
                "trained_by": metadata.get("trained_by", "N/A"),
                "training_timestamp": metadata.get("training_timestamp", "N/A")
            }
            logger.info(f"Retrieved lineage for {model_name} version {version}.")
            return lineage_info
        except ModelStoreError as e:
            logger.error(f"Failed to retrieve model lineage for {model_name} version {version}: {e}")
            raise ModelGovernanceError(f"Failed to get model lineage: {e}")
        except Exception as e:
            logger.critical(f"An unexpected error occurred during lineage retrieval for {model_name} version {version}: {e}", exc_info=True)
            raise ModelGovernanceError(f"Unexpected error during lineage retrieval: {e}")

