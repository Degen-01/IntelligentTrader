"""
Model Store and Governance for AI/ML trading models.
Manages model versions, metadata, performance metrics, and deployment status.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class ModelStoreError(Exception):
    """Custom exception for ModelStore operations."""
    pass

class ModelStore:
    def __init__(self, base_path: str = "models/model_store"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        logger.info(f"Initialized ModelStore at {self.base_path}")

    def _get_model_dir(self, model_name: str, version: str) -> str:
        """Returns the directory path for a specific model version."""
        return os.path.join(self.base_path, model_name, version)

    def _get_metadata_path(self, model_name: str, version: str) -> str:
        """Returns the path to the metadata file for a specific model version."""
        return os.path.join(self._get_model_dir(model_name, version), "metadata.json")

    def save_model_metadata(self,
                            model_name: str,
                            version: str,
                            metadata: Dict[str, Any],
                            overwrite: bool = False) -> None:
        """
        Saves metadata for a specific model version.
        
        Args:
            model_name: The name of the model (e.g., 'XGBoostClassifier', 'LSTMRegressor').
            version: The version string for the model (e.g., 'v1.0.0', '20231026-1530').
            metadata: A dictionary containing model metadata (e.g., training date, metrics, parameters).
            overwrite: If True, overwrite existing metadata. If False, raise error if exists.
        """
        model_version_dir = self._get_model_dir(model_name, version)
        os.makedirs(model_version_dir, exist_ok=True)
        metadata_path = self._get_metadata_path(model_name, version)

        if os.path.exists(metadata_path) and not overwrite:
            raise ModelStoreError(f"Metadata for {model_name} version {version} already exists. Set overwrite=True to force.")

        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Saved metadata for {model_name} version {version} to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata for {model_name} version {version}: {e}")
            raise ModelStoreError(f"Failed to save metadata: {e}")

    def load_model_metadata(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Loads metadata for a specific model version.
        
        Args:
            model_name: The name of the model.
            version: The version string for the model.
        
        Returns:
            A dictionary containing the model metadata.
        
        Raises:
            ModelStoreError: If metadata file does not exist or cannot be loaded.
        """
        metadata_path = self._get_metadata_path(model_name, version)
        if not os.path.exists(metadata_path):
            raise ModelStoreError(f"Metadata file not found for {model_name} version {version} at {metadata_path}")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.debug(f"Loaded metadata for {model_name} version {version} from {metadata_path}")
            return metadata
        except json.JSONDecodeError as e:
            raise ModelStoreError(f"Error decoding metadata JSON for {model_name} version {version}: {e}")
        except Exception as e:
            raise ModelStoreError(f"Failed to load metadata for {model_name} version {version}: {e}")

    def get_all_model_versions(self, model_name: str) -> List[str]:
        """
        Retrieves all available versions for a given model name.
        
        Args:
            model_name: The name of the model.
        
        Returns:
            A list of version strings, sorted in descending order (newest first).
        """
        model_path = os.path.join(self.base_path, model_name)
        if not os.path.exists(model_path):
            return []
        
        versions = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        # Attempt to sort versions intelligently (e.g., by date if YYYYMMDD format)
        try:
            versions.sort(key=lambda x: datetime.strptime(x.split('-')[0], '%Y%m%d') if '-' in x else x, reverse=True)
        except ValueError:
            versions.sort(reverse=True) # Fallback to alphabetical sort
            
        logger.debug(f"Found versions for {model_name}: {versions}")
        return versions

    def get_latest_model_version(self, model_name: str) -> Optional[str]:
        """
        Retrieves the latest available version for a given model name.
        
        Args:
            model_name: The name of the model.
        
        Returns:
            The latest version string or None if no versions exist.
        """
        versions = self.get_all_model_versions(model_name)
        return versions[0] if versions else None

    def get_deployed_model_version(self, model_name: str) -> Optional[str]:
        """
        Retrieves the currently deployed version of a model.
        Deployment status is indicated by a 'deployed.json' file in the model's root directory.
        """
        deployment_status_path = os.path.join(self.base_path, model_name, "deployed.json")
        if not os.path.exists(deployment_status_path):
            logger.warning(f"No deployed version found for {model_name}. 'deployed.json' does not exist.")
            return None
        try:
            with open(deployment_status_path, 'r') as f:
                data = json.load(f)
            deployed_version = data.get("version")
            if deployed_version:
                logger.info(f"Model {model_name} currently deployed version: {deployed_version}")
            return deployed_version
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding deployed.json for {model_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load deployed version for {model_name}: {e}")
            return None

    def set_deployed_model_version(self, model_name: str, version: str) -> None:
        """
        Sets a specific model version as currently deployed.
        Creates a 'deployed.json' file in the model's root directory.
        """
        model_root_dir = os.path.join(self.base_path, model_name)
        os.makedirs(model_root_dir, exist_ok=True)
        deployment_status_path = os.path.join(model_root_dir, "deployed.json")
        
        if not self.is_model_version_available(model_name, version):
            raise ModelStoreError(f"Cannot deploy {model_name} version {version}: version does not exist in store.")

        try:
            with open(deployment_status_path, 'w') as f:
                json.dump({"version": version, "deployed_at": datetime.now().isoformat()}, f, indent=4)
            logger.info(f"Set {model_name} version {version} as deployed.")
        except Exception as e:
            logger.error(f"Failed to set deployed version for {model_name} to {version}: {e}")
            raise ModelStoreError(f"Failed to set deployed version: {e}")

    def is_model_version_available(self, model_name: str, version: str) -> bool:
        """Checks if a specific model version exists in the store."""
        return os.path.exists(self._get_model_dir(model_name, version))

    def delete_model_version(self, model_name: str, version: str) -> None:
        """
        Deletes a specific model version and its metadata from the store.
        Prevents deletion of currently deployed version.
        """
        deployed_version = self.get_deployed_model_version(model_name)
        if deployed_version == version:
            raise ModelStoreError(f"Cannot delete deployed version {version} for {model_name}. Undeploy first.")
        
        model_version_dir = self._get_model_dir(model_name, version)
        if not os.path.exists(model_version_dir):
            logger.warning(f"Attempted to delete non-existent model version: {model_name} {version}")
            return

        import shutil
        try:
            shutil.rmtree(model_version_dir)
            logger.info(f"Deleted model {model_name} version {version} from {model_version_dir}")
        except Exception as e:
            logger.error(f"Failed to delete model {model_name} version {version}: {e}")
            raise ModelStoreError(f"Failed to delete model version: {e}")

    def cleanup_old_versions(self, model_name: str, keep_latest_n: int = 5) -> None:
        """
        Deletes old model versions for a given model, keeping only the latest N versions.
        Does not delete the currently deployed version even if it's outside the `keep_latest_n`.
        """
        all_versions = self.get_all_model_versions(model_name)
        deployed_version = self.get_deployed_model_version(model_name)
        
        if len(all_versions) <= keep_latest_n:
            logger.info(f"No old versions to clean up for {model_name}. Keeping all {len(all_versions)} versions.")
            return

        versions_to_delete = all_versions[keep_latest_n:]
        
        for version in versions_to_delete:
            if version == deployed_version:
                logger.info(f"Skipping deletion of deployed version {version} for {model_name}.")
                continue
            try:
                self.delete_model_version(model_name, version)
            except ModelStoreError as e:
                logger.warning(f"Could not delete {model_name} version {version} during cleanup: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during cleanup of {model_name} version {version}: {e}")

