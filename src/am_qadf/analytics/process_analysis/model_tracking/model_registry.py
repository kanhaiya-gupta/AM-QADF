"""
Model Registry for PBF-LB/M Systems

This module provides model versioning and registry capabilities for tracking
trained prediction models.
"""

import os
import pickle
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version metadata."""

    model_id: str
    model_type: str
    version: str
    training_date: datetime
    performance_metrics: Dict[str, float]
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    storage_path: Optional[str] = None
    file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (handles datetime serialization)."""
        data = asdict(self)
        # Convert datetime to ISO format string
        if isinstance(data.get("training_date"), datetime):
            data["training_date"] = data["training_date"].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary (handles datetime deserialization)."""
        if isinstance(data.get("training_date"), str):
            data["training_date"] = datetime.fromisoformat(data["training_date"])
        return cls(**data)


class ModelRegistry:
    """
    Model versioning and registry.

    This class provides capabilities to register, store, load, and manage
    trained prediction models with versioning and metadata tracking.
    """

    def __init__(self, storage_path: str = "models/"):
        """
        Initialize model registry.

        Args:
            storage_path: Path to store models and metadata
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Create metadata file
        self.metadata_file = self.storage_path / "model_metadata.json"
        self._models: Dict[str, ModelVersion] = {}
        self._load_metadata()

        logger.info(f"Model Registry initialized at {self.storage_path}")

    def _load_metadata(self) -> None:
        """Load model metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                    for model_id, model_data in data.items():
                        self._models[model_id] = ModelVersion.from_dict(model_data)
                logger.info(f"Loaded {len(self._models)} models from registry")
            except Exception as e:
                logger.warning(f"Error loading model metadata: {e}")
                self._models = {}
        else:
            self._models = {}

    def _save_metadata(self) -> None:
        """Save model metadata to file."""
        try:
            data = {model_id: model_version.to_dict() for model_id, model_version in self._models.items()}
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self._models)} models to registry")
        except Exception as e:
            logger.error(f"Error saving model metadata: {e}")

    def _generate_model_id(self, model_type: str, version: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{model_type}_{version}_{timestamp}"
        hash_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{model_type}_{version}_{hash_id}"

    def register_model(
        self,
        model: Any,  # Trained model object
        model_type: str,
        version: str,
        metadata: Dict[str, Any],
        performance_metrics: Dict[str, float],
        validation_metrics: Optional[Dict[str, float]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Register trained model in registry.

        Args:
            model: Trained model object (must be pickle-able)
            model_type: Type of model (e.g., 'random_forest', 'early_defect_predictor')
            version: Model version string (e.g., '1.0.0')
            metadata: Model metadata (e.g., feature names, config, etc.)
            performance_metrics: Performance metrics (R², RMSE, accuracy, etc.)
            validation_metrics: Validation metrics (cross-validation, etc.) - optional
            feature_importance: Feature importance scores - optional

        Returns:
            Model ID (unique identifier)
        """
        try:
            # Generate unique model ID
            model_id = self._generate_model_id(model_type, version)

            # Create storage directory for this model
            model_dir = self.storage_path / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save model to file
            model_file = model_dir / "model.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(model, f)

            # Create model version metadata
            model_version = ModelVersion(
                model_id=model_id,
                model_type=model_type,
                version=version,
                training_date=datetime.now(),
                performance_metrics=performance_metrics,
                validation_metrics=validation_metrics or {},
                feature_importance=feature_importance or {},
                metadata=metadata,
                storage_path=str(model_dir),
                file_path=str(model_file),
            )

            # Register in memory and save to disk
            self._models[model_id] = model_version
            self._save_metadata()

            logger.info(f"Registered model {model_id} (type: {model_type}, version: {version})")
            return model_id

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def load_model(self, model_id: str) -> Tuple[Any, ModelVersion]:
        """
        Load model and metadata from registry.

        Args:
            model_id: Model ID to load

        Returns:
            Tuple of (model_object, ModelVersion metadata)
        """
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found in registry")

        model_version = self._models[model_id]

        if model_version.file_path is None or not os.path.exists(model_version.file_path):
            raise FileNotFoundError(f"Model file not found: {model_version.file_path}")

        try:
            with open(model_version.file_path, "rb") as f:
                model = pickle.load(f)

            logger.info(f"Loaded model {model_id} from registry")
            return model, model_version

        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise

    def list_models(self, model_type: Optional[str] = None, version: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List models in registry with optional filters.

        Args:
            model_type: Filter by model type (optional)
            version: Filter by version (optional)

        Returns:
            List of model metadata dictionaries
        """
        filtered_models = []

        for model_id, model_version in self._models.items():
            # Apply filters
            if model_type is not None and model_version.model_type != model_type:
                continue
            if version is not None and model_version.version != version:
                continue

            filtered_models.append(model_version.to_dict())

        return filtered_models

    def compare_models(self, model_id1: str, model_id2: str) -> Dict[str, Any]:
        """
        Compare two model versions.

        Args:
            model_id1: First model ID
            model_id2: Second model ID

        Returns:
            Dictionary with comparison metrics
        """
        if model_id1 not in self._models:
            raise ValueError(f"Model {model_id1} not found in registry")
        if model_id2 not in self._models:
            raise ValueError(f"Model {model_id2} not found in registry")

        model1 = self._models[model_id1]
        model2 = self._models[model_id2]

        comparison = {
            "model1_id": model_id1,
            "model2_id": model_id2,
            "model1_type": model1.model_type,
            "model2_type": model2.model_type,
            "model1_version": model1.version,
            "model2_version": model2.version,
            "performance_comparison": {},
            "training_date_diff": (model1.training_date - model2.training_date).total_seconds(),
        }

        # Compare performance metrics
        common_metrics = set(model1.performance_metrics.keys()) & set(model2.performance_metrics.keys())
        for metric in common_metrics:
            val1 = model1.performance_metrics[metric]
            val2 = model2.performance_metrics[metric]
            comparison["performance_comparison"][metric] = {
                "model1": val1,
                "model2": val2,
                "difference": val1 - val2,
                "improvement": val1 > val2 if metric not in ["rmse", "mae", "mse"] else val1 < val2,
            }

        return comparison

    def delete_model(self, model_id: str) -> bool:
        """
        Delete model from registry.

        Args:
            model_id: Model ID to delete

        Returns:
            True if successful, False otherwise
        """
        if model_id not in self._models:
            logger.warning(f"Model {model_id} not found in registry")
            return False

        try:
            model_version = self._models[model_id]

            # Delete model file and directory
            if model_version.storage_path and os.path.exists(model_version.storage_path):
                import shutil

                shutil.rmtree(model_version.storage_path)

            # Remove from registry
            del self._models[model_id]
            self._save_metadata()

            logger.info(f"Deleted model {model_id} from registry")
            return True

        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False

    def get_latest_model(self, model_type: Optional[str] = None) -> Optional[str]:
        """
        Get latest model ID for given model type.

        Args:
            model_type: Model type (optional, returns latest of any type if None)

        Returns:
            Latest model ID or None if no models found
        """
        filtered_models = [
            (model_id, model_version)
            for model_id, model_version in self._models.items()
            if model_type is None or model_version.model_type == model_type
        ]

        if not filtered_models:
            return None

        # Sort by training date (most recent first)
        filtered_models.sort(key=lambda x: x[1].training_date, reverse=True)
        return filtered_models[0][0]
