"""
Virtual Experiment Results Storage

Store virtual experiment results in the warehouse.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Virtual experiment result data structure."""

    experiment_id: str
    model_id: str
    design_data: Dict[str, Any]
    results: Dict[str, Any]
    comparison_results: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = asdict(self)
        # Convert datetime to ISO string
        if isinstance(result["timestamp"], datetime):
            result["timestamp"] = result["timestamp"].isoformat()
        return result


class ExperimentStorage:
    """
    Storage client for virtual experiment results.
    """

    def __init__(self, mongo_client):
        """
        Initialize experiment storage.

        Args:
            mongo_client: MongoDB client
        """
        self.mongo_client = mongo_client
        self.collection_name = "virtual_experiments"
        logger.info("ExperimentStorage initialized")

    def store_experiment_result(self, result: ExperimentResult) -> str:
        """
        Store virtual experiment result.

        Args:
            result: ExperimentResult object

        Returns:
            Document ID
        """
        if not self.mongo_client or not self.mongo_client.connected:
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_name)

        document = result.to_dict()
        document["_id"] = result.experiment_id

        # Insert or update
        collection.replace_one({"_id": result.experiment_id}, document, upsert=True)

        logger.info(f"Stored experiment result: {result.experiment_id}")
        return result.experiment_id

    def store_experiment_design(self, experiment_id: str, design_data: Dict[str, Any]) -> str:
        """
        Store experiment design.

        Args:
            experiment_id: Experiment ID
            design_data: Design data dictionary

        Returns:
            Document ID
        """
        if not self.mongo_client or not self.mongo_client.connected:
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection("experiment_designs")

        document = {
            "_id": experiment_id,
            "design_data": design_data,
            "timestamp": datetime.now().isoformat(),
        }

        # Insert or update
        collection.replace_one({"_id": experiment_id}, document, upsert=True)

        logger.info(f"Stored experiment design: {experiment_id}")
        return experiment_id

    def store_comparison_results(self, experiment_id: str, model_id: str, comparison_data: Dict[str, Any]) -> str:
        """
        Store comparison results with warehouse data.

        Args:
            experiment_id: Experiment ID
            model_id: Model ID
            comparison_data: Comparison data dictionary

        Returns:
            Document ID
        """
        if not self.mongo_client or not self.mongo_client.connected:
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection("experiment_comparisons")

        comparison_id = f"{experiment_id}_{model_id}"
        document = {
            "_id": comparison_id,
            "experiment_id": experiment_id,
            "model_id": model_id,
            "comparison_data": comparison_data,
            "timestamp": datetime.now().isoformat(),
        }

        # Insert or update
        collection.replace_one({"_id": comparison_id}, document, upsert=True)

        logger.info(f"Stored comparison results: {comparison_id}")
        return comparison_id
