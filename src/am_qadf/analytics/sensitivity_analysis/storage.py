"""
Sensitivity Analysis Results Storage

Store sensitivity analysis results in the warehouse.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Sensitivity analysis result data structure."""

    model_id: str
    analysis_id: str
    method: str
    parameter_names: List[str]
    sensitivity_indices: Dict[str, float]
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    sample_size: int = 0
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


class SensitivityStorage:
    """
    Storage client for sensitivity analysis results.
    """

    def __init__(self, mongo_client):
        """
        Initialize sensitivity storage.

        Args:
            mongo_client: MongoDB client
        """
        self.mongo_client = mongo_client
        self.collection_name = "sensitivity_results"
        logger.info("SensitivityStorage initialized")

    def store_sensitivity_result(self, result: SensitivityResult) -> str:
        """
        Store sensitivity analysis result.

        Args:
            result: SensitivityResult object

        Returns:
            Document ID
        """
        if not self.mongo_client or not self.mongo_client.connected:
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_name)

        document = result.to_dict()
        document["_id"] = result.analysis_id

        # Insert or update
        collection.replace_one({"_id": result.analysis_id}, document, upsert=True)

        logger.info(f"Stored sensitivity result: {result.analysis_id}")
        return result.analysis_id

    def store_doe_design(self, model_id: str, design_id: str, design_data: Dict[str, Any]) -> str:
        """
        Store Design of Experiments (DoE) design.

        Args:
            model_id: Model ID
            design_id: Design ID
            design_data: Design data dictionary

        Returns:
            Document ID
        """
        if not self.mongo_client or not self.mongo_client.connected:
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection("doe_designs")

        document = {
            "_id": design_id,
            "model_id": model_id,
            "design_data": design_data,
            "timestamp": datetime.now().isoformat(),
        }

        # Insert or update
        collection.replace_one({"_id": design_id}, document, upsert=True)

        logger.info(f"Stored DoE design: {design_id}")
        return design_id

    def store_influence_rankings(self, model_id: str, ranking_id: str, rankings: Dict[str, float]) -> str:
        """
        Store process variable influence rankings.

        Args:
            model_id: Model ID
            ranking_id: Ranking ID
            rankings: Dictionary mapping variable names to influence scores

        Returns:
            Document ID
        """
        if not self.mongo_client or not self.mongo_client.connected:
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection("influence_rankings")

        document = {
            "_id": ranking_id,
            "model_id": model_id,
            "rankings": rankings,
            "timestamp": datetime.now().isoformat(),
        }

        # Insert or update
        collection.replace_one({"_id": ranking_id}, document, upsert=True)

        logger.info(f"Stored influence rankings: {ranking_id}")
        return ranking_id
