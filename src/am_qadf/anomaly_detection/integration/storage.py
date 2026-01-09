"""
Anomaly Detection Results Storage

Store anomaly detection results in the warehouse.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Anomaly detection result data structure."""

    model_id: str
    detection_id: str
    methods: List[str]
    anomalies: Dict[str, List[Dict[str, Any]]]  # Method -> list of anomalies
    anomaly_patterns: Optional[Dict[str, Any]] = None
    detection_metrics: Optional[Dict[str, float]] = None
    root_cause_analysis: Optional[Dict[str, Any]] = None
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
        # Recursively convert numpy arrays to lists
        result = self._convert_numpy_to_list(result)
        return result

    def _convert_numpy_to_list(self, obj: Any) -> Any:
        """Recursively convert numpy arrays to lists in nested structures."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_to_list(item) for item in obj]
        else:
            return obj


class AnomalyStorage:
    """
    Storage client for anomaly detection results.
    """

    def __init__(self, mongo_client):
        """
        Initialize anomaly storage.

        Args:
            mongo_client: MongoDB client
        """
        self.mongo_client = mongo_client
        self.collection_name = "anomaly_detections"
        logger.info("AnomalyStorage initialized")

    def store_anomaly_result(self, result: AnomalyResult) -> str:
        """
        Store anomaly detection result.

        Args:
            result: AnomalyResult object

        Returns:
            Document ID
        """
        if not self.mongo_client or not self.mongo_client.connected:
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_name)

        document = result.to_dict()
        document["_id"] = result.detection_id

        # Insert or update
        collection.replace_one({"_id": result.detection_id}, document, upsert=True)

        logger.info(f"Stored anomaly result: {result.detection_id}")
        return result.detection_id

    def store_anomaly_patterns(self, model_id: str, pattern_id: str, patterns: Dict[str, Any]) -> str:
        """
        Store anomaly patterns and clusters.

        Args:
            model_id: Model ID
            pattern_id: Pattern ID
            patterns: Pattern data dictionary

        Returns:
            Document ID
        """
        if not self.mongo_client or not self.mongo_client.connected:
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection("anomaly_patterns")

        document = {
            "_id": pattern_id,
            "model_id": model_id,
            "patterns": patterns,
            "timestamp": datetime.now().isoformat(),
        }

        # Insert or update
        collection.replace_one({"_id": pattern_id}, document, upsert=True)

        logger.info(f"Stored anomaly patterns: {pattern_id}")
        return pattern_id

    def store_detection_metrics(self, detection_id: str, metrics: Dict[str, float]) -> str:
        """
        Store detection performance metrics.

        Args:
            detection_id: Detection ID
            metrics: Metrics dictionary

        Returns:
            Document ID
        """
        if not self.mongo_client or not self.mongo_client.connected:
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection("detection_metrics")

        document = {
            "_id": detection_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        # Insert or update
        collection.replace_one({"_id": detection_id}, document, upsert=True)

        logger.info(f"Stored detection metrics: {detection_id}")
        return detection_id
