"""
Anomaly Detection Results Query

Query anomaly detection results from the warehouse.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class AnomalyQuery:
    """
    Query client for anomaly detection results.
    """

    def __init__(self, mongo_client):
        """
        Initialize anomaly query client.

        Args:
            mongo_client: MongoDB client
        """
        self.mongo_client = mongo_client
        self.collection_name = "anomaly_detections"
        logger.info("AnomalyQuery initialized")

    def query_anomalies(
        self,
        model_id: Optional[str] = None,
        anomaly_type: Optional[str] = None,
        severity: Optional[str] = None,
        method: Optional[str] = None,
        detection_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query anomaly detection results.

        Args:
            model_id: Model ID (optional)
            anomaly_type: Anomaly type (optional)
            severity: Severity level (optional)
            method: Detection method (optional)
            detection_id: Detection ID (optional)

        Returns:
            List of anomaly result documents
        """
        if not self.mongo_client or not self.mongo_client.connected:
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_name)

        # Build query
        query: Dict[str, Any] = {}
        if model_id:
            query["model_id"] = model_id
        if method:
            query["methods"] = {"$in": [method]}
        if detection_id:
            query["_id"] = detection_id

        results = list(collection.find(query))

        # Filter by anomaly type and severity if provided
        if anomaly_type or severity:
            filtered_results = []
            for result in results:
                anomalies = result.get("anomalies", {})
                for method_name, anomaly_list in anomalies.items():
                    for anomaly in anomaly_list:
                        if anomaly_type and anomaly.get("anomaly_type") != anomaly_type:
                            continue
                        if severity:
                            score = anomaly.get("anomaly_score", 0)
                            if severity == "high" and score < 0.8:
                                continue
                            elif severity == "medium" and (score < 0.5 or score >= 0.8):
                                continue
                            elif severity == "low" and score >= 0.5:
                                continue
                        filtered_results.append({**result, "anomaly": anomaly, "method": method_name})
            return filtered_results

        logger.info(f"Found {len(results)} anomaly detection results")
        return results

    def query_anomaly_results(
        self,
        model_id: Optional[str] = None,
        method: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query anomaly detection results in a user-friendly format.

        Args:
            model_id: Model ID (optional)
            method: Detection method (optional)
            limit: Maximum number of results to return (optional)

        Returns:
            List of formatted anomaly detection result dictionaries
        """
        # Query using the existing method
        raw_results = self.query_anomalies(model_id=model_id, method=method)

        # Format results for easier use
        formatted_results = []
        for result in raw_results:
            # Extract detection ID
            detection_id = result.get("_id", result.get("detection_id", "unknown"))
            if isinstance(detection_id, dict) and "$oid" in detection_id:
                detection_id = detection_id["$oid"]

            # Get methods used
            methods = result.get("methods", [])
            method_name = methods[0] if methods else "unknown"

            # Extract anomaly counts and rates from results
            results_dict = result.get("results", {})
            num_anomalies = 0
            anomaly_rate = 0.0
            total_points = result.get("total_points", 0)

            # Try to get anomaly info from results
            if method_name in results_dict:
                method_result = results_dict[method_name]
                if "error" not in method_result:
                    num_anomalies = method_result.get("num_anomalies", 0)
                    if total_points > 0:
                        anomaly_rate = num_anomalies / total_points
                    else:
                        # Try to calculate from anomaly_labels if available
                        anomaly_labels = result.get("anomaly_labels", [])
                        if anomaly_labels:
                            num_anomalies = sum(anomaly_labels)
                            anomaly_rate = num_anomalies / len(anomaly_labels) if len(anomaly_labels) > 0 else 0.0

            # Get timestamp
            timestamp = result.get("timestamp", result.get("created_at", "N/A"))
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()

            formatted_results.append(
                {
                    "detection_id": str(detection_id),
                    "model_id": result.get("model_id", "unknown"),
                    "method": method_name,
                    "methods": methods,
                    "num_anomalies": num_anomalies,
                    "total_points": total_points,
                    "anomaly_rate": anomaly_rate,
                    "timestamp": timestamp,
                    "detection_data_shape": result.get("detection_data_shape"),
                    "raw_result": result,  # Include raw result for advanced use
                }
            )

        # Apply limit if specified
        if limit:
            formatted_results = formatted_results[:limit]

        logger.info(f"Formatted {len(formatted_results)} anomaly detection results")
        return formatted_results

    def analyze_anomaly_trends(
        self,
        model_ids: Optional[List[str]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze anomaly trends over time.

        Args:
            model_ids: Optional list of model IDs
            time_range: Optional time range

        Returns:
            Dictionary with trend analysis
        """
        logger.info(f"Analyzing anomaly trends for {len(model_ids) if model_ids else 'all'} models")

        # Build query
        query = {}
        if model_ids:
            query["model_id"] = {"$in": model_ids}

        collection = self.mongo_client.get_collection(self.collection_name)
        results = list(collection.find(query))

        # Filter by time range if provided
        if time_range:
            start_time, end_time = time_range
            results = [r for r in results if start_time <= datetime.fromisoformat(r["timestamp"]) <= end_time]

        # Analyze trends
        trends = {
            "total_detections": len(results),
            "models_analyzed": len(set(r["model_id"] for r in results)),
            "methods_used": set(),
            "anomaly_counts_by_method": {},
            "anomaly_counts_by_model": {},
            "average_scores_by_method": {},
        }

        for result in results:
            model_id = result["model_id"]
            methods = result.get("methods", [])

            # Track methods
            trends["methods_used"].update(methods)

            # Count anomalies by method
            anomalies = result.get("anomalies", {})
            for method, anomaly_list in anomalies.items():
                if method not in trends["anomaly_counts_by_method"]:
                    trends["anomaly_counts_by_method"][method] = 0
                    trends["average_scores_by_method"][method] = []

                num_anomalies = len([a for a in anomaly_list if a.get("is_anomaly", False)])
                trends["anomaly_counts_by_method"][method] += num_anomalies

                scores = [a.get("anomaly_score", 0) for a in anomaly_list if a.get("is_anomaly", False)]
                trends["average_scores_by_method"][method].extend(scores)

            # Count anomalies by model
            total_anomalies = sum(
                len([a for a in anomaly_list if a.get("is_anomaly", False)]) for anomaly_list in anomalies.values()
            )
            if model_id not in trends["anomaly_counts_by_model"]:
                trends["anomaly_counts_by_model"][model_id] = 0
            trends["anomaly_counts_by_model"][model_id] += total_anomalies

        # Convert sets to lists and calculate averages
        trends["methods_used"] = list(trends["methods_used"])
        for method, scores in trends["average_scores_by_method"].items():
            if scores:
                trends["average_scores_by_method"][method] = sum(scores) / len(scores)
            else:
                trends["average_scores_by_method"][method] = 0.0

        return trends

    def query_anomaly_patterns(
        self, model_id: Optional[str] = None, pattern_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query and analyze anomaly patterns.

        Args:
            model_id: Optional model ID
            pattern_type: Optional pattern type

        Returns:
            List of pattern documents
        """
        logger.info(f"Querying anomaly patterns for model {model_id}")

        collection = self.mongo_client.get_collection("anomaly_patterns")

        query = {}
        if model_id:
            query["model_id"] = model_id
        if pattern_type:
            query["patterns.pattern_type"] = pattern_type

        patterns = list(collection.find(query))

        logger.info(f"Found {len(patterns)} anomaly patterns")
        return patterns
