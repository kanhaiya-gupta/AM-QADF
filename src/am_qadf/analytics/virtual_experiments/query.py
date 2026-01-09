"""
Virtual Experiment Results Query

Query virtual experiment results from the warehouse.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentQuery:
    """
    Query client for virtual experiment results.
    """

    def __init__(self, mongo_client):
        """
        Initialize experiment query client.

        Args:
            mongo_client: MongoDB client
        """
        self.mongo_client = mongo_client
        self.collection_name = "virtual_experiments"
        logger.info("ExperimentQuery initialized")

    def query_experiment_results(
        self,
        experiment_id: Optional[str] = None,
        model_id: Optional[str] = None,
        design_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query virtual experiment results.

        Args:
            experiment_id: Experiment ID (optional)
            model_id: Model ID (optional)
            design_type: Design type (optional)

        Returns:
            List of experiment result documents
        """
        if not self.mongo_client or not self.mongo_client.connected:
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_name)

        # Build query
        query = {}
        if experiment_id:
            query["_id"] = experiment_id
        if model_id:
            query["model_id"] = model_id
        if design_type:
            query["design_data.design_type"] = design_type

        results = list(collection.find(query))

        logger.info(f"Found {len(results)} experiment results")
        return results

    def compare_experiments_with_warehouse(self, experiment_id: str, model_id: str) -> Dict[str, Any]:
        """
        Compare experiment results with warehouse data.

        Args:
            experiment_id: Experiment ID
            model_id: Model ID to compare with

        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing experiment {experiment_id} with warehouse for model {model_id}")

        # Query experiment results
        experiment_results = self.query_experiment_results(experiment_id=experiment_id)

        if not experiment_results:
            return {}

        experiment = experiment_results[0]

        # Query comparison results if stored
        comparison_collection = self.mongo_client.get_collection("experiment_comparisons")
        comparison_id = f"{experiment_id}_{model_id}"
        comparison = comparison_collection.find_one({"_id": comparison_id})

        if comparison:
            return comparison.get("comparison_data", {})

        # Return experiment data for manual comparison
        return {
            "experiment_id": experiment_id,
            "model_id": model_id,
            "experiment_data": experiment.get("results", {}),
            "note": "Comparison results not found. Run comparison first.",
        }

    def analyze_experiment_trends(
        self,
        model_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze experiment trends.

        Args:
            model_id: Optional model ID
            time_range: Optional time range

        Returns:
            Dictionary with trend analysis
        """
        logger.info(f"Analyzing experiment trends for model {model_id}")

        results = self.query_experiment_results(model_id=model_id)

        if not results:
            return {}

        # Filter by time range if provided
        if time_range:
            start_time, end_time = time_range
            results = [r for r in results if start_time <= datetime.fromisoformat(r["timestamp"]) <= end_time]

        # Extract trends
        trends = {
            "num_experiments": len(results),
            "experiment_types": {},
            "design_types": {},
            "parameter_ranges": {},
        }

        for result in results:
            design_data = result.get("design_data", {})
            design_type = design_data.get("design_type", "unknown")

            if design_type not in trends["design_types"]:
                trends["design_types"][design_type] = 0
            trends["design_types"][design_type] += 1

            # Extract parameter ranges
            param_ranges = design_data.get("parameter_ranges", {})
            for param, (min_val, max_val) in param_ranges.items():
                if param not in trends["parameter_ranges"]:
                    trends["parameter_ranges"][param] = {"min": [], "max": []}
                trends["parameter_ranges"][param]["min"].append(min_val)
                trends["parameter_ranges"][param]["max"].append(max_val)

        # Calculate statistics
        for param, ranges in trends["parameter_ranges"].items():
            if ranges["min"] and ranges["max"]:
                trends["parameter_ranges"][param] = {
                    "min_range": (min(ranges["min"]), max(ranges["min"])),
                    "max_range": (min(ranges["max"]), max(ranges["max"])),
                    "avg_min": sum(ranges["min"]) / len(ranges["min"]),
                    "avg_max": sum(ranges["max"]) / len(ranges["max"]),
                }

        return trends
