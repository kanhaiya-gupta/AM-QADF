"""
Sensitivity Analysis Results Query

Query sensitivity analysis results from the warehouse.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class SensitivityQuery:
    """
    Query client for sensitivity analysis results.
    """

    def __init__(self, mongo_client):
        """
        Initialize sensitivity query client.

        Args:
            mongo_client: MongoDB client
        """
        self.mongo_client = mongo_client
        self.collection_name = "sensitivity_results"
        logger.info("SensitivityQuery initialized")

    def query_sensitivity_results(
        self,
        model_id: Optional[str] = None,
        method: Optional[str] = None,
        variable: Optional[str] = None,
        analysis_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query sensitivity analysis results.

        Args:
            model_id: Model ID (optional)
        method: Analysis method (optional)
            variable: Variable name (optional)
            analysis_id: Analysis ID (optional)

        Returns:
            List of sensitivity result documents
        """
        if not self.mongo_client or not self.mongo_client.connected:
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_name)

        # Build query
        query = {}
        if model_id:
            query["model_id"] = model_id
        if method:
            query["method"] = method
        if variable:
            query["parameter_names"] = {"$in": [variable]}
        if analysis_id:
            query["_id"] = analysis_id

        results = list(collection.find(query))

        logger.info(f"Found {len(results)} sensitivity results")
        return results

    def compare_sensitivity(self, model_ids: List[str], method: str = "sobol") -> pd.DataFrame:
        """
        Compare sensitivity across multiple models.

        Args:
            model_ids: List of model IDs
            method: Analysis method

        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing sensitivity for {len(model_ids)} models")

        all_results = []
        for model_id in model_ids:
            results = self.query_sensitivity_results(model_id=model_id, method=method)
            # Take only the first/latest result per model to avoid duplicates
            if results:
                # Sort by timestamp (most recent first) and take first
                results_sorted = sorted(results, key=lambda x: x.get("timestamp", ""), reverse=True)
                result = results_sorted[0]
                all_results.append(
                    {
                        "model_id": model_id,
                        "analysis_id": result.get("_id"),
                        "method": result.get("method"),
                        "sensitivity_indices": result.get("sensitivity_indices", {}),
                        "timestamp": result.get("timestamp"),
                    }
                )

        if not all_results:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        # Expand sensitivity indices
        if "sensitivity_indices" in df.columns:
            indices_df = pd.json_normalize(df["sensitivity_indices"])
            df = pd.concat([df.drop("sensitivity_indices", axis=1), indices_df], axis=1)

        return df

    def analyze_sensitivity_trends(
        self, model_id: str, time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity trends over time.

        Args:
            model_id: Model ID
            time_range: Optional time range

        Returns:
            Dictionary with trend analysis
        """
        logger.info(f"Analyzing sensitivity trends for model {model_id}")

        results = self.query_sensitivity_results(model_id=model_id)

        if not results:
            return {}

        # Filter by time range if provided
        if time_range:
            start_time, end_time = time_range
            results = [r for r in results if start_time <= datetime.fromisoformat(r["timestamp"]) <= end_time]

        # Extract trends
        trends: Dict[str, Dict[str, List[float]]] = {}
        for result in results:
            method = result.get("method")
            indices = result.get("sensitivity_indices", {})

            if method not in trends:
                trends[method] = {}

            for var, value in indices.items():
                if var not in trends[method]:
                    trends[method][var] = []
                trends[method][var].append(value)

        # Calculate statistics
        trend_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        for method, var_trends in trends.items():
            trend_stats[method] = {}
            for var, values in var_trends.items():
                if values:
                    trend_stats[method][var] = {
                        "mean": sum(values) / len(values),
                        "std": (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5,
                        "min": min(values),
                        "max": max(values),
                        "count": len(values),
                    }

        return trend_stats
