"""
SPC Data Storage

Store and retrieve SPC data, baselines, and results from MongoDB.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class SPCStorage:
    """
    Store and retrieve SPC data from MongoDB.

    Provides methods for:
    - Storing baseline statistics
    - Storing control chart results
    - Storing process capability results
    - Querying SPC history
    """

    def __init__(self, mongo_client: Any):
        """
        Initialize SPC storage.

        Args:
            mongo_client: MongoDB client instance
        """
        self.mongo_client = mongo_client
        self.collection_names = {
            "baselines": "spc_baselines",
            "control_charts": "spc_control_charts",
            "capability": "spc_capability",
            "multivariate": "spc_multivariate",
            "history": "spc_history",
        }

        # Check if MongoDB client is properly initialized and connected
        if not self.mongo_client:
            logger.warning("MongoDB client not provided - storage features will be limited")
        elif hasattr(self.mongo_client, "is_connected"):
            # MongoDBClient uses is_connected() method
            try:
                if self.mongo_client.is_connected():
                    logger.info("SPCStorage initialized with connected MongoDB client")
                else:
                    logger.warning("MongoDB client not connected - storage features will be limited")
            except Exception as e:
                logger.warning(f"MongoDB connection check failed: {e} - storage features will be limited")
        elif hasattr(self.mongo_client, "connected"):
            # Alternative client with 'connected' attribute
            if self.mongo_client.connected:
                logger.info("SPCStorage initialized with connected MongoDB client")
            else:
                logger.warning("MongoDB client not connected - storage features will be limited")
        else:
            logger.warning("MongoDB client format not recognized - storage features may be limited")

    def save_baseline(
        self, model_id: str, signal_name: str, baseline: Any, metadata: Optional[Dict] = None  # BaselineStatistics type
    ) -> str:
        """
        Save baseline statistics.

        Args:
            model_id: Model ID
            signal_name: Signal name
            baseline: BaselineStatistics object
            metadata: Optional additional metadata

        Returns:
            Document ID
        """
        if not self._check_connection():
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_names["baselines"])

        # Convert baseline to dict
        baseline_dict = self._baseline_to_dict(baseline)

        # Create document
        baseline_id = f"{model_id}_{signal_name}_{baseline.calculated_at.isoformat()}"
        document = {
            "_id": baseline_id,
            "model_id": model_id,
            "signal_name": signal_name,
            "baseline": baseline_dict,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "calculated_at": (
                baseline.calculated_at.isoformat()
                if hasattr(baseline.calculated_at, "isoformat")
                else str(baseline.calculated_at)
            ),
        }

        # Insert or update
        collection.replace_one({"_id": baseline_id}, document, upsert=True)

        logger.info(f"Saved baseline: {baseline_id}")
        return baseline_id

    def load_baseline(
        self, model_id: str, signal_name: str, baseline_id: Optional[str] = None
    ) -> Any:  # BaselineStatistics type
        """
        Load baseline statistics.

        Args:
            model_id: Model ID
            signal_name: Signal name
            baseline_id: Optional specific baseline ID (if None, loads most recent)

        Returns:
            BaselineStatistics object
        """
        if not self._check_connection():
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_names["baselines"])

        # Build query
        query = {"model_id": model_id, "signal_name": signal_name}
        if baseline_id:
            query["_id"] = baseline_id

        # Find document(s)
        if baseline_id:
            document = collection.find_one(query)
        else:
            # Get most recent
            document = collection.find_one(query, sort=[("calculated_at", -1)])

        if not document:
            raise ValueError(f"Baseline not found: {model_id}/{signal_name}")

        # Convert back to BaselineStatistics
        from .baseline_calculation import BaselineStatistics

        baseline_data = document["baseline"]
        baseline = BaselineStatistics(
            mean=baseline_data["mean"],
            std=baseline_data["std"],
            median=baseline_data["median"],
            min=baseline_data["min"],
            max=baseline_data["max"],
            range=baseline_data["range"],
            sample_size=baseline_data["sample_size"],
            subgroup_size=baseline_data["subgroup_size"],
            within_subgroup_std=baseline_data.get("within_subgroup_std"),
            between_subgroup_std=baseline_data.get("between_subgroup_std"),
            overall_std=baseline_data.get("overall_std"),
            calculated_at=(
                datetime.fromisoformat(baseline_data["calculated_at"])
                if isinstance(baseline_data["calculated_at"], str)
                else baseline_data["calculated_at"]
            ),
            metadata=baseline_data.get("metadata", {}),
        )

        return baseline

    def save_control_chart(
        self, model_id: str, chart_result: Any, metadata: Optional[Dict] = None  # ControlChartResult type
    ) -> str:
        """
        Save control chart result.

        Args:
            model_id: Model ID
            chart_result: ControlChartResult object
            metadata: Optional additional metadata

        Returns:
            Document ID
        """
        if not self._check_connection():
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_names["control_charts"])

        # Convert chart result to dict
        chart_dict = self._control_chart_to_dict(chart_result)

        # Create document ID
        chart_id = f"{model_id}_{chart_result.chart_type}_{datetime.now().isoformat()}"
        document = {
            "_id": chart_id,
            "model_id": model_id,
            "chart_result": chart_dict,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
        }

        # Insert or update
        collection.replace_one({"_id": chart_id}, document, upsert=True)

        logger.info(f"Saved control chart: {chart_id}")
        return chart_id

    def load_control_chart(self, model_id: str, chart_id: str) -> Any:  # ControlChartResult type
        """
        Load control chart result.

        Args:
            model_id: Model ID
            chart_id: Chart ID

        Returns:
            ControlChartResult object
        """
        if not self._check_connection():
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_names["control_charts"])

        # Find document
        document = collection.find_one({"_id": chart_id, "model_id": model_id})

        if not document:
            raise ValueError(f"Control chart not found: {chart_id}")

        # Convert back to ControlChartResult
        from .control_charts import ControlChartResult

        chart_data = document["chart_result"]
        chart_result = ControlChartResult(
            chart_type=chart_data["chart_type"],
            center_line=chart_data["center_line"],
            upper_control_limit=chart_data["upper_control_limit"],
            lower_control_limit=chart_data["lower_control_limit"],
            upper_warning_limit=chart_data.get("upper_warning_limit"),
            lower_warning_limit=chart_data.get("lower_warning_limit"),
            sample_values=np.array(chart_data["sample_values"]),
            sample_indices=np.array(chart_data["sample_indices"]),
            out_of_control_points=chart_data["out_of_control_points"],
            rule_violations=chart_data.get("rule_violations", {}),
            baseline_stats=chart_data.get("baseline_stats", {}),
            metadata=chart_data.get("metadata", {}),
        )

        return chart_result

    def save_capability_result(
        self, model_id: str, capability_result: Any, metadata: Optional[Dict] = None  # ProcessCapabilityResult type
    ) -> str:
        """
        Save process capability result.

        Args:
            model_id: Model ID
            capability_result: ProcessCapabilityResult object
            metadata: Optional additional metadata

        Returns:
            Document ID
        """
        if not self._check_connection():
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_names["capability"])

        # Convert capability result to dict
        capability_dict = asdict(capability_result)
        # Convert numpy arrays to lists
        for key, value in capability_dict.items():
            if isinstance(value, np.ndarray):
                capability_dict[key] = value.tolist()
            elif isinstance(value, np.generic):
                capability_dict[key] = float(value)

        # Create document ID
        capability_id = f"{model_id}_capability_{datetime.now().isoformat()}"
        document = {
            "_id": capability_id,
            "model_id": model_id,
            "capability_result": capability_dict,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
        }

        # Insert or update
        collection.replace_one({"_id": capability_id}, document, upsert=True)

        logger.info(f"Saved capability result: {capability_id}")
        return capability_id

    def save_multivariate_result(
        self, model_id: str, multivariate_result: Any, metadata: Optional[Dict] = None  # MultivariateSPCResult type
    ) -> str:
        """
        Save multivariate SPC result.

        Args:
            model_id: Model ID
            multivariate_result: MultivariateSPCResult object
            metadata: Optional additional metadata

        Returns:
            Document ID
        """
        if not self._check_connection():
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_names["multivariate"])

        # Convert to dict
        result_dict = self._multivariate_result_to_dict(multivariate_result)

        # Create document ID
        result_id = f"{model_id}_multivariate_{datetime.now().isoformat()}"
        document = {
            "_id": result_id,
            "model_id": model_id,
            "multivariate_result": result_dict,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
        }

        # Insert or update
        collection.replace_one({"_id": result_id}, document, upsert=True)

        logger.info(f"Saved multivariate result: {result_id}")
        return result_id

    def query_spc_history(
        self,
        model_id: str,
        signal_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        result_type: Optional[str] = None,  # 'baseline', 'chart', 'capability', 'multivariate'
    ) -> List[Dict]:
        """
        Query SPC history.

        Args:
            model_id: Model ID
            signal_name: Optional signal name filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            result_type: Optional result type filter

        Returns:
            List of history documents
        """
        if not self._check_connection():
            raise ConnectionError("MongoDB client not connected")

        # Build query
        query = {"model_id": model_id}

        if signal_name:
            query["signal_name"] = signal_name

        if start_time or end_time:
            query["created_at"] = {}
            if start_time:
                query["created_at"]["$gte"] = start_time.isoformat()
            if end_time:
                query["created_at"]["$lte"] = end_time.isoformat()

        results = []

        # Query relevant collections
        collections_to_query = []
        if result_type is None:
            collections_to_query = ["baselines", "control_charts", "capability", "multivariate"]
        elif result_type == "baseline":
            collections_to_query = ["baselines"]
        elif result_type == "chart":
            collections_to_query = ["control_charts"]
        elif result_type == "capability":
            collections_to_query = ["capability"]
        elif result_type == "multivariate":
            collections_to_query = ["multivariate"]

        for coll_name in collections_to_query:
            collection = self.mongo_client.get_collection(self.collection_names[coll_name])
            cursor = collection.find(query)
            # Handle sort - some MongoDB mocks don't support sort with list argument
            try:
                if hasattr(cursor, "sort"):
                    # Try with list first (pymongo style)
                    try:
                        docs = list(cursor.sort([("created_at", -1)]))
                    except TypeError:
                        # Fallback to positional args (older pymongo or mocks)
                        docs = list(cursor.sort("created_at", -1))
                else:
                    docs = list(cursor)
            except (TypeError, AttributeError):
                # If sort fails, just get documents and sort in Python
                docs = list(cursor)
                docs.sort(key=lambda x: x.get("created_at", 0), reverse=True)

            for doc in docs:
                doc["result_type"] = coll_name
            results.extend(docs)

        # Sort by created_at
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        logger.info(f"Found {len(results)} SPC history records")
        return results

    def _check_connection(self) -> bool:
        """Check if MongoDB client is connected."""
        if not self.mongo_client:
            return False
        # Check using is_connected() method (MongoDBClient)
        if hasattr(self.mongo_client, "is_connected"):
            try:
                result = self.mongo_client.is_connected()
                return bool(result)  # Ensure it's a Python bool
            except Exception:
                return False
        # Check using 'connected' attribute (alternative client format)
        if hasattr(self.mongo_client, "connected"):
            return bool(self.mongo_client.connected)
        # Assume connected if no connection check available
        return True

    def _baseline_to_dict(self, baseline: Any) -> Dict[str, Any]:
        """Convert BaselineStatistics to dictionary."""
        return {
            "mean": float(baseline.mean),
            "std": float(baseline.std),
            "median": float(baseline.median),
            "min": float(baseline.min),
            "max": float(baseline.max),
            "range": float(baseline.range),
            "sample_size": int(baseline.sample_size),
            "subgroup_size": int(baseline.subgroup_size),
            "within_subgroup_std": float(baseline.within_subgroup_std) if baseline.within_subgroup_std is not None else None,
            "between_subgroup_std": (
                float(baseline.between_subgroup_std) if baseline.between_subgroup_std is not None else None
            ),
            "overall_std": float(baseline.overall_std) if baseline.overall_std is not None else None,
            "calculated_at": (
                baseline.calculated_at.isoformat()
                if hasattr(baseline.calculated_at, "isoformat")
                else str(baseline.calculated_at)
            ),
            "metadata": baseline.metadata,
        }

    def _control_chart_to_dict(self, chart_result: Any) -> Dict[str, Any]:
        """Convert ControlChartResult to dictionary."""
        return {
            "chart_type": chart_result.chart_type,
            "center_line": float(chart_result.center_line),
            "upper_control_limit": float(chart_result.upper_control_limit),
            "lower_control_limit": float(chart_result.lower_control_limit),
            "upper_warning_limit": (
                float(chart_result.upper_warning_limit) if chart_result.upper_warning_limit is not None else None
            ),
            "lower_warning_limit": (
                float(chart_result.lower_warning_limit) if chart_result.lower_warning_limit is not None else None
            ),
            "sample_values": chart_result.sample_values.tolist(),
            "sample_indices": chart_result.sample_indices.tolist(),
            "out_of_control_points": chart_result.out_of_control_points,
            "rule_violations": {k: v for k, v in chart_result.rule_violations.items()},
            "baseline_stats": {
                k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in chart_result.baseline_stats.items()
            },
            "metadata": chart_result.metadata,
        }

    def _multivariate_result_to_dict(self, result: Any) -> Dict[str, Any]:
        """Convert MultivariateSPCResult to dictionary."""
        return {
            "hotelling_t2": result.hotelling_t2.tolist(),
            "ucl_t2": float(result.ucl_t2),
            "control_limits": {
                k: (
                    {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv for kk, vv in v.items()}
                    if isinstance(v, dict)
                    else v
                )
                for k, v in result.control_limits.items()
            },
            "out_of_control_points": result.out_of_control_points,
            "principal_components": result.principal_components.tolist() if result.principal_components is not None else None,
            "explained_variance": result.explained_variance.tolist() if result.explained_variance is not None else None,
            "contribution_analysis": {str(k): v for k, v in (result.contribution_analysis or {}).items()},
            "baseline_mean": result.baseline_mean.tolist(),
            "baseline_covariance": result.baseline_covariance.tolist(),
            "metadata": result.metadata,
        }
