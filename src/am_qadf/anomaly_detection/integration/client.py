"""
Anomaly Detection Client with Warehouse Data

Integration of anomaly detection with data warehouse.
Provides client for performing anomaly detection using warehouse data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging

# Import from new structure
try:
    from ..core.base_detector import (
        BaseAnomalyDetector,
        AnomalyDetectionResult,
        AnomalyDetectionConfig as BaseAnomalyDetectionConfig,
    )
    from ..core.types import AnomalyType
    from ..detectors.clustering import IsolationForestDetector, DBSCANDetector
    from ..detectors.statistical import ZScoreDetector

    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False
    BaseAnomalyDetector = None
    AnomalyDetectionResult = None
    BaseAnomalyDetectionConfig = None
    AnomalyType = None
    IsolationForestDetector = None
    DBSCANDetector = None
    ZScoreDetector = None

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection with warehouse data."""

    # Detection methods
    methods: List[str] = None  # e.g., ['isolation_forest', 'dbscan', 'z_score']

    # Detection scope
    signals: List[str] = None  # Signals to analyze (e.g., ['laser_power', 'density', 'temperature'])
    spatial_region: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
    layer_range: Optional[Tuple[int, int]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None

    # Detection parameters
    threshold: float = 0.5
    use_voxel_domain: bool = True
    voxel_resolution: float = 0.5

    # Historical data for training
    use_historical_training: bool = True
    historical_model_ids: Optional[List[str]] = None


class AnomalyDetectionClient:
    """
    Client for performing anomaly detection using warehouse data.

    Integrates anomaly detection with data warehouse and voxel domain.
    """

    def __init__(self, unified_query_client, voxel_domain_client=None):
        """
        Initialize anomaly detection client.

        Args:
            unified_query_client: UnifiedQueryClient for querying warehouse data
            voxel_domain_client: Optional VoxelDomainClient for voxel-based detection
        """
        self.unified_client = unified_query_client
        self.voxel_client = voxel_domain_client

        # Initialize detectors
        self.detectors = {}
        if ANOMALY_DETECTION_AVAILABLE:
            if IsolationForestDetector:
                self.detectors["isolation_forest"] = IsolationForestDetector
            if DBSCANDetector:
                self.detectors["dbscan"] = DBSCANDetector
            if ZScoreDetector:
                self.detectors["z_score"] = ZScoreDetector

        logger.info("AnomalyDetectionClient initialized")

    def query_fused_data(
        self,
        model_id: str,
        sources: List[str] = ["laser", "ispm", "ct"],
        layer_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Query and fuse multimodal data from multiple sources.

        Args:
            model_id: Model ID
            sources: List of data sources to query ('laser', 'ispm', 'ct')
            layer_range: Optional layer range tuple (start, end)

        Returns:
            DataFrame with fused multimodal data
        """
        logger.info(f"Querying fused data for model {model_id} from sources: {sources}")

        if not self.unified_client:
            logger.warning("UnifiedQueryClient not available")
            return pd.DataFrame()

        # Default layer range if not provided
        if layer_range is None:
            layer_range = (0, 100)

        # Query merged temporal data (handles laser and ispm)
        query_sources = [s for s in sources if s in ["laser", "ispm", "hatching"]]
        merged_data = {}

        if query_sources:
            try:
                merged_data = self.unified_client.merge_temporal_data(
                    model_id=model_id, layer_range=layer_range, sources=query_sources
                )
            except Exception as e:
                logger.warning(f"Error merging temporal data: {e}")
                merged_data = {}

        # Query CT data separately if requested
        ct_data = None
        if "ct" in sources and self.unified_client.ct_client:
            try:
                ct_scan = self.unified_client.ct_client.get_scan(model_id)
                if ct_scan:
                    # Extract CT data points
                    ct_data = {"points": [], "density": [], "porosity": []}
                    # CT data structure may vary, handle accordingly
                    if isinstance(ct_scan, dict):
                        if "voxel_grid" in ct_scan:
                            voxel_grid = ct_scan["voxel_grid"]
                            # Extract voxel coordinates and values
                            # This is a simplified extraction - may need adjustment based on actual CT data structure
                            pass
            except Exception as e:
                logger.warning(f"Error querying CT data: {e}")

        # Fuse data into DataFrame
        fused_rows = []

        # Process laser data
        if "laser" in sources and "sources" in merged_data and "laser" in merged_data["sources"]:
            laser_source = merged_data["sources"]["laser"]
            if "error" not in laser_source:
                points = laser_source.get("points", [])
                signals = laser_source.get("signals", {})

                # Map signal names to more readable column names
                signal_mapping = {
                    "power": "laser_power",
                    "velocity": "scan_speed",
                    "energy": "energy_density",
                }

                for i, point in enumerate(points):
                    row = {
                        "x": point[0] if len(point) > 0 else None,
                        "y": point[1] if len(point) > 1 else None,
                        "z": point[2] if len(point) > 2 else None,
                        "source": "laser",
                    }

                    # Add laser signals
                    for signal_key, signal_values in signals.items():
                        col_name = signal_mapping.get(signal_key, signal_key)
                        if i < len(signal_values):
                            row[col_name] = signal_values[i]

                    fused_rows.append(row)

        # Process ISPM data
        if "ispm" in sources and "sources" in merged_data and "ispm" in merged_data["sources"]:
            ispm_source = merged_data["sources"]["ispm"]
            if "error" not in ispm_source:
                points = ispm_source.get("points", [])
                signals = ispm_source.get("signals", {})

                # Map signal names
                signal_mapping = {
                    "temperature": "temperature",
                    "thermal": "temperature",
                    "melt_pool": "melt_pool_size",
                }

                for i, point in enumerate(points):
                    row = {
                        "x": point[0] if len(point) > 0 else None,
                        "y": point[1] if len(point) > 1 else None,
                        "z": point[2] if len(point) > 2 else None,
                        "source": "ispm",
                    }

                    # Add ISPM signals
                    for signal_key, signal_values in signals.items():
                        col_name = signal_mapping.get(signal_key, signal_key)
                        if i < len(signal_values):
                            row[col_name] = signal_values[i]

                    fused_rows.append(row)

        # Process CT data if available
        if ct_data and ct_data.get("points"):
            for i, point in enumerate(ct_data["points"]):
                row = {
                    "x": point[0] if len(point) > 0 else None,
                    "y": point[1] if len(point) > 1 else None,
                    "z": point[2] if len(point) > 2 else None,
                    "source": "ct",
                }

                if i < len(ct_data.get("density", [])):
                    row["density"] = ct_data["density"][i]
                if i < len(ct_data.get("porosity", [])):
                    row["porosity"] = ct_data["porosity"][i]

                fused_rows.append(row)

        # Convert to DataFrame
        if fused_rows:
            df = pd.DataFrame(fused_rows)

            # Try to fuse rows with similar coordinates (within tolerance)
            # Group by rounded coordinates to merge data from different sources
            if len(df) > 0:
                # Round coordinates to nearest 0.1mm for fusion
                df["x_rounded"] = df["x"].round(1)
                df["y_rounded"] = df["y"].round(1)
                df["z_rounded"] = df["z"].round(1)

                # Build aggregation dictionary dynamically based on available columns
                agg_dict = {
                    "x": "first",
                    "y": "first",
                    "z": "first",
                    "source": lambda x: ",".join(x.unique().astype(str)),  # Combine sources
                }

                # Add signal columns that exist in the DataFrame
                signal_columns = [
                    "laser_power",
                    "scan_speed",
                    "energy_density",
                    "temperature",
                    "melt_pool_size",
                    "density",
                    "porosity",
                ]

                for col in signal_columns:
                    if col in df.columns:
                        agg_dict[col] = "first"

                # Group by rounded coordinates and aggregate
                grouped = df.groupby(["x_rounded", "y_rounded", "z_rounded"], as_index=False).agg(agg_dict)

                # Drop rounded coordinate columns
                grouped = grouped.drop(columns=["x_rounded", "y_rounded", "z_rounded"])

                # Add layer_index if z coordinate is available
                if "z" in grouped.columns:
                    # Approximate layer from z coordinate (assuming 0.05mm layer height)
                    grouped["layer_index"] = (grouped["z"] / 0.05).astype(int)

                return grouped

        return pd.DataFrame()

    def query_historical_data(
        self,
        model_ids: Optional[List[str]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        signals: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Query historical data for training/comparison.

        Args:
            model_ids: Optional list of model IDs
            time_range: Optional time range
            signals: List of signals to query

        Returns:
            Dictionary with historical data
        """
        logger.info(f"Querying historical data for {len(model_ids) if model_ids else 'all'} models")

        if not model_ids:
            # Get all models (limited)
            if self.unified_client and self.unified_client.stl_client:
                models = self.unified_client.stl_client.list_models(limit=10)
                # Handle case where models might be a Mock or not iterable
                if models and hasattr(models, "__iter__") and not isinstance(models, (str, bytes)):
                    try:
                        model_ids = [
                            (m.get("model_id") if isinstance(m, dict) else getattr(m, "model_id", None)) for m in models
                        ]
                        model_ids = [mid for mid in model_ids if mid is not None]
                    except (TypeError, AttributeError):
                        model_ids = []
                else:
                    model_ids = []

        historical_data = {}
        for model_id in model_ids or []:
            # Query data for this model
            if self.voxel_client:
                # Use voxel domain
                voxel_data = self.voxel_client.map_signals_to_voxels(
                    model_id=model_id,
                    sources=signals or ["laser", "ispm", "ct"],
                    interpolation_method="nearest",
                )
                historical_data[model_id] = voxel_data
            else:
                # Query raw data
                if self.unified_client:
                    merged = self.unified_client.merge_temporal_data(
                        model_id=model_id,
                        layer_range=(0, 10),
                        sources=signals or ["laser", "ispm"],
                    )
                    historical_data[model_id] = merged

        return historical_data

    def detect_anomalies(
        self,
        model_id: str,
        config: AnomalyDetectionConfig,
        historical_data: Optional[Dict[str, Any]] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Detect anomalies using warehouse data.

        Args:
            model_id: Model ID
            config: Anomaly detection configuration
            historical_data: Optional historical data for training
            data: Optional DataFrame with fused data (if provided, will use this instead of querying)

        Returns:
            Dictionary with anomaly detection results including 'anomaly_labels'
        """
        logger.info(f"Detecting anomalies for model {model_id} using methods: {config.methods}")

        # Use provided DataFrame if available, otherwise query data
        if data is not None and not data.empty:
            # Use provided DataFrame
            logger.info(f"Using provided DataFrame with {len(data)} rows")
            # Extract numeric features (exclude coordinate and metadata columns)
            exclude_cols = ["x", "y", "z", "source", "layer_index", "model_id"]
            feature_cols = [
                col for col in data.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col])
            ]

            if not feature_cols:
                raise ValueError("No numeric feature columns found in provided data")

            detection_data = data[feature_cols].values
            data_length = len(data)
        elif config.use_voxel_domain and self.voxel_client:
            # Use voxel domain
            voxel_data = self.voxel_client.map_signals_to_voxels(
                model_id=model_id,
                sources=config.signals or ["laser", "ispm", "ct"],
                interpolation_method="nearest",
            )

            # Extract signal arrays
            signal_arrays = {}
            for signal in config.signals or voxel_data.available_signals:
                if signal in voxel_data.available_signals:
                    signal_arrays[signal] = voxel_data.get_signal_array(signal, default=0.0)

            # Prepare data for detection
            detection_data = self._prepare_voxel_data_for_detection(signal_arrays)
            data_length = detection_data.shape[0] if detection_data.size > 0 else 0
        else:
            # Query raw data
            if not self.unified_client:
                raise ValueError("UnifiedQueryClient required for raw data detection")

            merged_data = self.unified_client.merge_temporal_data(
                model_id=model_id,
                layer_range=config.layer_range or (0, 100),
                sources=config.signals or ["laser", "ispm"],
            )

            detection_data = self._prepare_raw_data_for_detection(merged_data)
            data_length = detection_data.shape[0] if detection_data.size > 0 else 0

        # Get historical data for training if needed
        if config.use_historical_training and historical_data is None:
            historical_data = self.query_historical_data(model_ids=config.historical_model_ids, signals=config.signals)

        # Train detectors on historical data
        trained_detectors = {}
        if historical_data:
            training_data = self._prepare_training_data(historical_data)
            for method in config.methods or ["isolation_forest"]:
                if method in self.detectors:
                    detector_class = self.detectors[method]
                    detector = detector_class()
                    detector.fit(training_data)
                    trained_detectors[method] = detector

        # Detect anomalies
        results = {}
        anomaly_labels = None

        # If no trained detectors and detection_data is empty, return empty results
        if detection_data.size == 0:
            logger.warning("No detection data available")
            return {
                "model_id": model_id,
                "methods": config.methods or ["isolation_forest"],
                "results": {},
                "detection_data_shape": (0,),
                "anomaly_labels": [],
                "timestamp": datetime.now(),
            }

        # If no trained detectors, train on the detection data itself (unsupervised)
        if not trained_detectors:
            logger.info("No historical training data, training detectors on current data")
            for method in config.methods or ["isolation_forest"]:
                if method in self.detectors:
                    try:
                        detector_class = self.detectors[method]
                        detector = detector_class()
                        detector.fit(detection_data)
                        trained_detectors[method] = detector
                    except Exception as e:
                        logger.warning(f"Could not train {method} detector: {e}")

        # Detect anomalies
        for method, detector in trained_detectors.items():
            try:
                anomaly_results = detector.predict(detection_data)

                # Extract labels (0 = normal, 1 = anomaly)
                if anomaly_results and len(anomaly_results) > 0:
                    # Check if results are AnomalyDetectionResult objects or simple values
                    if hasattr(anomaly_results[0], "is_anomaly"):
                        labels = [1 if r.is_anomaly else 0 for r in anomaly_results]
                        scores = [r.anomaly_score for r in anomaly_results]
                    else:
                        # Assume results are boolean or binary
                        labels = [1 if r else 0 for r in anomaly_results]
                        scores = [float(r) if isinstance(r, (int, float)) else 0.0 for r in anomaly_results]

                    # Use labels from first successful method
                    if anomaly_labels is None:
                        anomaly_labels = labels

                    results[method] = {
                        "anomalies": anomaly_results,
                        "num_anomalies": sum(labels),
                        "anomaly_scores": scores,
                        "anomaly_labels": labels,
                    }
                else:
                    results[method] = {"error": "No results returned"}
            except Exception as e:
                logger.error(f"Error detecting anomalies with {method}: {e}")
                results[method] = {"error": str(e)}

        # If no labels were generated, create empty list
        if anomaly_labels is None:
            anomaly_labels = [0] * data_length if data_length > 0 else []

        return {
            "model_id": model_id,
            "methods": config.methods or ["isolation_forest"],
            "results": results,
            "detection_data_shape": (detection_data.shape if hasattr(detection_data, "shape") else None),
            "anomaly_labels": anomaly_labels,
            "total_points": data_length,
            "num_anomalies": sum(anomaly_labels) if anomaly_labels else 0,
            "anomaly_rate": (sum(anomaly_labels) / len(anomaly_labels) if anomaly_labels and len(anomaly_labels) > 0 else 0.0),
            "timestamp": datetime.now(),
        }

    def _prepare_voxel_data_for_detection(self, signal_arrays: Dict[str, np.ndarray]) -> np.ndarray:
        """Prepare voxel data for anomaly detection."""
        # Stack signal arrays
        arrays = []
        for signal, array in signal_arrays.items():
            arrays.append(array.flatten())

        if arrays:
            return np.column_stack(arrays)
        return np.array([])

    def _prepare_raw_data_for_detection(self, merged_data: Dict[str, Any]) -> np.ndarray:
        """Prepare raw data for anomaly detection."""
        # Extract features from merged data
        features = []
        for source, data in merged_data.items():
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                features.append(df[numeric_cols].values)

        if features:
            # Combine features
            return np.vstack(features)
        return np.array([])

    def _prepare_training_data(self, historical_data: Dict[str, Any]) -> np.ndarray:
        """Prepare training data from historical data."""
        training_samples = []

        for model_id, data in historical_data.items():
            if hasattr(data, "get_signal_array"):
                # Voxel data
                signals = data.available_signals
                arrays = []
                for signal in signals:
                    arrays.append(data.get_signal_array(signal, default=0.0).flatten())
                if arrays:
                    training_samples.append(np.column_stack(arrays))
            elif isinstance(data, np.ndarray):
                # Direct numpy array
                if data.size > 0:
                    training_samples.append(data)
            elif isinstance(data, dict):
                # Raw merged data
                features = self._prepare_raw_data_for_detection(data)
                if features.size > 0:
                    training_samples.append(features)

        if training_samples:
            return np.vstack(training_samples)
        return np.array([])
