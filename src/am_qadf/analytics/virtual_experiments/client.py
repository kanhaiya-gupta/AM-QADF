"""
Virtual Experiment Client with Warehouse Data

Integration of virtual experiments with data warehouse.
Provides client for performing virtual experiments using warehouse data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import sys
from pathlib import Path
import importlib.util

# Import SpatialQuery for laser parameter queries
try:
    from ...data_warehouse_clients.query.base_query_client import SpatialQuery
except ImportError:
    try:
        from data_warehouse_clients.query.base_query_client import SpatialQuery
    except ImportError:
        # Fallback: try to get from sys.modules
        import sys

        if "base_query_client" in sys.modules:
            SpatialQuery = sys.modules["base_query_client"].SpatialQuery
        else:
            SpatialQuery = None

# Try to import virtual experiment modules
try:
    from ...virtual_environment.testing_frameworks.experiment_design import (
        VirtualExperimentDesigner,
        ExperimentType,
        DesignType,
        ExperimentConfig,
        ExperimentResult as VirtualExpResult,
    )

    VIRTUAL_EXPERIMENTS_AVAILABLE = True
except ImportError:
    # Fallback for direct module loading
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent.parent

    virtual_exp_dir = project_root / "src" / "data_pipeline" / "virtual_environment" / "testing_frameworks"

    VIRTUAL_EXPERIMENTS_AVAILABLE = False
    if virtual_exp_dir.exists():
        try:
            # Load experiment_design
            exp_design_path = virtual_exp_dir / "experiment_design.py"
            if exp_design_path.exists():
                spec = importlib.util.spec_from_file_location("experiment_design", exp_design_path)
                exp_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(exp_module)
                VirtualExperimentDesigner = getattr(exp_module, "VirtualExperimentDesigner", None)
                ExperimentType = getattr(exp_module, "ExperimentType", None)
                DesignType = getattr(exp_module, "DesignType", None)
                ExperimentConfig = getattr(exp_module, "ExperimentConfig", None)
                VirtualExpResult = getattr(exp_module, "ExperimentResult", None)
                VIRTUAL_EXPERIMENTS_AVAILABLE = True
        except Exception as e:
            logging.warning(f"Could not load virtual experiment modules: {e}")

logger = logging.getLogger(__name__)


@dataclass
class VirtualExperimentConfig:
    """Configuration for virtual experiment with warehouse data."""

    # Experiment type
    experiment_type: str = "parameter_sweep"  # "parameter_sweep", "optimization", "validation"

    # Base model
    base_model_id: str = None

    # Parameter ranges (from warehouse or specified)
    parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    use_warehouse_ranges: bool = True  # Use parameter ranges from warehouse

    # Design parameters
    design_type: str = "factorial"  # "factorial", "lhs", "random"
    num_samples: int = 100

    # Comparison settings
    compare_with_warehouse: bool = True
    comparison_metrics: List[str] = None  # Metrics to compare

    # Spatial/temporal scope
    spatial_region: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
    layer_range: Optional[Tuple[int, int]] = None


class VirtualExperimentClient:
    """
    Client for performing virtual experiments using warehouse data.

    Integrates virtual experiments with data warehouse.
    """

    def __init__(self, unified_query_client, voxel_domain_client=None):
        """
        Initialize virtual experiment client.

        Args:
            unified_query_client: UnifiedQueryClient for querying warehouse data
            voxel_domain_client: Optional VoxelDomainClient for voxel-based analysis
        """
        self.unified_client = unified_query_client
        self.voxel_client = voxel_domain_client

        # Initialize virtual experiment designer
        if VIRTUAL_EXPERIMENTS_AVAILABLE and VirtualExperimentDesigner:
            self.experiment_designer = VirtualExperimentDesigner()
        else:
            self.experiment_designer = None
            logger.warning("Virtual experiment modules not available")

        logger.info("VirtualExperimentClient initialized")

    def query_historical_builds(
        self,
        model_type: Optional[str] = None,
        process_conditions: Optional[List[str]] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Query historical build data from warehouse.

        Args:
            model_type: Optional model type filter
            process_conditions: Optional list of process condition filters (e.g., ['laser_power > 200'])
            limit: Maximum number of builds to return

        Returns:
            DataFrame with historical build data
        """
        logger.info(f"Querying historical builds (limit: {limit})")

        if not self.unified_client or not self.unified_client.stl_client:
            return pd.DataFrame()

        # Get list of models
        models = self.unified_client.stl_client.list_models(limit=limit)

        if not models:
            return pd.DataFrame()

        # Collect build data for each model
        build_data = []
        for model in models:
            model_id = model.get("model_id")

            # Query laser parameters for this model
            if self.unified_client.laser_client and SpatialQuery:
                try:
                    spatial_query = SpatialQuery(component_id=model_id)
                    laser_result = self.unified_client.laser_client.query(spatial=spatial_query)

                    if laser_result and laser_result.points:
                        # Convert QueryResult to list of dictionaries
                        laser_data = []
                        points = laser_result.points
                        signals = laser_result.signals

                        # Get signal values (using standard names)
                        power_values = signals.get("power", [])
                        velocity_values = signals.get("velocity", [])
                        energy_values = signals.get("energy", [])

                        # Try to get layer_index from MongoDB if laser_client uses MongoDB
                        # Create a mapping from coordinates to layer_index
                        coord_to_layer = {}
                        if (
                            hasattr(self.unified_client.laser_client, "use_mongodb")
                            and self.unified_client.laser_client.use_mongodb
                            and hasattr(self.unified_client.laser_client, "mongo_client")
                            and self.unified_client.laser_client.mongo_client
                        ):
                            try:
                                collection = self.unified_client.laser_client.mongo_client.get_collection("laser_monitoring_data")
                                # Query all documents for this model (limited to reasonable number)
                                cursor = collection.find({"model_id": model_id}).limit(10000)
                                for doc in cursor:
                                    coords = doc.get("spatial_coordinates", [])
                                    if len(coords) == 3:
                                        coord_key = tuple(coords)
                                        layer_idx = doc.get("layer_index", 0)
                                        coord_to_layer[coord_key] = layer_idx
                            except Exception as e:
                                logger.debug(f"Could not query layer_index from MongoDB: {e}")

                        for i, point in enumerate(points):
                            data_point = {
                                "x": point[0],
                                "y": point[1],
                                "z": point[2],
                                "laser_power": (power_values[i] if i < len(power_values) else None),
                                "scan_speed": (velocity_values[i] if i < len(velocity_values) else None),
                                "energy_density": (energy_values[i] if i < len(energy_values) else None),
                            }
                            # Add layer_index if available from coordinate mapping
                            if point in coord_to_layer:
                                data_point["layer_index"] = coord_to_layer[point]
                            else:
                                # Use z-coordinate to approximate layer (assuming 0.05mm layer height)
                                # This is a fallback approximation
                                data_point["layer_index"] = int(point[2] / 0.05) if point[2] >= 0 else 0

                            laser_data.append(data_point)

                        if laser_data:
                            # Limit to 1000 points if needed
                            if len(laser_data) > 1000:
                                laser_data = laser_data[:1000]

                            # Aggregate by layer
                            df = pd.DataFrame(laser_data)
                            if "layer_index" in df.columns and df["layer_index"].notna().any():
                                aggregated = (
                                    df.groupby("layer_index")
                                    .agg(
                                        {
                                            "laser_power": [
                                                "mean",
                                                "std",
                                                "min",
                                                "max",
                                            ],
                                            "scan_speed": ["mean", "std", "min", "max"],
                                            "energy_density": [
                                                "mean",
                                                "std",
                                                "min",
                                                "max",
                                            ],
                                        }
                                    )
                                    .reset_index()
                                )

                                # Flatten column names
                                aggregated.columns = [
                                    "_".join(col).strip("_") if col[1] else col[0] for col in aggregated.columns.values
                                ]
                                aggregated["model_id"] = model_id
                                build_data.append(aggregated)
                            else:
                                # If no layer_index, aggregate all data together
                                aggregated = pd.DataFrame(
                                    {
                                        "model_id": [model_id],
                                        "laser_power_mean": [df["laser_power"].mean()],
                                        "laser_power_std": [df["laser_power"].std()],
                                        "laser_power_min": [df["laser_power"].min()],
                                        "laser_power_max": [df["laser_power"].max()],
                                        "scan_speed_mean": [df["scan_speed"].mean()],
                                        "scan_speed_std": [df["scan_speed"].std()],
                                        "scan_speed_min": [df["scan_speed"].min()],
                                        "scan_speed_max": [df["scan_speed"].max()],
                                        "energy_density_mean": [df["energy_density"].mean()],
                                        "energy_density_std": [df["energy_density"].std()],
                                        "energy_density_min": [df["energy_density"].min()],
                                        "energy_density_max": [df["energy_density"].max()],
                                    }
                                )
                                build_data.append(aggregated)
                except Exception as e:
                    logger.warning(f"Error querying laser data for model {model_id}: {e}")
                    continue

        if build_data:
            return pd.concat(build_data, ignore_index=True)

        return pd.DataFrame()

    def get_parameter_ranges_from_warehouse(
        self,
        model_id: Optional[str] = None,
        model_ids: Optional[List[str]] = None,
        variables: List[str] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Get parameter ranges from warehouse data.

        Args:
            model_id: Optional single model ID (for compatibility with tests)
            model_ids: Optional list of model IDs to consider
            variables: List of variable names

        Returns:
            Dictionary mapping variable names to (min, max) ranges
        """
        # Handle model_id parameter (for test compatibility)
        if model_id is not None:
            model_ids = [model_id] if model_ids is None else model_ids

        logger.info(f"Getting parameter ranges from warehouse for {variables}")

        if not variables:
            variables = ["laser_power", "scan_speed", "energy_density"]

        if not self.unified_client or not self.unified_client.laser_client:
            return {}

        # Query laser parameters
        all_data = []
        if not SpatialQuery:
            logger.warning("SpatialQuery not available, cannot query laser parameters")
            return {}

        if model_ids:
            for model_id in model_ids:
                try:
                    spatial_query = SpatialQuery(component_id=model_id)
                    result = self.unified_client.laser_client.query(spatial=spatial_query)
                    if result and result.points:
                        # Convert QueryResult to list of dictionaries
                        points = result.points
                        signals = result.signals
                        power_values = signals.get("power", [])
                        velocity_values = signals.get("velocity", [])
                        energy_values = signals.get("energy", [])

                        for i, point in enumerate(points):
                            if i < 10000:  # Limit to 10000 points
                                all_data.append(
                                    {
                                        "x": point[0],
                                        "y": point[1],
                                        "z": point[2],
                                        "laser_power": (power_values[i] if i < len(power_values) else None),
                                        "scan_speed": (velocity_values[i] if i < len(velocity_values) else None),
                                        "energy_density": (energy_values[i] if i < len(energy_values) else None),
                                    }
                                )
                except Exception as e:
                    logger.warning(f"Error querying laser data for model {model_id}: {e}")
                    continue
        else:
            # Query from all models (limited)
            models = self.unified_client.stl_client.list_models(limit=10)
            for model in models:
                model_id = model.get("model_id")
                try:
                    spatial_query = SpatialQuery(component_id=model_id)
                    result = self.unified_client.laser_client.query(spatial=spatial_query)
                    if result and result.points:
                        # Convert QueryResult to list of dictionaries
                        points = result.points
                        signals = result.signals
                        power_values = signals.get("power", [])
                        velocity_values = signals.get("velocity", [])
                        energy_values = signals.get("energy", [])

                        for i, point in enumerate(points):
                            if i < 1000:  # Limit to 1000 points per model
                                all_data.append(
                                    {
                                        "x": point[0],
                                        "y": point[1],
                                        "z": point[2],
                                        "laser_power": (power_values[i] if i < len(power_values) else None),
                                        "scan_speed": (velocity_values[i] if i < len(velocity_values) else None),
                                        "energy_density": (energy_values[i] if i < len(energy_values) else None),
                                    }
                                )
                except Exception as e:
                    logger.warning(f"Error querying laser data for model {model_id}: {e}")
                    continue

        if not all_data:
            return {}

        # Calculate ranges
        df = pd.DataFrame(all_data)
        ranges = {}
        for var in variables:
            if var in df.columns:
                ranges[var] = (float(df[var].min()), float(df[var].max()))

        return ranges

    def design_experiment(
        self,
        base_model_id: str,
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        config: VirtualExperimentConfig = None,
    ) -> Dict[str, Any]:
        """
        Design virtual experiment based on warehouse data.

        Args:
            base_model_id: Base model ID
            parameter_ranges: Optional parameter ranges (if None, will query from warehouse)
            config: Experiment configuration

        Returns:
            Dictionary with experiment design
        """
        if config is None:
            config = VirtualExperimentConfig()
            config.base_model_id = base_model_id

        logger.info(f"Designing virtual experiment for model {base_model_id}")

        # Get parameter ranges from warehouse if not provided
        if parameter_ranges is None and config.use_warehouse_ranges:
            parameter_ranges = self.get_parameter_ranges_from_warehouse(model_ids=[base_model_id] if base_model_id else None)

        if not parameter_ranges:
            raise ValueError("No parameter ranges available. Provide ranges or ensure warehouse has data.")

        # Design experiment using virtual experiment designer
        if self.experiment_designer:
            # Generate design points
            design_points = self._generate_design_points(
                parameter_ranges=parameter_ranges,
                design_type=config.design_type,
                num_samples=config.num_samples,
            )

            return {
                "experiment_id": f"exp_{base_model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "base_model_id": base_model_id,
                "parameter_ranges": parameter_ranges,
                "design_points": design_points,
                "design_type": config.design_type,
                "num_samples": len(design_points),
                "timestamp": datetime.now(),
            }
        else:
            # Fallback: simple design
            design_points = self._generate_design_points(
                parameter_ranges=parameter_ranges,
                design_type=config.design_type,
                num_samples=config.num_samples,
            )

            return {
                "experiment_id": f"exp_{base_model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "base_model_id": base_model_id,
                "parameter_ranges": parameter_ranges,
                "design_points": design_points,
                "design_type": config.design_type,
                "num_samples": len(design_points),
                "timestamp": datetime.now(),
            }

    def _generate_design_points(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        design_type: str = "lhs",
        num_samples: int = 100,
    ) -> List[Dict[str, float]]:
        """Generate design points for experiment."""
        param_names = list(parameter_ranges.keys())
        param_bounds = [parameter_ranges[name] for name in param_names]

        if design_type == "lhs":
            # Latin Hypercube Sampling
            try:
                from scipy.stats import qmc

                sampler = qmc.LatinHypercube(d=len(param_names))
                samples = sampler.random(n=num_samples)

                # Scale to parameter ranges
                design_points = []
                for sample in samples:
                    point = {}
                    for i, name in enumerate(param_names):
                        min_val, max_val = param_bounds[i]
                        point[name] = min_val + sample[i] * (max_val - min_val)
                    design_points.append(point)
                return design_points
            except ImportError:
                # Fallback to random
                design_type = "random"

        if design_type == "random":
            # Random sampling
            design_points = []
            for _ in range(num_samples):
                point = {}
                for name in param_names:
                    min_val, max_val = parameter_ranges[name]
                    point[name] = np.random.uniform(min_val, max_val)
                design_points.append(point)
            return design_points

        # Default: grid sampling (limited)
        if len(param_names) <= 3:
            # Create grid
            grid_size = int(np.ceil(num_samples ** (1 / len(param_names))))
            design_points = []
            for name in param_names:
                min_val, max_val = parameter_ranges[name]
                values = np.linspace(min_val, max_val, grid_size)
                # Create all combinations
                if not design_points:
                    design_points = [{name: v} for v in values]
                else:
                    new_points = []
                    for point in design_points:
                        for v in values:
                            new_point = point.copy()
                            new_point[name] = v
                            new_points.append(new_point)
                    design_points = new_points
                    if len(design_points) > num_samples:
                        design_points = design_points[:num_samples]
            return design_points

        # Fallback to random
        return self._generate_design_points(parameter_ranges, "random", num_samples)

    def compare_with_warehouse(
        self, experiment_id: str, model_id: str, comparison_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare experiment results with warehouse data.

        Args:
            experiment_id: Experiment ID
            model_id: Model ID to compare with
            comparison_metrics: List of metrics to compare

        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing experiment {experiment_id} with warehouse data for model {model_id}")

        # This would typically query stored experiment results
        # and compare with warehouse data
        # For now, return placeholder structure

        return {
            "experiment_id": experiment_id,
            "model_id": model_id,
            "comparison_metrics": comparison_metrics or [],
            "timestamp": datetime.now(),
        }
