"""
Sensitivity Analysis Client with Warehouse Data

Integration of sensitivity analysis with data warehouse.
Provides client for performing sensitivity analysis using warehouse data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import sys
from pathlib import Path
import importlib.util

# Try to import sensitivity analysis modules
try:
    from ...processing.analytics.sensitivity_analysis import (
        GlobalSensitivityAnalyzer,
        SobolAnalyzer,
        MorrisAnalyzer,
        LocalSensitivityAnalyzer,
        ExperimentalDesigner,
        UncertaintyQuantifier,
    )

    SENSITIVITY_ANALYSIS_AVAILABLE = True
except ImportError:
    # Fallback for direct module loading
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent.parent

    sensitivity_dir = project_root / "src" / "data_pipeline" / "processing" / "analytics" / "sensitivity_analysis"

    SENSITIVITY_ANALYSIS_AVAILABLE = False
    if sensitivity_dir.exists():
        try:
            # Load global_analysis
            global_path = sensitivity_dir / "global_analysis.py"
            if global_path.exists():
                spec = importlib.util.spec_from_file_location("global_analysis", global_path)
                global_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(global_module)
                GlobalSensitivityAnalyzer = global_module.GlobalSensitivityAnalyzer
                SobolAnalyzer = getattr(global_module, "SobolAnalyzer", None)
                MorrisAnalyzer = getattr(global_module, "MorrisAnalyzer", None)
                SensitivityConfig = getattr(global_module, "SensitivityConfig", None)
                SENSITIVITY_ANALYSIS_AVAILABLE = True

            # Load local_analysis
            local_path = sensitivity_dir / "local_analysis.py"
            if local_path.exists():
                spec = importlib.util.spec_from_file_location("local_analysis", local_path)
                local_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(local_module)
                LocalSensitivityAnalyzer = local_module.LocalSensitivityAnalyzer

            # Load doe
            doe_path = sensitivity_dir / "doe.py"
            if doe_path.exists():
                spec = importlib.util.spec_from_file_location("doe", doe_path)
                doe_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(doe_module)
                ExperimentalDesigner = doe_module.ExperimentalDesigner

            # Load uncertainty
            uncertainty_path = sensitivity_dir / "uncertainty.py"
            if uncertainty_path.exists():
                spec = importlib.util.spec_from_file_location("uncertainty", uncertainty_path)
                uncertainty_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(uncertainty_module)
                UncertaintyQuantifier = uncertainty_module.UncertaintyQuantifier
        except Exception as e:
            logging.warning(f"Could not load Phase 9 sensitivity analysis modules: {e}")

logger = logging.getLogger(__name__)


@dataclass
class SensitivityAnalysisConfig:
    """Configuration for sensitivity analysis with warehouse data."""

    # Analysis method
    method: str = "sobol"  # "sobol", "morris", "local", "doe"

    # Parameter selection
    process_variables: List[str] = None  # Variables to analyze (e.g., ['laser_power', 'scan_speed'])
    measurement_variables: List[str] = None  # Measurement outputs (e.g., ['density', 'temperature'])

    # Spatial/temporal scope
    spatial_region: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None  # Bounding box
    layer_range: Optional[Tuple[int, int]] = None  # Layer range
    time_range: Optional[Tuple[datetime, datetime]] = None  # Time range

    # Analysis parameters
    sample_size: int = 1000
    confidence_level: float = 0.95

    # Use voxel domain
    use_voxel_domain: bool = False
    voxel_resolution: float = 0.5


class SensitivityAnalysisClient:
    """
    Client for performing sensitivity analysis using warehouse data.

    Integrates Phase 9 sensitivity analysis with Phase 12 data warehouse.
    """

    def __init__(self, unified_query_client, voxel_domain_client=None):
        """
        Initialize sensitivity analysis client.

        Args:
            unified_query_client: UnifiedQueryClient for querying warehouse data
            voxel_domain_client: Optional VoxelDomainClient for voxel-based analysis
        """
        self.unified_client = unified_query_client
        self.voxel_client = voxel_domain_client

        # Initialize sensitivity analysis modules
        if SENSITIVITY_ANALYSIS_AVAILABLE:
            # Disable parallel processing to avoid pickling issues with nested functions
            try:
                if "SensitivityConfig" in globals() and SensitivityConfig:
                    sa_config = SensitivityConfig()
                    sa_config.parallel_processing = False
                    self.global_analyzer = GlobalSensitivityAnalyzer(config=sa_config) if GlobalSensitivityAnalyzer else None
                else:
                    self.global_analyzer = GlobalSensitivityAnalyzer() if GlobalSensitivityAnalyzer else None
            except Exception as e:
                logger.warning(f"Could not configure sensitivity analyzer: {e}")
                # Fallback: create without config
                self.global_analyzer = GlobalSensitivityAnalyzer() if GlobalSensitivityAnalyzer else None
            self.local_analyzer = (
                LocalSensitivityAnalyzer() if "LocalSensitivityAnalyzer" in globals() and LocalSensitivityAnalyzer else None
            )
            self.doe_designer = (
                ExperimentalDesigner() if "ExperimentalDesigner" in globals() and ExperimentalDesigner else None
            )
            self.uncertainty_quantifier = (
                UncertaintyQuantifier() if "UncertaintyQuantifier" in globals() and UncertaintyQuantifier else None
            )
        else:
            self.global_analyzer = None
            self.local_analyzer = None
            self.doe_designer = None
            self.uncertainty_quantifier = None
            logger.warning("Sensitivity analysis modules not available")

        # Store surrogate model for picklable model function
        self._surrogate_model = None

        logger.info("SensitivityAnalysisClient initialized")

    def query_process_variables(
        self,
        model_id: str,
        variables: List[str],
        spatial_region: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
        layer_range: Optional[Tuple[int, int]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> pd.DataFrame:
        """
        Query process variables from warehouse.

        Args:
            model_id: Model ID
            variables: List of variable names (e.g., ['laser_power', 'scan_speed'])
            spatial_region: Optional bounding box
            layer_range: Optional layer range
            time_range: Optional time range

        Returns:
            DataFrame with process variables
        """
        logger.info(f"Querying process variables: {variables} for model {model_id}")

        # Use unified client's merge_temporal_data which handles the query interface correctly
        if self.unified_client:
            # Query using unified client's merge method
            merged_data = self.unified_client.merge_temporal_data(
                model_id=model_id,
                layer_range=layer_range or (0, 100),  # Default layer range if not provided
                sources=["laser"],
            )

            if merged_data and "sources" in merged_data and "laser" in merged_data["sources"]:
                laser_source = merged_data["sources"]["laser"]

                # Check if it's an error
                if "error" in laser_source:
                    logger.warning(f"Error querying laser data: {laser_source['error']}")
                    return pd.DataFrame()

                # Extract points and signals
                points = laser_source.get("points", [])
                signals = laser_source.get("signals", {})
                metadata = laser_source.get("metadata", {})

                if points and signals:
                    # Convert to DataFrame
                    data_dict: Dict[str, Any] = {}

                    # Add coordinates
                    if points:
                        coords = np.array(points)
                        data_dict["x"] = coords[:, 0] if coords.shape[1] > 0 else []
                        data_dict["y"] = coords[:, 1] if coords.shape[1] > 1 else []
                        data_dict["z"] = coords[:, 2] if coords.shape[1] > 2 else []

                    # Add signals (map to variable names)
                    signal_mapping = {
                        "power": "laser_power",
                        "velocity": "scan_speed",
                        "energy": "energy_density",
                    }

                    for signal_key, signal_values in signals.items():
                        var_name = signal_mapping.get(signal_key, signal_key)
                        if len(signal_values) == len(points):
                            data_dict[var_name] = signal_values

                    # Add metadata
                    if "layer_index" in metadata:
                        data_dict["layer_index"] = metadata["layer_index"]

                    df = pd.DataFrame(data_dict)

                    # Filter to requested variables
                    available_vars = [v for v in variables if v in df.columns]
                    if available_vars:
                        return df[available_vars + [col for col in ["x", "y", "z", "layer_index"] if col in df.columns]]

        logger.warning(f"No process variables found for {variables}")
        return pd.DataFrame()

    def query_measurement_data(
        self,
        model_id: str,
        sources: List[str] = ["ispm", "ct"],
        spatial_region: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
        layer_range: Optional[Tuple[int, int]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Query measurement data from warehouse.

        Args:
            model_id: Model ID
            sources: List of sources (e.g., ['ispm', 'ct'])
            spatial_region: Optional bounding box
            layer_range: Optional layer range
            time_range: Optional time range

        Returns:
            Dictionary mapping source names to DataFrames
        """
        logger.info(f"Querying measurement data from {sources} for model {model_id}")

        measurement_data = {}

        # Use unified client's merge methods which handle the query interface correctly
        if self.unified_client:
            # Query ISPM data
            if "ispm" in sources:
                try:
                    merged_ispm = self.unified_client.merge_temporal_data(
                        model_id=model_id,
                        layer_range=layer_range or (0, 100),
                        sources=["ispm"],
                    )

                    if merged_ispm and "sources" in merged_ispm and "ispm" in merged_ispm["sources"]:
                        ispm_source = merged_ispm["sources"]["ispm"]
                        if "error" not in ispm_source:
                            points = ispm_source.get("points", [])
                            signals = ispm_source.get("signals", {})
                            if points and signals:
                                # Convert to DataFrame
                                data_dict: Dict[str, Any] = {}
                                if points:
                                    coords = np.array(points)
                                    data_dict["x"] = coords[:, 0] if coords.shape[1] > 0 else []
                                    data_dict["y"] = coords[:, 1] if coords.shape[1] > 1 else []
                                    data_dict["z"] = coords[:, 2] if coords.shape[1] > 2 else []
                                for signal_key, signal_values in signals.items():
                                    if len(signal_values) == len(points):
                                        data_dict[signal_key] = signal_values
                                if data_dict:
                                    measurement_data["ispm"] = pd.DataFrame(data_dict)
                                    logger.info(f"Found {len(measurement_data['ispm'])} ISPM data points")
                        else:
                            logger.warning(f"ISPM query returned error: {ispm_source.get('error', 'Unknown error')}")
                    else:
                        logger.warning("ISPM query returned no data or invalid structure")
                except Exception as e:
                    logger.warning(f"Error querying ISPM data: {e}")

            # Query CT scan data
            if "ct" in sources:
                # Get model bounding box if spatial_region not provided
                if spatial_region is None:
                    try:
                        if self.unified_client.stl_client:
                            bbox = self.unified_client.stl_client.get_model_bounding_box(model_id)
                            if bbox:
                                # bbox is already a tuple (bbox_min, bbox_max)
                                spatial_region = bbox
                    except Exception as e:
                        logger.warning(f"Could not get model bounding box: {e}")

                # Only query if we have a bbox
                if spatial_region:
                    merged_ct = self.unified_client.merge_spatial_data(model_id=model_id, bbox=spatial_region, sources=["ct"])
                else:
                    merged_ct = None

                if merged_ct and "sources" in merged_ct and "ct" in merged_ct["sources"]:
                    ct_source = merged_ct["sources"]["ct"]
                    if "error" not in ct_source:
                        points = ct_source.get("points", [])
                        signals = ct_source.get("signals", {})
                        if points and signals:
                            # Convert to DataFrame
                            data_dict = {}
                            if points:
                                coords = np.array(points)
                                data_dict["x"] = coords[:, 0] if coords.shape[1] > 0 else []
                                data_dict["y"] = coords[:, 1] if coords.shape[1] > 1 else []
                                data_dict["z"] = coords[:, 2] if coords.shape[1] > 2 else []
                            for signal_key, signal_values in signals.items():
                                if len(signal_values) == len(points):
                                    data_dict[signal_key] = signal_values
                            if data_dict:
                                measurement_data["ct"] = pd.DataFrame(data_dict)

        return measurement_data

    def perform_sensitivity_analysis(self, model_id: str, config: SensitivityAnalysisConfig) -> Dict[str, Any]:
        """
        Perform sensitivity analysis using warehouse data.

        Args:
            model_id: Model ID
            config: Sensitivity analysis configuration

        Returns:
            Dictionary with sensitivity analysis results
        """
        logger.info(f"Performing {config.method} sensitivity analysis for model {model_id}")

        # Query process variables
        process_data = self.query_process_variables(
            model_id=model_id,
            variables=config.process_variables or ["laser_power", "scan_speed"],
            spatial_region=config.spatial_region,
            layer_range=config.layer_range,
            time_range=config.time_range,
        )

        if process_data.empty:
            raise ValueError(f"No process data found for model {model_id}")

        # Query measurement data
        # Handle both source names (['ispm', 'ct']) and signal names (['temperature', 'density'])
        measurement_sources_raw = config.measurement_variables or ["ispm", "ct"]

        # Check if these are source names or signal names
        valid_sources = ["ispm", "ct", "laser", "hatching"]
        if all(src in valid_sources for src in measurement_sources_raw):
            # These are source names - use directly
            measurement_sources = measurement_sources_raw
        else:
            # These are likely signal names - query from default sources
            logger.info(f"Interpreting {measurement_sources_raw} as signal names, querying from default sources")
            measurement_sources = ["ispm", "ct"]  # Default sources

        measurement_data = self.query_measurement_data(
            model_id=model_id,
            sources=measurement_sources,
            spatial_region=config.spatial_region,
            layer_range=config.layer_range,
            time_range=config.time_range,
        )

        if not measurement_data:
            # Provide more informative error message
            available_sources = []
            if self.unified_client:
                if self.unified_client.ispm_client:
                    available_sources.append("ispm")
                if self.unified_client.ct_client:
                    available_sources.append("ct")

            # Try to use process data as outputs if measurement data isn't available
            # This allows sensitivity analysis on process variables themselves
            logger.warning(f"No measurement data found. Attempting to use process data as outputs.")

            # Use process data columns that aren't in process_variables as outputs
            # If all process variables are inputs, use the last one as output (e.g., energy_density)
            output_cols = [
                col
                for col in process_data.columns
                if col not in config.process_variables
                and col not in ["x", "y", "z", "layer_index", "source", "model_id"]
                and pd.api.types.is_numeric_dtype(process_data[col])
            ]

            # If no output columns found, use a derived output (e.g., energy_density from laser_power and scan_speed)
            if not output_cols and len(config.process_variables) >= 2:
                # Use the last process variable as output (typically energy_density)
                # This allows analyzing how other variables affect it
                output_col = config.process_variables[-1]
                input_vars = config.process_variables[:-1]
                # Check if the output column exists in process_data
                if output_col not in process_data.columns:
                    # Can't use this column as output - fallback failed
                    error_msg = "No measurement data found"
                    logger.error(
                        f"{error_msg} for model {model_id}. Process data columns: {list(process_data.columns)}. Required: {config.process_variables}."
                    )
                    raise ValueError(error_msg)
                logger.info(f"No separate output column found. Using {output_col} as output (derived from {input_vars})")
                measurement_data["process_output"] = pd.DataFrame(
                    {
                        "x": process_data.get("x", [0] * len(process_data)),
                        "y": process_data.get("y", [0] * len(process_data)),
                        "z": process_data.get("z", [0] * len(process_data)),
                        output_col: process_data[output_col].values,
                    }
                )
            elif output_cols:
                logger.info(f"Using process data columns as outputs: {output_cols}")
                # Create a synthetic measurement_data dict
                for col in output_cols[:1]:  # Use first available output column
                    measurement_data["process_output"] = pd.DataFrame(
                        {
                            "x": process_data.get("x", [0] * len(process_data)),
                            "y": process_data.get("y", [0] * len(process_data)),
                            "z": process_data.get("z", [0] * len(process_data)),
                            col: process_data[col].values,
                        }
                    )
                    break

            if not measurement_data:
                error_msg = "No measurement data found"
                logger.error(
                    f"{error_msg} for model {model_id}. Queried sources: {measurement_sources}. Available clients: {available_sources}."
                )
                raise ValueError(error_msg)

        # Prepare data for analysis
        # If using process data as output, adjust process_variables
        if "process_output" in measurement_data:
            # We're using a process variable as output, so remove it from inputs
            output_col = None
            for col in measurement_data["process_output"].columns:
                if col not in ["x", "y", "z"]:
                    output_col = col
                    break

            if output_col and output_col in config.process_variables:
                # Remove output from process variables
                actual_process_vars = [v for v in config.process_variables if v != output_col]
                if not actual_process_vars:
                    # If all variables were used, use first two as inputs
                    actual_process_vars = (
                        config.process_variables[:-1] if len(config.process_variables) > 1 else config.process_variables
                    )
                X = process_data[actual_process_vars].values
                # Use the output column directly
                y = measurement_data["process_output"][output_col].values
            else:
                X = process_data[config.process_variables].values
                y = self._extract_measurement_outputs(
                    measurement_data,
                    config.measurement_variables,
                    target_length=len(process_data),
                )
        else:
            X = process_data[config.process_variables].values
            # Extract measurement outputs - need to align with process data length
            y = self._extract_measurement_outputs(
                measurement_data,
                config.measurement_variables,
                target_length=len(process_data),
            )

        if len(y) == 0:
            raise ValueError(f"Could not extract measurement outputs. Measurement data: {list(measurement_data.keys())}")

        # Align X and y to same length
        min_length = min(len(X), len(y))
        X = X[:min_length]
        y = y[:min_length]

        # Determine which variables to use for analysis
        if "process_output" in measurement_data:
            # We adjusted process_variables above
            output_col = None
            for col in measurement_data["process_output"].columns:
                if col not in ["x", "y", "z"]:
                    output_col = col
                    break
            if output_col and output_col in config.process_variables:
                analysis_variables = [v for v in config.process_variables if v != output_col]
                if not analysis_variables:
                    analysis_variables = (
                        config.process_variables[:-1] if len(config.process_variables) > 1 else config.process_variables
                    )
            else:
                analysis_variables = config.process_variables
        else:
            analysis_variables = config.process_variables

        # Check if method is supported (but only after we've confirmed we have data)
        if config.method not in ["sobol", "morris"]:
            raise ValueError(f"Unsupported method: {config.method}")

        # Check SALib availability - but only after we've confirmed we have valid data
        # The "No measurement data found" error should have been raised above if fallback failed
        if config.method == "sobol" and not self.global_analyzer:
            raise ValueError("SALib not available for Sobol analysis")
        if config.method == "morris" and not self.global_analyzer:
            raise ValueError("SALib not available for Morris analysis")

        # Get parameter bounds from data
        parameter_bounds = {var: (process_data[var].min(), process_data[var].max()) for var in analysis_variables}

        # Perform analysis based on method
        if config.method == "sobol" and self.global_analyzer:
            result = self.global_analyzer.analyze_sobol(
                model_function=self._create_model_function(X, y),
                parameter_bounds=parameter_bounds,
                parameter_names=analysis_variables,
            )
        elif config.method == "morris" and self.global_analyzer:
            result = self.global_analyzer.analyze_morris(
                model_function=self._create_model_function(X, y),
                parameter_bounds=parameter_bounds,
                parameter_names=analysis_variables,
            )
        else:
            raise ValueError(f"Unsupported method: {config.method}")

        return {
            "model_id": model_id,
            "method": config.method,
            "result": result,
            "parameter_bounds": parameter_bounds,
            "sample_size": len(process_data),
            "timestamp": datetime.now(),
        }

    def _build_surrogate_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Build a surrogate model from input-output data.

        Args:
            X: Input features
            y: Output values

        Returns:
            Surrogate model object
        """
        try:
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            return model
        except ImportError:
            # Fallback to simple linear model
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(X, y)
            return model

    def _extract_measurement_outputs(
        self,
        measurement_data: Dict[str, pd.DataFrame],
        variables: Optional[List[str]],
        target_length: Optional[int] = None,
    ) -> np.ndarray:
        """Extract measurement outputs from measurement data."""
        if not measurement_data:
            return np.array([])

        # Combine measurement data
        outputs = []
        for source, df in measurement_data.items():
            if df.empty:
                continue

            if variables:
                # Check if variable is a source name (e.g., 'ispm', 'ct')
                if source in variables:
                    # Use numeric columns from this source
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    # Prefer common measurement columns
                    preferred_cols = [
                        "temperature",
                        "peak_temperature",
                        "density",
                        "porosity",
                    ]
                    for col in preferred_cols:
                        if col in numeric_cols and len(df[col].values) > 0:
                            outputs.append(df[col].values)
                            break
                    else:
                        # If no preferred column, use first numeric column
                        if len(numeric_cols) > 0:
                            first_col = numeric_cols[0]
                            if len(df[first_col].values) > 0:
                                outputs.append(df[first_col].values)
                else:
                    # Try to find variables in the dataframe
                    for var in variables:
                        if var in df.columns and len(df[var].values) > 0:
                            outputs.append(df[var].values)
            else:
                # Use all numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                # Prefer common measurement columns
                preferred_cols = [
                    "temperature",
                    "peak_temperature",
                    "density",
                    "porosity",
                ]
                for col in preferred_cols:
                    if col in numeric_cols and len(df[col].values) > 0:
                        outputs.append(df[col].values)
                        break
                else:
                    # If no preferred column, use first numeric column
                    if len(numeric_cols) > 0:
                        first_col = numeric_cols[0]
                        if len(df[first_col].values) > 0:
                            outputs.append(df[first_col].values)

        if not outputs:
            return np.array([])

        # Find the minimum length to align all outputs
        min_length = min(len(arr) for arr in outputs)
        if target_length:
            min_length = min(min_length, target_length)

        # Truncate all to same length and combine
        aligned_outputs = [arr[:min_length] for arr in outputs]

        # Take mean across all measurement sources as a single output
        if aligned_outputs:
            return np.mean(aligned_outputs, axis=0)

        return np.array([])

    def _create_model_function(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """Create model function from data."""
        from sklearn.ensemble import RandomForestRegressor

        # Train a surrogate model and store it
        self._surrogate_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        self._surrogate_model.fit(X, y)

        # Create a simple function that references the stored model
        # This will work with sequential processing (parallel disabled)
        def model_function(params):
            if self._surrogate_model is None:
                raise ValueError("Surrogate model not trained")
            return self._surrogate_model.predict(params.reshape(1, -1))[0]

        return model_function
