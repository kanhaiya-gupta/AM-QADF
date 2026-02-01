"""
Grid Naming and ID Generation Module

Provides centralized functions for generating consistent grid names and IDs
following the AM-QADF naming convention.

Naming Format: {source}_{grid_type}_{resolution}_{stage}_{timestamp}
"""

from datetime import datetime
from typing import Optional, Dict, Tuple, Any
from enum import Enum
import re


class GridSource(Enum):
    """Data source types for grids."""
    LASER = "laser"
    CT = "ct"
    ISPM = "ispm"
    HATCHING = "hatching"
    FUSED = "fused"


class GridType(Enum):
    """Grid type options."""
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    MULTIRES = "multires"


class GridStage(Enum):
    """Processing stages for grids."""
    EMPTY = "empty"
    MAPPED = "mapped"
    ALIGNED = "aligned"
    CORRECTED = "corrected"
    PROCESSED = "processed"
    FUSED = "fused"


class GridNaming:
    """
    Centralized grid naming and ID generation.
    
    Follows the naming convention:
    {source}_{grid_type}_{resolution}_{stage}_{timestamp}
    """

    @staticmethod
    def format_resolution(resolution: float) -> str:
        """
        Format resolution value for use in grid name.
        
        Converts resolution in mm to integer representation (× 100).
        Example: 0.5 mm → "50", 1.0 mm → "100"
        
        Args:
            resolution: Resolution in mm
            
        Returns:
            Formatted resolution string
        """
        return str(int(resolution * 100))

    @staticmethod
    def parse_resolution(resolution_str: str) -> float:
        """
        Parse resolution string back to float value.
        
        Converts integer representation back to mm.
        Example: "50" → 0.5 mm, "100" → 1.0 mm
        
        Args:
            resolution_str: Resolution string from grid name
            
        Returns:
            Resolution in mm
        """
        return int(resolution_str) / 100.0

    @staticmethod
    def generate_grid_name(
        source: str,
        grid_type: str,
        resolution: float,
        stage: str,
        timestamp: Optional[str] = None,
        fusion_strategy: Optional[str] = None,
    ) -> str:
        """
        Generate grid name following the naming convention.
        
        Format: {source}_{grid_type}_{resolution}_{stage}_{timestamp}
        
        Args:
            source: Data source (laser, ct, ispm, hatching, fused)
            grid_type: Grid type (uniform, adaptive, multires)
            resolution: Resolution in mm
            stage: Processing stage (empty, mapped, aligned, corrected, fused)
            timestamp: Optional timestamp (YYYYMMDD_HHMMSS). If None, current time is used.
            fusion_strategy: Optional fusion strategy name (for fused grids)
            
        Returns:
            Grid name string
            
        Examples:
            >>> GridNaming.generate_grid_name("laser", "uniform", 0.5, "mapped")
            'laser_uniform_50_mapped_20250105_120000'
            
            >>> GridNaming.generate_grid_name("fused", "uniform", 0.5, "fused", fusion_strategy="weighted_avg")
            'fused_weighted_avg_50_fused_20250105_120000'
        """
        # Validate inputs
        if source not in [s.value for s in GridSource]:
            raise ValueError(f"Invalid source: {source}. Must be one of {[s.value for s in GridSource]}")
        
        if grid_type not in [t.value for t in GridType]:
            raise ValueError(f"Invalid grid_type: {grid_type}. Must be one of {[t.value for t in GridType]}")
        
        if stage not in [s.value for s in GridStage]:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {[s.value for s in GridStage]}")
        
        # Format resolution
        resolution_str = GridNaming.format_resolution(resolution)
        
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Special handling for fused grids
        if source == GridSource.FUSED.value:
            if fusion_strategy:
                # Format: fused_{grid_type}_{resolution}_{strategy}_{stage}_{timestamp}
                return f"fused_{grid_type}_{resolution_str}_{fusion_strategy}_{stage}_{timestamp}"
            else:
                # Format: fused_{grid_type}_{resolution}_{stage}_{timestamp}
                return f"fused_{grid_type}_{resolution_str}_{stage}_{timestamp}"
        
        # Standard format: {source}_{grid_type}_{resolution}_{stage}_{timestamp}
        return f"{source}_{grid_type}_{resolution_str}_{stage}_{timestamp}"

    @staticmethod
    def parse_grid_name(grid_name: str) -> Dict[str, str]:
        """
        Parse grid name to extract components.
        
        Args:
            grid_name: Grid name string
            
        Returns:
            Dictionary with keys: source, grid_type, resolution, stage, timestamp, fusion_strategy (if applicable)
            
        Examples:
            >>> GridNaming.parse_grid_name("laser_uniform_50_mapped_20250105_120000")
            {'source': 'laser', 'grid_type': 'uniform', 'resolution': '50', 
             'stage': 'mapped', 'timestamp': '20250105_120000'}
        """
        # Pattern for standard grid names
        # {source}_{grid_type}_{resolution}_{stage}_{timestamp}
        standard_pattern = r"^([a-z]+)_(uniform|adaptive|multires)_(\d+)_([a-z]+)_(\d{8}_\d{6})$"
        
        # Pattern for fused grids with strategy
        # Format: fused_{grid_type}_{resolution}_{strategy}_{stage}_{timestamp}
        fused_pattern = r"^fused_(uniform|adaptive|multires)_(\d+)_([a-z_]+)_([a-z]+)_(\d{8}_\d{6})$"
        
        # Pattern for fused grids without strategy
        # Format: fused_{grid_type}_{resolution}_{stage}_{timestamp}
        fused_no_strategy_pattern = r"^fused_(uniform|adaptive|multires)_(\d+)_([a-z]+)_(\d{8}_\d{6})$"
        
        # Try standard pattern first
        match = re.match(standard_pattern, grid_name)
        if match:
            source, grid_type, resolution, stage, timestamp = match.groups()
            return {
                "source": source,
                "grid_type": grid_type,
                "resolution": resolution,
                "stage": stage,
                "timestamp": timestamp,
            }
        
        # Try fused pattern with strategy
        match = re.match(fused_pattern, grid_name)
        if match:
            grid_type, resolution, fusion_strategy, stage, timestamp = match.groups()
            return {
                "source": "fused",
                "grid_type": grid_type,
                "resolution": resolution,
                "stage": stage,
                "timestamp": timestamp,
                "fusion_strategy": fusion_strategy,
            }
        
        # Try fused pattern without strategy
        match = re.match(fused_no_strategy_pattern, grid_name)
        if match:
            grid_type, resolution, stage, timestamp = match.groups()
            return {
                "source": "fused",
                "grid_type": grid_type,
                "resolution": resolution,
                "stage": stage,
                "timestamp": timestamp,
            }
        
        raise ValueError(f"Invalid grid name format: {grid_name}")

    @staticmethod
    def generate_grid_id(
        model_id: str,
        grid_name: str,
    ) -> str:
        """
        Generate unique grid ID from model ID and grid name.
        
        Format: {model_id}_{grid_name}
        
        Args:
            model_id: Model identifier
            grid_name: Grid name
            
        Returns:
            Grid ID string
        """
        # Use first 8 characters of model_id for brevity
        model_short = model_id[:8] if len(model_id) > 8 else model_id
        return f"{model_short}_{grid_name}"

    @staticmethod
    def generate_empty_grid_name(
        source: str,
        grid_type: str,
        resolution: float,
    ) -> str:
        """
        Generate name for empty grid (Step 1).
        
        Args:
            source: Data source
            grid_type: Grid type
            resolution: Resolution in mm
            
        Returns:
            Grid name for empty grid
        """
        return GridNaming.generate_grid_name(
            source=source,
            grid_type=grid_type,
            resolution=resolution,
            stage=GridStage.EMPTY.value,
        )

    @staticmethod
    def generate_mapped_grid_name(
        source: str,
        grid_type: str,
        resolution: float,
        timestamp: Optional[str] = None,
    ) -> str:
        """
        Generate name for mapped grid (Step 2).
        
        Args:
            source: Data source
            grid_type: Grid type
            resolution: Resolution in mm
            timestamp: Optional timestamp
            
        Returns:
            Grid name for mapped grid
        """
        return GridNaming.generate_grid_name(
            source=source,
            grid_type=grid_type,
            resolution=resolution,
            stage=GridStage.MAPPED.value,
            timestamp=timestamp,
        )

    @staticmethod
    def generate_mapped_grid_name_with_method(
        source: str,
        grid_type: str,
        resolution: float,
        method: str,
        method_params: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> str:
        """
        Generate name for mapped grid with method and parameters (Step 2).
        
        Format: {source}_{grid_type}_{resolution}_mapped_{method}_{params}_{timestamp}
        
        Args:
            source: Data source
            grid_type: Grid type
            resolution: Resolution in mm
            method: Mapping method name (e.g., 'nearest', 'linear', 'idw', 'gaussian_kde', 'rbf')
            method_params: Optional dictionary of method parameters to include in name
                Common parameters:
                - 'k' or 'k_neighbors': Number of neighbors
                - 'power' or 'p': Power parameter (for IDW)
                - 'bandwidth' or 'bw': Bandwidth (for KDE)
                - 'kernel': Kernel type (for RBF)
                - 'radius' or 'r': Radius limit
                - 'smoothing' or 's': Smoothing parameter
                - 'epsilon' or 'eps': Epsilon parameter
                - 'adaptive': Boolean for adaptive methods
            timestamp: Optional timestamp (YYYYMMDD_HHMMSS). If None, current time is used.
            
        Returns:
            Grid name for mapped grid with method and parameters
            
        Examples:
            >>> GridNaming.generate_mapped_grid_name_with_method("laser", "uniform", 1.0, "nearest")
            'laser_uniform_100_mapped_nearest_20250114_163000'
            
            >>> GridNaming.generate_mapped_grid_name_with_method("laser", "uniform", 1.0, "linear", {"k": 8})
            'laser_uniform_100_mapped_linear_k8_20250114_163000'
            
            >>> GridNaming.generate_mapped_grid_name_with_method("ct", "uniform", 0.5, "idw", {"power": 2.0, "k": 8})
            'ct_uniform_50_mapped_idw_p2.0_k8_20250114_163000'
        """
        # Generate base mapped grid name
        base_name = GridNaming.generate_mapped_grid_name(
            source=source,
            grid_type=grid_type,
            resolution=resolution,
            timestamp=None  # We'll add our own timestamp at the end
        )
        
        # Extract base without timestamp: {source}_{grid_type}_{resolution}_mapped
        # Timestamp format is YYYYMMDD_HHMMSS (has underscore), so remove last 2 parts
        base_without_timestamp = base_name.rsplit('_', 2)[0]
        
        # Build parameter string
        param_parts = []
        if method_params:
            # Format common parameters
            if 'k' in method_params or 'k_neighbors' in method_params:
                k_val = method_params.get('k') or method_params.get('k_neighbors')
                param_parts.append(f"k{int(k_val)}")
            
            if 'power' in method_params or 'p' in method_params:
                p_val = method_params.get('power') or method_params.get('p')
                param_parts.append(f"p{p_val}")
            
            if 'bandwidth' in method_params or 'bw' in method_params:
                bw_val = method_params.get('bandwidth') or method_params.get('bw')
                param_parts.append(f"bw{bw_val}")
            
            if 'kernel' in method_params:
                kernel_val = method_params.get('kernel')
                # Shorten common kernel names
                kernel_map = {
                    'gaussian': 'gauss',
                    'multiquadric': 'mq',
                    'inverse_multiquadric': 'imq',
                    'thin_plate_spline': 'tps'
                }
                kernel_str = kernel_map.get(kernel_val, kernel_val)
                param_parts.append(kernel_str)
            
            if 'radius' in method_params or 'r' in method_params:
                r_val = method_params.get('radius') or method_params.get('r')
                if r_val is not None:
                    param_parts.append(f"r{r_val}")
            
            if 'smoothing' in method_params or 's' in method_params:
                s_val = method_params.get('smoothing') or method_params.get('s')
                if s_val is not None and s_val != 0:
                    param_parts.append(f"s{s_val}")
            
            if 'epsilon' in method_params or 'eps' in method_params:
                eps_val = method_params.get('epsilon') or method_params.get('eps')
                if eps_val is not None:
                    param_parts.append(f"eps{eps_val}")
            
            if method_params.get('adaptive', False):
                param_parts.append("adaptive")
        
        # Build parameter string
        param_str = ""
        if param_parts:
            param_str = "_" + "_".join(param_parts)
        
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Final grid name: {source}_{grid_type}_{resolution}_mapped_{method}{param_str}_{timestamp}
        return f"{base_without_timestamp}_{method}{param_str}_{timestamp}"

    @staticmethod
    def generate_aligned_grid_name(
        source: str,
        grid_type: str,
        resolution: float,
        alignment_mode: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> str:
        """
        Generate name for aligned grid (Step 3).
        
        Format: {source}_{grid_type}_{resolution}_aligned_{mode}_{timestamp}
        If mode is None: {source}_{grid_type}_{resolution}_aligned_{timestamp}
        
        Args:
            source: Data source
            grid_type: Grid type
            resolution: Resolution in mm
            alignment_mode: Optional alignment mode ('temporal', 'spatial', or 'both')
            timestamp: Optional timestamp (YYYYMMDD_HHMMSS). If None, current time is used.
            
        Returns:
            Grid name for aligned grid
            
        Examples:
            >>> GridNaming.generate_aligned_grid_name("laser", "uniform", 1.0)
            'laser_uniform_100_aligned_20250114_163000'
            
            >>> GridNaming.generate_aligned_grid_name("laser", "uniform", 1.0, "both")
            'laser_uniform_100_aligned_both_20250114_163000'
        """
        # Generate base aligned grid name
        base_name = GridNaming.generate_grid_name(
            source=source,
            grid_type=grid_type,
            resolution=resolution,
            stage=GridStage.ALIGNED.value,
            timestamp=None  # We'll add our own timestamp at the end
        )
        
        # Extract base without timestamp: {source}_{grid_type}_{resolution}_aligned
        # Timestamp format is YYYYMMDD_HHMMSS (has underscore), so remove last 2 parts
        base_without_timestamp = base_name.rsplit('_', 2)[0]
        
        # Add mode if provided
        if alignment_mode:
            # Validate alignment mode
            valid_modes = ['temporal', 'spatial', 'both']
            if alignment_mode not in valid_modes:
                raise ValueError(f"Invalid alignment_mode: {alignment_mode}. Must be one of {valid_modes}")
            base_without_timestamp = f"{base_without_timestamp}_{alignment_mode}"
        
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Final grid name: {source}_{grid_type}_{resolution}_aligned_{mode?}_{timestamp}
        return f"{base_without_timestamp}_{timestamp}"

    @staticmethod
    def generate_corrected_grid_name(
        source: str,
        grid_type: str,
        resolution: float,
        correction_type: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> str:
        """
        Generate name for corrected grid (Step 4).
        
        Format: {source}_{grid_type}_{resolution}_corrected_{type}_{timestamp}
        If type is None: {source}_{grid_type}_{resolution}_corrected_{timestamp}
        
        Args:
            source: Data source
            grid_type: Grid type
            resolution: Resolution in mm
            correction_type: Optional correction type ('scaling', 'rotation', 'warping', or 'combined')
            timestamp: Optional timestamp (YYYYMMDD_HHMMSS). If None, current time is used.
            
        Returns:
            Grid name for corrected grid
            
        Examples:
            >>> GridNaming.generate_corrected_grid_name("laser", "uniform", 1.0)
            'laser_uniform_100_corrected_20250114_163000'
            
            >>> GridNaming.generate_corrected_grid_name("laser", "uniform", 1.0, "scaling")
            'laser_uniform_100_corrected_scaling_20250114_163000'
        """
        # Generate base corrected grid name
        base_name = GridNaming.generate_grid_name(
            source=source,
            grid_type=grid_type,
            resolution=resolution,
            stage=GridStage.CORRECTED.value,
            timestamp=None  # We'll add our own timestamp at the end
        )
        
        # Extract base without timestamp: {source}_{grid_type}_{resolution}_corrected
        # Timestamp format is YYYYMMDD_HHMMSS (has underscore), so remove last 2 parts
        base_without_timestamp = base_name.rsplit('_', 2)[0]
        
        # Add correction type if provided
        if correction_type:
            # Validate correction type
            valid_types = ['scaling', 'rotation', 'warping', 'combined']
            if correction_type not in valid_types:
                raise ValueError(f"Invalid correction_type: {correction_type}. Must be one of {valid_types}")
            base_without_timestamp = f"{base_without_timestamp}_{correction_type}"
        
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Final grid name: {source}_{grid_type}_{resolution}_corrected_{type?}_{timestamp}
        return f"{base_without_timestamp}_{timestamp}"

    @staticmethod
    def generate_processed_grid_name(
        source: str,
        grid_type: str,
        resolution: float,
        processing_type: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> str:
        """
        Generate name for processed grid (signal processing).
        
        Format: {source}_{grid_type}_{resolution}_processed_{type}_{timestamp}
        If type is None: {source}_{grid_type}_{resolution}_processed_{timestamp}
        
        Args:
            source: Data source
            grid_type: Grid type
            resolution: Resolution in mm
            processing_type: Optional processing type ('smoothing', 'noise_reduction', 'derived', or 'combined')
            timestamp: Optional timestamp (YYYYMMDD_HHMMSS). If None, current time is used.
            
        Returns:
            Grid name for processed grid
            
        Examples:
            >>> GridNaming.generate_processed_grid_name("laser", "uniform", 1.0)
            'laser_uniform_100_processed_20250114_163000'
            
            >>> GridNaming.generate_processed_grid_name("laser", "uniform", 1.0, "smoothing")
            'laser_uniform_100_processed_smoothing_20250114_163000'
        """
        # Generate base processed grid name
        base_name = GridNaming.generate_grid_name(
            source=source,
            grid_type=grid_type,
            resolution=resolution,
            stage=GridStage.PROCESSED.value,
            timestamp=None  # We'll add our own timestamp at the end
        )
        
        # Extract base without timestamp: {source}_{grid_type}_{resolution}_processed
        # Timestamp format is YYYYMMDD_HHMMSS (has underscore), so remove last 2 parts
        base_without_timestamp = base_name.rsplit('_', 2)[0]
        
        # Add processing type if provided
        if processing_type:
            # Validate processing type
            valid_types = ['smoothing', 'noise_reduction', 'derived', 'combined']
            if processing_type not in valid_types:
                raise ValueError(f"Invalid processing_type: {processing_type}. Must be one of {valid_types}")
            base_without_timestamp = f"{base_without_timestamp}_{processing_type}"
        
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Final grid name: {source}_{grid_type}_{resolution}_processed_{type?}_{timestamp}
        return f"{base_without_timestamp}_{timestamp}"

    @staticmethod
    def generate_fused_grid_name(
        fusion_strategy: str,
        resolution: float,
        grid_type: str = GridType.UNIFORM.value,
        timestamp: Optional[str] = None,
    ) -> str:
        """
        Generate name for fused grid (Step 5).
        
        Format: fused_{grid_type}_{resolution}_{strategy}_{stage}_{timestamp}
        
        Args:
            fusion_strategy: Fusion strategy name (e.g., "weighted_avg", "quality_based")
            resolution: Resolution in mm
            grid_type: Grid type (uniform, adaptive, multires). Defaults to 'uniform'
            timestamp: Optional timestamp
            
        Returns:
            Grid name for fused grid
            
        Examples:
            >>> GridNaming.generate_fused_grid_name("weighted_avg", 0.5, "uniform")
            'fused_uniform_50_weighted_avg_fused_20250114_163000'
            
            >>> GridNaming.generate_fused_grid_name("quality_based", 1.0, "adaptive")
            'fused_adaptive_100_quality_based_fused_20250114_163000'
        """
        return GridNaming.generate_grid_name(
            source=GridSource.FUSED.value,
            grid_type=grid_type,
            resolution=resolution,
            stage=GridStage.FUSED.value,
            timestamp=timestamp,
            fusion_strategy=fusion_strategy,
        )

    @staticmethod
    def validate_grid_name(grid_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate grid name format.
        
        Args:
            grid_name: Grid name to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            GridNaming.parse_grid_name(grid_name)
            return True, None
        except ValueError as e:
            return False, str(e)

    @staticmethod
    def get_stage_from_name(grid_name: str) -> Optional[str]:
        """
        Extract stage from grid name.
        
        Args:
            grid_name: Grid name
            
        Returns:
            Stage string or None if parsing fails
        """
        try:
            parsed = GridNaming.parse_grid_name(grid_name)
            return parsed.get("stage")
        except ValueError:
            return None

    @staticmethod
    def get_source_from_name(grid_name: str) -> Optional[str]:
        """
        Extract source from grid name.
        
        Args:
            grid_name: Grid name
            
        Returns:
            Source string or None if parsing fails
        """
        try:
            parsed = GridNaming.parse_grid_name(grid_name)
            return parsed.get("source")
        except ValueError:
            return None

    @staticmethod
    def get_resolution_from_name(grid_name: str) -> Optional[float]:
        """
        Extract resolution from grid name.
        
        Args:
            grid_name: Grid name
            
        Returns:
            Resolution in mm or None if parsing fails
        """
        try:
            parsed = GridNaming.parse_grid_name(grid_name)
            resolution_str = parsed.get("resolution")
            if resolution_str:
                return GridNaming.parse_resolution(resolution_str)
            return None
        except (ValueError, KeyError):
            return None
