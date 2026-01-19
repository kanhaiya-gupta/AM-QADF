"""
Multi-Source Data Fusion Module

Comprehensive fusion module for combining multiple source grids into a unified fused grid.
Preserves all original signals, creates source-specific fused signals, and generates
multi-source fused signals for matching signal types.

This module implements the complete fusion structure with:
- Original signals (source-prefixed, preserved as-is)
- Source-specific fused signals (all with _fused suffix)
- Multi-source fused signals (fused from matching signal types)
- Comprehensive metadata for traceability and analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from enum import Enum

from .data_fusion import FusionStrategy, DataFusion
from .fusion_methods import get_fusion_method


class MultiSourceFusion:
    """
    Comprehensive multi-source data fusion for voxel grids.
    
    Combines signals from multiple sources (laser, ISPM, hatching, CT, etc.)
    into a unified fused grid with complete signal preservation and metadata.
    """
    
    def __init__(
        self,
        default_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
        use_quality_scores: bool = True,
        normalize_weights: bool = True,
    ):
        """
        Initialize multi-source fusion.
        
        Args:
            default_strategy: Default fusion strategy for multi-source fusion
            use_quality_scores: Whether to use quality scores for weighting
            normalize_weights: Whether to normalize source weights
        """
        self.default_strategy = default_strategy
        self.use_quality_scores = use_quality_scores
        self.normalize_weights = normalize_weights
        self.fusion_engine = DataFusion(default_strategy=default_strategy)
    
    def fuse_sources(
        self,
        source_grids: Dict[str, Dict[str, Any]],
        source_weights: Optional[Dict[str, float]] = None,
        quality_scores: Optional[Dict[str, float]] = None,
        fusion_strategy: Optional[FusionStrategy] = None,
        grid_name: Optional[str] = None,
        grid_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fuse multiple source grids into a comprehensive fused grid.
        
        Args:
            source_grids: Dictionary mapping source names to grid data
                Each grid data should contain:
                - 'signal_arrays': Dict[str, np.ndarray] - signal arrays
                - 'metadata': Dict - grid metadata
                - 'grid_id': str - source grid ID
                - 'grid_name': str - source grid name
                - 'quality_score': float - optional quality score
                - 'coverage': float - optional coverage score
            source_weights: Optional weights for each source
            quality_scores: Optional quality scores for each source
            fusion_strategy: Fusion strategy (None = use default)
            grid_name: Name for the fused grid
            grid_id: ID for the fused grid
            
        Returns:
            Complete fused grid dictionary with:
            - signal_arrays: All signals (original + source-specific fused + multi-source fused)
            - metadata: Comprehensive metadata
        """
        if not source_grids:
            raise ValueError("At least one source grid must be provided")
        
        fusion_strategy = fusion_strategy or self.default_strategy
        fusion_timestamp = datetime.now().isoformat()
        
        # Step 1: Collect and prefix all signals with source names
        all_signals = self._collect_and_prefix_signals(source_grids)
        
        # Step 2: Create source-specific fused signals (all with _fused suffix)
        source_specific_fused = self._create_source_specific_fused(
            source_grids, all_signals
        )
        
        # Step 3: Identify signal types and group matching signals
        signal_groups = self._group_signals_by_type(all_signals)
        
        # Step 4: Create multi-source fused signals
        multi_source_fused, fusion_metadata = self._create_multi_source_fused(
            signal_groups,
            source_specific_fused,
            source_weights,
            quality_scores,
            fusion_strategy,
            fusion_timestamp
        )
        
        # Step 5: Calculate signal statistics
        signal_statistics = self._calculate_signal_statistics(
            all_signals, source_specific_fused, multi_source_fused
        )
        
        # Step 6: Build comprehensive metadata
        metadata = self._build_metadata(
            source_grids,
            all_signals,
            source_specific_fused,
            multi_source_fused,
            signal_groups,
            fusion_metadata,
            signal_statistics,
            source_weights,
            quality_scores,
            fusion_strategy,
            fusion_timestamp,
            grid_name,
            grid_id
        )
        
        # Step 7: Combine all signals
        signal_arrays = {}
        signal_arrays.update(all_signals)  # Original signals
        signal_arrays.update(source_specific_fused)  # Source-specific fused
        signal_arrays.update(multi_source_fused)  # Multi-source fused
        
        return {
            'signal_arrays': signal_arrays,
            'metadata': metadata
        }
    
    def _collect_and_prefix_signals(
        self, source_grids: Dict[str, Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """
        Collect all signals from source grids and prefix with source names.
        
        Args:
            source_grids: Dictionary of source grids
            
        Returns:
            Dictionary mapping prefixed signal names to arrays
        """
        all_signals = {}
        
        for source_name, grid_data in source_grids.items():
            signal_arrays = grid_data.get('signal_arrays', {})
            
            for signal_name, signal_array in signal_arrays.items():
                # Prefix signal name with source if not already prefixed
                if not signal_name.startswith(f"{source_name}_"):
                    prefixed_name = f"{source_name}_{signal_name}"
                else:
                    prefixed_name = signal_name
                
                all_signals[prefixed_name] = signal_array
        
        return all_signals
    
    def _create_source_specific_fused(
        self,
        source_grids: Dict[str, Dict[str, Any]],
        all_signals: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Create source-specific fused signals (all with _fused suffix).
        
        For now, these are just copies of the original signals with _fused suffix.
        In the future, this could apply source-specific processing.
        
        Args:
            source_grids: Dictionary of source grids
            all_signals: Original prefixed signals
            
        Returns:
            Dictionary of source-specific fused signals
        """
        source_specific_fused = {}
        
        for signal_name, signal_array in all_signals.items():
            fused_name = f"{signal_name}_fused"
            # For now, just copy the signal
            # In future, could apply source-specific processing here
            source_specific_fused[fused_name] = signal_array.copy()
        
        return source_specific_fused
    
    def _group_signals_by_type(
        self, all_signals: Dict[str, np.ndarray]
    ) -> Dict[str, List[str]]:
        """
        Group signals by their base type (removing source prefix and _fused suffix).
        
        Args:
            all_signals: Dictionary of prefixed signals
            
        Returns:
            Dictionary mapping signal types to lists of signal names
        """
        signal_groups = {}
        
        for signal_name in all_signals.keys():
            # Remove source prefix and _fused suffix to get base type
            base_type = self._extract_signal_type(signal_name)
            
            if base_type not in signal_groups:
                signal_groups[base_type] = []
            signal_groups[base_type].append(signal_name)
        
        return signal_groups
    
    def _extract_signal_type(self, signal_name: str) -> str:
        """
        Extract base signal type from signal name.
        
        Examples:
            'laser_power' -> 'power'
            'ispm_temperature' -> 'temperature'
            'hatching_temperature_fused' -> 'temperature'
            'temperature_fused' -> 'temperature'
        
        Args:
            signal_name: Full signal name
            
        Returns:
            Base signal type
        """
        # Remove _fused suffix if present
        if signal_name.endswith('_fused'):
            signal_name = signal_name[:-6]
        
        # Remove source prefix (common sources)
        sources = ['laser', 'ispm', 'hatching', 'ct', 'thermal_camera']
        for source in sources:
            if signal_name.startswith(f"{source}_"):
                return signal_name[len(source) + 1:]
        
        # If no source prefix, return as-is
        return signal_name
    
    def _create_multi_source_fused(
        self,
        signal_groups: Dict[str, List[str]],
        source_specific_fused: Dict[str, np.ndarray],
        source_weights: Optional[Dict[str, float]],
        quality_scores: Optional[Dict[str, float]],
        fusion_strategy: FusionStrategy,
        fusion_timestamp: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """
        Create multi-source fused signals for matching signal types.
        
        Args:
            signal_groups: Dictionary mapping signal types to signal names
            source_specific_fused: Source-specific fused signals
            source_weights: Optional source weights
            quality_scores: Optional quality scores
            fusion_strategy: Fusion strategy
            fusion_timestamp: Timestamp for fusion
            
        Returns:
            Tuple of (multi_source_fused_signals, fusion_metadata)
        """
        multi_source_fused = {}
        fusion_metadata = {}
        
        for signal_type, signal_names in signal_groups.items():
            # Only fuse if multiple sources have this signal type
            if len(signal_names) < 2:
                continue
            
            # Get corresponding _fused signal names
            fused_signal_names = [f"{name}_fused" for name in signal_names]
            
            # Check if all fused signals exist
            available_fused = [
                name for name in fused_signal_names
                if name in source_specific_fused
            ]
            
            if len(available_fused) < 2:
                continue
            
            # Collect signals to fuse
            signals_to_fuse = {}
            sources_for_fusion = []
            
            for fused_name in available_fused:
                # Extract source from signal name
                source = self._extract_source_from_signal_name(fused_name)
                signals_to_fuse[source] = source_specific_fused[fused_name]
                sources_for_fusion.append(source)
            
            # Prepare weights and quality scores
            weights = None
            if source_weights:
                weights = {
                    source: source_weights.get(source, 1.0)
                    for source in sources_for_fusion
                }
            
            quality = None
            if quality_scores:
                quality = {
                    source: quality_scores.get(source, 0.8)
                    for source in sources_for_fusion
                }
            
            # Normalize weights if requested
            if weights and self.normalize_weights:
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v / total_weight for k, v in weights.items()}
            
            # Fuse signals
            fused_signal = self._fuse_signal_arrays(
                signals_to_fuse, fusion_strategy, weights, quality
            )
            
            # Store fused signal
            fused_signal_name = f"{signal_type}_fused"
            multi_source_fused[fused_signal_name] = fused_signal
            
            # Calculate fusion quality metrics
            fusion_score = self._calculate_single_fusion_score(
                signals_to_fuse, fused_signal, quality
            )
            consistency_score = self._calculate_consistency_score(
                signals_to_fuse, fused_signal
            )
            
            # Store fusion metadata
            fusion_metadata[fused_signal_name] = {
                'signal_type': signal_type,
                'sources': sources_for_fusion,
                'source_signals': available_fused,
                'strategy': fusion_strategy.value,
                'weights': weights or {},
                'quality_scores': quality or {},
                'fusion_timestamp': fusion_timestamp,
                'fusion_score': fusion_score,
                'consistency_score': consistency_score
            }
        
        return multi_source_fused, fusion_metadata
    
    def _extract_source_from_signal_name(self, signal_name: str) -> str:
        """
        Extract source name from signal name.
        
        Args:
            signal_name: Signal name (e.g., 'laser_power_fused')
            
        Returns:
            Source name (e.g., 'laser')
        """
        # Remove _fused suffix
        if signal_name.endswith('_fused'):
            signal_name = signal_name[:-6]
        
        # Extract source prefix
        parts = signal_name.split('_', 1)
        if len(parts) > 1:
            return parts[0]
        return 'unknown'
    
    def _fuse_signal_arrays(
        self,
        signals: Dict[str, np.ndarray],
        strategy: FusionStrategy,
        weights: Optional[Dict[str, float]] = None,
        quality_scores: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Fuse multiple signal arrays using specified strategy.
        
        Args:
            signals: Dictionary mapping source names to signal arrays
            strategy: Fusion strategy
            weights: Optional weights for each source
            quality_scores: Optional quality scores for each source
            
        Returns:
            Fused signal array
        """
        if not signals:
            raise ValueError("No signals to fuse")
        
        # Get fusion method
        fusion_method = get_fusion_method(strategy)
        
        # Convert weights to list format if needed
        weight_dict = None
        if weights:
            weight_dict = weights
        
        # Fuse signals
        fused = fusion_method.fuse(
            signals=signals,
            weights=weight_dict,
            quality_scores=quality_scores
        )
        
        return fused
    
    def _calculate_signal_statistics(
        self,
        original_signals: Dict[str, np.ndarray],
        source_specific_fused: Dict[str, np.ndarray],
        multi_source_fused: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics for all signals.
        
        Args:
            original_signals: Original signals
            source_specific_fused: Source-specific fused signals
            multi_source_fused: Multi-source fused signals
            
        Returns:
            Dictionary mapping signal names to statistics
        """
        statistics = {}
        all_signals = {**original_signals, **source_specific_fused, **multi_source_fused}
        
        for signal_name, signal_array in all_signals.items():
            if not isinstance(signal_array, np.ndarray):
                continue
            
            # Calculate statistics
            valid_mask = ~np.isnan(signal_array) if np.issubdtype(signal_array.dtype, np.floating) else (signal_array != 0)
            valid_data = signal_array[valid_mask]
            
            stats = {
                'shape': list(signal_array.shape),
                'dtype': str(signal_array.dtype),
                'min': float(np.min(valid_data)) if len(valid_data) > 0 else 0.0,
                'max': float(np.max(valid_data)) if len(valid_data) > 0 else 0.0,
                'mean': float(np.mean(valid_data)) if len(valid_data) > 0 else 0.0,
                'std': float(np.std(valid_data)) if len(valid_data) > 0 else 0.0,
                'non_zero_count': int(np.count_nonzero(valid_mask)),
                'total_count': int(signal_array.size),
                'coverage': float(np.count_nonzero(valid_mask) / signal_array.size) if signal_array.size > 0 else 0.0
            }
            
            statistics[signal_name] = stats
        
        return statistics
    
    def _build_metadata(
        self,
        source_grids: Dict[str, Dict[str, Any]],
        original_signals: Dict[str, np.ndarray],
        source_specific_fused: Dict[str, np.ndarray],
        multi_source_fused: Dict[str, np.ndarray],
        signal_groups: Dict[str, List[str]],
        fusion_metadata: Dict[str, Dict[str, Any]],
        signal_statistics: Dict[str, Dict[str, Any]],
        source_weights: Optional[Dict[str, float]],
        quality_scores: Optional[Dict[str, float]],
        fusion_strategy: FusionStrategy,
        fusion_timestamp: str,
        grid_name: Optional[str],
        grid_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Build comprehensive metadata for the fused grid.
        
        Args:
            source_grids: Source grids data
            original_signals: Original signals
            source_specific_fused: Source-specific fused signals
            multi_source_fused: Multi-source fused signals
            signal_groups: Signal groups by type
            fusion_metadata: Multi-source fusion metadata
            signal_statistics: Signal statistics
            source_weights: Source weights
            quality_scores: Quality scores
            fusion_strategy: Fusion strategy
            fusion_timestamp: Fusion timestamp
            grid_name: Grid name
            grid_id: Grid ID
            
        Returns:
            Comprehensive metadata dictionary
        """
        # Categorize signals
        original_list = list(original_signals.keys())
        source_specific_list = list(source_specific_fused.keys())
        multi_source_list = list(multi_source_fused.keys())
        
        # Identify unique and shared signals
        unique_signals = []
        shared_signals = []
        
        for signal_type, signal_names in signal_groups.items():
            if len(signal_names) == 1:
                unique_signals.extend(signal_names)
            else:
                shared_signals.append(signal_type)
        
        # Build source mapping
        source_mapping = {}
        for source_name, grid_data in source_grids.items():
            original_sigs = [
                sig for sig in original_list
                if sig.startswith(f"{source_name}_")
            ]
            fused_sigs = [
                sig for sig in source_specific_list
                if sig.startswith(f"{source_name}_")
            ]
            
            source_mapping[source_name] = {
                'original_signals': original_sigs,
                'fused_signals': fused_sigs,
                'grid_id': grid_data.get('grid_id', ''),
                'grid_name': grid_data.get('grid_name', ''),
                'source_type': source_name,
                'quality_score': grid_data.get('quality_score', 0.8),
                'coverage': grid_data.get('coverage', 1.0)
            }
        
        # Get grid properties from first source
        first_source = list(source_grids.values())[0]
        first_metadata = first_source.get('metadata', {})
        
        # Build provenance
        provenance = {
            'created_by': 'multi_source_fusion',
            'creation_method': 'comprehensive_fusion',
            'source_grids': [
                {
                    'source': source_name,
                    'grid_id': grid_data.get('grid_id', ''),
                    'grid_name': grid_data.get('grid_name', ''),
                    'stage': self._extract_stage_from_grid_name(
                        grid_data.get('grid_name', '')
                    ),
                    'signals_contributed': [
                        sig for sig in original_list
                        if sig.startswith(f"{source_name}_")
                    ]
                }
                for source_name, grid_data in source_grids.items()
            ],
            'processing_chain': [
                'signal_mapping',
                'alignment',
                'correction',
                'processing',
                'fusion'
            ]
        }
        
        # Calculate fusion metrics
        fusion_metrics = self._calculate_fusion_metrics(
            source_grids, multi_source_fused, fusion_metadata
        )
        
        # Build metadata
        metadata = {
            # Grid metadata
            'grid_name': grid_name or f"fused_{fusion_timestamp}",
            'grid_id': grid_id or '',
            'grid_shape': list(first_metadata.get('grid_shape', [9, 9, 9])),
            'resolution': first_metadata.get('resolution', 1.0),
            'bbox_min': first_metadata.get('bbox_min', [-50, -50, 0]),
            'bbox_max': first_metadata.get('bbox_max', [50, 50, 100]),
            'creation_timestamp': fusion_timestamp,
            
            # Fusion metadata
            'fusion_applied': True,
            'fusion_timestamp': fusion_timestamp,
            'fusion_strategy': fusion_strategy.value,
            'fusion_sources': list(source_grids.keys()),
            'fusion_method': 'voxel_domain_fusion',
            
            # Signal categorization
            'signal_categories': {
                'original': original_list,
                'source_specific_fused': source_specific_list,
                'multi_source_fused': multi_source_list,
                'unique_signals': unique_signals,
                'shared_signals': shared_signals
            },
            
            # Source mapping
            'source_mapping': source_mapping,
            
            # Multi-source fusion metadata
            'multi_source_fusion': fusion_metadata,
            
            # Signal statistics
            'signal_statistics': signal_statistics,
            
            # Fusion quality metrics
            'fusion_metrics': fusion_metrics,
            
            # Configuration metadata
            'configuration_metadata': {
                'fusion_applied': True,
                'fusion_strategy': fusion_strategy.value,
                'normalize_weights': self.normalize_weights,
                'use_quality_scores': self.use_quality_scores,
                'source_weights': source_weights or {},
                'grid_parameters': {
                    'resolution': first_metadata.get('resolution', 1.0),
                    'bbox_min': first_metadata.get('bbox_min', [-50, -50, 0]),
                    'bbox_max': first_metadata.get('bbox_max', [50, 50, 100])
                }
            },
            
            # Provenance
            'provenance': provenance
        }
        
        return metadata
    
    def _extract_stage_from_grid_name(self, grid_name: str) -> str:
        """Extract processing stage from grid name."""
        if '_fused_' in grid_name:
            return 'fused'
        elif '_processed_' in grid_name:
            return 'processed'
        elif '_corrected_' in grid_name:
            return 'corrected'
        elif '_aligned_' in grid_name:
            return 'aligned'
        elif '_mapped_' in grid_name:
            return 'mapped'
        else:
            return 'raw'
    
    def _calculate_single_fusion_score(
        self,
        signals: Dict[str, np.ndarray],
        fused_signal: np.ndarray,
        quality_scores: Optional[Dict[str, float]]
    ) -> float:
        """Calculate fusion score for a single fused signal."""
        if not signals or fused_signal.size == 0:
            return 0.0
        
        # Calculate weighted average of quality scores
        if quality_scores:
            weights = list(quality_scores.values())
            if weights:
                return float(np.mean(weights))
        
        # Default score based on number of sources
        return min(0.9, 0.7 + 0.1 * len(signals))
    
    def _calculate_consistency_score(
        self,
        signals: Dict[str, np.ndarray],
        fused_signal: np.ndarray
    ) -> float:
        """Calculate consistency score between source signals and fused signal."""
        if not signals or fused_signal.size == 0:
            return 0.0
        
        # Calculate correlation between signals
        signal_list = list(signals.values())
        if len(signal_list) < 2:
            return 1.0
        
        # Calculate mean absolute difference between signals
        differences = []
        for i in range(len(signal_list)):
            for j in range(i + 1, len(signal_list)):
                diff = np.abs(signal_list[i] - signal_list[j])
                valid_diff = diff[~np.isnan(diff)] if np.issubdtype(diff.dtype, np.floating) else diff
                if len(valid_diff) > 0:
                    differences.append(np.mean(valid_diff))
        
        if not differences:
            return 0.85
        
        # Normalize consistency (lower difference = higher consistency)
        mean_diff = np.mean(differences)
        
        # Find max value across all signals
        max_vals = []
        for sig in signal_list:
            if np.issubdtype(sig.dtype, np.floating):
                valid_sig = sig[~np.isnan(sig)]
            else:
                valid_sig = sig[sig != 0]
            if len(valid_sig) > 0:
                max_vals.append(np.max(valid_sig))
        
        max_val = max(max_vals) if max_vals else 1.0
        
        if max_val > 0:
            consistency = 1.0 - min(1.0, mean_diff / max_val)
        else:
            consistency = 0.85
        
        return float(max(0.0, min(1.0, consistency)))
    
    def _calculate_fusion_metrics(
        self,
        source_grids: Dict[str, Dict[str, Any]],
        multi_source_fused: Dict[str, np.ndarray],
        fusion_metadata: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall fusion quality metrics."""
        # Calculate average quality score
        quality_scores = [
            grid_data.get('quality_score', 0.8)
            for grid_data in source_grids.values()
        ]
        avg_quality = np.mean(quality_scores) if quality_scores else 0.8
        
        # Calculate average coverage
        coverages = [
            grid_data.get('coverage', 1.0)
            for grid_data in source_grids.values()
        ]
        avg_coverage = np.mean(coverages) if coverages else 1.0
        
        # Calculate fusion scores from metadata
        fusion_scores = [
            meta.get('fusion_score', 0.9)
            for meta in fusion_metadata.values()
        ]
        avg_fusion_score = np.mean(fusion_scores) if fusion_scores else 0.9
        
        consistency_scores = [
            meta.get('consistency_score', 0.85)
            for meta in fusion_metadata.values()
        ]
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.85
        
        # Count signals correctly
        total_original = sum(
            len(grid_data.get('signal_arrays', {}))
            for grid_data in source_grids.values()
        )
        
        return {
            'overall_fusion_score': float(avg_fusion_score),
            'coverage': float(avg_coverage),
            'consistency_score': float(avg_consistency),
            'quality_score': float(avg_quality),
            'signal_count': {
                'original': total_original,
                'source_specific_fused': total_original,  # One per original signal
                'multi_source_fused': len(multi_source_fused),
                'total': total_original * 2 + len(multi_source_fused)
            }
        }


__all__ = ['MultiSourceFusion']
