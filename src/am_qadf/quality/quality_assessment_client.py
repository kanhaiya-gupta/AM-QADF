"""
Quality Assessment Client

Main client for comprehensive quality assessment of voxel domain data.
Integrates data quality, signal quality, alignment accuracy, and completeness analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import sys
import importlib.util
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import quality assessment components with fallback for direct loading
try:
    from .data_quality import DataQualityAnalyzer, DataQualityMetrics
    from .signal_quality import SignalQualityAnalyzer, SignalQualityMetrics
    from .alignment_accuracy import AlignmentAccuracyAnalyzer, AlignmentAccuracyMetrics
    from .completeness import (
        CompletenessAnalyzer,
        CompletenessMetrics,
        GapFillingStrategy,
    )
except ImportError:
    # Fallback for direct module loading (e.g., in notebooks)
    current_file = Path(__file__)
    module_dir = current_file.parent

    # Load data_quality
    data_quality_path = module_dir / "data_quality.py"
    spec_dq = importlib.util.spec_from_file_location("data_quality", data_quality_path)
    dq_module = importlib.util.module_from_spec(spec_dq)
    spec_dq.loader.exec_module(dq_module)
    DataQualityAnalyzer = dq_module.DataQualityAnalyzer
    DataQualityMetrics = dq_module.DataQualityMetrics

    # Load signal_quality
    signal_quality_path = module_dir / "signal_quality.py"
    spec_sq = importlib.util.spec_from_file_location("signal_quality", signal_quality_path)
    sq_module = importlib.util.module_from_spec(spec_sq)
    spec_sq.loader.exec_module(sq_module)
    SignalQualityAnalyzer = sq_module.SignalQualityAnalyzer
    SignalQualityMetrics = sq_module.SignalQualityMetrics

    # Load alignment_accuracy
    alignment_path = module_dir / "alignment_accuracy.py"
    spec_align = importlib.util.spec_from_file_location("alignment_accuracy", alignment_path)
    align_module = importlib.util.module_from_spec(spec_align)
    spec_align.loader.exec_module(align_module)
    AlignmentAccuracyAnalyzer = align_module.AlignmentAccuracyAnalyzer
    AlignmentAccuracyMetrics = align_module.AlignmentAccuracyMetrics

    # Load completeness
    completeness_path = module_dir / "completeness.py"
    spec_comp = importlib.util.spec_from_file_location("completeness", completeness_path)
    comp_module = importlib.util.module_from_spec(spec_comp)
    spec_comp.loader.exec_module(comp_module)
    CompletenessAnalyzer = comp_module.CompletenessAnalyzer
    CompletenessMetrics = comp_module.CompletenessMetrics
    GapFillingStrategy = comp_module.GapFillingStrategy


class QualityAssessmentClient:
    """
    Client for comprehensive quality assessment of voxel domain data.

    Provides:
    - Data quality metrics (completeness, coverage, consistency, accuracy, reliability)
    - Signal quality assessment (SNR, uncertainty, confidence)
    - Alignment accuracy validation (coordinate, temporal, spatial)
    - Completeness checks and gap filling
    """

    def __init__(self, max_acceptable_error: float = 0.1, noise_floor: float = 1e-6, enable_validation: bool = True):
        """
        Initialize the quality assessment client.

        Args:
            max_acceptable_error: Maximum acceptable alignment error (mm)
            noise_floor: Minimum noise level for SNR calculation
            enable_validation: Whether to enable validation capabilities
        """
        self.data_quality_analyzer = DataQualityAnalyzer()
        self.signal_quality_analyzer = SignalQualityAnalyzer(noise_floor=noise_floor)
        self.alignment_analyzer = AlignmentAccuracyAnalyzer(max_acceptable_error=max_acceptable_error)
        self.completeness_analyzer = CompletenessAnalyzer()

        # Initialize validation client if enabled
        self.enable_validation = enable_validation
        self.validation_client = None
        if enable_validation:
            try:
                from ..validation import ValidationClient, ValidationConfig

                validation_config = ValidationConfig(max_acceptable_error=max_acceptable_error, correlation_threshold=0.85)
                self.validation_client = ValidationClient(config=validation_config)
                logger.info("Validation client initialized")
            except ImportError as e:
                logger.warning(f"Validation module not available: {e}. Validation features disabled.")
                self.enable_validation = False

    def assess_data_quality(
        self,
        voxel_data: Any,
        signals: Optional[List[str]] = None,
        layer_range: Optional[Tuple[int, int]] = None,
    ) -> DataQualityMetrics:
        """
        Assess overall data quality.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to check (None = all signals)
            layer_range: (min_layer, max_layer) range for temporal coverage

        Returns:
            DataQualityMetrics object
        """
        return self.data_quality_analyzer.assess_quality(voxel_data, signals=signals, layer_range=layer_range)

    def assess_signal_quality(
        self,
        signal_name: str,
        signal_array: np.ndarray,
        noise_estimate: Optional[np.ndarray] = None,
        measurement_uncertainty: Optional[float] = None,
        store_maps: bool = True,
    ) -> SignalQualityMetrics:
        """
        Assess quality for a single signal.

        Args:
            signal_name: Name of the signal
            signal_array: Signal array
            noise_estimate: Optional noise estimate
            measurement_uncertainty: Optional measurement uncertainty
            store_maps: Whether to store per-voxel quality maps

        Returns:
            SignalQualityMetrics object
        """
        return self.signal_quality_analyzer.assess_signal_quality(
            signal_name,
            signal_array,
            noise_estimate,
            measurement_uncertainty,
            store_maps,
        )

    def assess_all_signals(
        self,
        voxel_data: Any,
        signals: Optional[List[str]] = None,
        store_maps: bool = True,
    ) -> Dict[str, SignalQualityMetrics]:
        """
        Assess quality for all signals.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to check (None = all signals)
            store_maps: Whether to store per-voxel quality maps

        Returns:
            Dictionary mapping signal names to SignalQualityMetrics
        """
        if signals is None:
            signals = list(voxel_data.available_signals) if hasattr(voxel_data, "available_signals") else []

        results = {}
        for signal in signals:
            try:
                signal_array = voxel_data.get_signal_array(signal, default=0.0)
                results[signal] = self.assess_signal_quality(signal, signal_array, store_maps=store_maps)
            except Exception as e:
                print(f"⚠️ Error assessing signal {signal}: {e}")
                continue

        return results

    def assess_alignment_accuracy(
        self,
        voxel_data: Any,
        coordinate_transformer: Optional[Any] = None,
        reference_data: Optional[Any] = None,
    ) -> AlignmentAccuracyMetrics:
        """
        Assess alignment accuracy.

        Args:
            voxel_data: Voxel domain data object
            coordinate_transformer: Optional coordinate system transformer
            reference_data: Optional reference data for comparison

        Returns:
            AlignmentAccuracyMetrics object
        """
        return self.alignment_analyzer.assess_alignment_accuracy(voxel_data, coordinate_transformer, reference_data)

    def assess_completeness(
        self,
        voxel_data: Any,
        signals: Optional[List[str]] = None,
        store_details: bool = True,
    ) -> CompletenessMetrics:
        """
        Assess data completeness.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to check (None = all signals)
            store_details: Whether to store detailed information

        Returns:
            CompletenessMetrics object
        """
        return self.completeness_analyzer.assess_completeness(voxel_data, signals=signals, store_details=store_details)

    def fill_gaps(
        self,
        signal_array: np.ndarray,
        strategy: GapFillingStrategy = GapFillingStrategy.LINEAR,
    ) -> np.ndarray:
        """
        Fill missing data gaps.

        Args:
            signal_array: Signal array with missing data
            strategy: Gap filling strategy

        Returns:
            Filled signal array
        """
        return self.completeness_analyzer.fill_gaps(signal_array, strategy)

    def comprehensive_assessment(
        self,
        voxel_data: Any,
        signals: Optional[List[str]] = None,
        layer_range: Optional[Tuple[int, int]] = None,
        coordinate_transformer: Optional[Any] = None,
        store_maps: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to check (None = all signals)
            layer_range: (min_layer, max_layer) range for temporal coverage
            coordinate_transformer: Optional coordinate system transformer
            store_maps: Whether to store per-voxel quality maps

        Returns:
            Dictionary containing all quality assessment results
        """
        print("🔍 Starting comprehensive quality assessment...")

        # Data quality
        print("  📊 Assessing data quality...")
        data_quality = self.assess_data_quality(voxel_data, signals, layer_range)

        # Signal quality
        print("  📈 Assessing signal quality...")
        signal_quality = self.assess_all_signals(voxel_data, signals, store_maps=store_maps)

        # Alignment accuracy
        print("  🎯 Assessing alignment accuracy...")
        alignment_accuracy = self.assess_alignment_accuracy(voxel_data, coordinate_transformer)

        # Completeness
        print("  ✅ Assessing completeness...")
        completeness = self.assess_completeness(voxel_data, signals, store_details=True)

        print("✅ Quality assessment complete")

        return {
            "data_quality": data_quality,
            "signal_quality": signal_quality,
            "alignment_accuracy": alignment_accuracy,
            "completeness": completeness,
            "summary": {
                "overall_quality_score": (
                    (
                        data_quality.completeness * 0.3
                        + np.mean([sq.quality_score for sq in signal_quality.values()]) * 0.3
                        + alignment_accuracy.alignment_score * 0.2
                        + completeness.completeness_ratio * 0.2
                    )
                    if signal_quality
                    else 0.0
                ),
                "data_quality_score": data_quality.completeness,
                "signal_quality_score": (
                    np.mean([sq.quality_score for sq in signal_quality.values()]) if signal_quality else 0.0
                ),
                "alignment_score": alignment_accuracy.alignment_score,
                "completeness_score": completeness.completeness_ratio,
            },
        }

    def generate_quality_report(self, assessment_results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """
        Generate a human-readable quality report.

        Args:
            assessment_results: Results from comprehensive_assessment()
            output_file: Optional file path to save report

        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("QUALITY ASSESSMENT REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary
        summary = assessment_results.get("summary", {})
        report_lines.append("📊 SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Overall Quality Score: {summary.get('overall_quality_score', 0.0):.2%}")
        report_lines.append(f"Data Quality Score: {summary.get('data_quality_score', 0.0):.2%}")
        report_lines.append(f"Signal Quality Score: {summary.get('signal_quality_score', 0.0):.2%}")
        report_lines.append(f"Alignment Score: {summary.get('alignment_score', 0.0):.2%}")
        report_lines.append(f"Completeness Score: {summary.get('completeness_score', 0.0):.2%}")
        report_lines.append("")

        # Data Quality
        data_quality = assessment_results.get("data_quality")
        if data_quality:
            report_lines.append("📊 DATA QUALITY METRICS")
            report_lines.append("-" * 80)
            report_lines.append(f"Completeness: {data_quality.completeness:.2%}")
            report_lines.append(f"Spatial Coverage: {data_quality.coverage_spatial:.2%}")
            report_lines.append(f"Temporal Coverage: {data_quality.coverage_temporal:.2%}")
            report_lines.append(f"Consistency Score: {data_quality.consistency_score:.2%}")
            report_lines.append(f"Accuracy Score: {data_quality.accuracy_score:.2%}")
            report_lines.append(f"Reliability Score: {data_quality.reliability_score:.2%}")
            report_lines.append(f"Filled Voxels: {data_quality.filled_voxels:,} / {data_quality.total_voxels:,}")
            report_lines.append(f"Sources: {data_quality.sources_count}")
            report_lines.append(f"Missing Regions: {len(data_quality.missing_regions)}")
            report_lines.append("")

        # Signal Quality
        signal_quality = assessment_results.get("signal_quality", {})
        if signal_quality:
            report_lines.append("📈 SIGNAL QUALITY METRICS")
            report_lines.append("-" * 80)
            for signal_name, metrics in signal_quality.items():
                report_lines.append(f"\n{signal_name}:")
                report_lines.append(f"  Quality Score: {metrics.quality_score:.2%}")
                report_lines.append(
                    f"  SNR: {metrics.snr_mean:.2f} ± {metrics.snr_std:.2f} dB (range: {metrics.snr_min:.2f} - {metrics.snr_max:.2f})"
                )
                report_lines.append(f"  Uncertainty: {metrics.uncertainty_mean:.4f}")
                report_lines.append(f"  Confidence: {metrics.confidence_mean:.2%}")
            report_lines.append("")

        # Alignment Accuracy
        alignment = assessment_results.get("alignment_accuracy")
        if alignment:
            report_lines.append("🎯 ALIGNMENT ACCURACY")
            report_lines.append("-" * 80)
            report_lines.append(f"Alignment Score: {alignment.alignment_score:.2%}")
            report_lines.append(f"Coordinate Error: {alignment.coordinate_alignment_error:.4f} mm")
            report_lines.append(f"Temporal Error: {alignment.temporal_alignment_error:.4f}")
            report_lines.append(f"Spatial Registration Error: {alignment.spatial_registration_error:.4f} mm")
            report_lines.append(f"Residual Error: {alignment.residual_error_mean:.4f} ± {alignment.residual_error_std:.4f} mm")
            report_lines.append("")

        # Completeness
        completeness = assessment_results.get("completeness")
        if completeness:
            report_lines.append("✅ COMPLETENESS METRICS")
            report_lines.append("-" * 80)
            report_lines.append(f"Completeness Ratio: {completeness.completeness_ratio:.2%}")
            report_lines.append(f"Spatial Coverage: {completeness.spatial_coverage:.2%}")
            report_lines.append(f"Temporal Coverage: {completeness.temporal_coverage:.2%}")
            report_lines.append(f"Missing Voxels: {completeness.missing_voxels_count:,}")
            report_lines.append(f"Missing Regions: {completeness.missing_regions_count}")
            report_lines.append(f"Gap Fillable Ratio: {completeness.gap_fillable_ratio:.2%}")
            report_lines.append("")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            print(f"✅ Report saved to: {output_file}")

        return report

    def validate_quality_assessment(
        self,
        voxel_data: Any,
        reference_data: Any,
        validation_type: str = "mpm",
        signals: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate quality assessment against reference data (MPM or ground truth).

        Args:
            voxel_data: Voxel domain data with quality assessment results
            reference_data: Reference data for validation (MPM system outputs or ground truth)
            validation_type: Type of validation ('mpm', 'accuracy', 'statistical', or 'comprehensive')
            signals: List of signal names to validate (None = all signals)

        Returns:
            Dictionary containing validation results

        Raises:
            RuntimeError: If validation is not enabled or validation client not available
        """
        if not self.enable_validation or self.validation_client is None:
            raise RuntimeError(
                "Validation not available. Initialize QualityAssessmentClient with enable_validation=True "
                "and ensure validation module is installed."
            )

        # Perform quality assessment
        assessment_results = self.comprehensive_assessment(voxel_data, signals=signals)

        # Extract metrics for comparison
        framework_metrics = {
            "overall_quality_score": assessment_results["summary"]["overall_quality_score"],
            "data_quality_score": assessment_results["summary"]["data_quality_score"],
            "signal_quality_score": assessment_results["summary"]["signal_quality_score"],
            "alignment_score": assessment_results["summary"]["alignment_score"],
            "completeness_score": assessment_results["summary"]["completeness_score"],
        }

        # Add signal-specific metrics if available
        if signals is None and hasattr(voxel_data, "available_signals"):
            signals = list(voxel_data.available_signals)

        if signals and "signal_quality" in assessment_results:
            for signal_name in signals:
                if signal_name in assessment_results["signal_quality"]:
                    sq = assessment_results["signal_quality"][signal_name]
                    framework_metrics[f"{signal_name}_snr"] = sq.snr_mean
                    framework_metrics[f"{signal_name}_quality_score"] = sq.quality_score

        # Perform validation based on type
        validation_results = {}

        if validation_type in ["mpm", "comprehensive"]:
            # MPM comparison
            try:
                mpm_results = self.validation_client.compare_with_mpm(
                    framework_metrics,
                    reference_data if isinstance(reference_data, dict) else {"reference": reference_data},
                    metrics=list(framework_metrics.keys()) if isinstance(reference_data, dict) else None,
                )
                validation_results["mpm_comparison"] = mpm_results
                logger.info(f"MPM comparison completed: {len(mpm_results)} metrics compared")
            except Exception as e:
                logger.error(f"MPM comparison failed: {e}")
                validation_results["mpm_comparison"] = {"error": str(e)}

        if validation_type in ["accuracy", "comprehensive"]:
            # Accuracy validation
            try:
                # Convert assessment results to arrays for accuracy validation
                if isinstance(reference_data, dict) and "quality_scores" in reference_data:
                    # Assume reference_data contains corresponding quality scores
                    accuracy_result = self.validation_client.validate_accuracy(
                        np.array(list(framework_metrics.values())),
                        np.array(list(reference_data["quality_scores"].values())),
                        validation_type="quality",
                    )
                    validation_results["accuracy"] = accuracy_result
                    logger.info("Accuracy validation completed")
            except Exception as e:
                logger.error(f"Accuracy validation failed: {e}")
                validation_results["accuracy"] = {"error": str(e)}

        if validation_type in ["statistical", "comprehensive"]:
            # Statistical validation
            try:
                if isinstance(reference_data, dict) and "quality_scores" in reference_data:
                    framework_array = np.array(list(framework_metrics.values()))
                    reference_array = np.array(list(reference_data["quality_scores"].values()))
                    stat_result = self.validation_client.perform_statistical_test(
                        "t_test", framework_array, reference_array, alternative="two-sided"
                    )
                    validation_results["statistical"] = stat_result
                    logger.info("Statistical validation completed")
            except Exception as e:
                logger.error(f"Statistical validation failed: {e}")
                validation_results["statistical"] = {"error": str(e)}

        return {
            "framework_metrics": framework_metrics,
            "validation_type": validation_type,
            "validation_results": validation_results,
            "assessment_results": assessment_results,
        }

    def benchmark_quality_assessment(
        self,
        voxel_data: Any,
        signals: Optional[List[str]] = None,
        iterations: int = 1,
        warmup_iterations: int = 0,
    ) -> Optional[Any]:
        """
        Benchmark quality assessment performance.

        Args:
            voxel_data: Voxel domain data to assess
            signals: List of signal names to check (None = all signals)
            iterations: Number of iterations to run for benchmarking
            warmup_iterations: Number of warmup iterations (excluded from timing)

        Returns:
            BenchmarkResult if validation is enabled, None otherwise
        """
        if not self.enable_validation or self.validation_client is None:
            logger.warning("Benchmarking not available. Validation client not initialized.")
            return None

        return self.validation_client.benchmark_operation(
            self.comprehensive_assessment,
            voxel_data,
            signals=signals,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
        )
