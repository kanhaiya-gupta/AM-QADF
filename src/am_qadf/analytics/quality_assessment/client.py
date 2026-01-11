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

    def __init__(
        self,
        max_acceptable_error: float = 0.1,
        noise_floor: float = 1e-6,
        enable_spc: bool = True,
        enable_validation: bool = False,
        mongo_client: Optional[Any] = None,
    ):
        """
        Initialize the quality assessment client.

        Args:
            max_acceptable_error: Maximum acceptable alignment error (mm)
            noise_floor: Minimum noise level for SNR calculation
            enable_spc: Whether to enable SPC integration (default: True)
            enable_validation: Whether to enable validation capabilities (default: False)
            mongo_client: Optional MongoDB client for SPC storage
        """
        self.data_quality_analyzer = DataQualityAnalyzer()
        self.signal_quality_analyzer = SignalQualityAnalyzer(noise_floor=noise_floor)
        self.alignment_analyzer = AlignmentAccuracyAnalyzer(max_acceptable_error=max_acceptable_error)
        self.completeness_analyzer = CompletenessAnalyzer()

        # Initialize SPC client if enabled
        self.enable_spc = enable_spc
        self.spc_client = None
        if enable_spc:
            try:
                from ..spc import SPCClient, SPCConfig

                spc_config = SPCConfig(control_limit_sigma=3.0, subgroup_size=5, enable_warnings=True)
                self.spc_client = SPCClient(config=spc_config, mongo_client=mongo_client)
                logging.getLogger(__name__).info("SPC integration enabled in QualityAssessmentClient")
            except ImportError as e:
                logging.getLogger(__name__).warning(f"SPC module not available: {e}. SPC features disabled.")
                self.enable_spc = False

        # Initialize validation client if enabled
        self.enable_validation = enable_validation
        self.validation_client = None
        if enable_validation:
            try:
                from ...validation import ValidationClient, ValidationConfig

                validation_config = ValidationConfig(max_acceptable_error=max_acceptable_error, correlation_threshold=0.85)
                self.validation_client = ValidationClient(config=validation_config)
                logger.info("Validation client initialized")
            except ImportError as e:
                logger.warning(f"Validation module not available: {e}. Validation features disabled.")
                self.enable_validation = False

    def assess_data_quality(
        self,
        voxel_data: Any,
        model_id: Optional[str] = None,
        signals: Optional[List[str]] = None,
        layer_range: Optional[Tuple[int, int]] = None,
    ) -> DataQualityMetrics:
        """
        Assess overall data quality.

        Args:
            voxel_data: Voxel domain data object
            model_id: Model ID (UUID of the STL model/sample being studied) - required for traceability
            signals: List of signal names to check (None = all signals)
            layer_range: (min_layer, max_layer) range for temporal coverage

        Returns:
            DataQualityMetrics object
        """
        return self.data_quality_analyzer.assess_quality(
            voxel_data, signals=signals, layer_range=layer_range, model_id=model_id
        )

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

    def assess_with_spc(
        self,
        voxel_data: Any,
        signals: Optional[List[str]] = None,
        specification_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        spc_config: Optional[Any] = None,  # SPCConfig type
    ) -> Dict[str, Any]:
        """
        Assess quality with Statistical Process Control analysis.

        Performs quality assessment and applies SPC analysis to quality metrics.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to check (None = all signals)
            specification_limits: Optional dictionary mapping signal/metric names to (USL, LSL)
            spc_config: Optional SPCConfig for SPC analysis

        Returns:
            Dictionary containing quality assessment and SPC analysis results
        """
        if not self.enable_spc or self.spc_client is None:
            raise ValueError("SPC integration not enabled. Set enable_spc=True in __init__")

        # Perform standard quality assessment
        quality_results = self.comprehensive_assessment(voxel_data, signals=signals, store_maps=True)

        # Extract quality metrics for SPC analysis
        quality_metrics = {}

        # Add data quality metrics
        if "data_quality" in quality_results:
            dq = quality_results["data_quality"]
            quality_metrics["completeness"] = np.array([dq.completeness])
            quality_metrics["coverage_spatial"] = np.array([dq.coverage_spatial])
            quality_metrics["coverage_temporal"] = np.array([dq.coverage_temporal])
            quality_metrics["consistency"] = np.array([dq.consistency_score])

        # Add signal quality metrics (if multiple samples available)
        if "signal_quality" in quality_results:
            for signal_name, sq in quality_results["signal_quality"].items():
                # For SPC, we need time series data - here we use aggregated metrics
                # In practice, this would come from historical data
                quality_metrics[f"{signal_name}_quality"] = np.array([sq.quality_score])
                quality_metrics[f"{signal_name}_snr"] = np.array([sq.snr_mean])

        # Add alignment metrics
        if "alignment_accuracy" in quality_results:
            align = quality_results["alignment_accuracy"]
            quality_metrics["alignment_score"] = np.array([align.alignment_score])

        # Perform SPC analysis on quality metrics
        spc_results = {}
        if quality_metrics:
            try:
                spc_results = self.spc_client.integrate_with_quality_assessment(
                    quality_metrics, specification_limits=specification_limits, config=spc_config
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"Error in SPC analysis: {e}")
                spc_results = {"error": str(e)}

        return {
            "quality_assessment": quality_results,
            "spc_analysis": spc_results,
            "summary": {
                **quality_results.get("summary", {}),
                "spc_enabled": True,
                "spc_metrics_analyzed": len(quality_metrics),
            },
        }

    def monitor_quality_with_spc(
        self,
        quality_metrics_history: Dict[str, np.ndarray],
        baseline_data: Optional[Dict[str, np.ndarray]] = None,
        specification_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        spc_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Monitor quality metrics using SPC control charts.

        Args:
            quality_metrics_history: Dictionary mapping metric names to time series arrays
            baseline_data: Optional baseline data for establishing control limits
            specification_limits: Optional dictionary mapping metric names to (USL, LSL)
            spc_config: Optional SPCConfig

        Returns:
            Dictionary with SPC monitoring results for each metric
        """
        if not self.enable_spc or self.spc_client is None:
            raise ValueError("SPC integration not enabled. Set enable_spc=True in __init__")

        results = {}

        for metric_name, metric_data in quality_metrics_history.items():
            try:
                # Establish baseline if provided
                baseline = None
                if baseline_data and metric_name in baseline_data:
                    baseline = self.spc_client.establish_baseline(baseline_data[metric_name], config=spc_config)

                # Create control chart
                chart_result = self.spc_client.create_control_chart(metric_data, chart_type="individual", config=spc_config)

                # Detect rule violations
                violations = []
                if hasattr(self.spc_client, "detect_rule_violations"):
                    try:
                        violations = self.spc_client.detect_rule_violations(chart_result, rule_set="western_electric")
                    except Exception as e:
                        import logging

                        logging.getLogger(__name__).warning(f"Error detecting rule violations for {metric_name}: {e}")

                # Process capability if specification limits provided
                capability = None
                if specification_limits and metric_name in specification_limits:
                    try:
                        capability = self.spc_client.analyze_process_capability(
                            metric_data, specification_limits[metric_name], config=spc_config
                        )
                    except Exception as e:
                        import logging

                        logging.getLogger(__name__).warning(f"Error in capability analysis for {metric_name}: {e}")

                results[metric_name] = {
                    "control_chart": chart_result,
                    "rule_violations": violations,
                    "process_capability": capability,
                    "baseline": baseline,
                    "n_samples": len(metric_data),
                    "out_of_control_points": (
                        len(chart_result.out_of_control_points) if hasattr(chart_result, "out_of_control_points") else 0
                    ),
                }

            except Exception as e:
                import logging

                logging.getLogger(__name__).error(f"Error monitoring {metric_name} with SPC: {e}")
                results[metric_name] = {"error": str(e)}

        return results

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
