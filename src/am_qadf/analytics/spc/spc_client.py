"""
Statistical Process Control (SPC) Client

Main client interface for SPC operations.
Provides unified interface for all SPC capabilities.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator, Callable, Generator
from dataclasses import dataclass, field
import logging
from datetime import datetime

from .control_charts import ControlChartGenerator, ControlChartResult
from .baseline_calculation import BaselineCalculator, BaselineStatistics, AdaptiveLimitsCalculator

logger = logging.getLogger(__name__)


@dataclass
class SPCConfig:
    """Configuration for SPC operations."""

    control_limit_sigma: float = 3.0  # Standard deviations for control limits
    subgroup_size: int = 5  # Subgroup size for X-bar charts
    baseline_sample_size: int = 100  # Minimum samples for baseline
    adaptive_limits: bool = False  # Enable adaptive control limits
    update_frequency: Optional[int] = None  # Samples between limit updates
    specification_limits: Optional[Tuple[float, float]] = None  # USL, LSL
    target_value: Optional[float] = None  # Target/center value
    enable_warnings: bool = True  # Enable warning limits (2-sigma)
    warning_sigma: float = 2.0  # Warning limit multiplier


class SPCClient:
    """
    Main client interface for Statistical Process Control operations.

    Provides:
    - Control chart generation (X-bar, R, S, Individual, Moving Range)
    - Process capability analysis (Phase 2)
    - Multivariate SPC (Phase 4)
    - Control rule detection (Phase 3)
    - Baseline calculation and management
    - Integration with quality assessment
    """

    def __init__(self, config: Optional[SPCConfig] = None, mongo_client: Optional[Any] = None):
        """
        Initialize SPC client.

        Args:
            config: Optional SPCConfig for default settings
            mongo_client: Optional MongoDB client for data storage
        """
        self.config = config if config is not None else SPCConfig()
        self.mongo_client = mongo_client

        # Initialize components
        self.chart_generator = ControlChartGenerator()
        self.baseline_calc = BaselineCalculator()
        self.adaptive_calc = AdaptiveLimitsCalculator()

        # Process capability analyzer (Phase 2)
        try:
            from .process_capability import ProcessCapabilityAnalyzer

            self.capability_analyzer = ProcessCapabilityAnalyzer()
        except Exception as e:
            logger.warning(f"Process capability analyzer not available: {e}")
            self.capability_analyzer = None

        # Control rule detector (Phase 3)
        try:
            from .control_rules import ControlRuleDetector

            self.rule_detector = ControlRuleDetector()
        except Exception as e:
            logger.warning(f"Control rule detector not available: {e}")
            self.rule_detector = None

        # Multivariate SPC analyzer (Phase 4)
        try:
            from .multivariate_spc import MultivariateSPCAnalyzer

            self.multivariate_analyzer = MultivariateSPCAnalyzer()
        except Exception as e:
            logger.warning(f"Multivariate SPC analyzer not available: {e}")
            self.multivariate_analyzer = None

        # Storage (Phase 7)
        self.storage = None
        if mongo_client:
            try:
                from .spc_storage import SPCStorage

                self.storage = SPCStorage(mongo_client)
            except Exception as e:
                logger.warning(f"SPC storage not available: {e}")

        logger.info("SPCClient initialized")

    def create_control_chart(
        self,
        data: np.ndarray,
        chart_type: str = "xbar_r",
        subgroup_size: Optional[int] = None,
        config: Optional[SPCConfig] = None,
    ) -> Union[ControlChartResult, Tuple[ControlChartResult, ...]]:
        """
        Create control chart(s).

        Args:
            data: Input data array
            chart_type: Type of chart ('xbar', 'r', 's', 'individual', 'moving_range', 'xbar_r', 'xbar_s')
            subgroup_size: Subgroup size (None uses config default)
            config: Optional SPCConfig (overrides instance config)

        Returns:
            ControlChartResult or tuple of results for paired charts
        """
        use_config = config if config is not None else self.config
        subgroup_size = subgroup_size if subgroup_size is not None else use_config.subgroup_size

        try:
            if chart_type == "xbar":
                return self.chart_generator.create_xbar_chart(data, subgroup_size, use_config)
            elif chart_type == "r":
                return self.chart_generator.create_r_chart(data, subgroup_size, use_config)
            elif chart_type == "s":
                return self.chart_generator.create_s_chart(data, subgroup_size, use_config)
            elif chart_type == "individual":
                return self.chart_generator.create_individual_chart(data, use_config)
            elif chart_type == "moving_range":
                return self.chart_generator.create_moving_range_chart(data, window_size=2, config=use_config)
            elif chart_type == "xbar_r":
                return self.chart_generator.create_xbar_r_charts(data, subgroup_size, use_config)
            elif chart_type == "xbar_s":
                return self.chart_generator.create_xbar_s_charts(data, subgroup_size, use_config)
            else:
                raise ValueError(f"Unknown chart type: {chart_type}")
        except Exception as e:
            logger.error(f"Error creating control chart {chart_type}: {e}")
            raise

    def establish_baseline(
        self,
        historical_data: np.ndarray,
        subgroup_size: Optional[int] = None,
        config: Optional[SPCConfig] = None,
        model_id: Optional[str] = None,
        signal_name: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> BaselineStatistics:
        """
        Establish baseline statistics from historical data.

        Args:
            historical_data: Historical process data
            subgroup_size: Subgroup size (None uses config default)
            config: Optional SPCConfig
            model_id: Optional model ID for storage
            signal_name: Optional signal name for storage
            metadata: Optional additional metadata for storage

        Returns:
            BaselineStatistics object
        """
        use_config = config if config is not None else self.config
        subgroup_size = subgroup_size if subgroup_size is not None else use_config.subgroup_size

        baseline = self.baseline_calc.calculate_baseline(historical_data, subgroup_size=subgroup_size, config=use_config)

        if self.storage and model_id and signal_name:
            try:
                # Store baseline
                self.storage.save_baseline(model_id, signal_name, baseline, metadata)
            except Exception as e:
                logger.warning(f"Could not store baseline: {e}")

        return baseline

    def update_baseline_adaptive(
        self, current_baseline: BaselineStatistics, new_data: np.ndarray, method: str = "exponential_smoothing"
    ) -> BaselineStatistics:
        """
        Update baseline statistics adaptively.

        Args:
            current_baseline: Current baseline statistics
            new_data: New process data
            method: Update method ('exponential_smoothing', 'cumulative', 'window')

        Returns:
            Updated BaselineStatistics
        """
        updated = self.baseline_calc.update_baseline(current_baseline, new_data, method=method)

        if self.storage:
            try:
                # Store updated baseline
                logger.debug("Baseline update storage not yet implemented")
            except Exception as e:
                logger.warning(f"Could not store updated baseline: {e}")

        return updated

    def detect_rule_violations(
        self, chart_result: ControlChartResult, rule_set: str = "western_electric"
    ) -> List[Any]:  # List[ControlRuleViolation] - avoiding circular import
        """
        Detect control rule violations.

        Args:
            chart_result: Control chart result to analyze
            rule_set: Rule set to use ('western_electric', 'nelson')

        Returns:
            List of ControlRuleViolation objects
        """
        if self.rule_detector is None:
            raise NotImplementedError("Control rule detector not available")

        if rule_set == "western_electric":
            violations = self.rule_detector.detect_western_electric_rules(chart_result)
        elif rule_set == "nelson":
            violations = self.rule_detector.detect_nelson_rules(chart_result)
        else:
            violations = self.rule_detector.detect_custom_rules(chart_result, [rule_set])

        # Store violations in chart result
        for violation in violations:
            chart_result.rule_violations[violation.rule_name] = violation.affected_points

        return violations

    def analyze_process_capability(
        self,
        data: np.ndarray,
        specification_limits: Tuple[float, float],
        target_value: Optional[float] = None,
        config: Optional[SPCConfig] = None,
    ) -> Any:  # ProcessCapabilityResult
        """
        Analyze process capability.

        Args:
            data: Process data
            specification_limits: (USL, LSL) tuple
            target_value: Optional target value
            config: Optional SPCConfig

        Returns:
            ProcessCapabilityResult object
        """
        if self.capability_analyzer is None:
            raise NotImplementedError("Process capability analyzer not available")

        use_config = config if config is not None else self.config
        target_value = target_value if target_value is not None else use_config.target_value

        result = self.capability_analyzer.calculate_capability(
            data, specification_limits, target_value=target_value, subgroup_size=use_config.subgroup_size, config=use_config
        )

        if self.storage:
            try:
                # Store capability result
                model_id = "unknown"  # Should be passed as parameter
                self.storage.save_capability_result(model_id, result, metadata={})
            except Exception as e:
                logger.warning(f"Could not store capability result: {e}")

        return result

    def create_multivariate_chart(
        self,
        data: np.ndarray,
        method: str = "hotelling_t2",
        baseline_data: Optional[np.ndarray] = None,
        config: Optional[SPCConfig] = None,
    ) -> Any:  # MultivariateSPCResult
        """
        Create multivariate SPC chart.

        Args:
            data: Multivariate process data (n_samples x n_variables)
            method: Method ('hotelling_t2', 'pca')
            baseline_data: Optional baseline data
            config: Optional SPCConfig

        Returns:
            MultivariateSPCResult object
        """
        if self.multivariate_analyzer is None:
            raise NotImplementedError("Multivariate SPC analyzer not available")

        use_config = config if config is not None else self.config

        if method == "hotelling_t2":
            result = self.multivariate_analyzer.create_hotelling_t2_chart(data, baseline_data=baseline_data, config=use_config)
        elif method == "pca":
            result = self.multivariate_analyzer.create_pca_chart(data, baseline_data=baseline_data, config=use_config)
        else:
            raise ValueError(f"Unknown multivariate method: {method}. Use 'hotelling_t2' or 'pca'")

        if self.storage:
            try:
                # Store multivariate result
                model_id = "unknown"  # Should be passed as parameter
                self.storage.save_multivariate_result(model_id, result, metadata={})
            except Exception as e:
                logger.warning(f"Could not store multivariate result: {e}")

        return result

    def comprehensive_spc_analysis(
        self,
        data: np.ndarray,
        specification_limits: Optional[Tuple[float, float]] = None,
        chart_types: Optional[List[str]] = None,
        config: Optional[SPCConfig] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive SPC analysis.

        Args:
            data: Process data
            specification_limits: Optional (USL, LSL) tuple
            chart_types: List of chart types to create (None = auto-select)
            config: Optional SPCConfig

        Returns:
            Dictionary with all analysis results
        """
        use_config = config if config is not None else self.config

        # Auto-select chart types if not specified
        if chart_types is None:
            # Default: X-bar and R charts for subgrouped data, I-MR for individual data
            if use_config.subgroup_size > 1:
                chart_types = ["xbar_r"]
            else:
                chart_types = ["individual", "moving_range"]

        results = {}

        # Create control charts
        charts = {}
        for chart_type in chart_types:
            try:
                chart_result = self.create_control_chart(data, chart_type, config=use_config)
                if isinstance(chart_result, tuple):
                    # Paired charts
                    for i, chart in enumerate(chart_result):
                        charts[f"{chart_type}_{i}"] = chart
                else:
                    charts[chart_type] = chart_result
            except Exception as e:
                logger.error(f"Error creating {chart_type} chart: {e}")
                continue

        results["control_charts"] = charts

        # Establish baseline
        try:
            baseline = self.establish_baseline(data, config=use_config)
            results["baseline"] = baseline
        except Exception as e:
            logger.error(f"Error establishing baseline: {e}")
            results["baseline"] = None

        # Process capability (if specification limits provided)
        if specification_limits is not None:
            try:
                capability = self.analyze_process_capability(
                    data, specification_limits, target_value=use_config.target_value, config=use_config
                )
                results["process_capability"] = capability
            except (NotImplementedError, AttributeError) as e:
                logger.warning(f"Process capability analysis not available: {e}")
                results["process_capability"] = None
            except Exception as e:
                logger.error(f"Error in process capability analysis: {e}")
                results["process_capability"] = None
        else:
            results["process_capability"] = None

        # Detect rule violations for control charts
        if self.rule_detector is not None:
            for chart_name, chart_result in charts.items():
                if isinstance(chart_result, ControlChartResult):
                    try:
                        violations = self.detect_rule_violations(chart_result, rule_set="western_electric")
                        results.setdefault("rule_violations", {})[chart_name] = violations
                    except Exception as e:
                        logger.warning(f"Error detecting rule violations for {chart_name}: {e}")

        # Summary statistics
        results["summary"] = {
            "n_samples": len(data),
            "charts_created": len(charts),
            "ooc_points": sum(
                len(chart.out_of_control_points) for chart in charts.values() if isinstance(chart, ControlChartResult)
            ),
            "baseline_established": results["baseline"] is not None,
            "capability_analyzed": results["process_capability"] is not None,
        }

        return results

    def integrate_with_quality_assessment(
        self,
        quality_metrics: Dict[str, np.ndarray],
        specification_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        config: Optional[SPCConfig] = None,
    ) -> Dict[str, Any]:
        """
        Integrate SPC with quality assessment metrics.

        Args:
            quality_metrics: Dictionary mapping metric names to data arrays
            specification_limits: Optional dictionary mapping metric names to (USL, LSL)
            config: Optional SPCConfig

        Returns:
            Dictionary with SPC analysis for each quality metric
        """
        use_config = config if config is not None else self.config
        results = {}

        for metric_name, metric_data in quality_metrics.items():
            try:
                spec_limits = specification_limits.get(metric_name) if specification_limits else None

                analysis = self.comprehensive_spc_analysis(metric_data, specification_limits=spec_limits, config=use_config)
                results[metric_name] = analysis
            except Exception as e:
                logger.error(f"Error analyzing quality metric {metric_name}: {e}")
                results[metric_name] = {"error": str(e)}

        return results

    def monitor_streaming_data(
        self,
        data_stream: Iterator[np.ndarray],
        baseline: BaselineStatistics,
        callback: Optional[Callable] = None,
        config: Optional[SPCConfig] = None,
    ) -> Generator[ControlChartResult, None, None]:
        """
        Monitor streaming data in real-time.

        Args:
            data_stream: Iterator of data batches
            baseline: Baseline statistics
            callback: Optional callback function for each result
            config: Optional SPCConfig

        Yields:
            ControlChartResult objects for each batch
        """
        # Will be enhanced in Phase 7 with adaptive limits
        use_config = config if config is not None else self.config

        for batch in data_stream:
            try:
                # Create control chart for this batch
                chart_result = self.create_control_chart(batch, chart_type="individual", config=use_config)

                # Callback if provided
                if callback:
                    callback(chart_result)

                yield chart_result
            except Exception as e:
                logger.error(f"Error processing streaming batch: {e}")
                continue

    def generate_spc_report(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """
        Generate SPC analysis report.

        Args:
            results: Results from comprehensive_spc_analysis
            output_file: Optional file path to save report

        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("STATISTICAL PROCESS CONTROL (SPC) ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary
        summary = results.get("summary", {})
        report_lines.append("📊 SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Samples: {summary.get('n_samples', 0):,}")
        report_lines.append(f"Charts Created: {summary.get('charts_created', 0)}")
        report_lines.append(f"Out-of-Control Points: {summary.get('ooc_points', 0)}")
        report_lines.append(f"Baseline Established: {summary.get('baseline_established', False)}")
        report_lines.append(f"Capability Analyzed: {summary.get('capability_analyzed', False)}")
        report_lines.append("")

        # Control Charts
        charts = results.get("control_charts", {})
        if charts:
            report_lines.append("📈 CONTROL CHARTS")
            report_lines.append("-" * 80)
            for chart_name, chart_result in charts.items():
                if isinstance(chart_result, ControlChartResult):
                    report_lines.append(f"\n{chart_name.upper()} Chart ({chart_result.chart_type}):")
                    report_lines.append(f"  Center Line (CL): {chart_result.center_line:.4f}")
                    report_lines.append(f"  Upper Control Limit (UCL): {chart_result.upper_control_limit:.4f}")
                    report_lines.append(f"  Lower Control Limit (LCL): {chart_result.lower_control_limit:.4f}")
                    report_lines.append(f"  Out-of-Control Points: {len(chart_result.out_of_control_points)}")
                    if chart_result.out_of_control_points:
                        report_lines.append(
                            f"  OOC Indices: {chart_result.out_of_control_points[:10]}{'...' if len(chart_result.out_of_control_points) > 10 else ''}"
                        )
            report_lines.append("")

        # Baseline
        baseline = results.get("baseline")
        if baseline:
            report_lines.append("📊 BASELINE STATISTICS")
            report_lines.append("-" * 80)
            report_lines.append(f"Mean: {baseline.mean:.4f}")
            report_lines.append(f"Standard Deviation: {baseline.std:.4f}")
            report_lines.append(f"Median: {baseline.median:.4f}")
            report_lines.append(f"Min: {baseline.min:.4f}")
            report_lines.append(f"Max: {baseline.max:.4f}")
            report_lines.append(f"Range: {baseline.range:.4f}")
            report_lines.append(f"Sample Size: {baseline.sample_size:,}")
            report_lines.append(f"Subgroup Size: {baseline.subgroup_size}")
            if baseline.within_subgroup_std is not None:
                report_lines.append(f"Within-Subgroup Std: {baseline.within_subgroup_std:.4f}")
            if baseline.between_subgroup_std is not None:
                report_lines.append(f"Between-Subgroup Std: {baseline.between_subgroup_std:.4f}")
            report_lines.append("")

        # Process Capability
        capability = results.get("process_capability")
        if capability:
            report_lines.append("📊 PROCESS CAPABILITY")
            report_lines.append("-" * 80)
            # Will be implemented in Phase 2
            report_lines.append("Process capability analysis results (details to be added in Phase 2)")
            report_lines.append("")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            logger.info(f"Report saved to: {output_file}")

        return report
