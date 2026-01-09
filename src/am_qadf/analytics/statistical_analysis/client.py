"""
Advanced Analytics Client

Main client for comprehensive statistical analysis of voxel domain data.
Integrates descriptive statistics, correlation analysis, trend analysis, and pattern recognition.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import sys
import importlib.util
from pathlib import Path

# Import statistical analysis components with fallback for direct loading
try:
    from .descriptive_stats import DescriptiveStatsAnalyzer, DescriptiveStatistics
    from .correlation import CorrelationAnalyzer, CorrelationResults
    from .trends import TrendAnalyzer, TrendResults
    from .patterns import PatternAnalyzer, PatternResults
except ImportError:
    # Fallback for direct module loading (e.g., in notebooks)
    current_file = Path(__file__)
    module_dir = current_file.parent

    # Load descriptive_stats
    desc_path = module_dir / "descriptive_stats.py"
    spec_desc = importlib.util.spec_from_file_location("descriptive_stats", desc_path)
    desc_module = importlib.util.module_from_spec(spec_desc)
    spec_desc.loader.exec_module(desc_module)
    DescriptiveStatsAnalyzer = desc_module.DescriptiveStatsAnalyzer
    DescriptiveStatistics = desc_module.DescriptiveStatistics

    # Load correlation
    corr_path = module_dir / "correlation.py"
    spec_corr = importlib.util.spec_from_file_location("correlation", corr_path)
    corr_module = importlib.util.module_from_spec(spec_corr)
    spec_corr.loader.exec_module(corr_module)
    CorrelationAnalyzer = corr_module.CorrelationAnalyzer
    CorrelationResults = corr_module.CorrelationResults

    # Load trends
    trends_path = module_dir / "trends.py"
    spec_trends = importlib.util.spec_from_file_location("trends", trends_path)
    trends_module = importlib.util.module_from_spec(spec_trends)
    spec_trends.loader.exec_module(trends_module)
    TrendAnalyzer = trends_module.TrendAnalyzer
    TrendResults = trends_module.TrendResults

    # Load patterns
    patterns_path = module_dir / "patterns.py"
    spec_patterns = importlib.util.spec_from_file_location("patterns", patterns_path)
    patterns_module = importlib.util.module_from_spec(spec_patterns)
    spec_patterns.loader.exec_module(patterns_module)
    PatternAnalyzer = patterns_module.PatternAnalyzer
    PatternResults = patterns_module.PatternResults


class AdvancedAnalyticsClient:
    """
    Client for comprehensive statistical analysis of voxel domain data.

    Provides:
    - Descriptive statistics (mean, median, std, min, max, percentiles, skewness, kurtosis)
    - Correlation analysis (signal correlations, spatial/temporal autocorrelations)
    - Trend analysis (temporal trends, spatial trends, build progression)
    - Pattern recognition (spatial clusters, periodic patterns, anomalies, process patterns)
    """

    def __init__(self, mongo_client=None):
        """
        Initialize the advanced analytics client.

        Args:
            mongo_client: Optional MongoDB client for storing results (currently not used)
        """
        self.mongo_client = mongo_client
        self.stats_analyzer = DescriptiveStatsAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()

    def calculate_descriptive_statistics(
        self, voxel_data: Any, signals: Optional[List[str]] = None
    ) -> Dict[str, DescriptiveStatistics]:
        """
        Calculate descriptive statistics for signals.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to analyze (None = all signals)

        Returns:
            Dictionary mapping signal names to DescriptiveStatistics
        """
        return self.stats_analyzer.assess_all_signals(voxel_data, signals)

    def analyze_correlations(
        self,
        voxel_data: Any,
        signals: Optional[List[str]] = None,
        include_spatial: bool = True,
        include_temporal: bool = True,
    ) -> CorrelationResults:
        """
        Analyze correlations between signals and spatial/temporal autocorrelations.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to analyze (None = all signals)
            include_spatial: Whether to calculate spatial autocorrelations
            include_temporal: Whether to calculate temporal autocorrelations

        Returns:
            CorrelationResults object
        """
        return self.correlation_analyzer.analyze_correlations(voxel_data, signals, include_spatial, include_temporal)

    def analyze_trends(
        self,
        voxel_data: Any,
        signals: Optional[List[str]] = None,
        include_spatial: bool = True,
    ) -> TrendResults:
        """
        Analyze temporal and spatial trends.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to analyze (None = all signals)
            include_spatial: Whether to analyze spatial trends

        Returns:
            TrendResults object
        """
        return self.trend_analyzer.analyze_trends(voxel_data, signals, include_spatial)

    def analyze_patterns(
        self,
        voxel_data: Any,
        signals: Optional[List[str]] = None,
        include_anomalies: bool = True,
        include_process: bool = True,
    ) -> PatternResults:
        """
        Identify patterns in signals.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to analyze (None = all signals)
            include_anomalies: Whether to detect anomaly patterns
            include_process: Whether to identify process patterns

        Returns:
            PatternResults object
        """
        return self.pattern_analyzer.analyze_patterns(voxel_data, signals, include_anomalies, include_process)

    def comprehensive_analysis(
        self,
        voxel_data: Any,
        signals: Optional[List[str]] = None,
        include_spatial: bool = True,
        include_temporal: bool = True,
        include_anomalies: bool = True,
        include_process: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to analyze (None = all signals)
            include_spatial: Whether to include spatial analysis
            include_temporal: Whether to include temporal analysis
            include_anomalies: Whether to detect anomalies
            include_process: Whether to identify process patterns

        Returns:
            Dictionary containing all analysis results
        """
        print("📊 Starting comprehensive statistical analysis...")

        # Descriptive statistics
        print("  📈 Calculating descriptive statistics...")
        descriptive_stats = self.calculate_descriptive_statistics(voxel_data, signals)

        # Correlation analysis
        print("  🔗 Analyzing correlations...")
        correlations = self.analyze_correlations(voxel_data, signals, include_spatial, include_temporal)

        # Trend analysis
        print("  📉 Analyzing trends...")
        trends = self.analyze_trends(voxel_data, signals, include_spatial)

        # Pattern recognition
        print("  🔍 Identifying patterns...")
        patterns = self.analyze_patterns(voxel_data, signals, include_anomalies, include_process)

        print("✅ Statistical analysis complete")

        return {
            "descriptive_statistics": descriptive_stats,
            "correlations": correlations,
            "trends": trends,
            "patterns": patterns,
            "summary": {
                "num_signals_analyzed": len(descriptive_stats),
                "num_correlations": len(correlations.signal_correlations),
                "num_temporal_trends": len(trends.temporal_trends),
                "num_spatial_patterns": (
                    sum(len(p.get("clusters", [])) for p in patterns.spatial_patterns.values())
                    if patterns.spatial_patterns
                    else 0
                ),
            },
        }

    def generate_analysis_report(self, analysis_results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """
        Generate a human-readable analysis report.

        Args:
            analysis_results: Results from comprehensive_analysis()
            output_file: Optional file path to save report

        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ADVANCED STATISTICAL ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary
        summary = analysis_results.get("summary", {})
        report_lines.append("📊 SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Signals Analyzed: {summary.get('num_signals_analyzed', 0)}")
        report_lines.append(f"Correlations Found: {summary.get('num_correlations', 0)}")
        report_lines.append(f"Temporal Trends: {summary.get('num_temporal_trends', 0)}")
        report_lines.append(f"Spatial Patterns: {summary.get('num_spatial_patterns', 0)}")
        report_lines.append("")

        # Descriptive Statistics
        descriptive_stats = analysis_results.get("descriptive_statistics", {})
        if descriptive_stats:
            report_lines.append("📈 DESCRIPTIVE STATISTICS")
            report_lines.append("-" * 80)
            for signal_name, stats in descriptive_stats.items():
                report_lines.append(f"\n{signal_name}:")
                report_lines.append(f"  Mean: {stats.mean:.4f}")
                report_lines.append(f"  Median: {stats.median:.4f}")
                report_lines.append(f"  Std: {stats.std:.4f}")
                report_lines.append(f"  Range: [{stats.min:.4f}, {stats.max:.4f}]")
                report_lines.append(f"  Skewness: {stats.skewness:.4f}")
                report_lines.append(f"  Kurtosis: {stats.kurtosis:.4f}")
                report_lines.append(
                    f"  Valid: {stats.valid_count:,} / {stats.total_count:,} ({stats.valid_count/stats.total_count*100:.1f}%)"
                )
            report_lines.append("")

        # Correlations
        correlations = analysis_results.get("correlations")
        if correlations and correlations.signal_correlations:
            report_lines.append("🔗 CORRELATIONS")
            report_lines.append("-" * 80)
            report_lines.append("Signal Correlations:")
            for (signal1, signal2), corr in list(correlations.signal_correlations.items())[:10]:  # Show top 10
                report_lines.append(f"  {signal1} ↔ {signal2}: {corr:.4f}")
            if len(correlations.signal_correlations) > 10:
                report_lines.append(f"  ... and {len(correlations.signal_correlations) - 10} more")
            report_lines.append("")

        # Trends
        trends = analysis_results.get("trends")
        if trends and trends.temporal_trends:
            report_lines.append("📉 TRENDS")
            report_lines.append("-" * 80)
            report_lines.append("Temporal Trends:")
            for signal_name, trend in trends.temporal_trends.items():
                direction = trend.get("trend_direction", "unknown")
                slope = trend.get("slope", 0.0)
                r_value = trend.get("r_value", 0.0)
                report_lines.append(f"  {signal_name}: {direction} (slope={slope:.4f}, r={r_value:.4f})")
            report_lines.append("")

        # Patterns
        patterns = analysis_results.get("patterns")
        if patterns:
            report_lines.append("🔍 PATTERNS")
            report_lines.append("-" * 80)
            if patterns.spatial_patterns:
                report_lines.append("Spatial Patterns:")
                for signal_name, pattern in patterns.spatial_patterns.items():
                    num_clusters = pattern.get("num_clusters", 0)
                    report_lines.append(f"  {signal_name}: {num_clusters} clusters detected")

            if patterns.anomaly_patterns:
                report_lines.append("\nAnomaly Patterns:")
                for signal_name, anomalies in patterns.anomaly_patterns.items():
                    num_anomalies = anomalies.get("num_anomalies", 0)
                    fraction = anomalies.get("anomaly_fraction", 0.0)
                    report_lines.append(f"  {signal_name}: {num_anomalies} anomalies ({fraction*100:.2f}%)")
            report_lines.append("")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            print(f"✅ Report saved to: {output_file}")

        return report

    def analyze(
        self,
        model_id: str = None,
        voxel_grid: Any = None,
        signals: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform statistical analysis on voxel data.

        This is a convenience method that wraps comprehensive_analysis().

        Args:
            model_id: Optional model ID (for storage/retrieval)
            voxel_grid: Voxel domain data object
            signals: List of signal names to analyze (None = all signals)
            **kwargs: Additional arguments passed to comprehensive_analysis()

        Returns:
            Dictionary containing all analysis results
        """
        if voxel_grid is None:
            raise ValueError("voxel_grid is required")

        return self.comprehensive_analysis(voxel_data=voxel_grid, signals=signals, **kwargs)

    def query_results(self, model_id: str = None, analysis_type: str = None, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Query stored analysis results.

        Args:
            model_id: Model ID to query
            analysis_type: Type of analysis to query (e.g., 'descriptive', 'correlation')
            **kwargs: Additional query filters

        Returns:
            Analysis results dictionary or None if not found
        """
        # If mongo_client is available, query from storage
        if self.mongo_client:
            try:
                collection = self.mongo_client.get_collection("analytics_results")
                query = {}
                if model_id:
                    query["model_id"] = model_id
                if analysis_type:
                    query["analysis_type"] = analysis_type
                query.update(kwargs)

                result = collection.find_one(query)
                return result
            except Exception:
                # If storage query fails, return None
                return None

        # No storage available
        return None


# Alias for backward compatibility and notebook usage
StatisticalAnalysisClient = AdvancedAnalyticsClient
