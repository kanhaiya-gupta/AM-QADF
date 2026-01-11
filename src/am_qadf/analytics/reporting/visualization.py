"""
Visualization for PBF-LB/M Analytics

This module provides comprehensive visualization capabilities for PBF-LB/M
analytics results, including sensitivity analysis visualization, statistical
analysis visualization, and process analysis visualization.
"""

# Set matplotlib to use non-interactive backend for headless environments
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""

    # Plot parameters
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = "whitegrid"  # "whitegrid", "darkgrid", "white", "dark"

    # Color parameters
    color_palette: str = "viridis"  # "viridis", "plasma", "inferno", "magma"
    alpha: float = 0.7

    # Output parameters
    output_directory: str = "plots"
    save_format: str = "png"  # "png", "pdf", "svg"

    # Analysis parameters
    confidence_level: float = 0.95


@dataclass
class VisualizationResult:
    """Result of visualization generation."""

    success: bool
    visualization_type: str
    plot_paths: List[str]
    generation_time: float
    error_message: Optional[str] = None


class AnalysisVisualizer:
    """
    Analysis visualizer for PBF-LB/M analytics.

    This class provides comprehensive visualization capabilities for
    analytics results including sensitivity analysis, statistical analysis,
    and process analysis visualization.
    """

    def __init__(self, config: VisualizationConfig = None):
        """Initialize the visualizer."""
        self.config = config or VisualizationConfig()

        # Set matplotlib style - try seaborn styles, fallback to default
        try:
            if "seaborn-v0_8" in plt.style.available:
                plt.style.use("seaborn-v0_8")
            elif "seaborn" in plt.style.available:
                plt.style.use("seaborn")
            else:
                # Use seaborn's set_style if available, otherwise use default
                try:
                    sns.set_style(self.config.style)
                except:
                    plt.style.use("default")
        except Exception:
            # Fallback to default style if anything fails
            try:
                sns.set_style(self.config.style)
            except:
                plt.style.use("default")

        try:
            sns.set_palette(self.config.color_palette)
        except Exception:
            pass  # If palette setting fails, continue with default

        # Create output directory
        import os

        os.makedirs(self.config.output_directory, exist_ok=True)

        logger.info("Analysis Visualizer initialized")

    def visualize_sensitivity_analysis(
        self,
        sensitivity_results: Dict[str, Any],
        plot_title: str = "Sensitivity Analysis",
    ) -> VisualizationResult:
        """
        Visualize sensitivity analysis results.

        Args:
            sensitivity_results: Dictionary containing sensitivity analysis results
            plot_title: Title for the plots

        Returns:
            VisualizationResult: Visualization generation result
        """
        try:
            start_time = datetime.now()
            plot_paths = []

            # Generate Sobol indices plot
            if "sobol_analysis" in sensitivity_results:
                sobol_plot_path = self._plot_sobol_indices(sensitivity_results["sobol_analysis"], plot_title)
                plot_paths.append(sobol_plot_path)

            # Generate Morris screening plot
            if "morris_analysis" in sensitivity_results:
                morris_plot_path = self._plot_morris_screening(sensitivity_results["morris_analysis"], plot_title)
                plot_paths.append(morris_plot_path)

            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()

            # Create result
            result = VisualizationResult(
                success=True,
                visualization_type="SensitivityAnalysis",
                plot_paths=plot_paths,
                generation_time=generation_time,
            )

            logger.info(f"Sensitivity analysis visualization completed: {generation_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error in sensitivity analysis visualization: {e}")
            return VisualizationResult(
                success=False,
                visualization_type="SensitivityAnalysis",
                plot_paths=[],
                generation_time=0.0,
                error_message=str(e),
            )

    def visualize_statistical_analysis(
        self,
        statistical_results: Dict[str, Any],
        plot_title: str = "Statistical Analysis",
    ) -> VisualizationResult:
        """
        Visualize statistical analysis results.

        Args:
            statistical_results: Dictionary containing statistical analysis results
            plot_title: Title for the plots

        Returns:
            VisualizationResult: Visualization generation result
        """
        try:
            start_time = datetime.now()
            plot_paths = []

            # Generate PCA plot
            if "pca_analysis" in statistical_results:
                pca_plot_path = self._plot_pca_results(statistical_results["pca_analysis"], plot_title)
                plot_paths.append(pca_plot_path)

            # Generate correlation heatmap
            if "correlation_analysis" in statistical_results:
                corr_plot_path = self._plot_correlation_heatmap(statistical_results["correlation_analysis"], plot_title)
                plot_paths.append(corr_plot_path)

            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()

            # Create result
            result = VisualizationResult(
                success=True,
                visualization_type="StatisticalAnalysis",
                plot_paths=plot_paths,
                generation_time=generation_time,
            )

            logger.info(f"Statistical analysis visualization completed: {generation_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error in statistical analysis visualization: {e}")
            return VisualizationResult(
                success=False,
                visualization_type="StatisticalAnalysis",
                plot_paths=[],
                generation_time=0.0,
                error_message=str(e),
            )

    def visualize_process_analysis(
        self, process_results: Dict[str, Any], plot_title: str = "Process Analysis"
    ) -> VisualizationResult:
        """
        Visualize process analysis results.

        Args:
            process_results: Dictionary containing process analysis results
            plot_title: Title for the plots

        Returns:
            VisualizationResult: Visualization generation result
        """
        try:
            start_time = datetime.now()
            plot_paths = []

            # Generate parameter importance plot
            if "parameter_analysis" in process_results:
                param_plot_path = self._plot_parameter_importance(process_results["parameter_analysis"], plot_title)
                plot_paths.append(param_plot_path)

            # Generate quality prediction plot
            if "quality_analysis" in process_results:
                quality_plot_path = self._plot_quality_predictions(process_results["quality_analysis"], plot_title)
                plot_paths.append(quality_plot_path)

            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()

            # Create result
            result = VisualizationResult(
                success=True,
                visualization_type="ProcessAnalysis",
                plot_paths=plot_paths,
                generation_time=generation_time,
            )

            logger.info(f"Process analysis visualization completed: {generation_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error in process analysis visualization: {e}")
            return VisualizationResult(
                success=False,
                visualization_type="ProcessAnalysis",
                plot_paths=[],
                generation_time=0.0,
                error_message=str(e),
            )

    def _plot_sobol_indices(self, sobol_results: Any, plot_title: str) -> str:
        """Plot Sobol sensitivity indices."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        # Extract parameter names and indices
        parameter_names = sobol_results.parameter_names
        sensitivity_indices = sobol_results.sensitivity_indices

        # Extract first-order indices
        s1_indices = [sensitivity_indices.get(f"S1_{name}", 0) for name in parameter_names]
        st_indices = [sensitivity_indices.get(f"ST_{name}", 0) for name in parameter_names]

        # Create bar plot
        x = np.arange(len(parameter_names))
        width = 0.35

        ax.bar(
            x - width / 2,
            s1_indices,
            width,
            label="First-order (S1)",
            alpha=self.config.alpha,
        )
        ax.bar(
            x + width / 2,
            st_indices,
            width,
            label="Total-order (ST)",
            alpha=self.config.alpha,
        )

        ax.set_xlabel("Parameters")
        ax.set_ylabel("Sensitivity Index")
        ax.set_title(f"{plot_title} - Sobol Indices")
        ax.set_xticks(x)
        ax.set_xticklabels(parameter_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/sobol_indices_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        return plot_path

    def _plot_morris_screening(self, morris_results: Any, plot_title: str) -> str:
        """Plot Morris screening results."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        # Extract parameter names and indices
        parameter_names = morris_results.parameter_names
        sensitivity_indices = morris_results.sensitivity_indices

        # Extract mu_star indices
        mu_star_indices = [sensitivity_indices.get(f"mu_star_{name}", 0) for name in parameter_names]

        # Create bar plot
        x = np.arange(len(parameter_names))
        ax.bar(x, mu_star_indices, alpha=self.config.alpha)

        ax.set_xlabel("Parameters")
        ax.set_ylabel("Mu* (Elementary Effects)")
        ax.set_title(f"{plot_title} - Morris Screening")
        ax.set_xticks(x)
        ax.set_xticklabels(parameter_names, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/morris_screening_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        return plot_path

    def _plot_pca_results(self, pca_results: Any, plot_title: str) -> str:
        """Plot PCA results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figure_size)

        # Plot explained variance
        explained_variance = pca_results.explained_variance["explained_variance_ratio"]
        cumulative_variance = pca_results.explained_variance["cumulative_variance_ratio"]

        ax1.bar(
            range(1, len(explained_variance) + 1),
            explained_variance,
            alpha=self.config.alpha,
        )
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Explained Variance Ratio")
        ax1.set_title("Explained Variance by Component")
        ax1.grid(True, alpha=0.3)

        # Plot cumulative variance
        ax2.plot(
            range(1, len(cumulative_variance) + 1),
            cumulative_variance,
            "o-",
            alpha=self.config.alpha,
        )
        ax2.set_xlabel("Principal Component")
        ax2.set_ylabel("Cumulative Explained Variance")
        ax2.set_title("Cumulative Explained Variance")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"{plot_title} - PCA Results")
        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/pca_results_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        return plot_path

    def _plot_correlation_heatmap(self, correlation_results: Any, plot_title: str) -> str:
        """Plot correlation heatmap."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        # Extract correlation matrix
        correlation_matrix = correlation_results.analysis_results["correlation_matrix"]
        feature_names = correlation_results.analysis_results["feature_names"]

        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            xticklabels=feature_names,
            yticklabels=feature_names,
            annot=True,
            cmap="coolwarm",
            center=0,
            ax=ax,
        )

        ax.set_title(f"{plot_title} - Correlation Heatmap")
        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/correlation_heatmap_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        return plot_path

    def _plot_parameter_importance(self, parameter_results: Any, plot_title: str) -> str:
        """Plot parameter importance."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        # Extract parameter importance
        parameter_importance = parameter_results.parameter_importance
        parameter_names = list(parameter_importance.keys())
        importance_values = list(parameter_importance.values())

        # Create bar plot
        x = np.arange(len(parameter_names))
        ax.bar(x, importance_values, alpha=self.config.alpha)

        ax.set_xlabel("Parameters")
        ax.set_ylabel("Importance")
        ax.set_title(f"{plot_title} - Parameter Importance")
        ax.set_xticks(x)
        ax.set_xticklabels(parameter_names, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/parameter_importance_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        return plot_path

    def _plot_quality_predictions(self, quality_results: Any, plot_title: str) -> str:
        """Plot quality predictions."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        # Extract quality predictions
        quality_predictions = quality_results.quality_predictions

        # Create histogram
        ax.hist(quality_predictions, bins=30, alpha=self.config.alpha, edgecolor="black")

        ax.set_xlabel("Quality Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{plot_title} - Quality Predictions Distribution")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/quality_predictions_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        return plot_path


class SensitivityVisualizer(AnalysisVisualizer):
    """Specialized sensitivity analysis visualizer."""

    def __init__(self, config: VisualizationConfig = None):
        super().__init__(config)
        self.visualization_type = "Sensitivity"

    def visualize(
        self,
        sensitivity_results: Dict[str, Any],
        plot_title: str = "Sensitivity Analysis",
    ) -> VisualizationResult:
        """Visualize sensitivity analysis results."""
        return self.visualize_sensitivity_analysis(sensitivity_results, plot_title)


class QualityDashboardGenerator:
    """
    Quality Dashboard Generator for AM-QADF.

    This class provides capabilities to generate quality dashboards from
    quality assessment data stored in MongoDB or provided directly.
    Supports historical, real-time, and comparison dashboards.
    """

    def __init__(self, config: VisualizationConfig = None):
        """Initialize the quality dashboard generator."""
        self.config = config or VisualizationConfig()
        logger.info("Quality Dashboard Generator initialized")

        # Try to import validation module for comparison features
        try:
            from ...validation import ValidationClient

            self.validation_available = True
            logger.info("Validation module available - comparison features enabled")
        except ImportError:
            self.validation_available = False
            logger.warning("Validation module not available - comparison features disabled")

        # Try to import SPC module for control charts
        try:
            from ..spc import SPCClient, ControlChartResult

            self.spc_available = True
            logger.info("SPC module available - control charts enabled")
        except ImportError:
            self.spc_available = False
            logger.warning("SPC module not available - control charts disabled")

    def generate_dashboard(
        self,
        quality_data: Union[pd.DataFrame, Dict[str, Any]],
        dashboard_type: str = "historical",
        output_path: Optional[str] = None,
    ) -> VisualizationResult:
        """
        Generate a quality dashboard.

        Args:
            quality_data: Quality assessment data (DataFrame or dict)
            dashboard_type: Type of dashboard ('historical', 'realtime', 'comparison')
            output_path: Optional path to save dashboard

        Returns:
            VisualizationResult: Dashboard generation result
        """
        try:
            start_time = datetime.now()

            # Convert dict to DataFrame if needed
            if isinstance(quality_data, dict):
                quality_data = pd.DataFrame([quality_data])

            if len(quality_data) == 0:
                return VisualizationResult(
                    success=False,
                    visualization_type="QualityDashboard",
                    plot_paths=[],
                    generation_time=0.0,
                    error_message="No quality data provided",
                )

            plot_paths = []

            # Generate time series plot
            if "timestamp" in quality_data.columns:
                time_series_path = self._plot_quality_timeseries(quality_data, dashboard_type)
                plot_paths.append(time_series_path)

            # Generate metrics summary
            metrics_path = self._plot_quality_metrics(quality_data)
            plot_paths.append(metrics_path)

            # Generate SPC control charts if available and data supports it
            if self.spc_available and len(quality_data) >= 10:
                try:
                    spc_charts_path = self._plot_spc_control_charts(quality_data)
                    if spc_charts_path:
                        plot_paths.append(spc_charts_path)
                except Exception as e:
                    logger.warning(f"Could not generate SPC control charts: {e}")

            generation_time = (datetime.now() - start_time).total_seconds()

            result = VisualizationResult(
                success=True,
                visualization_type=f"QualityDashboard_{dashboard_type}",
                plot_paths=plot_paths,
                generation_time=generation_time,
            )

            logger.info(f"Quality dashboard generated: {generation_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error generating quality dashboard: {e}")
            return VisualizationResult(
                success=False,
                visualization_type="QualityDashboard",
                plot_paths=[],
                generation_time=0.0,
                error_message=str(e),
            )

    def _plot_quality_timeseries(self, quality_data: pd.DataFrame, dashboard_type: str) -> str:
        """Plot quality metrics time series."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        # Get timestamp column
        if "timestamp" in quality_data.columns:
            timestamps = pd.to_datetime(quality_data["timestamp"])
        else:
            timestamps = quality_data.index

        # Plot overall quality score
        if "overall_score" in quality_data.columns:
            ax.plot(
                timestamps,
                quality_data["overall_score"],
                label="Overall Quality",
                linewidth=2,
                marker="o",
                markersize=4,
            )

        # Plot other metrics if available
        if "completeness" in quality_data.columns:
            ax.plot(
                timestamps,
                quality_data["completeness"],
                label="Completeness",
                linewidth=2,
                linestyle="--",
                alpha=0.7,
            )

        if "coverage" in quality_data.columns:
            ax.plot(
                timestamps,
                quality_data["coverage"],
                label="Coverage",
                linewidth=2,
                linestyle=":",
                alpha=0.7,
            )

        if "consistency" in quality_data.columns:
            ax.plot(
                timestamps,
                quality_data["consistency"],
                label="Consistency",
                linewidth=2,
                linestyle="-.",
                alpha=0.7,
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Quality Score")
        ax.set_title(f"Quality Metrics Over Time - {dashboard_type.title()} Dashboard")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/quality_timeseries_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        return plot_path

    def _plot_quality_metrics(self, quality_data: pd.DataFrame) -> str:
        """Plot quality metrics summary."""
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        axes = axes.flatten()

        metrics_to_plot = [
            ("overall_score", "Overall Score"),
            ("completeness", "Completeness"),
            ("coverage", "Coverage"),
            ("consistency", "Consistency"),
        ]

        for idx, (metric_col, metric_name) in enumerate(metrics_to_plot):
            if metric_col in quality_data.columns:
                values = quality_data[metric_col].dropna()
                if len(values) > 0:
                    axes[idx].hist(values, bins=20, alpha=self.config.alpha, edgecolor="black")
                    axes[idx].axvline(
                        values.mean(),
                        color="r",
                        linestyle="--",
                        label=f"Mean: {values.mean():.2f}",
                    )
                    axes[idx].set_xlabel(metric_name)
                    axes[idx].set_ylabel("Frequency")
                    axes[idx].set_title(f"{metric_name} Distribution")
                    axes[idx].legend()
                    axes[idx].grid(True, alpha=0.3)
                else:
                    axes[idx].text(0.5, 0.5, "No data", ha="center", va="center")
                    axes[idx].set_title(metric_name)
            else:
                axes[idx].text(0.5, 0.5, "No data", ha="center", va="center")
                axes[idx].set_title(metric_name)

        plt.suptitle("Quality Metrics Summary")
        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/quality_metrics_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        return plot_path

    def _plot_spc_control_charts(self, quality_data: pd.DataFrame) -> Optional[str]:
        """Plot SPC control charts for quality metrics."""
        if not self.spc_available:
            return None

        try:
            from ..spc import SPCClient, SPCConfig, ControlChartResult

            # Create SPC client
            spc_config = SPCConfig(control_limit_sigma=3.0, subgroup_size=5, enable_warnings=True)
            spc_client = SPCClient(config=spc_config)

            # Identify metrics suitable for control charts (numeric columns with enough data)
            metrics_cols = [
                col
                for col in quality_data.columns
                if col not in ["timestamp"] and quality_data[col].dtype in ["float64", "int64"]
            ]

            if len(metrics_cols) == 0 or len(quality_data) < 10:
                logger.warning("Insufficient data for control charts")
                return None

            # Create subplots for control charts
            n_metrics = min(len(metrics_cols), 4)  # Limit to 4 charts
            fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics))
            if n_metrics == 1:
                axes = [axes]

            plot_paths = []

            for idx, metric_col in enumerate(metrics_cols[:n_metrics]):
                try:
                    # Get metric values
                    values = quality_data[metric_col].dropna().values

                    if len(values) < 10:
                        axes[idx].text(
                            0.5,
                            0.5,
                            f"Insufficient data for {metric_col}",
                            ha="center",
                            va="center",
                            transform=axes[idx].transAxes,
                        )
                        axes[idx].set_title(f"Control Chart: {metric_col}")
                        continue

                    # Create control chart
                    chart_result = spc_client.create_control_chart(values, chart_type="individual", config=spc_config)

                    # Plot control chart
                    sample_indices = chart_result.sample_indices
                    sample_values = chart_result.sample_values

                    # Plot data points
                    axes[idx].plot(
                        sample_indices, sample_values, "b-", marker="o", markersize=3, label="Data", linewidth=1, alpha=0.7
                    )

                    # Plot center line
                    axes[idx].axhline(
                        chart_result.center_line,
                        color="g",
                        linestyle="-",
                        linewidth=2,
                        label=f"CL: {chart_result.center_line:.3f}",
                    )

                    # Plot control limits
                    axes[idx].axhline(
                        chart_result.upper_control_limit,
                        color="r",
                        linestyle="--",
                        linewidth=1.5,
                        label=f"UCL: {chart_result.upper_control_limit:.3f}",
                    )
                    axes[idx].axhline(
                        chart_result.lower_control_limit,
                        color="r",
                        linestyle="--",
                        linewidth=1.5,
                        label=f"LCL: {chart_result.lower_control_limit:.3f}",
                    )

                    # Plot warning limits if available
                    if chart_result.upper_warning_limit is not None:
                        axes[idx].axhline(
                            chart_result.upper_warning_limit,
                            color="orange",
                            linestyle=":",
                            linewidth=1,
                            alpha=0.7,
                            label="UWL",
                        )
                    if chart_result.lower_warning_limit is not None:
                        axes[idx].axhline(
                            chart_result.lower_warning_limit,
                            color="orange",
                            linestyle=":",
                            linewidth=1,
                            alpha=0.7,
                            label="LWL",
                        )

                    # Highlight out-of-control points
                    if chart_result.out_of_control_points:
                        ooc_indices = [sample_indices[i] for i in chart_result.out_of_control_points]
                        ooc_values = [sample_values[i] for i in chart_result.out_of_control_points]
                        axes[idx].scatter(
                            ooc_indices,
                            ooc_values,
                            color="red",
                            marker="x",
                            s=100,
                            zorder=5,
                            label=f"OOC ({len(chart_result.out_of_control_points)})",
                        )

                    axes[idx].set_xlabel("Sample Index")
                    axes[idx].set_ylabel(metric_col)
                    axes[idx].set_title(f"Control Chart: {metric_col} (Type: {chart_result.chart_type})")
                    axes[idx].legend(loc="upper right", fontsize=8)
                    axes[idx].grid(True, alpha=0.3)

                except Exception as e:
                    logger.warning(f"Error creating control chart for {metric_col}: {e}")
                    axes[idx].text(0.5, 0.5, f"Error: {str(e)[:50]}", ha="center", va="center", transform=axes[idx].transAxes)
                    axes[idx].set_title(f"Control Chart: {metric_col} (Error)")

            plt.suptitle("SPC Control Charts for Quality Metrics", fontsize=14, fontweight="bold")
            plt.tight_layout(rect=[0, 0, 1, 0.98])

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = f"{self.config.output_directory}/spc_control_charts_{timestamp}.{self.config.save_format}"
            plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches="tight")
            plt.close()

            return plot_path

        except ImportError as e:
            logger.warning(f"SPC module not available for control charts: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating SPC control charts: {e}")
            return None

    def add_spc_control_charts(
        self,
        dashboard: VisualizationResult,
        quality_data: Union[pd.DataFrame, Dict[str, Any]],
        metric_names: Optional[List[str]] = None,
    ) -> VisualizationResult:
        """
        Add SPC control charts to existing dashboard.

        Args:
            dashboard: Existing dashboard VisualizationResult
            quality_data: Quality data (DataFrame or dict)
            metric_names: Optional list of metric names to create charts for

        Returns:
            Updated VisualizationResult with control charts added
        """
        if not self.spc_available:
            logger.warning("SPC module not available, cannot add control charts")
            return dashboard

        try:
            # Convert to DataFrame if needed
            if isinstance(quality_data, dict):
                quality_df = pd.DataFrame([quality_data])
            else:
                quality_df = quality_data.copy()

            # Generate control charts
            spc_path = self._plot_spc_control_charts(quality_df)

            if spc_path:
                # Add to existing plot paths
                dashboard.plot_paths.append(spc_path)
                logger.info(f"Added SPC control charts to dashboard: {spc_path}")

            return dashboard

        except Exception as e:
            logger.error(f"Error adding SPC control charts: {e}")
            return dashboard

    def create_comparison_dashboard(
        self,
        framework_data: Union[pd.DataFrame, Dict[str, Any]],
        reference_data: Union[pd.DataFrame, Dict[str, Any]],
        comparison_type: str = "mpm",
        output_path: Optional[str] = None,
    ) -> VisualizationResult:
        """
        Create comparison dashboard showing framework vs. reference (MPM or ground truth).

        Args:
            framework_data: Framework-generated quality data
            reference_data: Reference quality data (MPM system or ground truth)
            comparison_type: Type of comparison ('mpm', 'ground_truth', 'baseline')
            output_path: Optional path to save dashboard

        Returns:
            VisualizationResult with comparison dashboard plots
        """
        if not self.validation_available:
            return VisualizationResult(
                success=False,
                visualization_type="ComparisonDashboard",
                plot_paths=[],
                generation_time=0.0,
                error_message="Validation module not available. Install validation module for comparison features.",
            )

        try:
            start_time = datetime.now()

            # Convert to DataFrames if needed
            if isinstance(framework_data, dict):
                framework_df = pd.DataFrame([framework_data])
            else:
                framework_df = framework_data.copy()

            if isinstance(reference_data, dict):
                reference_df = pd.DataFrame([reference_data])
            else:
                reference_df = reference_data.copy()

            plot_paths = []

            # Generate side-by-side comparison plot
            comparison_path = self._plot_side_by_side_comparison(framework_df, reference_df, comparison_type)
            plot_paths.append(comparison_path)

            # Generate correlation plot
            correlation_path = self._plot_correlation_comparison(framework_df, reference_df, comparison_type)
            plot_paths.append(correlation_path)

            # Generate difference plot
            difference_path = self._plot_difference_analysis(framework_df, reference_df, comparison_type)
            plot_paths.append(difference_path)

            generation_time = (datetime.now() - start_time).total_seconds()

            result = VisualizationResult(
                success=True,
                visualization_type=f"ComparisonDashboard_{comparison_type}",
                plot_paths=plot_paths,
                generation_time=generation_time,
            )

            logger.info(f"Comparison dashboard generated: {generation_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error generating comparison dashboard: {e}")
            return VisualizationResult(
                success=False,
                visualization_type="ComparisonDashboard",
                plot_paths=[],
                generation_time=0.0,
                error_message=str(e),
            )

    def add_validation_metrics(
        self,
        dashboard: VisualizationResult,
        validation_results: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> VisualizationResult:
        """
        Add validation metrics to existing dashboard.

        Args:
            dashboard: Existing dashboard VisualizationResult
            validation_results: Validation results from ValidationClient
            output_path: Optional path to save updated dashboard

        Returns:
            Updated VisualizationResult with validation plots added
        """
        if not self.validation_available:
            logger.warning("Validation module not available, skipping validation metrics")
            return dashboard

        try:
            plot_paths = list(dashboard.plot_paths) if dashboard.plot_paths else []

            # Add validation summary plot
            if "mpm_comparison" in validation_results:
                validation_path = self._plot_validation_summary(validation_results)
                plot_paths.append(validation_path)

            # Add accuracy metrics plot if available
            if "accuracy" in validation_results and validation_results["accuracy"]:
                accuracy_path = self._plot_accuracy_metrics(validation_results["accuracy"])
                plot_paths.append(accuracy_path)

            # Add statistical test results if available
            if "statistical" in validation_results and validation_results["statistical"]:
                stat_path = self._plot_statistical_results(validation_results["statistical"])
                plot_paths.append(stat_path)

            updated_result = VisualizationResult(
                success=True,
                visualization_type=f"{dashboard.visualization_type}_with_validation",
                plot_paths=plot_paths,
                generation_time=dashboard.generation_time,
            )

            return updated_result

        except Exception as e:
            logger.error(f"Error adding validation metrics: {e}")
            return dashboard

    def _plot_side_by_side_comparison(
        self,
        framework_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        comparison_type: str,
    ) -> str:
        """Plot side-by-side comparison of metrics."""
        # Find common metrics
        common_metrics = set(framework_df.columns) & set(reference_df.columns)
        common_metrics = [m for m in common_metrics if framework_df[m].dtype in ["float64", "int64"]]

        if not common_metrics:
            common_metrics = ["overall_score", "completeness", "coverage", "consistency"]

        n_metrics = min(len(common_metrics), 4)
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        axes = axes.flatten()

        for idx, metric in enumerate(common_metrics[:n_metrics]):
            if metric in framework_df.columns and metric in reference_df.columns:
                framework_vals = framework_df[metric].dropna()
                reference_vals = reference_df[metric].dropna()

                if len(framework_vals) > 0 and len(reference_vals) > 0:
                    x = np.arange(max(len(framework_vals), len(reference_vals)))
                    width = 0.35

                    axes[idx].bar(
                        x[: len(framework_vals)] - width / 2,
                        framework_vals.values,
                        width,
                        label="Framework",
                        alpha=self.config.alpha,
                    )
                    axes[idx].bar(
                        x[: len(reference_vals)] + width / 2,
                        reference_vals.values,
                        width,
                        label=comparison_type.title(),
                        alpha=self.config.alpha,
                    )

                    axes[idx].set_xlabel("Sample")
                    axes[idx].set_ylabel(metric.replace("_", " ").title())
                    axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison')
                    axes[idx].legend()
                    axes[idx].grid(True, alpha=0.3)

        plt.suptitle(f"Framework vs. {comparison_type.title()} Comparison")
        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/comparison_side_by_side_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        return plot_path

    def _plot_correlation_comparison(
        self,
        framework_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        comparison_type: str,
    ) -> str:
        """Plot correlation between framework and reference metrics."""
        common_metrics = set(framework_df.columns) & set(reference_df.columns)
        common_metrics = [m for m in common_metrics if framework_df[m].dtype in ["float64", "int64"]]

        if not common_metrics:
            return ""

        fig, ax = plt.subplots(figsize=self.config.figure_size)

        for metric in common_metrics[:5]:  # Limit to 5 metrics
            framework_vals = framework_df[metric].dropna()
            reference_vals = reference_df[metric].dropna()

            min_len = min(len(framework_vals), len(reference_vals))
            if min_len > 0:
                framework_subset = framework_vals[:min_len]
                reference_subset = reference_vals[:min_len]

                # Calculate correlation
                try:
                    correlation = np.corrcoef(framework_subset, reference_subset)[0, 1]
                    if not np.isnan(correlation):
                        ax.scatter(
                            framework_subset,
                            reference_subset,
                            alpha=self.config.alpha,
                            label=f"{metric} (r={correlation:.3f})",
                        )
                except Exception:
                    pass

        # Add diagonal line
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Perfect agreement")

        ax.set_xlabel("Framework Values")
        ax.set_ylabel(f"{comparison_type.title()} Values")
        ax.set_title(f"Framework vs. {comparison_type.title()} Correlation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/comparison_correlation_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        return plot_path

    def _plot_difference_analysis(
        self,
        framework_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        comparison_type: str,
    ) -> str:
        """Plot difference analysis between framework and reference."""
        common_metrics = set(framework_df.columns) & set(reference_df.columns)
        common_metrics = [m for m in common_metrics if framework_df[m].dtype in ["float64", "int64"]]

        if not common_metrics:
            return ""

        differences = {}
        for metric in common_metrics:
            framework_vals = framework_df[metric].dropna()
            reference_vals = reference_df[metric].dropna()
            min_len = min(len(framework_vals), len(reference_vals))
            if min_len > 0:
                diff = framework_vals[:min_len].values - reference_vals[:min_len].values
                differences[metric] = diff

        if not differences:
            return ""

        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        axes = axes.flatten()

        for idx, (metric, diff) in enumerate(list(differences.items())[:4]):
            axes[idx].hist(diff, bins=20, alpha=self.config.alpha, edgecolor="black")
            axes[idx].axvline(np.mean(diff), color="r", linestyle="--", label=f"Mean: {np.mean(diff):.4f}")
            axes[idx].axvline(0, color="k", linestyle="-", alpha=0.3, label="Zero difference")
            axes[idx].set_xlabel("Difference (Framework - Reference)")
            axes[idx].set_ylabel("Frequency")
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Differences')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle(f"Framework vs. {comparison_type.title()} Difference Analysis")
        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/comparison_difference_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        return plot_path

    def _plot_validation_summary(self, validation_results: Dict[str, Any]) -> str:
        """Plot validation summary from validation results."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        if "mpm_comparison" in validation_results:
            mpm_results = validation_results["mpm_comparison"]
            if isinstance(mpm_results, dict):
                metrics = []
                correlations = []
                for metric_name, result in mpm_results.items():
                    if result and hasattr(result, "correlation"):
                        metrics.append(metric_name.replace("_", " ").title()[:20])
                        correlations.append(result.correlation)

                if metrics:
                    ax.barh(metrics, correlations, alpha=self.config.alpha)
                    ax.axvline(0.85, color="r", linestyle="--", label="Threshold (0.85)")
                    ax.set_xlabel("Correlation Coefficient")
                    ax.set_ylabel("Metric")
                    ax.set_title("MPM Comparison - Correlation Summary")
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/validation_summary_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        return plot_path

    def _plot_accuracy_metrics(self, accuracy_result: Any) -> str:
        """Plot accuracy validation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        axes = axes.flatten()

        if hasattr(accuracy_result, "rmse"):
            axes[0].bar(["RMSE"], [accuracy_result.rmse], alpha=self.config.alpha, color="blue")
            axes[0].set_ylabel("Error")
            axes[0].set_title("Root Mean Square Error")
            axes[0].grid(True, alpha=0.3, axis="y")

        if hasattr(accuracy_result, "mae"):
            axes[1].bar(["MAE"], [accuracy_result.mae], alpha=self.config.alpha, color="green")
            axes[1].set_ylabel("Error")
            axes[1].set_title("Mean Absolute Error")
            axes[1].grid(True, alpha=0.3, axis="y")

        if hasattr(accuracy_result, "r2_score"):
            axes[2].bar(["R²"], [accuracy_result.r2_score], alpha=self.config.alpha, color="orange")
            axes[2].set_ylabel("Score")
            axes[2].set_title("R² Score")
            axes[2].set_ylim([0, 1])
            axes[2].grid(True, alpha=0.3, axis="y")

        if hasattr(accuracy_result, "within_tolerance"):
            status = "Valid" if accuracy_result.within_tolerance else "Invalid"
            color = "green" if accuracy_result.within_tolerance else "red"
            axes[3].bar(["Status"], [1], alpha=self.config.alpha, color=color)
            axes[3].text(0, 0.5, status, ha="center", va="center", fontsize=14, fontweight="bold")
            axes[3].set_title("Validation Status")
            axes[3].set_ylim([0, 1])
            axes[3].set_yticks([])

        plt.suptitle("Accuracy Validation Metrics")
        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/accuracy_metrics_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        return plot_path

    def _plot_statistical_results(self, stat_result: Any) -> str:
        """Plot statistical validation results."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        if hasattr(stat_result, "test_statistic") and hasattr(stat_result, "p_value"):
            # Create visualization of test results
            x = np.arange(2)
            values = [stat_result.test_statistic, stat_result.p_value]
            labels = ["Test Statistic", "P-value"]
            colors = ["blue", "green" if stat_result.is_significant else "red"]

            bars = ax.bar(labels, values, alpha=self.config.alpha, color=colors)

            # Add significance level line for p-value
            if hasattr(stat_result, "significance_level"):
                ax.axhline(
                    stat_result.significance_level, color="r", linestyle="--", label=f"α = {stat_result.significance_level}"
                )

            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.4f}", ha="center", va="bottom")

            ax.set_ylabel("Value")
            ax.set_title(f"Statistical Test: {stat_result.test_name}")
            if hasattr(stat_result, "significance_level"):
                ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/statistical_results_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        return plot_path
