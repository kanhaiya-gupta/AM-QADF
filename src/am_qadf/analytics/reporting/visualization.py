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
    """

    def __init__(self, config: VisualizationConfig = None):
        """Initialize the quality dashboard generator."""
        self.config = config or VisualizationConfig()
        logger.info("Quality Dashboard Generator initialized")

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
