"""
Temporal Anomaly Visualization

Provides time-series visualization of anomalies including:
- Anomaly scores over time/layers
- Layer-by-layer analysis
- Anomaly timeline
- Trend analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TemporalAnomalyVisualizer:
    """
    Visualizer for temporal anomaly detection results.
    """

    def __init__(self):
        """Initialize temporal anomaly visualizer."""
        self.time_series: Optional[np.ndarray] = None
        self.layer_numbers: Optional[np.ndarray] = None
        self.anomaly_scores: Optional[np.ndarray] = None
        self.anomaly_labels: Optional[np.ndarray] = None

    def set_data(
        self,
        time_series: Optional[np.ndarray] = None,
        layer_numbers: Optional[np.ndarray] = None,
        anomaly_scores: np.ndarray = None,
        anomaly_labels: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set data for visualization.

        Args:
            time_series: Optional array of timestamps or time indices
            layer_numbers: Optional array of layer numbers
            anomaly_scores: Array of anomaly scores
            anomaly_labels: Optional array of boolean anomaly labels
        """
        if time_series is not None:
            self.time_series = np.asarray(time_series)
        if layer_numbers is not None:
            self.layer_numbers = np.asarray(layer_numbers)
        self.anomaly_scores = np.asarray(anomaly_scores)
        if anomaly_labels is not None:
            self.anomaly_labels = np.asarray(anomaly_labels, dtype=bool)
        else:
            self.anomaly_labels = None

    def plot_time_series(
        self,
        threshold: Optional[float] = None,
        title: str = "Anomaly Scores Over Time",
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot anomaly scores over time.

        Args:
            threshold: Optional threshold line to display
            title: Plot title
            show: Whether to show the plot immediately

        Returns:
            Matplotlib figure
        """
        if self.anomaly_scores is None:
            logger.warning("Anomaly scores not set. Call set_data() first.")
            return None

        # Determine x-axis
        if self.time_series is not None:
            x_data = self.time_series
            x_label = "Time"
        elif self.layer_numbers is not None:
            x_data = self.layer_numbers
            x_label = "Layer Number"
        else:
            x_data = np.arange(len(self.anomaly_scores))
            x_label = "Sample Index"

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot scores
        ax.plot(
            x_data,
            self.anomaly_scores,
            "b-",
            alpha=0.6,
            linewidth=1,
            label="Anomaly Score",
        )

        # Highlight anomalies if labels available
        if self.anomaly_labels is not None:
            anomaly_x = x_data[self.anomaly_labels]
            anomaly_scores = self.anomaly_scores[self.anomaly_labels]
            ax.scatter(
                anomaly_x,
                anomaly_scores,
                c="red",
                s=50,
                alpha=0.8,
                label="Detected Anomalies",
                zorder=5,
            )

        # Add threshold line
        if threshold is not None:
            ax.axhline(
                threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Threshold: {threshold:.3f}",
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Anomaly Score")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()

        return fig

    def plot_layer_analysis(self, title: str = "Layer-by-Layer Anomaly Analysis", show: bool = True) -> plt.Figure:
        """
        Plot layer-by-layer anomaly analysis.

        Args:
            title: Plot title
            show: Whether to show the plot immediately

        Returns:
            Matplotlib figure
        """
        if self.layer_numbers is None or self.anomaly_scores is None:
            logger.warning("Layer numbers or anomaly scores not set.")
            return None

        # Group by layer
        unique_layers = np.unique(self.layer_numbers)
        layer_scores = []
        layer_counts = []
        layer_anomaly_counts = []

        for layer in unique_layers:
            mask = self.layer_numbers == layer
            layer_scores.append(np.mean(self.anomaly_scores[mask]))
            layer_counts.append(np.sum(mask))
            if self.anomaly_labels is not None:
                layer_anomaly_counts.append(np.sum(self.anomaly_labels[mask]))
            else:
                layer_anomaly_counts.append(0)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Average anomaly score per layer
        ax = axes[0]
        ax.plot(unique_layers, layer_scores, "o-", linewidth=2, markersize=8)
        ax.set_xlabel("Layer Number")
        ax.set_ylabel("Average Anomaly Score")
        ax.set_title("Average Anomaly Score by Layer")
        ax.grid(True, alpha=0.3)

        # Plot 2: Anomaly count per layer
        ax = axes[1]
        ax.bar(unique_layers, layer_anomaly_counts, alpha=0.7, color="red")
        ax.set_xlabel("Layer Number")
        ax.set_ylabel("Number of Anomalies")
        ax.set_title("Anomaly Count by Layer")
        ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_anomaly_timeline(self, title: str = "Anomaly Timeline", show: bool = True) -> plt.Figure:
        """
        Plot chronological timeline of anomalies.

        Args:
            title: Plot title
            show: Whether to show the plot immediately

        Returns:
            Matplotlib figure
        """
        if self.anomaly_labels is None:
            logger.warning("Anomaly labels not set.")
            return None

        # Determine x-axis
        if self.time_series is not None:
            x_data = self.time_series
            x_label = "Time"
        elif self.layer_numbers is not None:
            x_data = self.layer_numbers
            x_label = "Layer Number"
        else:
            x_data = np.arange(len(self.anomaly_labels))
            x_label = "Sample Index"

        anomaly_indices = np.where(self.anomaly_labels)[0]

        if len(anomaly_indices) == 0:
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.text(0.5, 0.5, "No anomalies detected", transform=ax.transAxes, ha="center")
            return fig

        fig, ax = plt.subplots(figsize=(14, 4))

        # Plot timeline
        for idx in anomaly_indices:
            ax.axvline(x_data[idx], color="red", alpha=0.6, linewidth=2)

        # Add score overlay
        if self.anomaly_scores is not None:
            anomaly_scores = self.anomaly_scores[anomaly_indices]
            ax.scatter(
                x_data[anomaly_indices],
                anomaly_scores,
                c="red",
                s=100,
                alpha=0.8,
                zorder=5,
                label="Anomalies",
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Anomaly Score" if self.anomaly_scores is not None else "")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        if show:
            plt.show()

        return fig

    def plot_trend_analysis(
        self,
        window_size: int = 10,
        title: str = "Anomaly Trend Analysis",
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot long-term trend analysis with moving average.

        Args:
            window_size: Size of moving average window
            title: Plot title
            show: Whether to show the plot immediately

        Returns:
            Matplotlib figure
        """
        if self.anomaly_scores is None:
            logger.warning("Anomaly scores not set.")
            return None

        # Determine x-axis
        if self.time_series is not None:
            x_data = self.time_series
            x_label = "Time"
        elif self.layer_numbers is not None:
            x_data = self.layer_numbers
            x_label = "Layer Number"
        else:
            x_data = np.arange(len(self.anomaly_scores))
            x_label = "Sample Index"

        # Calculate moving average
        if len(self.anomaly_scores) >= window_size:
            moving_avg = np.convolve(self.anomaly_scores, np.ones(window_size) / window_size, mode="valid")
            moving_avg_x = x_data[window_size - 1 :]
        else:
            moving_avg = np.array([np.mean(self.anomaly_scores)])
            moving_avg_x = np.array([np.mean(x_data)])

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot raw scores
        ax.plot(
            x_data,
            self.anomaly_scores,
            "b-",
            alpha=0.3,
            linewidth=1,
            label="Raw Scores",
        )

        # Plot moving average
        ax.plot(
            moving_avg_x,
            moving_avg,
            "r-",
            linewidth=2,
            label=f"Moving Average (window={window_size})",
        )

        # Highlight anomalies
        if self.anomaly_labels is not None:
            anomaly_x = x_data[self.anomaly_labels]
            anomaly_scores = self.anomaly_scores[self.anomaly_labels]
            ax.scatter(
                anomaly_x,
                anomaly_scores,
                c="red",
                s=50,
                alpha=0.8,
                label="Detected Anomalies",
                zorder=5,
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Anomaly Score")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()

        return fig
