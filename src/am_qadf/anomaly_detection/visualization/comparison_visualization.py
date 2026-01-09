"""
Comparison Visualization

Provides visualization for comparing multiple anomaly detection methods including:
- Method comparison charts
- Performance metrics visualization
- ROC/PR curves
- Confusion matrices
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

try:
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class ComparisonVisualizer:
    """
    Visualizer for comparing multiple anomaly detection methods.
    """

    def __init__(self):
        """Initialize comparison visualizer."""
        self.method_results: Dict[str, Dict[str, Any]] = {}

    def add_method_results(
        self,
        method_name: str,
        metrics: Any,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        y_scores: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add results from a detection method.

        Args:
            method_name: Name of the method
            metrics: Metrics object (with precision, recall, f1_score, etc.)
            y_true: Optional true labels
            y_pred: Optional predicted labels
            y_scores: Optional anomaly scores
        """
        self.method_results[method_name] = {
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_scores": y_scores,
        }

    def plot_method_comparison(
        self,
        metric: str = "f1_score",
        title: str = "Method Comparison",
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot comparison of methods by a specific metric.

        Args:
            metric: Metric to compare ('f1_score', 'precision', 'recall', 'roc_auc')
            title: Plot title
            show: Whether to show the plot immediately

        Returns:
            Matplotlib figure
        """
        if not self.method_results:
            logger.warning("No method results added.")
            return None

        method_names = list(self.method_results.keys())
        metric_values = []

        for name in method_names:
            metrics = self.method_results[name]["metrics"]
            value = getattr(metrics, metric, 0.0)
            if value is None:
                value = 0.0
            metric_values.append(value)

        # Sort by metric value
        sorted_pairs = sorted(zip(method_names, metric_values), key=lambda x: x[1], reverse=True)
        method_names, metric_values = zip(*sorted_pairs)

        fig, ax = plt.subplots(figsize=(12, 8))

        bars = ax.barh(
            method_names,
            metric_values,
            color=plt.cm.viridis(np.linspace(0, 1, len(method_names))),
        )

        # Add value labels
        for i, (name, value) in enumerate(zip(method_names, metric_values)):
            ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=10)

        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_xlim([0, max(metric_values) * 1.2 if max(metric_values) > 0 else 1.0])

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_precision_recall_comparison(self, title: str = "Precision-Recall Comparison", show: bool = True) -> plt.Figure:
        """
        Plot precision-recall scatter for all methods.

        Args:
            title: Plot title
            show: Whether to show the plot immediately

        Returns:
            Matplotlib figure
        """
        if not self.method_results:
            logger.warning("No method results added.")
            return None

        method_names = list(self.method_results.keys())
        precisions = []
        recalls = []

        for name in method_names:
            metrics = self.method_results[name]["metrics"]
            precisions.append(metrics.precision)
            recalls.append(metrics.recall)

        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(
            recalls,
            precisions,
            s=200,
            alpha=0.7,
            c=range(len(method_names)),
            cmap="viridis",
            edgecolors="black",
            linewidth=2,
        )

        for i, name in enumerate(method_names):
            ax.annotate(
                name,
                (recalls[i], precisions[i]),
                fontsize=10,
                alpha=0.9,
                fontweight="bold",
            )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_roc_curves(self, title: str = "ROC Curves Comparison", show: bool = True) -> plt.Figure:
        """
        Plot ROC curves for all methods.

        Args:
            title: Plot title
            show: Whether to show the plot immediately

        Returns:
            Matplotlib figure
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available for ROC curves")
            return None

        if not self.method_results:
            logger.warning("No method results added.")
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.method_results)))

        for i, (name, results) in enumerate(self.method_results.items()):
            y_true = results["y_true"]
            y_scores = results["y_scores"]

            if y_true is None or y_scores is None:
                continue

            try:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = results["metrics"].roc_auc
                if roc_auc is None:
                    continue

                ax.plot(
                    fpr,
                    tpr,
                    label=f"{name} (AUC = {roc_auc:.3f})",
                    linewidth=2,
                    color=colors[i],
                )
            except Exception as e:
                logger.warning(f"Error plotting ROC for {name}: {e}")

        ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_confusion_matrices(
        self,
        methods: Optional[List[str]] = None,
        title: str = "Confusion Matrices",
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot confusion matrices for specified methods.

        Args:
            methods: List of method names to plot (if None, plot all)
            title: Plot title
            show: Whether to show the plot immediately

        Returns:
            Matplotlib figure
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available for confusion matrices")
            return None

        if not self.method_results:
            logger.warning("No method results added.")
            return None

        if methods is None:
            methods = list(self.method_results.keys())

        n_methods = len(methods)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        # Ensure axes is always a flat list of Axes objects
        if n_methods == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten().tolist()
        else:
            axes = list(axes) if not isinstance(axes, list) else axes

        for i, method_name in enumerate(methods):
            if method_name not in self.method_results:
                continue

            results = self.method_results[method_name]
            y_true = results["y_true"]
            y_pred = results["y_pred"]

            if y_true is None or y_pred is None:
                continue

            ax = axes[i]
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"],
            )
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            ax.set_title(method_name)

        # Hide unused subplots
        for i in range(n_methods, len(axes)):
            axes[i].axis("off")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_comprehensive_comparison(self, title: str = "Comprehensive Method Comparison", show: bool = True) -> plt.Figure:
        """
        Create comprehensive comparison with multiple subplots.

        Args:
            title: Plot title
            show: Whether to show the plot immediately

        Returns:
            Matplotlib figure
        """
        if not self.method_results:
            logger.warning("No method results added.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # Plot 1: F1-Score comparison
        ax = axes[0, 0]
        method_names = list(self.method_results.keys())
        f1_scores = [getattr(self.method_results[name]["metrics"], "f1_score", 0.0) for name in method_names]

        sorted_pairs = sorted(zip(method_names, f1_scores), key=lambda x: x[1], reverse=True)
        method_names, f1_scores = zip(*sorted_pairs)

        bars = ax.barh(
            method_names,
            f1_scores,
            color=plt.cm.viridis(np.linspace(0, 1, len(method_names))),
        )
        ax.set_xlabel("F1-Score")
        ax.set_title("F1-Score Comparison")
        ax.grid(True, alpha=0.3, axis="x")
        for i, (name, score) in enumerate(zip(method_names, f1_scores)):
            ax.text(score + 0.01, i, f"{score:.3f}", va="center", fontsize=9)

        # Plot 2: Precision-Recall
        ax = axes[0, 1]
        precisions = [self.method_results[name]["metrics"].precision for name in method_names]
        recalls = [self.method_results[name]["metrics"].recall for name in method_names]
        ax.scatter(
            recalls,
            precisions,
            s=200,
            alpha=0.7,
            c=range(len(method_names)),
            cmap="viridis",
            edgecolors="black",
            linewidth=2,
        )
        for i, name in enumerate(method_names):
            ax.annotate(name, (recalls[i], precisions[i]), fontsize=9, alpha=0.9)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Trade-off")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

        # Plot 3: ROC-AUC (if available)
        ax = axes[1, 0]
        roc_aucs = []
        valid_methods = []
        for name in method_names:
            roc_auc = getattr(self.method_results[name]["metrics"], "roc_auc", None)
            if roc_auc is not None and roc_auc > 0:
                roc_aucs.append(roc_auc)
                valid_methods.append(name)

        if valid_methods:
            bars = ax.barh(
                valid_methods,
                roc_aucs,
                color=plt.cm.plasma(np.linspace(0, 1, len(valid_methods))),
            )
            ax.set_xlabel("ROC-AUC")
            ax.set_title("ROC-AUC Comparison")
            ax.grid(True, alpha=0.3, axis="x")
            for i, (name, auc) in enumerate(zip(valid_methods, roc_aucs)):
                ax.text(auc + 0.01, i, f"{auc:.3f}", va="center", fontsize=9)
        else:
            ax.text(
                0.5,
                0.5,
                "ROC-AUC not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("ROC-AUC Comparison")

        # Plot 4: Metrics summary table
        ax = axes[1, 1]
        ax.axis("off")
        table_data = []
        for name in method_names[:10]:  # Limit to top 10
            m = self.method_results[name]["metrics"]
            roc = m.roc_auc if m.roc_auc else 0.0
            table_data.append(
                [
                    name[:20],  # Truncate long names
                    f"{m.precision:.3f}",
                    f"{m.recall:.3f}",
                    f"{m.f1_score:.3f}",
                    f"{roc:.3f}",
                ]
            )

        table = ax.table(
            cellText=table_data,
            colLabels=["Method", "Precision", "Recall", "F1", "ROC-AUC"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax.set_title("Metrics Summary", fontsize=11, fontweight="bold", pad=20)

        plt.tight_layout()

        if show:
            plt.show()

        return fig
