"""
Multivariate SPC

Multivariate Statistical Process Control for monitoring multiple correlated variables simultaneously.
Includes Hotelling T² charts and PCA-based SPC.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from scipy import stats

# Try to import scikit-learn for PCA
try:
    from sklearn.decomposition import PCA

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("scikit-learn not available, PCA-based SPC will be limited")

logger = logging.getLogger(__name__)


@dataclass
class MultivariateSPCResult:
    """Result of multivariate SPC analysis."""

    hotelling_t2: np.ndarray  # Hotelling T² statistics
    ucl_t2: float  # Upper control limit for T²
    control_limits: Dict[str, float]  # Control limits for each component
    out_of_control_points: List[int]  # Indices of OOC points
    baseline_mean: np.ndarray  # Baseline mean vector
    baseline_covariance: np.ndarray  # Baseline covariance matrix
    principal_components: Optional[np.ndarray] = None  # PCA components
    explained_variance: Optional[np.ndarray] = None  # Explained variance ratio
    contribution_analysis: Optional[Dict[int, List[str]]] = None  # Variable contributions for OOC points
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultivariateSPCAnalyzer:
    """
    Multivariate SPC analyzer.

    Provides methods for:
    - Hotelling T² control chart for multivariate data
    - PCA-based SPC (dimensionality reduction)
    - Contribution analysis (identify variables contributing to OOC)
    - Detection of multivariate outliers
    """

    def __init__(self):
        """Initialize multivariate SPC analyzer."""
        self.sklearn_available = SKLEARN_AVAILABLE

    def create_hotelling_t2_chart(
        self,
        data: np.ndarray,
        baseline_data: Optional[np.ndarray] = None,
        config: Optional[Any] = None,  # SPCConfig type
        alpha: float = 0.05,
    ) -> MultivariateSPCResult:
        """
        Create Hotelling T² control chart.

        Hotelling T² = (x - μ)ᵀ Σ⁻¹ (x - μ)
        where x is observation, μ is mean vector, Σ is covariance matrix

        For Phase II (monitoring): UCL_T² = p(n+1)(n-1)/(n(n-p)) × F(α, p, n-p)

        Args:
            data: Multivariate process data (n_samples x n_variables)
            baseline_data: Optional baseline data for Phase I estimation (if None, uses first part of data)
            config: Optional SPCConfig
            alpha: Significance level for control limit (default: 0.05)

        Returns:
            MultivariateSPCResult object
        """
        data = np.asarray(data)

        if data.ndim != 2:
            raise ValueError(f"Data must be 2D (n_samples x n_variables), got shape {data.shape}")

        n_samples, p = data.shape

        if p < 2:
            raise ValueError("Multivariate SPC requires at least 2 variables")

        if n_samples < p + 1:
            raise ValueError(f"Need at least {p + 1} samples for {p} variables")

        # Determine baseline data
        if baseline_data is not None:
            baseline_data = np.asarray(baseline_data)
            if baseline_data.ndim != 2:
                raise ValueError("Baseline data must be 2D")
            n_baseline = baseline_data.shape[0]
            if n_baseline < p + 1:
                raise ValueError(f"Need at least {p + 1} baseline samples for {p} variables")
        else:
            # Use first 2/3 of data as baseline (Phase I)
            n_baseline = max(p + 1, int(0.67 * n_samples))
            baseline_data = data[:n_baseline]
            logger.info(f"Using first {n_baseline} samples as baseline (Phase I)")

        # Calculate baseline statistics
        baseline_mean = np.mean(baseline_data, axis=0)
        baseline_cov = np.cov(baseline_data, rowvar=False, ddof=1)

        # Handle singular or near-singular covariance matrix
        try:
            baseline_cov_inv = np.linalg.inv(baseline_cov)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            logger.warning("Covariance matrix is singular, using pseudo-inverse")
            baseline_cov_inv = np.linalg.pinv(baseline_cov)

        # Calculate Hotelling T² for all data points
        hotelling_t2 = np.zeros(n_samples)
        for i in range(n_samples):
            if np.any(np.isfinite(data[i])):
                deviation = data[i] - baseline_mean
                t2 = np.dot(np.dot(deviation, baseline_cov_inv), deviation)
                hotelling_t2[i] = t2
            else:
                hotelling_t2[i] = np.nan

        # Calculate control limit for Phase II (monitoring)
        # UCL_T² = p(n+1)(n-1)/(n(n-p)) × F(α, p, n-p)
        n_baseline = baseline_data.shape[0]
        if n_baseline > p:
            f_critical = stats.f.ppf(1 - alpha, p, n_baseline - p)
            ucl_t2 = (p * (n_baseline + 1) * (n_baseline - 1)) / (n_baseline * (n_baseline - p)) * f_critical
        else:
            # Fallback: use chi-square approximation
            ucl_t2 = stats.chi2.ppf(1 - alpha, p)
            logger.warning(f"Using chi-square approximation for UCL (n={n_baseline}, p={p})")

        # Detect out-of-control points
        valid_mask = np.isfinite(hotelling_t2)
        ooc_indices = [i for i in range(n_samples) if valid_mask[i] and hotelling_t2[i] > ucl_t2]

        # Calculate control limits for individual variables (for reference)
        control_limits = {}
        for j in range(p):
            var_mean = baseline_mean[j]
            var_std = np.sqrt(baseline_cov[j, j])
            control_limits[f"variable_{j}"] = {
                "mean": float(var_mean),
                "ucl": float(var_mean + 3 * var_std),
                "lcl": float(var_mean - 3 * var_std),
            }

        # Contribution analysis for OOC points
        contribution_analysis = {}
        if ooc_indices and len(ooc_indices) > 0:
            for idx in ooc_indices[:10]:  # Limit to first 10 for performance
                contributions = self._calculate_contribution(data[idx], baseline_mean, baseline_cov_inv)
                contribution_analysis[idx] = contributions

        return MultivariateSPCResult(
            hotelling_t2=hotelling_t2,
            ucl_t2=ucl_t2,
            control_limits=control_limits,
            out_of_control_points=ooc_indices,
            principal_components=None,
            explained_variance=None,
            contribution_analysis=contribution_analysis,
            baseline_mean=baseline_mean,
            baseline_covariance=baseline_cov,
            metadata={
                "n_samples": n_samples,
                "n_variables": p,
                "n_baseline": n_baseline,
                "alpha": alpha,
                "phase": "II",  # Monitoring phase
            },
        )

    def create_pca_chart(
        self,
        data: np.ndarray,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
        baseline_data: Optional[np.ndarray] = None,
        config: Optional[Any] = None,
        alpha: float = 0.05,
    ) -> MultivariateSPCResult:
        """
        Create PCA-based multivariate SPC chart.

        Uses Principal Component Analysis to reduce dimensionality,
        then applies Hotelling T² in the reduced space.

        Args:
            data: Multivariate process data (n_samples x n_variables)
            n_components: Number of principal components (None = auto-select based on variance)
            variance_threshold: Cumulative variance threshold for component selection (default: 0.95)
            baseline_data: Optional baseline data (if None, uses first part of data)
            config: Optional SPCConfig
            alpha: Significance level for control limit (default: 0.05)

        Returns:
            MultivariateSPCResult object
        """
        if not self.sklearn_available:
            raise ImportError("scikit-learn is required for PCA-based SPC. Install with: pip install scikit-learn")

        from sklearn.decomposition import PCA as SklearnPCA

        data = np.asarray(data)

        if data.ndim != 2:
            raise ValueError(f"Data must be 2D (n_samples x n_variables), got shape {data.shape}")

        n_samples, p = data.shape

        # Determine baseline data
        if baseline_data is not None:
            baseline_data = np.asarray(baseline_data)
        else:
            n_baseline = max(p + 1, int(0.67 * n_samples))
            baseline_data = data[:n_baseline]

        # Fit PCA on baseline data
        pca = SklearnPCA(n_components=n_components)
        pca.fit(baseline_data)

        # Determine number of components
        if n_components is None:
            # Select components to explain variance_threshold of variance
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_var >= variance_threshold) + 1
            n_components = min(n_components, p)  # Can't exceed original dimensions
            logger.info(f"Selected {n_components} components to explain {cumsum_var[n_components-1]:.2%} of variance")

        # Transform all data to principal component space
        pca_data_full = pca.transform(data)
        pca_baseline_full = pca.transform(baseline_data)

        # Use only selected components for SPC calculations
        pca_data = pca_data_full[:, :n_components]
        pca_baseline = pca_baseline_full[:, :n_components]

        # Calculate statistics in PC space
        pc_mean = np.mean(pca_baseline, axis=0)
        pc_cov = np.cov(pca_baseline, rowvar=False, ddof=1)

        # Ensure pc_cov is 2D (handle case where np.cov returns scalar)
        if pc_cov.ndim == 0:
            pc_cov = np.array([[pc_cov]])
        elif pc_cov.ndim == 1:
            pc_cov = np.diag(pc_cov)

        try:
            pc_cov_inv = np.linalg.inv(pc_cov)
        except np.linalg.LinAlgError:
            pc_cov_inv = np.linalg.pinv(pc_cov)
            logger.warning("PC covariance matrix is singular, using pseudo-inverse")

        # Calculate Hotelling T² in PC space
        hotelling_t2 = np.zeros(n_samples)
        for i in range(n_samples):
            if np.all(np.isfinite(pca_data[i])):
                deviation = pca_data[i] - pc_mean
                t2 = np.dot(np.dot(deviation, pc_cov_inv), deviation)
                hotelling_t2[i] = t2
            else:
                hotelling_t2[i] = np.nan

        # Calculate control limit
        n_baseline = pca_baseline.shape[0]
        if n_baseline > n_components:
            f_critical = stats.f.ppf(1 - alpha, n_components, n_baseline - n_components)
            ucl_t2 = (
                (n_components * (n_baseline + 1) * (n_baseline - 1)) / (n_baseline * (n_baseline - n_components)) * f_critical
            )
        else:
            ucl_t2 = stats.chi2.ppf(1 - alpha, n_components)

        # Detect out-of-control points
        valid_mask = np.isfinite(hotelling_t2)
        ooc_indices = [i for i in range(n_samples) if valid_mask[i] and hotelling_t2[i] > ucl_t2]

        # Control limits for PCs
        control_limits = {}
        for j in range(n_components):
            pc_mean_val = pc_mean[j]
            pc_std = np.sqrt(pc_cov[j, j])
            control_limits[f"PC_{j+1}"] = {
                "mean": float(pc_mean_val),
                "ucl": float(pc_mean_val + 3 * pc_std),
                "lcl": float(pc_mean_val - 3 * pc_std),
            }

        # Calculate contributions in original variable space
        contribution_analysis = {}
        if ooc_indices:
            for idx in ooc_indices[:10]:
                # Transform back to original space for contribution analysis
                # Use full PC data (all components) for inverse transform
                # Pad pc_mean to full dimensions for inverse transform
                pc_mean_full = np.zeros(pca_baseline_full.shape[1])
                pc_mean_full[:n_components] = pc_mean

                # Use full PC data for inverse transform
                original_point = pca.inverse_transform([pca_data_full[idx]])[0]
                original_mean = pca.inverse_transform([pc_mean_full])[0]
                original_deviation = original_point - original_mean

                contributions = {f"variable_{i}": float(abs(dev)) for i, dev in enumerate(original_deviation)}
                # Sort by magnitude
                contributions = dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))
                contribution_analysis[idx] = list(contributions.keys())[:5]  # Top 5 contributors

        return MultivariateSPCResult(
            hotelling_t2=hotelling_t2,
            ucl_t2=ucl_t2,
            control_limits=control_limits,
            out_of_control_points=ooc_indices,
            principal_components=pca.components_[:n_components],
            explained_variance=pca.explained_variance_ratio_[:n_components],
            contribution_analysis=contribution_analysis,
            baseline_mean=pc_mean,  # Mean in PC space
            baseline_covariance=pc_cov,  # Covariance in PC space
            metadata={
                "n_samples": n_samples,
                "n_variables": p,
                "n_components": n_components,
                "variance_explained": float(np.sum(pca.explained_variance_ratio_[:n_components])),
                "alpha": alpha,
                "method": "pca",
            },
        )

    def calculate_contribution(
        self,
        result: MultivariateSPCResult,
        out_of_control_index: int,
        original_data: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Calculate variable contributions to an out-of-control point.

        Args:
            result: MultivariateSPCResult
            out_of_control_index: Index of the OOC point
            original_data: Original data array (n_samples x n_variables)
            variable_names: Optional list of variable names

        Returns:
            Dictionary mapping variable names to contribution values
        """
        if out_of_control_index >= original_data.shape[0]:
            raise ValueError(f"Index {out_of_control_index} out of range")

        observation = original_data[out_of_control_index]
        baseline_mean = result.baseline_mean

        # Calculate Mahalanobis distance contribution per variable
        # Simplified: use squared deviation weighted by inverse covariance
        if result.baseline_covariance.ndim == 2:
            cov_inv = np.linalg.pinv(result.baseline_covariance)
            deviation = observation - baseline_mean

            contributions = {}
            for i in range(len(observation)):
                # Contribution is approximately: deviation[i]^2 / variance[i]
                var_name = variable_names[i] if variable_names and i < len(variable_names) else f"variable_{i}"
                contribution = abs(deviation[i]) * np.sqrt(abs(cov_inv[i, i]))
                contributions[var_name] = float(contribution)

            # Normalize contributions
            total = sum(contributions.values())
            if total > 0:
                contributions = {k: v / total for k, v in contributions.items()}
        else:
            # Fallback: simple squared deviation
            deviation = observation - baseline_mean
            contributions = {}
            for i in range(len(observation)):
                var_name = variable_names[i] if variable_names and i < len(variable_names) else f"variable_{i}"
                contributions[var_name] = float(deviation[i] ** 2)

            total = sum(contributions.values())
            if total > 0:
                contributions = {k: v / total for k, v in contributions.items()}

        return contributions

    def detect_multivariate_outliers(
        self, data: np.ndarray, baseline_mean: np.ndarray, baseline_cov: np.ndarray, alpha: float = 0.05
    ) -> List[int]:
        """
        Detect multivariate outliers using Mahalanobis distance.

        Args:
            data: Process data (n_samples x n_variables)
            baseline_mean: Baseline mean vector
            baseline_cov: Baseline covariance matrix
            alpha: Significance level

        Returns:
            List of outlier indices
        """
        data = np.asarray(data)
        n_samples, p = data.shape

        try:
            cov_inv = np.linalg.inv(baseline_cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(baseline_cov)

        # Calculate Mahalanobis distances (Hotelling T²)
        mahalanobis_distances = np.zeros(n_samples)
        for i in range(n_samples):
            if np.any(np.isfinite(data[i])):
                deviation = data[i] - baseline_mean
                md2 = np.dot(np.dot(deviation, cov_inv), deviation)
                mahalanobis_distances[i] = md2
            else:
                mahalanobis_distances[i] = np.nan

        # Critical value (chi-square)
        critical_value = stats.chi2.ppf(1 - alpha, p)

        # Find outliers
        outliers = [
            i for i in range(n_samples) if np.isfinite(mahalanobis_distances[i]) and mahalanobis_distances[i] > critical_value
        ]

        return outliers

    def _calculate_contribution(
        self, observation: np.ndarray, baseline_mean: np.ndarray, cov_inv: np.ndarray
    ) -> Dict[int, float]:
        """Internal helper for contribution calculation."""
        deviation = observation - baseline_mean
        contributions = {}

        for i in range(len(observation)):
            # Simplified contribution: deviation weighted by diagonal of inverse covariance
            contribution = abs(deviation[i]) * np.sqrt(abs(cov_inv[i, i]))
            contributions[i] = float(contribution) if np.isfinite(contribution) else 0.0

        # Normalize
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}

        return contributions
