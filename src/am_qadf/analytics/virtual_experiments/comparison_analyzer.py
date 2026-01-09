"""
Comparison Analyzer for Virtual Experiments

This module provides comparison capabilities between virtual experiment results
and Phase 9 sensitivity analysis predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy.stats import pearsonr
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparison analysis."""

    success: bool
    parameter_rankings_virtual: Dict[str, float]
    parameter_rankings_sensitivity: Dict[str, float]
    ranking_correlation: float
    agreement_metrics: Dict[str, float]
    discrepancies: List[str]
    error_message: Optional[str] = None


class ComparisonAnalyzer:
    """
    Analyzer for comparing virtual experiment results with sensitivity analysis.

    Compares parameter importance rankings, identifies agreements and discrepancies,
    and provides validation metrics.
    """

    def __init__(self):
        """Initialize the comparison analyzer."""
        logger.info("Comparison Analyzer initialized")

    def compare_parameter_importance(
        self,
        virtual_results: Dict[str, Any],
        sensitivity_results: Dict[str, Any],
        response_name: str = "quality",
    ) -> ComparisonResult:
        """
        Compare parameter importance rankings between virtual experiments and sensitivity analysis.

        Args:
            virtual_results: Virtual experiment analysis results
            sensitivity_results: Phase 9 sensitivity analysis results
            response_name: Response variable name to compare

        Returns:
            ComparisonResult: Comparison results
        """
        try:
            # Extract parameter rankings from virtual results
            virtual_rankings = self._extract_rankings_from_virtual(virtual_results, response_name)

            # Extract parameter rankings from sensitivity results
            sensitivity_rankings = self._extract_rankings_from_sensitivity(sensitivity_results, response_name)

            # Calculate ranking correlation
            ranking_corr = self._calculate_ranking_correlation(virtual_rankings, sensitivity_rankings)

            # Calculate agreement metrics
            agreement = self._calculate_agreement_metrics(virtual_rankings, sensitivity_rankings)

            # Identify discrepancies
            discrepancies = self._identify_discrepancies(virtual_rankings, sensitivity_rankings)

            return ComparisonResult(
                success=True,
                parameter_rankings_virtual=virtual_rankings,
                parameter_rankings_sensitivity=sensitivity_rankings,
                ranking_correlation=ranking_corr,
                agreement_metrics=agreement,
                discrepancies=discrepancies,
            )

        except Exception as e:
            logger.error(f"Error comparing parameter importance: {e}")
            return ComparisonResult(
                success=False,
                parameter_rankings_virtual={},
                parameter_rankings_sensitivity={},
                ranking_correlation=0.0,
                agreement_metrics={},
                discrepancies=[],
                error_message=str(e),
            )

    def _extract_rankings_from_virtual(self, virtual_results: Dict[str, Any], response_name: str) -> Dict[str, float]:
        """Extract parameter importance rankings from virtual experiment results."""
        rankings = {}

        # Try to extract from correlations
        if "correlations" in virtual_results and response_name in virtual_results["correlations"]:
            correlations = virtual_results["correlations"][response_name]
            for param, corr_data in correlations.items():
                if isinstance(corr_data, dict) and "pearson" in corr_data:
                    rankings[param] = abs(corr_data["pearson"])
                elif isinstance(corr_data, (int, float)):
                    rankings[param] = abs(corr_data)

        # Normalize rankings (0-1 scale)
        if rankings:
            max_val = max(rankings.values()) if rankings.values() else 1.0
            if max_val > 0:
                rankings = {k: v / max_val for k, v in rankings.items()}

        return rankings

    def _extract_rankings_from_sensitivity(self, sensitivity_results: Dict[str, Any], response_name: str) -> Dict[str, float]:
        """Extract parameter importance rankings from sensitivity analysis results."""
        rankings = {}

        # Try Sobol indices first
        if "sensitivity_indices" in sensitivity_results:
            indices = sensitivity_results["sensitivity_indices"]
            for key, value in indices.items():
                if key.startswith("S1_"):
                    param = key[3:]  # Remove 'S1_' prefix
                    if isinstance(value, (int, float)):
                        rankings[param] = abs(value)
                    elif isinstance(value, np.ndarray):
                        rankings[param] = abs(float(value.item() if value.size == 1 else value[0]))

        # Try Morris indices
        if not rankings and "sensitivity_indices" in sensitivity_results:
            indices = sensitivity_results["sensitivity_indices"]
            for key, value in indices.items():
                if key.startswith("mu_star_"):
                    param = key[8:]  # Remove 'mu_star_' prefix
                    if isinstance(value, (int, float)):
                        rankings[param] = abs(value)
                    elif isinstance(value, np.ndarray):
                        rankings[param] = abs(float(value.item() if value.size == 1 else value[0]))

        # Normalize rankings
        if rankings:
            max_val = max(rankings.values()) if rankings.values() else 1.0
            if max_val > 0:
                rankings = {k: v / max_val for k, v in rankings.items()}

        return rankings

    def _calculate_ranking_correlation(self, rankings1: Dict[str, float], rankings2: Dict[str, float]) -> float:
        """Calculate correlation between two parameter rankings."""
        # Find common parameters
        common_params = set(rankings1.keys()) & set(rankings2.keys())

        if len(common_params) < 2:
            return 0.0

        # Extract values for common parameters
        values1 = [rankings1[param] for param in common_params]
        values2 = [rankings2[param] for param in common_params]

        try:
            corr, _ = pearsonr(values1, values2)
            return float(corr)
        except:
            return 0.0

    def _calculate_agreement_metrics(self, rankings1: Dict[str, float], rankings2: Dict[str, float]) -> Dict[str, float]:
        """Calculate agreement metrics between rankings."""
        common_params = set(rankings1.keys()) & set(rankings2.keys())

        if len(common_params) == 0:
            return {"agreement": 0.0, "top3_agreement": 0.0}

        # Calculate overall agreement (1 - mean absolute difference)
        differences = [abs(rankings1[p] - rankings2[p]) for p in common_params]
        agreement = 1.0 - (np.mean(differences) if differences else 0.0)

        # Top 3 agreement
        top3_1 = sorted(rankings1.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_2 = sorted(rankings2.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_params_1 = {p for p, _ in top3_1}
        top3_params_2 = {p for p, _ in top3_2}
        top3_agreement = len(top3_params_1 & top3_params_2) / 3.0

        return {
            "agreement": float(agreement),
            "top3_agreement": float(top3_agreement),
            "common_parameters": len(common_params),
        }

    def _identify_discrepancies(self, rankings1: Dict[str, float], rankings2: Dict[str, float]) -> List[str]:
        """Identify significant discrepancies between rankings."""
        discrepancies = []
        common_params = set(rankings1.keys()) & set(rankings2.keys())

        for param in common_params:
            diff = abs(rankings1[param] - rankings2[param])
            if diff > 0.3:  # Significant difference threshold
                discrepancies.append(
                    f"{param}: Virtual={rankings1[param]:.3f}, Sensitivity={rankings2[param]:.3f}, Diff={diff:.3f}"
                )

        return discrepancies
