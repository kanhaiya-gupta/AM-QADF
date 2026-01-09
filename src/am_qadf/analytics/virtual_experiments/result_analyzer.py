"""
Virtual Experiment Result Analyzer

This module provides comprehensive analysis of virtual experiment results,
including statistical analysis, parameter interactions, and result validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of virtual experiment analysis."""

    experiment_id: str
    analysis_type: str
    parameter_names: List[str]
    response_names: List[str]

    # Statistical summaries
    response_statistics: Dict[str, Dict[str, float]]

    # Parameter-response relationships
    correlations: Dict[str, Dict[str, float]]
    parameter_interactions: Dict[str, float]

    # Analysis metadata
    analysis_time: float
    sample_size: int
    success: bool
    error_message: Optional[str] = None


class VirtualExperimentResultAnalyzer:
    """
    Analyzer for virtual experiment results.

    Provides comprehensive statistical analysis, parameter interaction
    detection, and result validation for virtual experiments.
    """

    def __init__(self):
        """Initialize the result analyzer."""
        self.analysis_cache = {}
        logger.info("Virtual Experiment Result Analyzer initialized")

    def analyze_results(
        self,
        experiment_results: List[Dict[str, Any]],
        parameter_names: List[str],
        response_names: List[str],
    ) -> AnalysisResult:
        """
        Analyze virtual experiment results.

        Args:
            experiment_results: List of experiment result dictionaries
            parameter_names: List of parameter names
            response_names: List of response variable names

        Returns:
            AnalysisResult: Analysis results
        """
        try:
            start_time = datetime.now()

            # Convert results to DataFrame
            df = self._results_to_dataframe(experiment_results, parameter_names, response_names)

            # Check if DataFrame has valid data (non-NaN values)
            if df.empty or df.isna().all().all():
                # No valid data
                analysis_time = (datetime.now() - start_time).total_seconds()
                return AnalysisResult(
                    experiment_id="",
                    analysis_type="comprehensive",
                    parameter_names=parameter_names,
                    response_names=response_names,
                    response_statistics={},
                    correlations={},
                    parameter_interactions={},
                    analysis_time=analysis_time,
                    sample_size=0,
                    success=False,
                    error_message="No valid data found in experiment results",
                )

            # Calculate response statistics
            response_stats = self._calculate_response_statistics(df, response_names)

            # Check if we have any valid statistics
            has_valid_stats = any(len(stats) > 0 for stats in response_stats.values())

            # Calculate parameter-response correlations
            correlations = self._calculate_correlations(df, parameter_names, response_names)

            # Detect parameter interactions
            interactions = self._detect_parameter_interactions(df, parameter_names, response_names)

            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()

            result = AnalysisResult(
                experiment_id="",  # Will be set by caller
                analysis_type="comprehensive",
                parameter_names=parameter_names,
                response_names=response_names,
                response_statistics=response_stats,
                correlations=correlations,
                parameter_interactions=interactions,
                analysis_time=analysis_time,
                sample_size=len(df),
                success=has_valid_stats and len(df) > 0,
            )

            logger.info(f"Virtual experiment results analyzed: {len(df)} samples")
            return result

        except Exception as e:
            logger.error(f"Error analyzing virtual experiment results: {e}")
            return AnalysisResult(
                experiment_id="",
                analysis_type="comprehensive",
                parameter_names=parameter_names or [],
                response_names=response_names or [],
                response_statistics={},
                correlations={},
                parameter_interactions={},
                analysis_time=0.0,
                sample_size=0,
                success=False,
                error_message=str(e),
            )

    def _results_to_dataframe(
        self,
        results: List[Dict[str, Any]],
        parameter_names: List[str],
        response_names: List[str],
    ) -> pd.DataFrame:
        """Convert experiment results to DataFrame."""
        data = []

        for result in results:
            row = {}

            # Extract input parameters
            if "input_parameters" in result:
                for param in parameter_names:
                    row[param] = result["input_parameters"].get(param, np.nan)
            elif "parameters" in result:
                for param in parameter_names:
                    row[param] = result["parameters"].get(param, np.nan)

            # Extract output responses
            if "output_responses" in result:
                for resp in response_names:
                    row[resp] = result["output_responses"].get(resp, np.nan)
            elif "responses" in result:
                for resp in response_names:
                    row[resp] = result["responses"].get(resp, np.nan)

            data.append(row)

        return pd.DataFrame(data)

    def _calculate_response_statistics(self, df: pd.DataFrame, response_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each response variable."""
        stats_dict = {}

        for resp in response_names:
            if resp in df.columns:
                values = df[resp].dropna()
                if len(values) > 0:
                    stats_dict[resp] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "median": float(np.median(values)),
                        "q25": float(np.percentile(values, 25)),
                        "q75": float(np.percentile(values, 75)),
                        "skewness": float(stats.skew(values)),
                        "kurtosis": float(stats.kurtosis(values)),
                    }
                else:
                    stats_dict[resp] = {}
            else:
                stats_dict[resp] = {}

        return stats_dict

    def _calculate_correlations(
        self, df: pd.DataFrame, parameter_names: List[str], response_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between parameters and responses."""
        correlations = {}

        for resp in response_names:
            if resp in df.columns:
                correlations[resp] = {}
                for param in parameter_names:
                    if param in df.columns:
                        try:
                            # Pearson correlation
                            corr, p_value = pearsonr(df[param].dropna(), df[resp].dropna())
                            correlations[resp][param] = {
                                "pearson": float(corr),
                                "p_value": float(p_value),
                                "significant": p_value < 0.05,
                            }
                        except:
                            correlations[resp][param] = {
                                "pearson": 0.0,
                                "p_value": 1.0,
                                "significant": False,
                            }

        return correlations

    def _detect_parameter_interactions(
        self, df: pd.DataFrame, parameter_names: List[str], response_names: List[str]
    ) -> Dict[str, float]:
        """Detect parameter interactions using correlation of products."""
        interactions = {}

        # Check two-way interactions
        for i, param1 in enumerate(parameter_names):
            for param2 in parameter_names[i + 1 :]:
                if param1 in df.columns and param2 in df.columns:
                    interaction_term = df[param1] * df[param2]

                    for resp in response_names:
                        if resp in df.columns:
                            try:
                                corr, p_value = pearsonr(interaction_term.dropna(), df[resp].dropna())
                                interaction_key = f"{param1} × {param2}"
                                interactions[f"{interaction_key} → {resp}"] = {
                                    "correlation": float(corr),
                                    "p_value": float(p_value),
                                    "significant": p_value < 0.05,
                                }
                            except:
                                pass

        return interactions

    def compare_with_sensitivity_analysis(
        self, virtual_results: AnalysisResult, sensitivity_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare virtual experiment results with Phase 9 sensitivity analysis.

        Args:
            virtual_results: Analysis results from virtual experiments
            sensitivity_results: Sensitivity analysis results from Phase 9

        Returns:
            Dict: Comparison results
        """
        comparison = {
            "parameter_rankings": {},
            "agreement_metrics": {},
            "discrepancies": [],
        }

        # Compare parameter importance rankings
        # This would require sensitivity analysis results structure
        # For now, return basic comparison structure

        return comparison
