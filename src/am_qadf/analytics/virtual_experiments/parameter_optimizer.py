"""
Parameter Optimizer for Virtual Experiments

This module provides parameter optimization capabilities including single-objective
and multi-objective optimization from virtual experiment results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy.optimize import minimize, differential_evolution
import warnings

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""

    success: bool
    optimal_parameters: Dict[str, float]
    optimal_objectives: Dict[str, float]
    optimization_method: str
    iterations: int
    convergence_info: Dict[str, Any]
    pareto_front: Optional[List[Dict[str, Any]]] = None  # For multi-objective
    error_message: Optional[str] = None


class ParameterOptimizer:
    """
    Parameter optimizer for virtual experiment results.

    Provides single-objective and multi-objective optimization capabilities
    to identify optimal parameter sets from virtual experiment results.
    """

    def __init__(self):
        """Initialize the parameter optimizer."""
        self.optimization_cache = {}
        logger.info("Parameter Optimizer initialized")

    def optimize_single_objective(
        self,
        experiment_results: List[Dict[str, Any]],
        parameter_names: List[str],
        objective_name: str,
        maximize: bool = True,
        constraints: Dict[str, Tuple[float, float]] = None,
    ) -> OptimizationResult:
        """
        Optimize parameters for a single objective.

        Args:
            experiment_results: List of experiment result dictionaries
            parameter_names: List of parameter names
            objective_name: Name of objective to optimize
            maximize: Whether to maximize (True) or minimize (False)
            constraints: Parameter constraints {name: (min, max)}

        Returns:
            OptimizationResult: Optimization results
        """
        try:
            # Convert results to DataFrame
            df = self._results_to_dataframe(experiment_results, parameter_names, [objective_name])

            # Create objective function from data (interpolation/surrogate model)
            objective_func = self._create_surrogate_model(df, parameter_names, objective_name)

            # Get parameter bounds
            if constraints is None:
                constraints = self._get_parameter_bounds(df, parameter_names)

            # Initial guess (best point from data)
            if maximize:
                best_idx = df[objective_name].idxmax()
            else:
                best_idx = df[objective_name].idxmin()

            initial_guess = [df.loc[best_idx, param] for param in parameter_names]

            # Optimize
            bounds = [constraints[param] for param in parameter_names]

            if maximize:
                result = minimize(
                    lambda x: -objective_func(x),
                    initial_guess,
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                optimal_value = -result.fun
            else:
                result = minimize(objective_func, initial_guess, method="L-BFGS-B", bounds=bounds)
                optimal_value = result.fun

            optimal_params = {param: float(result.x[i]) for i, param in enumerate(parameter_names)}

            return OptimizationResult(
                success=result.success,
                optimal_parameters=optimal_params,
                optimal_objectives={objective_name: float(optimal_value)},
                optimization_method="L-BFGS-B",
                iterations=result.nit if hasattr(result, "nit") else 0,
                convergence_info={"message": result.message},
                error_message=None if result.success else str(result.message),
            )

        except Exception as e:
            logger.error(f"Error in single-objective optimization: {e}")
            return OptimizationResult(
                success=False,
                optimal_parameters={},
                optimal_objectives={},
                optimization_method="L-BFGS-B",
                iterations=0,
                convergence_info={},
                error_message=str(e),
            )

    def optimize_multi_objective(
        self,
        experiment_results: List[Dict[str, Any]],
        parameter_names: List[str],
        objective_names: List[str],
        objective_directions: List[str] = None,  # 'maximize' or 'minimize'
        constraints: Dict[str, Tuple[float, float]] = None,
    ) -> OptimizationResult:
        """
        Optimize parameters for multiple objectives (Pareto optimization).

        Args:
            experiment_results: List of experiment result dictionaries
            parameter_names: List of parameter names
            objective_names: List of objective names
            objective_directions: List of 'maximize' or 'minimize' for each objective
            constraints: Parameter constraints

        Returns:
            OptimizationResult: Optimization results with Pareto front
        """
        try:
            # Convert results to DataFrame
            df = self._results_to_dataframe(experiment_results, parameter_names, objective_names)

            # Default: maximize all objectives
            if objective_directions is None:
                objective_directions = ["maximize"] * len(objective_names)

            # Normalize objectives (all to maximize)
            df_normalized = df.copy()
            for i, (obj, direction) in enumerate(zip(objective_names, objective_directions)):
                if direction == "minimize":
                    df_normalized[obj] = -df[obj]

            # Find Pareto front
            pareto_front = self._find_pareto_front(df_normalized, parameter_names, objective_names)

            # Select representative solution (e.g., closest to ideal point)
            ideal_point = {obj: df_normalized[obj].max() for obj in objective_names}
            representative = self._select_representative_solution(pareto_front, ideal_point, objective_names)

            return OptimizationResult(
                success=True,
                optimal_parameters=representative["parameters"],
                optimal_objectives=representative["objectives"],
                optimization_method="pareto",
                iterations=len(pareto_front),
                convergence_info={"pareto_solutions": len(pareto_front)},
                pareto_front=pareto_front,
            )

        except Exception as e:
            logger.error(f"Error in multi-objective optimization: {e}")
            return OptimizationResult(
                success=False,
                optimal_parameters={},
                optimal_objectives={},
                optimization_method="pareto",
                iterations=0,
                convergence_info={},
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

            # Extract parameters
            if "input_parameters" in result:
                for param in parameter_names:
                    row[param] = result["input_parameters"].get(param, np.nan)

            # Extract responses
            if "output_responses" in result:
                for resp in response_names:
                    row[resp] = result["output_responses"].get(resp, np.nan)

            data.append(row)

        return pd.DataFrame(data)

    def _create_surrogate_model(self, df: pd.DataFrame, parameter_names: List[str], response_name: str) -> Callable:
        """Create a simple surrogate model (interpolation) from data."""
        from sklearn.ensemble import RandomForestRegressor

        # Prepare data
        X = df[parameter_names].values
        y = df[response_name].values

        # Remove NaN
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        if len(X_clean) == 0:
            # Fallback: return mean
            mean_val = np.nanmean(y)
            return lambda x: mean_val

        # Train simple model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_clean, y_clean)

        def surrogate(x):
            return model.predict([x])[0]

        return surrogate

    def _get_parameter_bounds(self, df: pd.DataFrame, parameter_names: List[str]) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds from data."""
        bounds = {}
        for param in parameter_names:
            if param in df.columns:
                values = df[param].dropna()
                if len(values) > 0:
                    bounds[param] = (float(values.min()), float(values.max()))
                else:
                    bounds[param] = (0.0, 1.0)
            else:
                bounds[param] = (0.0, 1.0)
        return bounds

    def _find_pareto_front(
        self, df: pd.DataFrame, parameter_names: List[str], objective_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Find Pareto-optimal solutions."""
        pareto_solutions = []

        for idx, row in df.iterrows():
            is_pareto = True

            # Check if this solution is dominated by any other
            for other_idx, other_row in df.iterrows():
                if idx == other_idx:
                    continue

                # Check if other_row dominates row
                all_better = all(other_row[obj] >= row[obj] for obj in objective_names)
                at_least_one_better = any(other_row[obj] > row[obj] for obj in objective_names)

                if all_better and at_least_one_better:
                    is_pareto = False
                    break

            if is_pareto:
                pareto_solutions.append(
                    {
                        "parameters": {param: float(row[param]) for param in parameter_names},
                        "objectives": {obj: float(row[obj]) for obj in objective_names},
                    }
                )

        return pareto_solutions

    def _select_representative_solution(
        self,
        pareto_front: List[Dict[str, Any]],
        ideal_point: Dict[str, float],
        objective_names: List[str],
    ) -> Dict[str, Any]:
        """Select representative solution from Pareto front (closest to ideal)."""
        if not pareto_front:
            return {"parameters": {}, "objectives": {}}

        # Find solution closest to ideal point
        min_distance = float("inf")
        representative = pareto_front[0]

        for solution in pareto_front:
            distance = sum((solution["objectives"][obj] - ideal_point[obj]) ** 2 for obj in objective_names)
            if distance < min_distance:
                min_distance = distance
                representative = solution

        return representative
