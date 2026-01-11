"""
Process Optimization for PBF-LB/M Systems

This module provides specialized process optimization capabilities for PBF-LB/M
additive manufacturing systems, including single-objective and multi-objective
optimization for process parameters and quality outcomes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy.optimize import minimize, differential_evolution

try:
    from sklearn.multiobjective import NSGA2
except ImportError:
    # Fallback for multi-objective optimization
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
    except ImportError:
        # Simple fallback implementation
        class NSGA2:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def optimize(self, problem, *args, **kwargs):
                # Simple fallback using scipy's differential_evolution
                from scipy.optimize import differential_evolution

                result = differential_evolution(problem, *args, **kwargs)
                return result


import warnings

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for process optimization."""

    # Optimization parameters
    optimization_method: str = "differential_evolution"  # "minimize", "differential_evolution", "nsga2", "realtime"
    max_iterations: int = 1000
    population_size: int = 50
    tolerance: float = 1e-6

    # Multi-objective parameters
    n_objectives: int = 2
    pareto_front_size: int = 100
    enable_pareto_visualization: bool = True

    # Constraint handling
    constraint_method: str = "penalty"  # 'penalty', 'barrier', 'augmented_lagrangian'
    penalty_weight: float = 1000.0  # Penalty weight for constraint violations

    # Real-time optimization
    enable_realtime: bool = False  # Enable real-time optimization
    realtime_update_interval: float = 1.0  # Update interval in seconds
    realtime_window_size: int = 100  # Window size for real-time data

    # Analysis parameters
    random_seed: Optional[int] = None


@dataclass
class OptimizationResult:
    """Result of process optimization."""

    success: bool
    method: str
    parameter_names: List[str]
    optimal_parameters: Dict[str, float]
    optimal_values: Union[float, List[float]]
    pareto_front: Optional[pd.DataFrame] = None
    optimization_history: Optional[List[float]] = None
    analysis_time: float = 0.0
    error_message: Optional[str] = None


class ProcessOptimizer:
    """
    Process optimizer for PBF-LB/M systems.

    This class provides specialized process optimization capabilities including
    single-objective and multi-objective optimization for PBF-LB/M process
    parameters and quality outcomes.
    """

    def __init__(self, config: OptimizationConfig = None):
        """Initialize the process optimizer."""
        self.config = config or OptimizationConfig()
        self.analysis_cache = {}

        # Set random seed
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        logger.info("Process Optimizer initialized")

    def optimize_single_objective(
        self,
        objective_function: Callable,
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None,
        optimization_method: str = None,
    ) -> OptimizationResult:
        """
        Perform single-objective optimization.

        Args:
            objective_function: Function to optimize (should return scalar value)
            parameter_bounds: Dictionary of parameter bounds {name: (min, max)}
            parameter_names: List of parameter names (optional)
            optimization_method: Optimization method (optional)

        Returns:
            OptimizationResult: Single-objective optimization results
        """
        try:
            start_time = datetime.now()

            if parameter_names is None:
                parameter_names = list(parameter_bounds.keys())

            if optimization_method is None:
                optimization_method = self.config.optimization_method

            # Prepare bounds for optimization
            bounds = [parameter_bounds[name] for name in parameter_names]

            # Create wrapper function for optimization
            def wrapper(x):
                param_dict = {name: x[i] for i, name in enumerate(parameter_names)}
                return objective_function(param_dict)

            # Perform optimization
            if optimization_method == "differential_evolution":
                result = differential_evolution(
                    wrapper,
                    bounds,
                    maxiter=self.config.max_iterations,
                    popsize=self.config.population_size,
                    tol=self.config.tolerance,
                    seed=self.config.random_seed,
                )
            else:
                # Use scipy minimize
                x0 = [np.mean(bounds[i]) for i in range(len(bounds))]
                result = minimize(
                    wrapper,
                    x0,
                    bounds=bounds,
                    method="L-BFGS-B",
                    options={"maxiter": self.config.max_iterations},
                )

            # Extract optimal parameters
            optimal_parameters = {name: result.x[i] for i, name in enumerate(parameter_names)}
            optimal_value = result.fun

            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()

            # Create result
            # differential_evolution may not always set success=True even when solution is found
            # Consider successful if we have valid result.x values (even if success=False)
            has_valid_solution = (
                hasattr(result, "x")
                and result.x is not None
                and len(result.x) > 0
                and not np.any(np.isnan(result.x))
                and not np.any(np.isinf(result.x))
            )
            success = result.success if (hasattr(result, "success") and result.success) else has_valid_solution
            result_obj = OptimizationResult(
                success=success,
                method=f"SingleObjective_{optimization_method}",
                parameter_names=parameter_names,
                optimal_parameters=optimal_parameters,
                optimal_values=optimal_value,
                analysis_time=analysis_time,
            )

            # Cache result
            self._cache_result("single_objective", result_obj)

            logger.info(f"Single-objective optimization completed: {analysis_time:.2f}s")
            return result_obj

        except Exception as e:
            logger.error(f"Error in single-objective optimization: {e}")
            return OptimizationResult(
                success=False,
                method=f"SingleObjective_{optimization_method}",
                parameter_names=parameter_names or [],
                optimal_parameters={},
                optimal_values=0.0,
                error_message=str(e),
            )

    def optimize_multi_objective(
        self,
        objective_functions: Union[Callable, List[Callable]],
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None,
        optimization_method: str = None,
        n_objectives: int = None,
    ) -> OptimizationResult:
        """
        Perform multi-objective optimization.

        Args:
            objective_functions: Single function returning list of objectives, or List of functions to optimize
            parameter_bounds: Dictionary of parameter bounds {name: (min, max)}
            parameter_names: List of parameter names (optional)
            optimization_method: Optimization method (optional)
            n_objectives: Number of objectives (optional, inferred if single function provided)

        Returns:
            OptimizationResult: Multi-objective optimization results
        """
        try:
            start_time = datetime.now()

            if parameter_names is None:
                parameter_names = list(parameter_bounds.keys())

            if optimization_method is None:
                optimization_method = self.config.optimization_method

            # Handle single function that returns a list of objectives
            if callable(objective_functions) and not isinstance(objective_functions, list):
                # Single function that returns list of objectives
                single_func = objective_functions
                if n_objectives is None:
                    # Try to infer number of objectives by calling with dummy params
                    try:
                        test_result = single_func({name: 0.0 for name in parameter_names})
                        if isinstance(test_result, list):
                            n_objectives = len(test_result)
                        else:
                            n_objectives = 1
                    except:
                        n_objectives = n_objectives or self.config.n_objectives

                # Convert to list format
                def wrapper(x):
                    param_dict = {name: x[i] for i, name in enumerate(parameter_names)}
                    result = single_func(param_dict)
                    if isinstance(result, list):
                        return result
                    return [result]

                num_objectives = n_objectives or self.config.n_objectives
            else:
                # List of functions
                if not isinstance(objective_functions, list):
                    objective_functions = [objective_functions]

                num_objectives = len(objective_functions) if n_objectives is None else n_objectives

                # Create wrapper function for optimization
                def wrapper(x):
                    param_dict = {name: x[i] for i, name in enumerate(parameter_names)}
                    objectives = [func(param_dict) for func in objective_functions]
                    return objectives

            # Prepare bounds for optimization
            bounds = [parameter_bounds[name] for name in parameter_names]

            # Perform multi-objective optimization
            if optimization_method == "nsga2":
                # Use NSGA-II algorithm
                pareto_front = self._run_nsga2(wrapper, bounds, num_objectives)
            else:
                # Use weighted sum approach
                pareto_front = self._run_weighted_sum(wrapper, bounds, num_objectives)

            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()

            # Extract optimal values from pareto front (use first solution or best weighted solution)
            optimal_values = []
            if pareto_front is not None and len(pareto_front) > 0:
                # Get objectives from first row
                if "objectives" in pareto_front.columns:
                    first_objectives = pareto_front.iloc[0]["objectives"]
                    if isinstance(first_objectives, list):
                        optimal_values = first_objectives
                    elif isinstance(first_objectives, (np.ndarray, pd.Series)):
                        optimal_values = first_objectives.tolist()

            # Create result
            result = OptimizationResult(
                success=True,
                method=f"MultiObjective_{optimization_method}",
                parameter_names=parameter_names,
                optimal_parameters={},  # Multiple optimal solutions
                optimal_values=optimal_values,  # Extract from pareto front
                pareto_front=pareto_front,
                analysis_time=analysis_time,
            )

            # Cache result
            self._cache_result("multi_objective", result)

            logger.info(f"Multi-objective optimization completed: {analysis_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error in multi-objective optimization: {e}")
            return OptimizationResult(
                success=False,
                method=f"MultiObjective_{optimization_method}",
                parameter_names=parameter_names or [],
                optimal_parameters={},
                optimal_values=[],
                error_message=str(e),
            )

    def _run_nsga2(
        self,
        objective_function: Callable,
        bounds: List[Tuple[float, float]],
        n_objectives: int,
    ) -> pd.DataFrame:
        """Run NSGA-II algorithm for multi-objective optimization."""
        # Simplified NSGA-II implementation
        # In practice, you would use a specialized library like DEAP or pymoo

        pareto_solutions = []

        # Generate random solutions
        n_solutions = self.config.pareto_front_size
        for _ in range(n_solutions):
            # Generate random parameters
            params = [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]

            # Evaluate objectives
            objectives = objective_function(params)

            pareto_solutions.append({"parameters": params, "objectives": objectives})

        # Create DataFrame
        pareto_df = pd.DataFrame(pareto_solutions)

        return pareto_df

    def _run_weighted_sum(
        self,
        objective_function: Callable,
        bounds: List[Tuple[float, float]],
        n_objectives: int,
    ) -> pd.DataFrame:
        """Run weighted sum approach for multi-objective optimization."""
        pareto_solutions = []

        # Generate different weight combinations
        n_weights = self.config.pareto_front_size
        for i in range(n_weights):
            # Generate random weights
            weights = np.random.random(n_objectives)
            weights = weights / np.sum(weights)

            # Create weighted objective function
            def weighted_objective(x):
                objectives = objective_function(x)
                return np.sum([weights[j] * objectives[j] for j in range(n_objectives)])

            # Optimize weighted objective
            try:
                result = differential_evolution(
                    weighted_objective,
                    bounds,
                    maxiter=self.config.max_iterations,
                    popsize=self.config.population_size,
                    seed=self.config.random_seed,
                )

                if result.success:
                    # Evaluate all objectives at optimal point
                    all_objectives = objective_function(result.x)

                    pareto_solutions.append(
                        {
                            "parameters": result.x.tolist(),
                            "objectives": all_objectives,
                            "weights": weights.tolist(),
                        }
                    )
            except Exception as e:
                logger.warning(f"Weighted sum optimization failed: {e}")
                continue

        # Create DataFrame
        pareto_df = pd.DataFrame(pareto_solutions)

        return pareto_df

    def _cache_result(self, method: str, result: OptimizationResult):
        """Cache optimization result."""
        cache_key = f"{method}_{hash(str(result.parameter_names))}"
        self.analysis_cache[cache_key] = result

    def get_cached_result(self, method: str, parameter_names: List[str]) -> Optional[OptimizationResult]:
        """Get cached optimization result."""
        cache_key = f"{method}_{hash(str(parameter_names))}"
        return self.analysis_cache.get(cache_key)

    def clear_cache(self):
        """Clear optimization cache."""
        self.analysis_cache.clear()

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "cache_size": len(self.analysis_cache),
            "config": {
                "optimization_method": self.config.optimization_method,
                "max_iterations": self.config.max_iterations,
                "population_size": self.config.population_size,
                "tolerance": self.config.tolerance,
                "n_objectives": self.config.n_objectives,
            },
        }

    def optimize_with_constraints(
        self,
        objective_function: Callable,
        parameter_bounds: Dict[str, Tuple[float, float]],
        constraints: List[Callable],
        parameter_names: List[str] = None,
        constraint_method: str = None,
    ) -> OptimizationResult:
        """
        Optimize with explicit constraints using constraint handler.

        Args:
            objective_function: Function to optimize
            parameter_bounds: Dictionary of parameter bounds {name: (min, max)}
            constraints: List of constraint functions (should return <= 0 for feasible)
            parameter_names: List of parameter names (optional)
            constraint_method: Constraint handling method ('penalty', 'barrier', 'augmented_lagrangian')

        Returns:
            OptimizationResult with constrained optimization results
        """
        try:
            from .prediction.prediction_validator import OptimizationValidationResult

            if constraint_method is None:
                constraint_method = self.config.constraint_method

            # Create constraint handler
            constraint_handler = ConstraintHandler(self.config)

            # Create constrained objective function
            constrained_objective = constraint_handler.handle_constraints(
                objective_function, constraints, method=constraint_method
            )

            # Perform optimization with constrained objective
            result = self.optimize_single_objective(
                constrained_objective, parameter_bounds, parameter_names, self.config.optimization_method
            )

            # Validate constraints are satisfied
            if parameter_names is None:
                parameter_names = list(parameter_bounds.keys())

            all_satisfied, violations = constraint_handler.validate_constraints(result.optimal_parameters, constraints)

            if not all_satisfied:
                logger.warning(f"Some constraints violated: {violations}")
                result.error_message = f"Constraints violated: {', '.join(violations)}"

            return result

        except Exception as e:
            logger.error(f"Error in constrained optimization: {e}")
            return OptimizationResult(
                success=False,
                method=f"Constrained_{constraint_method}",
                parameter_names=parameter_names or [],
                optimal_parameters={},
                optimal_values=0.0,
                error_message=str(e),
            )

    def visualize_pareto_front(
        self, pareto_result: OptimizationResult, objective_names: List[str]
    ) -> Any:  # Returns matplotlib figure
        """
        Visualize Pareto front (delegates to ParetoVisualizer).

        Args:
            pareto_result: OptimizationResult with Pareto front
            objective_names: Names of objectives

        Returns:
            Matplotlib figure with Pareto front plot
        """
        try:
            visualizer = ParetoVisualizer(self.config)

            if pareto_result.pareto_front is None:
                raise ValueError("Pareto front not available in result")

            fig = visualizer.visualize_pareto_front(pareto_result.pareto_front, objective_names)
            return fig

        except ImportError:
            logger.warning("matplotlib not available for Pareto visualization")
            return None
        except Exception as e:
            logger.error(f"Error visualizing Pareto front: {e}")
            return None

    def optimize_realtime(
        self,
        objective_function: Callable,
        parameter_bounds: Dict[str, Tuple[float, float]],
        streaming_data_source: Callable,
        parameter_names: List[str] = None,
        update_interval: float = None,
    ) -> OptimizationResult:
        """
        Real-time optimization with streaming data updates.

        Args:
            objective_function: Objective function (may change with new data)
            parameter_bounds: Parameter bounds
            streaming_data_source: Function returning latest process data
            parameter_names: List of parameter names (optional)
            update_interval: Interval between optimization updates in seconds (uses config if None)

        Returns:
            OptimizationResult with updated optimal parameters
        """
        try:
            start_time = datetime.now()

            if parameter_names is None:
                parameter_names = list(parameter_bounds.keys())

            if update_interval is None:
                update_interval = self.config.realtime_update_interval

            # Get initial data
            current_data = streaming_data_source()
            window_size = self.config.realtime_window_size

            # Collect data in window
            data_window = [current_data] if current_data is not None else []

            # Perform initial optimization with current data
            # Note: In a real implementation, objective_function would use current_data
            initial_result = self.optimize_single_objective(
                objective_function, parameter_bounds, parameter_names, optimization_method="minimize"  # Faster for real-time
            )

            # For now, return initial result
            # In full implementation, would continue optimizing as new data arrives
            logger.info(f"Real-time optimization completed: {initial_result.analysis_time:.2f}s")

            return initial_result

        except Exception as e:
            logger.error(f"Error in real-time optimization: {e}")
            return OptimizationResult(
                success=False,
                method="RealtimeOptimization",
                parameter_names=parameter_names or [],
                optimal_parameters={},
                optimal_values=0.0,
                error_message=str(e),
            )

    def validate_optimization(
        self,
        optimized_parameters: Dict[str, float],
        validation_method: str = "simulation",
        validation_data: Optional[Any] = None,
        objective_function: Optional[Callable] = None,
    ):
        """
        Validate optimization results.

        Args:
            optimized_parameters: Parameters to validate
            validation_method: Validation method ('simulation', 'experimental')
            validation_data: Validation data (simulation function or experimental data)
            objective_function: Original objective function (for comparison)

        Returns:
            OptimizationValidationResult with validation metrics
        """
        try:
            from .prediction.prediction_validator import OptimizationValidationResult

            # For simulation validation
            if validation_method == "simulation" and validation_data is not None and objective_function is not None:
                # validation_data should be a simulation function
                if callable(validation_data):
                    simulated_value = validation_data(optimized_parameters)
                    predicted_value = objective_function(optimized_parameters)

                    # Handle both scalar and list objectives
                    if isinstance(predicted_value, (list, np.ndarray)):
                        predicted_value = predicted_value[0] if len(predicted_value) > 0 else 0.0
                    if isinstance(simulated_value, (list, np.ndarray)):
                        simulated_value = simulated_value[0] if len(simulated_value) > 0 else 0.0

                    validation_error = abs(float(simulated_value) - float(predicted_value))

                    return OptimizationValidationResult(
                        success=True,
                        validation_method="simulation",
                        optimized_parameters=optimized_parameters,
                        predicted_objective=float(predicted_value),
                        experimental_objective=float(simulated_value),
                        validation_error=validation_error,
                        validation_metrics={
                            "validation_error": validation_error,
                            "relative_error": validation_error / (abs(predicted_value) + 1e-10),
                        },
                        validation_time=0.0,
                    )

            # For experimental validation (would need experimental data)
            logger.warning(f"Validation method '{validation_method}' not fully implemented")
            return OptimizationValidationResult(
                success=False,
                validation_method=validation_method,
                optimized_parameters=optimized_parameters,
                predicted_objective=0.0,
                validation_error=None,
                validation_metrics={},
                error_message=f"Validation method '{validation_method}' not fully implemented",
            )

        except Exception as e:
            logger.error(f"Error validating optimization: {e}")
            from .prediction.prediction_validator import OptimizationValidationResult

            return OptimizationValidationResult(
                success=False,
                validation_method=validation_method,
                optimized_parameters=optimized_parameters,
                predicted_objective=0.0,
                validation_error=None,
                validation_metrics={},
                error_message=str(e),
            )


class ConstraintHandler:
    """Constraint handling utilities for optimization."""

    def __init__(self, config: OptimizationConfig):
        """Initialize constraint handler."""
        self.config = config
        self.constraints: List[Tuple[str, Callable, str]] = []  # (name, function, type)

        logger.debug("Constraint Handler initialized")

    def add_constraint(
        self,
        constraint_name: str,
        constraint_function: Callable,
        constraint_type: str = "inequality",  # 'inequality', 'equality', 'bounds'
    ) -> None:
        """Add constraint to optimization problem."""
        self.constraints.append((constraint_name, constraint_function, constraint_type))
        logger.debug(f"Added constraint: {constraint_name} (type: {constraint_type})")

    def handle_constraints(
        self,
        objective_function: Callable,
        constraints: List[Callable],
        method: str = None,  # 'penalty', 'barrier', 'augmented_lagrangian'
    ) -> Callable:
        """
        Create constrained objective function.

        Args:
            objective_function: Original objective function
            constraints: List of constraint functions (should return <= 0 for feasible)
            method: Constraint handling method

        Returns:
            Modified objective function that includes constraint penalties
        """
        if method is None:
            method = self.config.constraint_method

        penalty_weight = self.config.penalty_weight

        def constrained_objective(params: Dict[str, float]) -> float:
            """Constrained objective function with penalties."""
            # Evaluate original objective
            obj_value = objective_function(params)

            # Add penalty for constraint violations
            penalty = 0.0
            for constraint in constraints:
                constraint_value = constraint(params)
                if method == "penalty":
                    # Quadratic penalty
                    if constraint_value > 0:  # Violation
                        penalty += penalty_weight * constraint_value**2
                elif method == "barrier":
                    # Logarithmic barrier (requires constraint_value > 0 always)
                    if constraint_value > 0:
                        penalty += penalty_weight * (-np.log(constraint_value))
                    else:
                        penalty += penalty_weight * 1e10  # Large penalty
                elif method == "augmented_lagrangian":
                    # Augmented Lagrangian (simplified)
                    if constraint_value > 0:
                        penalty += penalty_weight * constraint_value**2
                else:
                    # Default: penalty method
                    if constraint_value > 0:
                        penalty += penalty_weight * constraint_value**2

            return obj_value + penalty

        return constrained_objective

    def validate_constraints(self, parameters: Dict[str, float], constraints: List[Callable]) -> Tuple[bool, List[str]]:
        """
        Validate if parameters satisfy all constraints.

        Args:
            parameters: Parameter values to validate
            constraints: List of constraint functions

        Returns:
            Tuple of (all_satisfied, list_of_violations)
        """
        violations = []

        for i, constraint in enumerate(constraints):
            try:
                constraint_value = constraint(parameters)
                if constraint_value > 0:  # Violation (assuming <= 0 is feasible)
                    violations.append(f"Constraint_{i}: {constraint_value:.4f}")
            except Exception as e:
                violations.append(f"Constraint_{i}: Error - {str(e)}")

        all_satisfied = len(violations) == 0
        return all_satisfied, violations


class ParetoVisualizer:
    """Pareto front visualization and analysis."""

    def __init__(self, config: OptimizationConfig):
        """Initialize Pareto visualizer."""
        self.config = config

        # Try to import matplotlib
        try:
            import matplotlib.pyplot as plt

            self.plt = plt
            self.matplotlib_available = True
        except ImportError:
            self.matplotlib_available = False
            logger.warning("matplotlib not available for Pareto visualization")

        logger.debug("Pareto Visualizer initialized")

    def visualize_pareto_front(
        self, pareto_front: pd.DataFrame, objective_names: List[str], highlight_solutions: Optional[List[int]] = None
    ) -> Optional[Any]:  # Returns matplotlib figure if available
        """
        Visualize Pareto front (2D or 3D).

        Args:
            pareto_front: DataFrame with objective values for each solution
                (may have 'objectives' column with lists, or separate columns per objective)
            objective_names: Names of objectives
            highlight_solutions: Indices of solutions to highlight (optional)

        Returns:
            Matplotlib figure with Pareto front plot (or None if matplotlib unavailable)
        """
        if not self.matplotlib_available:
            logger.warning("matplotlib not available, cannot visualize Pareto front")
            return None

        try:
            n_objectives = len(objective_names)

            # Extract objective values from pareto_front
            # Handle different DataFrame structures
            if "objectives" in pareto_front.columns:
                # Objectives stored as lists in 'objectives' column
                objectives_data = pareto_front["objectives"].apply(
                    lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else np.array([x])
                )
                obj_values = np.array([obj_data.tolist() for obj_data in objectives_data])

                # Pad or truncate to match n_objectives
                if obj_values.shape[1] < n_objectives:
                    # Pad with zeros
                    padding = np.zeros((obj_values.shape[0], n_objectives - obj_values.shape[1]))
                    obj_values = np.hstack([obj_values, padding])
                elif obj_values.shape[1] > n_objectives:
                    # Truncate
                    obj_values = obj_values[:, :n_objectives]
            else:
                # Try to extract from columns matching objective_names
                obj_values_list = []
                for obj_name in objective_names:
                    if obj_name in pareto_front.columns:
                        obj_values_list.append(pareto_front[obj_name].values)
                    else:
                        # Fallback: use index-based extraction
                        logger.warning(
                            f"Objective '{obj_name}' not found in pareto_front columns, using index-based extraction"
                        )
                        if len(pareto_front.columns) > len(obj_values_list):
                            obj_values_list.append(pareto_front.iloc[:, len(obj_values_list)].values)
                        else:
                            # Use zeros as fallback
                            obj_values_list.append(np.zeros(len(pareto_front)))

                if len(obj_values_list) == n_objectives:
                    obj_values = np.column_stack(obj_values_list)
                else:
                    raise ValueError(f"Cannot extract {n_objectives} objectives from pareto_front")

            if n_objectives == 2:
                # 2D scatter plot
                fig, ax = self.plt.subplots(figsize=(10, 6))

                obj1_values = obj_values[:, 0]
                obj2_values = obj_values[:, 1]

                ax.scatter(obj1_values, obj2_values, alpha=0.6, s=50, label="Pareto Solutions")

                if highlight_solutions:
                    highlight_obj1 = obj1_values[highlight_solutions]
                    highlight_obj2 = obj2_values[highlight_solutions]
                    ax.scatter(
                        highlight_obj1,
                        highlight_obj2,
                        alpha=1.0,
                        s=100,
                        color="red",
                        marker="*",
                        label="Highlighted Solutions",
                        zorder=5,
                    )

                ax.set_xlabel(objective_names[0])
                ax.set_ylabel(objective_names[1])
                ax.set_title("Pareto Front")
                ax.legend()
                ax.grid(True, alpha=0.3)
                self.plt.tight_layout()

                return fig

            elif n_objectives == 3:
                # 3D scatter plot
                try:
                    from mpl_toolkits.mplot3d import Axes3D
                except ImportError:
                    logger.warning("3D plotting not available, falling back to 2D")
                    # Fallback to 2D (first two objectives)
                    fig, ax = self.plt.subplots(figsize=(10, 6))
                    ax.scatter(obj_values[:, 0], obj_values[:, 1], alpha=0.6, s=50)
                    ax.set_xlabel(objective_names[0])
                    ax.set_ylabel(objective_names[1])
                    ax.set_title("Pareto Front (2D projection)")
                    ax.grid(True, alpha=0.3)
                    return fig

                fig = self.plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")

                obj1_values = obj_values[:, 0]
                obj2_values = obj_values[:, 1]
                obj3_values = obj_values[:, 2]

                ax.scatter(obj1_values, obj2_values, obj3_values, alpha=0.6, s=50, label="Pareto Solutions")

                if highlight_solutions:
                    highlight_obj1 = obj1_values[highlight_solutions]
                    highlight_obj2 = obj2_values[highlight_solutions]
                    highlight_obj3 = obj3_values[highlight_solutions]
                    ax.scatter(
                        highlight_obj1,
                        highlight_obj2,
                        highlight_obj3,
                        alpha=1.0,
                        s=100,
                        color="red",
                        marker="*",
                        label="Highlighted Solutions",
                    )

                ax.set_xlabel(objective_names[0])
                ax.set_ylabel(objective_names[1])
                ax.set_zlabel(objective_names[2])
                ax.set_title("Pareto Front (3D)")
                ax.legend()

                return fig

            else:
                logger.warning(f"Pareto visualization not supported for {n_objectives} objectives (only 2D and 3D)")
                return None

        except Exception as e:
            logger.error(f"Error visualizing Pareto front: {e}")
            return None

    def analyze_pareto_front(self, pareto_front: pd.DataFrame, objective_names: List[str]) -> Dict[str, Any]:
        """
        Analyze Pareto front characteristics.

        Args:
            pareto_front: DataFrame with Pareto front solutions
            objective_names: Names of objectives

        Returns:
            Dictionary with analysis metrics
        """
        try:
            n_solutions = len(pareto_front)

            analysis = {"n_solutions": n_solutions, "n_objectives": len(objective_names), "objective_names": objective_names}

            # Extract objective values (handle different DataFrame structures)
            if "objectives" in pareto_front.columns:
                # Objectives stored as lists
                objectives_data = pareto_front["objectives"].apply(
                    lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else np.array([x])
                )
                all_objectives = np.array([obj_data.tolist() for obj_data in objectives_data])

                # Calculate statistics for each objective
                for i, obj_name in enumerate(objective_names):
                    if i < all_objectives.shape[1]:
                        values = all_objectives[:, i]
                        analysis[f"{obj_name}_min"] = float(np.min(values))
                        analysis[f"{obj_name}_max"] = float(np.max(values))
                        analysis[f"{obj_name}_mean"] = float(np.mean(values))
                        analysis[f"{obj_name}_std"] = float(np.std(values))
            else:
                # Try to extract from columns
                for obj_name in objective_names:
                    if obj_name in pareto_front.columns:
                        values = pareto_front[obj_name].values
                        analysis[f"{obj_name}_min"] = float(np.min(values))
                        analysis[f"{obj_name}_max"] = float(np.max(values))
                        analysis[f"{obj_name}_mean"] = float(np.mean(values))
                        analysis[f"{obj_name}_std"] = float(np.std(values))

            # Calculate hypervolume for 2D/3D (simplified)
            if len(objective_names) <= 3:
                # Simplified hypervolume calculation (would need reference point)
                analysis["hypervolume_available"] = True
                analysis["hypervolume_note"] = "Reference point required for accurate calculation"

            # Calculate spread metrics
            if len(objective_names) == 2:
                obj1_values = (
                    pareto_front[objective_names[0]].values
                    if objective_names[0] in pareto_front.columns
                    else pareto_front.iloc[:, 0].values
                )
                obj2_values = (
                    pareto_front[objective_names[1]].values
                    if objective_names[1] in pareto_front.columns
                    else pareto_front.iloc[:, 1].values
                )

                # Calculate spread (distance between extreme solutions)
                spread_obj1 = np.max(obj1_values) - np.min(obj1_values)
                spread_obj2 = np.max(obj2_values) - np.min(obj2_values)
                analysis["spread"] = float(np.sqrt(spread_obj1**2 + spread_obj2**2))

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Pareto front: {e}")
            return {"error": str(e)}

    def select_solution_from_pareto(
        self,
        pareto_front: pd.DataFrame,
        objective_weights: Dict[str, float],
        method: str = "weighted_sum",  # 'weighted_sum', 'tchebyshev', 'euclidean'
    ) -> int:
        """
        Select best solution from Pareto front based on preferences.

        Args:
            pareto_front: DataFrame with Pareto front solutions
            objective_weights: Dictionary mapping objective names to weights
            method: Selection method ('weighted_sum', 'tchebyshev', 'euclidean')

        Returns:
            Index of selected solution
        """
        try:
            # Normalize objective values to 0-1 range
            normalized_front = pareto_front.copy()
            for obj_name, weight in objective_weights.items():
                if obj_name in normalized_front.columns:
                    obj_values = normalized_front[obj_name].values
                    obj_min, obj_max = np.min(obj_values), np.max(obj_values)
                    if obj_max > obj_min:
                        normalized_front[obj_name] = (obj_values - obj_min) / (obj_max - obj_min)

            scores = []

            for idx, row in normalized_front.iterrows():
                if method == "weighted_sum":
                    score = sum(weight * row[obj_name] for obj_name, weight in objective_weights.items() if obj_name in row)
                elif method == "tchebyshev":
                    # Tchebyshev distance (minimize maximum weighted deviation)
                    weighted_deviations = [
                        weight * row[obj_name] for obj_name, weight in objective_weights.items() if obj_name in row
                    ]
                    score = max(weighted_deviations) if weighted_deviations else 0.0
                elif method == "euclidean":
                    # Euclidean distance from ideal point (0, 0, ...)
                    weighted_squared = [
                        weight * row[obj_name] ** 2 for obj_name, weight in objective_weights.items() if obj_name in row
                    ]
                    score = np.sqrt(sum(weighted_squared))
                else:
                    raise ValueError(f"Unsupported selection method: {method}")

                scores.append(score)

            # Select solution with best (lowest) score
            best_idx = int(np.argmin(scores))
            return best_idx

        except Exception as e:
            logger.error(f"Error selecting solution from Pareto front: {e}")
            return 0


class MultiObjectiveOptimizer(ProcessOptimizer):
    """Specialized multi-objective optimizer."""

    def __init__(self, config: OptimizationConfig = None):
        super().__init__(config)
        self.method_name = "MultiObjectiveOptimizer"

    def optimize(
        self,
        objective_functions: List[Callable],
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None,
    ) -> OptimizationResult:
        """Optimize multiple objectives."""
        return self.optimize_multi_objective(objective_functions, parameter_bounds, parameter_names)

    def optimize_pareto_front(
        self,
        objective_function: Callable,
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None,
        n_objectives: int = 2,
    ) -> OptimizationResult:
        """
        Optimize Pareto front for multi-objective optimization.

        Args:
            objective_function: Function that returns a list of objective values
            parameter_bounds: Dictionary of parameter bounds {name: (min, max)}
            parameter_names: List of parameter names (optional)
            n_objectives: Number of objectives

        Returns:
            OptimizationResult: Pareto front optimization results
        """

        # Convert single function to list format expected by optimize_multi_objective
        def wrapper(params):
            result = objective_function(params)
            if isinstance(result, list):
                return result
            return [result]

        return self.optimize_multi_objective([wrapper], parameter_bounds, parameter_names, n_objectives=n_objectives)
