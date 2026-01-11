# Process Analysis: Prediction and Optimization API Reference

## Overview

API reference for prediction, model tracking, and enhanced optimization capabilities in the Process Analysis module.

## Prediction Module

### EarlyDefectPredictor

Early defect prediction before build completion.

```python
from am_qadf.analytics.process_analysis.prediction import (
    EarlyDefectPredictor,
    PredictionConfig,
    EarlyDefectPredictionResult
)

predictor = EarlyDefectPredictor(config: PredictionConfig)
```

#### Methods

##### `train_early_prediction_model(process_data: pd.DataFrame, defect_labels: np.ndarray, feature_names: List[str] = None, early_horizon: int = None) -> EarlyDefectPredictionResult`

Train model to predict defects early in build process.

**Parameters**:
- `process_data` (pd.DataFrame): Process data with features (sensor readings over time)
- `defect_labels` (np.ndarray): Binary labels (0=no defect, 1=defect) for completed builds
- `feature_names` (List[str], optional): Feature names. If None, inferred from process_data.
- `early_horizon` (int, optional): Number of samples before completion to use. Uses config if None.

**Returns**: `EarlyDefectPredictionResult` with:
- `success` (bool): Whether training succeeded
- `model_type` (str): Type of model used
- `defect_probability` (np.ndarray): Probability of defect for each sample
- `defect_prediction` (np.ndarray): Binary defect prediction (0=no defect, 1=defect)
- `prediction_confidence` (np.ndarray): Confidence in prediction (0-1)
- `early_prediction_accuracy` (float): Accuracy of early predictions
- `prediction_horizon` (int): Number of samples ahead predicted
- `model_performance` (Dict[str, float]): Metrics (accuracy, precision, recall, f1_score, roc_auc)
- `feature_importance` (Optional[Dict[str, float]]): Feature importance scores
- `analysis_time` (float): Training time in seconds
- `error_message` (Optional[str]): Error message if failed

**Example**:
```python
config = PredictionConfig(model_type='random_forest', early_prediction_horizon=100)
predictor = EarlyDefectPredictor(config)
result = predictor.train_early_prediction_model(
    process_data, defect_labels, feature_names=['temp', 'power', 'speed']
)
```

##### `predict_early_defect(partial_process_data: pd.DataFrame, build_progress: float) -> Tuple[np.ndarray, np.ndarray]`

Predict defect probability for partial build data.

**Parameters**:
- `partial_process_data` (pd.DataFrame): Process data from partial build
- `build_progress` (float): Current build progress (0.0-1.0)

**Returns**: Tuple of `(defect_probability, prediction_confidence)` arrays

**Raises**: `ValueError` if model not trained

**Example**:
```python
partial_data = get_partial_build_data(build_id='build_001')
defect_prob, confidence = predictor.predict_early_defect(partial_data, build_progress=0.3)
```

##### `update_model_with_new_data(new_process_data: pd.DataFrame, new_defect_labels: np.ndarray, early_horizon: int = None) -> EarlyDefectPredictionResult`

Update model with new training data (incremental learning).

**Parameters**:
- `new_process_data` (pd.DataFrame): New process data
- `new_defect_labels` (np.ndarray): New defect labels
- `early_horizon` (int, optional): Horizon for new data

**Returns**: `EarlyDefectPredictionResult` from retrained model

##### `get_feature_importance() -> Optional[Dict[str, float]]`

Get feature importance for early defect prediction.

**Returns**: Dictionary mapping feature names to importance scores, or None if not available

---

### TimeSeriesPredictor

Time-series forecasting for quality metrics.

```python
from am_qadf.analytics.process_analysis.prediction import (
    TimeSeriesPredictor,
    TimeSeriesPredictionResult
)

predictor = TimeSeriesPredictor(config: PredictionConfig)
```

#### Methods

##### `forecast_quality_metric(historical_data: np.ndarray, forecast_horizon: int = None, model_type: str = 'arima') -> TimeSeriesPredictionResult`

Forecast quality metric using time-series model.

**Parameters**:
- `historical_data` (np.ndarray): Historical time-series data (1D array)
- `forecast_horizon` (int, optional): Number of steps ahead. Uses config if None.
- `model_type` (str): Model type - `'arima'`, `'exponential_smoothing'`, `'moving_average'`, `'prophet'`

**Returns**: `TimeSeriesPredictionResult` with:
- `success` (bool): Whether forecasting succeeded
- `model_type` (str): Type of model used
- `forecast` (np.ndarray): Forecasted values
- `forecast_lower_bound` (np.ndarray): Lower confidence bound
- `forecast_upper_bound` (np.ndarray): Upper confidence bound
- `forecast_horizon` (int): Forecast horizon
- `historical_data` (np.ndarray): Historical data used
- `model_performance` (Dict[str, float]): Performance metrics (MAPE, RMSE, etc.)
- `trend_components` (Optional[np.ndarray]): Trend components if decomposed
- `seasonality_components` (Optional[np.ndarray]): Seasonality components if decomposed
- `analysis_time` (float): Analysis time in seconds
- `error_message` (Optional[str]): Error message if failed

**Example**:
```python
historical_quality = process_data['quality'].values
result = predictor.forecast_quality_metric(
    historical_quality, forecast_horizon=10, model_type='arima'
)
```

##### `forecast_process_parameter(parameter_history: pd.DataFrame, parameter_name: str, forecast_horizon: int = None) -> TimeSeriesPredictionResult`

Forecast specific process parameter.

**Parameters**:
- `parameter_history` (pd.DataFrame): DataFrame with parameter history
- `parameter_name` (str): Name of parameter to forecast
- `forecast_horizon` (int, optional): Number of steps ahead

**Returns**: `TimeSeriesPredictionResult` with forecast for parameter

**Raises**: `ValueError` if parameter not found in history

**Example**:
```python
param_history = pd.DataFrame({'temperature': temp_values, 'power': power_values})
result = predictor.forecast_process_parameter(
    param_history, parameter_name='temperature', forecast_horizon=10
)
```

##### `detect_anomalies_in_forecast(forecast: TimeSeriesPredictionResult, actual_values: np.ndarray) -> np.ndarray`

Detect anomalies by comparing forecast to actual values.

**Parameters**:
- `forecast` (TimeSeriesPredictionResult): Forecast result
- `actual_values` (np.ndarray): Actual values to compare against

**Returns**: Array of boolean anomaly flags (True = anomaly detected)

**Raises**: `ValueError` if forecast not successful or length mismatch

**Example**:
```python
actual = get_actual_quality_values()
anomalies = predictor.detect_anomalies_in_forecast(forecast_result, actual)
```

---

### PredictionValidator

Validation workflows for prediction models.

```python
from am_qadf.analytics.process_analysis.prediction import (
    PredictionValidator,
    OptimizationValidationResult
)

validator = PredictionValidator(config: PredictionConfig)
```

#### Methods

##### `cross_validate_model(predictor: Union[QualityPredictor, EarlyDefectPredictor], process_data: pd.DataFrame, quality_target: str, n_folds: int = None, validation_method: str = None) -> Dict[str, float]`

Perform cross-validation on prediction model.

**Parameters**:
- `predictor`: Trained predictor model (QualityPredictor or EarlyDefectPredictor)
- `process_data` (pd.DataFrame): DataFrame containing process data
- `quality_target` (str): Name of quality target variable
- `n_folds` (int, optional): Number of folds. Uses config if None.
- `validation_method` (str, optional): Method - `'kfold'`, `'stratified'`, `'time_series_split'`. Uses config if None.

**Returns**: Dictionary with mean and std of performance metrics across folds:
- `'mean_r2'` / `'mean_accuracy'`: Mean metric value
- `'std_r2'` / `'std_accuracy'`: Standard deviation
- `'n_folds'`: Number of folds used
- `'validation_method'`: Method used
- `'analysis_time'`: Time taken

**Example**:
```python
cv_result = validator.cross_validate_model(
    quality_predictor, process_data, quality_target='quality',
    n_folds=5, validation_method='kfold'
)
```

##### `validate_with_experimental_data(predictor: Union[QualityPredictor, EarlyDefectPredictor], predicted_data: pd.DataFrame, experimental_data: pd.DataFrame, quality_target: str) -> OptimizationValidationResult`

Validate predictions against experimental data.

**Parameters**:
- `predictor`: Trained predictor model
- `predicted_data` (pd.DataFrame): Data used for predictions
- `experimental_data` (pd.DataFrame): Experimental validation data with actual values
- `quality_target` (str): Name of quality target variable

**Returns**: `OptimizationValidationResult` with:
- `success` (bool): Whether validation succeeded
- `validation_method` (str): 'experimental'
- `predicted_objective` (float): Mean predicted value
- `experimental_objective` (float): Mean experimental value
- `validation_error` (float): MAE between predicted and experimental
- `validation_metrics` (Dict[str, float]): Metrics (RMSE, MAE, R², Accuracy, etc.)
- `experimental_data` (pd.DataFrame): Experimental data used
- `validation_time` (float): Validation time in seconds

**Raises**: `ValueError` if quality target not found in experimental data

**Example**:
```python
validation_result = validator.validate_with_experimental_data(
    quality_predictor, predicted_data, experimental_data, quality_target='quality'
)
```

##### `calculate_prediction_intervals(predictions: np.ndarray, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]`

Calculate prediction intervals for uncertainty quantification.

**Parameters**:
- `predictions` (np.ndarray): Array of predictions
- `confidence_level` (float): Confidence level (default: 0.95)

**Returns**: Tuple of `(lower_bound, upper_bound)` arrays

**Example**:
```python
predictions = model.predict(test_data)
lower, upper = validator.calculate_prediction_intervals(predictions, confidence_level=0.95)
```

---

## Model Tracking Module

### ModelRegistry

Model versioning and registry.

```python
from am_qadf.analytics.process_analysis.model_tracking import (
    ModelRegistry,
    ModelVersion
)

registry = ModelRegistry(storage_path: str = 'models/')
```

#### Methods

##### `register_model(model: Any, model_type: str, version: str, metadata: Optional[Dict[str, Any]] = None, performance_metrics: Optional[Dict[str, float]] = None, validation_metrics: Optional[Dict[str, float]] = None, feature_importance: Optional[Dict[str, float]] = None) -> str`

Register trained model in registry.

**Parameters**:
- `model`: Trained model object (e.g., sklearn model, tensorflow model)
- `model_type` (str): Type of model (e.g., 'RandomForestRegressor', 'ARIMA')
- `version` (str): Version string (e.g., '1.0', '2.1-beta')
- `metadata` (Dict[str, Any], optional): Additional metadata
- `performance_metrics` (Dict[str, float], optional): Initial performance metrics
- `validation_metrics` (Dict[str, float], optional): Cross-validation metrics
- `feature_importance` (Dict[str, float], optional): Feature importance scores

**Returns**: Model ID (unique identifier)

**Example**:
```python
model_id = registry.register_model(
    model=trained_model,
    model_type='RandomForestRegressor',
    version='1.0',
    metadata={'feature_names': ['temp', 'power']},
    performance_metrics={'r2_score': 0.85, 'rmse': 0.1}
)
```

##### `load_model(model_id: str) -> Tuple[Any, ModelVersion]`

Load model and its metadata from registry.

**Parameters**:
- `model_id` (str): Unique identifier of the model

**Returns**: Tuple of `(trained_model_object, ModelVersion_metadata)`

**Raises**: `ValueError` if model not found, `FileNotFoundError` if model file missing

**Example**:
```python
model, model_version = registry.load_model(model_id)
```

##### `list_models(model_type: Optional[str] = None, version: Optional[str] = None) -> List[Dict[str, Any]]`

List models in registry with optional filters.

**Parameters**:
- `model_type` (str, optional): Filter by model type
- `version` (str, optional): Filter by version

**Returns**: List of model metadata dictionaries

**Example**:
```python
all_models = registry.list_models()
rf_models = registry.list_models(model_type='RandomForestRegressor')
v1_models = registry.list_models(version='1.0')
```

##### `compare_models(model_id1: str, model_id2: str) -> Dict[str, Any]`

Compare two model versions.

**Parameters**:
- `model_id1` (str): ID of first model
- `model_id2` (str): ID of second model

**Returns**: Dictionary with comparison results including performance metrics differences

**Raises**: `ValueError` if either model not found

**Example**:
```python
comparison = registry.compare_models('model_001', 'model_002')
```

##### `delete_model(model_id: str) -> bool`

Delete a model from the registry and its storage.

**Parameters**:
- `model_id` (str): ID of the model to delete

**Returns**: True if successful, False otherwise

---

### ModelPerformanceTracker

Track and monitor model performance over time.

```python
from am_qadf.analytics.process_analysis.model_tracking import (
    ModelPerformanceTracker,
    ModelPerformanceMetrics
)

tracker = ModelPerformanceTracker(
    model_id: str,
    model_registry: ModelRegistry,
    history_size: int = 100
)
```

#### Methods

##### `evaluate_model_performance(model: Any, test_data: pd.DataFrame, quality_target: str, evaluation_date: datetime = None, feature_names: Optional[List[str]] = None) -> ModelPerformanceMetrics`

Evaluate model performance on new test data.

**Parameters**:
- `model`: Trained model object (must have predict method)
- `test_data` (pd.DataFrame): DataFrame with test data
- `quality_target` (str): Name of quality target variable
- `evaluation_date` (datetime, optional): Date of evaluation. Uses current time if None.
- `feature_names` (List[str], optional): Feature names. Inferred if None.

**Returns**: `ModelPerformanceMetrics` with:
- `model_id` (str): Model ID
- `model_type` (str): Model type
- `version` (str): Model version
- `training_date` (datetime): Original training date
- `performance_metrics` (Dict[str, float]): Current performance (R², RMSE, MAE, Accuracy, F1, etc.)
- `validation_metrics` (Dict[str, float]): Validation metrics
- `feature_importance` (Dict[str, float]): Feature importance
- `drift_score` (float): Drift score (0-1)
- `last_evaluated` (datetime): Last evaluation timestamp
- `evaluation_count` (int): Number of evaluations
- `metadata` (Dict[str, Any]): Additional metadata

**Example**:
```python
metrics = tracker.evaluate_model_performance(
    model, test_data, quality_target='quality'
)
```

##### `detect_performance_degradation(metric_name: str = 'r2_score', threshold: float = 0.1, min_evaluations: int = 5) -> Tuple[bool, Optional[float]]`

Detect if model performance has degraded significantly compared to baseline.

**Parameters**:
- `metric_name` (str): The metric to monitor (e.g., 'r2_score', 'accuracy')
- `threshold` (float): Percentage degradation (e.g., 0.1 for 10%) to trigger alert
- `min_evaluations` (int): Minimum number of evaluations in history to start detecting

**Returns**: Tuple of `(degradation_detected, degradation_percentage)`

**Example**:
```python
detected, degradation_pct = tracker.detect_performance_degradation(
    metric_name='r2_score', threshold=0.1, min_evaluations=5
)
```

##### `get_performance_trend(metric_name: str) -> Dict[str, Any]`

Get performance trend for specific metric from history.

**Parameters**:
- `metric_name` (str): The metric for which to get the trend

**Returns**: Dictionary with:
- `'metric_name'` (str): Metric analyzed
- `'trend'` (str): Trend direction ('improving', 'degrading', 'stable', 'insufficient_data')
- `'slope'` (float): Linear regression slope
- `'avg_change_per_eval'` (float): Average change per evaluation
- `'initial_value'` (float): First value in history
- `'latest_value'` (float): Last value in history
- `'data_points'` (int): Number of data points

**Example**:
```python
trend = tracker.get_performance_trend('r2_score')
```

##### `calculate_drift_score(current_data: pd.DataFrame, training_data: pd.DataFrame) -> float`

Calculate model drift score (data distribution change).

**Parameters**:
- `current_data` (pd.DataFrame): DataFrame of current input data
- `training_data` (pd.DataFrame): DataFrame of reference input data (training data)

**Returns**: Drift score (0-1, higher = more drift)

**Example**:
```python
drift_score = tracker.calculate_drift_score(current_data, training_data)
```

##### `get_performance_history() -> List[Dict[str, Any]]`

Get performance history as list of dictionaries.

**Returns**: List of performance history entries with metrics, timestamps, and drift scores

---

### ModelMonitor

Monitor model performance and drift in production.

```python
from am_qadf.analytics.process_analysis.model_tracking import (
    ModelMonitor,
    ModelMonitoringConfig
)

monitor = ModelMonitor(
    model_id: str,
    model_registry: ModelRegistry,
    performance_tracker: ModelPerformanceTracker,
    monitoring_config: Optional[ModelMonitoringConfig] = None
)
```

#### Methods

##### `start_monitoring(data_source_callable: Callable[[], pd.DataFrame], reference_data_for_drift: pd.DataFrame)`

Start continuous monitoring threads for performance and drift.

**Parameters**:
- `data_source_callable`: Function that returns new batch of data for evaluation
- `reference_data_for_drift`: Baseline data (training data) for drift detection

**Example**:
```python
def get_new_data():
    return fetch_latest_process_data()

monitor.start_monitoring(get_new_data, training_data)
```

##### `stop_monitoring()`

Stop all continuous monitoring threads.

---

## Enhanced Optimization

### ProcessOptimizer (Enhanced)

Enhanced process optimizer with new capabilities.

```python
from am_qadf.analytics.process_analysis.optimization import (
    ProcessOptimizer,
    OptimizationConfig,
    OptimizationResult
)

optimizer = ProcessOptimizer(config: OptimizationConfig)
```

#### Enhanced Methods

##### `optimize_with_constraints(objective_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], constraints: List[Callable], parameter_names: List[str] = None, constraint_method: str = None) -> OptimizationResult`

Optimize with explicit constraints using constraint handler.

**Parameters**:
- `objective_function` (Callable): Function to optimize
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Parameter bounds {name: (min, max)}
- `constraints` (List[Callable]): List of constraint functions (should return <= 0 for feasible)
- `parameter_names` (List[str], optional): Parameter names. Inferred if None.
- `constraint_method` (str, optional): Method - `'penalty'`, `'barrier'`, `'augmented_lagrangian'`. Uses config if None.

**Returns**: `OptimizationResult` with constrained optimization results

**Example**:
```python
def energy_constraint(params):
    return (params['laser_power'] / params['scan_speed']) - 0.3  # <= 0 for feasible

result = optimizer.optimize_with_constraints(
    objective_function, parameter_bounds, [energy_constraint],
    constraint_method='penalty'
)
```

##### `optimize_realtime(objective_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], streaming_data_source: Callable[[], Dict[str, float]], parameter_names: List[str] = None, update_interval: float = None) -> OptimizationResult`

Perform real-time optimization with streaming data updates.

**Parameters**:
- `objective_function` (Callable): Function to optimize (takes params and latest data, returns scalar)
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Parameter bounds
- `streaming_data_source` (Callable): Callable that returns latest streaming data as dict
- `parameter_names` (List[str], optional): Parameter names. Inferred if None.
- `update_interval` (float, optional): Interval in seconds. Uses config if None.

**Returns**: `OptimizationResult` with latest optimal parameters

**Example**:
```python
def get_latest_data():
    return fetch_streaming_data()

result = optimizer.optimize_realtime(
    objective_function, parameter_bounds, get_latest_data,
    update_interval=1.0
)
```

##### `visualize_pareto_front(pareto_result: OptimizationResult, objective_names: List[str]) -> Any`

Visualize Pareto front (delegates to ParetoVisualizer).

**Parameters**:
- `pareto_result` (OptimizationResult): Multi-objective optimization result
- `objective_names` (List[str]): Names of objectives for labeling

**Returns**: Matplotlib figure

---

## Enhanced Quality Predictor

### QualityPredictor (Enhanced)

Enhanced quality predictor with new capabilities.

```python
from am_qadf.analytics.process_analysis.quality_analysis import (
    QualityPredictor,
    QualityAnalysisConfig
)

predictor = QualityPredictor(config: QualityAnalysisConfig)
```

#### Enhanced Methods

##### `predict_early_defect(partial_process_data: pd.DataFrame, build_progress: float, early_horizon: Optional[int] = None) -> EarlyDefectPredictionResult`

Predict defects early in build using the EarlyDefectPredictor.

**Parameters**:
- `partial_process_data` (pd.DataFrame): Process data from partial build
- `build_progress` (float): Current build progress (0.0-1.0)
- `early_horizon` (int, optional): Samples before completion to use

**Returns**: `EarlyDefectPredictionResult` with defect probabilities and confidence

##### `forecast_quality_timeseries(historical_quality: np.ndarray, forecast_horizon: int, model_type: str = 'arima') -> TimeSeriesPredictionResult`

Forecast quality using time-series model via TimeSeriesPredictor.

**Parameters**:
- `historical_quality` (np.ndarray): Historical time-series data
- `forecast_horizon` (int): Number of steps ahead to forecast
- `model_type` (str): Type of time-series model

**Returns**: `TimeSeriesPredictionResult` with forecast and confidence intervals

##### `cross_validate(X: np.ndarray, y: np.ndarray, model: Any = None, validation_method: str = None, n_folds: int = None) -> Dict[str, float]`

Perform cross-validation using PredictionValidator.

**Parameters**:
- `X` (np.ndarray): Feature data
- `y` (np.ndarray): Target data
- `model` (Any, optional): Model to cross-validate. Uses trained model if None.
- `validation_method` (str, optional): Cross-validation method
- `n_folds` (int, optional): Number of folds

**Returns**: Dictionary of cross-validation metrics

---

## Configuration Classes

### PredictionConfig

```python
@dataclass
class PredictionConfig:
    model_type: str = 'random_forest'
    enable_early_prediction: bool = True
    early_prediction_horizon: int = 100
    time_series_forecast_horizon: int = 10
    enable_deep_learning: bool = False
    validation_method: str = 'cross_validation'
    n_folds: int = 5
    test_size: float = 0.2
    random_seed: Optional[int] = None
    confidence_threshold: float = 0.7
```

### ModelMonitoringConfig

```python
@dataclass
class ModelMonitoringConfig:
    enable_performance_monitoring: bool = True
    performance_check_interval_seconds: float = 3600.0
    performance_degradation_threshold: float = 0.1
    primary_performance_metric: str = 'r2_score'
    min_evaluations_for_degradation: int = 5
    enable_drift_monitoring: bool = True
    drift_check_interval_seconds: float = 86400.0
    drift_p_value_threshold: float = 0.05
    drift_features_subset: Optional[List[str]] = None
    enable_retraining_trigger: bool = True
    retraining_trigger_threshold_degradation: float = 0.15
    retraining_trigger_threshold_drift_features: int = 3
    alert_on_degradation: bool = True
    alert_on_drift: bool = True
```

---

## Result Classes

### EarlyDefectPredictionResult

```python
@dataclass
class EarlyDefectPredictionResult:
    success: bool
    model_type: str
    defect_probability: np.ndarray
    defect_prediction: np.ndarray
    prediction_confidence: np.ndarray
    early_prediction_accuracy: float
    prediction_horizon: int
    model_performance: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    analysis_time: float = 0.0
    error_message: Optional[str] = None
```

### TimeSeriesPredictionResult

```python
@dataclass
class TimeSeriesPredictionResult:
    success: bool
    model_type: str
    forecast: np.ndarray
    forecast_lower_bound: np.ndarray
    forecast_upper_bound: np.ndarray
    forecast_horizon: int
    historical_data: np.ndarray
    model_performance: Dict[str, float]
    trend_components: Optional[np.ndarray] = None
    seasonality_components: Optional[np.ndarray] = None
    analysis_time: float = 0.0
    error_message: Optional[str] = None
```

### ModelVersion

```python
@dataclass
class ModelVersion:
    model_id: str
    model_type: str
    version: str
    training_date: datetime
    performance_metrics: Dict[str, float]
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    storage_path: Optional[str] = None
    file_path: Optional[str] = None
```

### ModelPerformanceMetrics

```python
@dataclass
class ModelPerformanceMetrics:
    model_id: str
    model_type: str
    version: str
    training_date: datetime
    performance_metrics: Dict[str, float]
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    drift_score: float = 0.0
    last_evaluated: datetime = field(default_factory=datetime.now)
    evaluation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

**Parent**: [Process Analysis Prediction Module](../05-modules/process-analysis-prediction.md)
