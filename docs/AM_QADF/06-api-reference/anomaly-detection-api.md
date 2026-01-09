# Anomaly Detection Module API Reference

## Overview

The Anomaly Detection module provides various anomaly detection algorithms for identifying outliers in voxel domain data.

## AnomalyDetectionClient

Main client for anomaly detection operations.

```python
from am_qadf.anomaly_detection import AnomalyDetectionClient

client = AnomalyDetectionClient(mongo_client: Optional[MongoDBClient] = None)
```

### Methods

#### `detect_anomalies(voxel_data: Any, signal_name: str, detector_type: str = 'zscore', **kwargs) -> AnomalyDetectionResult`

Detect anomalies in a signal.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object (VoxelGrid)
- `signal_name` (str): Name of the signal to analyze
- `detector_type` (str): Detector type ('zscore', 'iqr', 'isolation_forest', 'dbscan', etc.)
- `**kwargs`: Additional detector-specific parameters

**Returns**: `AnomalyDetectionResult` object

**Example**:
```python
result = client.detect_anomalies(
    voxel_data=grid,
    signal_name='power',
    detector_type='isolation_forest',
    contamination=0.1
)
```

#### `query_anomalies(model_id: str, signal_name: Optional[str] = None, detector_type: Optional[str] = None) -> List[AnomalyDetectionResult]`

Query stored anomaly detection results.

**Parameters**:
- `model_id` (str): Model identifier
- `signal_name` (Optional[str]): Filter by signal name
- `detector_type` (Optional[str]): Filter by detector type

**Returns**: List of `AnomalyDetectionResult` objects

#### `store_anomalies(result: AnomalyDetectionResult, model_id: str) -> None`

Store anomaly detection results.

**Parameters**:
- `result` (AnomalyDetectionResult): Detection result
- `model_id` (str): Model identifier

---

## BaseAnomalyDetector

Abstract base class for all anomaly detectors.

```python
from am_qadf.anomaly_detection import BaseAnomalyDetector

class MyDetector(BaseAnomalyDetector):
    def detect(self, data: np.ndarray, **kwargs) -> AnomalyDetectionResult:
        # Implementation
        pass
```

### Abstract Methods

#### `detect(data: np.ndarray, **kwargs) -> AnomalyDetectionResult`

Detect anomalies in data.

**Parameters**:
- `data` (np.ndarray): Data array
- `**kwargs`: Detector-specific parameters

**Returns**: `AnomalyDetectionResult` object

---

## Statistical Detectors

### ZScoreDetector

Z-score based anomaly detection.

```python
from am_qadf.anomaly_detection import ZScoreDetector

detector = ZScoreDetector(threshold: float = 3.0)
```

#### Methods

##### `detect(data: np.ndarray, threshold: float = 3.0) -> AnomalyDetectionResult`

Detect anomalies using Z-score.

**Parameters**:
- `data` (np.ndarray): Data array
- `threshold` (float): Z-score threshold (default: 3.0)

**Returns**: `AnomalyDetectionResult` object

### IQRDetector

Interquartile range based anomaly detection.

```python
from am_qadf.anomaly_detection import IQRDetector

detector = IQRDetector(factor: float = 1.5)
```

#### Methods

##### `detect(data: np.ndarray, factor: float = 1.5) -> AnomalyDetectionResult`

Detect anomalies using IQR.

**Parameters**:
- `data` (np.ndarray): Data array
- `factor` (float): IQR factor (default: 1.5)

**Returns**: `AnomalyDetectionResult` object

---

## Clustering Detectors

### DBSCANDetector

DBSCAN clustering based anomaly detection.

```python
from am_qadf.anomaly_detection import DBSCANDetector

detector = DBSCANDetector(eps: float = 0.5, min_samples: int = 5)
```

#### Methods

##### `detect(data: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> AnomalyDetectionResult`

Detect anomalies using DBSCAN.

**Parameters**:
- `data` (np.ndarray): Data array
- `eps` (float): Maximum distance for clustering
- `min_samples` (int): Minimum samples per cluster

**Returns**: `AnomalyDetectionResult` object

### IsolationForestDetector

Isolation Forest based anomaly detection.

```python
from am_qadf.anomaly_detection import IsolationForestDetector

detector = IsolationForestDetector(contamination: float = 0.1)
```

#### Methods

##### `detect(data: np.ndarray, contamination: float = 0.1) -> AnomalyDetectionResult`

Detect anomalies using Isolation Forest.

**Parameters**:
- `data` (np.ndarray): Data array
- `contamination` (float): Expected proportion of anomalies (0-1)

**Returns**: `AnomalyDetectionResult` object

---

## ML-Based Detectors

### AutoencoderDetector

Autoencoder based anomaly detection.

```python
from am_qadf.anomaly_detection import AutoencoderDetector

detector = AutoencoderDetector(threshold: float = 0.1)
```

#### Methods

##### `detect(data: np.ndarray, threshold: float = 0.1) -> AnomalyDetectionResult`

Detect anomalies using autoencoder reconstruction error.

**Parameters**:
- `data` (np.ndarray): Data array
- `threshold` (float): Reconstruction error threshold

**Returns**: `AnomalyDetectionResult` object

---

## AnomalyDetectionResult

Result container for anomaly detection.

```python
@dataclass
class AnomalyDetectionResult:
    anomaly_mask: np.ndarray  # Boolean mask (True = anomaly)
    anomaly_scores: np.ndarray  # Anomaly scores
    detector_type: str  # Detector type name
    parameters: Dict[str, Any]  # Detector parameters
    metadata: Dict[str, Any]  # Additional metadata
```

### Attributes

- **anomaly_mask** (`np.ndarray`): Boolean mask indicating anomalies
- **anomaly_scores** (`np.ndarray`): Anomaly scores (higher = more anomalous)
- **detector_type** (`str`): Name of detector used
- **parameters** (`Dict[str, Any]`): Detector parameters
- **metadata** (`Dict[str, Any]`): Additional metadata

---

## Related

- [Anomaly Detection Module Documentation](../05-modules/anomaly-detection.md) - Module overview
- [All API References](README.md) - Other API references

---

**Parent**: [API Reference](README.md)

