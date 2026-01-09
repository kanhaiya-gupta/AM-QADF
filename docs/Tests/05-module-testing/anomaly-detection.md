# Anomaly Detection Module - Testing Guide

## Test Structure

```
tests/unit/anomaly_detection/
├── core/
│   ├── test_base_detector.py
│   └── test_types.py
├── detectors/
│   ├── statistical/
│   ├── clustering/
│   ├── ensemble/
│   ├── machine_learning/
│   └── rule_based/
├── integration/
│   ├── test_client.py
│   ├── test_query.py
│   └── test_storage.py
├── utils/
│   ├── test_preprocessing.py
│   ├── test_synthetic_anomalies.py
│   └── test_voxel_detection.py
├── evaluation/
│   ├── test_metrics.py
│   ├── test_comparison.py
│   └── test_cross_validation.py
├── reporting/
│   └── test_report_generator.py
└── visualization/
    ├── test_spatial_visualization.py
    ├── test_temporal_visualization.py
    └── test_comparison_visualization.py
```

## Key Tests

### Detection Algorithms
- Statistical detectors (Z-score, IQR, Mahalanobis, etc.)
- Clustering detectors (DBSCAN, Isolation Forest, LOF, etc.)
- Machine learning detectors (Autoencoder, LSTM, VAE, etc.)
- Rule-based detectors (Threshold, Pattern, etc.)
- Ensemble detectors (Voting, Weighted)

### Detection Accuracy
- False positive/negative rates
- Detection accuracy
- Performance with large voxel grids

### Integration
- Client interface
- Query interface
- Storage operations

## Coverage Target

**80%+**

## Example Tests

```python
def test_z_score_detector_accuracy():
    """Test Z-score detector accuracy."""
    
def test_isolation_forest_detection():
    """Test Isolation Forest detection."""
    
def test_ensemble_detection():
    """Test ensemble detector combinations."""
```

## Running Anomaly Detection Module Tests

```bash
# Run all anomaly detection tests
pytest tests/unit/anomaly_detection/ -m unit

# Run by submodule
pytest tests/unit/anomaly_detection/core/              # Core detectors
pytest tests/unit/anomaly_detection/detectors/         # All detectors
pytest tests/unit/anomaly_detection/integration/       # Integration
pytest tests/unit/anomaly_detection/evaluation/        # Evaluation
pytest tests/unit/anomaly_detection/utils/             # Utilities

# Run specific detector types
pytest tests/unit/anomaly_detection/detectors/statistical/      # Statistical
pytest tests/unit/anomaly_detection/detectors/clustering/        # Clustering
pytest tests/unit/anomaly_detection/detectors/machine_learning/  # ML
pytest tests/unit/anomaly_detection/detectors/rule_based/        # Rule-based
pytest tests/unit/anomaly_detection/detectors/ensemble/          # Ensemble

# Run specific detector
pytest tests/unit/anomaly_detection/detectors/statistical/test_z_score.py
pytest tests/unit/anomaly_detection/detectors/clustering/test_isolation_forest.py

# Run with coverage
pytest tests/unit/anomaly_detection/ --cov=am_qadf.anomaly_detection --cov-report=term-missing
```

## Related

- [Anomaly Detection Test Plan](../anomaly_detection_Test_Plan.md) - Detailed test plan

---

**Parent**: [Module Testing Guides](README.md)

