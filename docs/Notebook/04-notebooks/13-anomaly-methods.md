# Notebook 13: Anomaly Detection Methods

**File**: `13_Anomaly_Detection_Methods.ipynb`  
**Category**: Anomaly Detection  
**Duration**: 90-120 minutes

## Purpose

This notebook teaches you how to detect anomalies using various detection methods. You'll learn statistical, clustering, ML-based, and rule-based anomaly detection techniques.

## Learning Objectives

By the end of this notebook, you will:

- ✅ Use statistical anomaly detectors (Z-Score, IQR, Mahalanobis, Grubbs)
- ✅ Apply clustering-based detectors (DBSCAN, Isolation Forest, LOF, One-Class SVM, K-Means)
- ✅ Use ML-based detectors (Autoencoder, LSTM, VAE, Random Forest)
- ✅ Implement rule-based detectors (Threshold, Pattern, Spatial, Temporal, Multi-Signal)
- ✅ Compare detection methods

## Topics Covered

### Statistical Methods

- **Z-Score**: Standard deviation-based detection
- **IQR**: Interquartile range method
- **Mahalanobis Distance**: Multivariate distance
- **Grubbs Test**: Outlier test

### Clustering Methods

- **DBSCAN**: Density-based clustering
- **Isolation Forest**: Isolation-based detection
- **LOF**: Local Outlier Factor
- **One-Class SVM**: Support vector machine
- **K-Means**: K-means clustering

### ML-Based Methods

- **Autoencoder**: Reconstruction error-based
- **LSTM**: Time series anomaly detection
- **VAE**: Variational autoencoder
- **Random Forest**: Tree-based detection

### Rule-Based Methods

- **Threshold**: Value threshold rules
- **Pattern**: Pattern-based rules
- **Spatial**: Spatial pattern rules
- **Temporal**: Temporal pattern rules
- **Multi-Signal**: Multi-signal rules

### Ensemble Methods

- **Voting**: Majority voting
- **Weighted**: Weighted ensemble

## Interactive Widgets

### Top Panel

- **Detector Type**: Dropdown (Statistical/Clustering/ML/Rule-Based/Ensemble)
- **Signal Selector**: Dropdown to select signal
- **Execute Detection**: Button to execute detection
- **Compare Detectors**: Button to compare methods

### Left Panel

- **Dynamic Configuration**: Accordion sections
  - **Statistical**: Method, threshold parameters
  - **Clustering**: Algorithm, parameters
  - **ML**: Model type, training options
  - **Rule-Based**: Rule definition
  - **Ensemble**: Ensemble configuration

### Center Panel

- **Visualization Modes**: Radio buttons
  - **Temporal**: Temporal anomaly visualization
  - **Spatial**: Spatial anomaly visualization
  - **Comparison**: Method comparison
  - **Metrics**: Detection metrics visualization

### Right Panel

- **Detection Metrics**: Precision, recall, F1-score
- **Anomaly Statistics**: Anomaly count, percentage
- **Detector Performance**: Performance metrics
- **Comparison Results**: Method comparison
- **Export Options**: Export detection results

### Bottom Panel

- **Status Display**: Current operation status
- **Progress Bar**: Detection progress
- **Info Display**: Additional information

## Usage

### Step 1: Select Detector

1. Choose detector type
2. Select signal to analyze
3. Review signal information

### Step 2: Configure Detector

1. Configure detector-specific parameters
2. Set detection thresholds
3. Configure training options if ML-based

### Step 3: Execute Detection

1. Click "Execute Detection" button
2. Wait for detection to complete
3. Review detection results

### Step 4: Compare Methods

1. Click "Compare Detectors"
2. View comparison visualization
3. Review performance metrics
4. Export results

## Example Workflow

1. **Select Type**: Choose "Clustering"
2. **Select Method**: Choose "Isolation Forest"
3. **Select Signal**: Choose temperature signal
4. **Configure**: Set contamination parameter
5. **Execute**: Run detection
6. **Compare**: Compare with DBSCAN

## Key Takeaways

1. **Method Selection**: Choose appropriate detection method
2. **Parameter Tuning**: Adjust parameters for optimal performance
3. **Signal Selection**: Select relevant signals for detection
4. **Comparison**: Compare multiple methods
5. **Validation**: Validate detection results

## Related Notebooks

- **Previous**: [12: Virtual Experiments](12-virtual-experiments.md)
- **Next**: [14: Anomaly Detection Workflow](14-anomaly-workflow.md)
- **Related**: [09: Statistical Analysis](09-statistical.md)

## Related Documentation

- **[Anomaly Detection Module](../../AM_QADF/05-modules/anomaly-detection.md)** - Anomaly detection details
- **[Anomaly Detection API](../../AM_QADF/06-api-reference/anomaly-detection-api.md)** - Anomaly detection API

---

**Last Updated**: 2024

