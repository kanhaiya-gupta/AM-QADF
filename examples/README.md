# AM-QADF Usage Examples

This directory contains example scripts demonstrating how to use the AM-QADF framework.

## Example Scripts

### Basic Examples

- **[basic_usage.py](basic_usage.py)** - Basic framework usage
  - Initialize clients
  - Query data
  - Create voxel grid
  - Map signals to voxels

### Module-Specific Examples

- **[signal_mapping_example.py](signal_mapping_example.py)** - Signal mapping and interpolation
  - Nearest neighbor interpolation
  - Linear interpolation
  - IDW interpolation
  - KDE interpolation
  - RBF interpolation

- **[fusion_example.py](fusion_example.py)** - Multi-source data fusion
  - Weighted average fusion
  - Median fusion
  - Quality-based fusion
  - Compare fusion strategies

- **[quality_assessment_example.py](quality_assessment_example.py)** - Quality assessment
  - Data completeness analysis
  - Signal quality (SNR, uncertainty)
  - Alignment accuracy
  - Quality metrics

- **[analytics_example.py](analytics_example.py)** - Statistical and sensitivity analysis
  - Descriptive statistics
  - Correlation analysis
  - Sensitivity analysis (Sobol, Morris)
  - Process analysis

- **[anomaly_detection_example.py](anomaly_detection_example.py)** - Anomaly detection
  - Statistical detectors (Z-score, IQR)
  - Clustering detectors (DBSCAN, Isolation Forest)
  - ML-based detectors (Autoencoder)
  - Compare detection results

- **[visualization_example.py](visualization_example.py)** - 3D visualization
  - 3D voxel grid rendering
  - 2D slice visualization
  - Signal distribution plots
  - Interactive widgets (Jupyter)

### Complete Workflow

- **[complete_workflow_example.py](complete_workflow_example.py)** - End-to-end workflow
  - Complete pipeline from data query to visualization
  - Demonstrates all major framework capabilities
  - Best starting point for understanding the full workflow

## Running Examples

### Prerequisites

1. **MongoDB**: Ensure MongoDB is running (for examples that use database)
   ```bash
   # Check MongoDB status
   mongosh --eval "db.version()"
   ```

2. **Python Dependencies**: Install required packages
   ```bash
   pip install numpy scipy scikit-learn
   # Optional dependencies:
   pip install pyvista matplotlib  # For visualization
   pip install tensorflow keras     # For ML-based anomaly detection
   pip install salib                # For sensitivity analysis
   ```

### Running Examples

```bash
# Basic usage
python examples/basic_usage.py

# Signal mapping
python examples/signal_mapping_example.py

# Data fusion
python examples/fusion_example.py

# Quality assessment
python examples/quality_assessment_example.py

# Analytics
python examples/analytics_example.py

# Anomaly detection
python examples/anomaly_detection_example.py

# Visualization
python examples/visualization_example.py

# Complete workflow
python examples/complete_workflow_example.py
```

### Configuration

Before running examples, you may need to:

1. **Update Model ID**: Replace `"example_model_id"` with an actual model ID from your database
2. **MongoDB Connection**: Update connection string if MongoDB is not on localhost
3. **Data Availability**: Ensure your database contains data for the specified model

## Example Output

Each example script provides:
- Step-by-step progress messages
- Results and statistics
- Error handling and troubleshooting tips
- Recommendations for further use

## Notes

- **Sample Data**: Some examples generate synthetic data for demonstration
- **MongoDB**: Examples gracefully handle MongoDB unavailability (with warnings)
- **Dependencies**: Examples handle missing optional dependencies gracefully
- **Performance**: Examples use reduced data sizes for faster execution

## Related Documentation

- [Quick Start Guide](../docs/AM_QADF/04-quick-start.md)
- [Module Documentation](../docs/AM_QADF/05-modules/README.md)
- [API Reference](../docs/AM_QADF/06-api-reference/README.md)
- [Examples Documentation](../docs/AM_QADF/07-examples/README.md)

## Contributing

When adding new examples:
1. Follow the existing structure and style
2. Include comprehensive comments
3. Handle errors gracefully
4. Provide clear output messages
5. Update this README

---

**Last Updated**: 2024

