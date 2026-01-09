"""
Unit tests for data preprocessing utilities.

Tests for PreprocessingConfig, DataPreprocessor, and extract_features_from_fused_data.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
from am_qadf.anomaly_detection.utils.preprocessing import (
    PreprocessingConfig,
    DataPreprocessor,
    extract_features_from_fused_data,
)


class TestPreprocessingConfig:
    """Test suite for PreprocessingConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating PreprocessingConfig with default values."""
        config = PreprocessingConfig()

        assert config.normalization_method == "standard"
        assert config.missing_data_strategy == "mean"
        assert config.remove_outliers is False
        assert config.outlier_method == "iqr"
        assert config.outlier_threshold == 3.0
        assert config.feature_selection is False
        assert config.feature_selection_method == "variance"
        assert config.min_variance == 0.01
        assert config.reduce_dimensions is False
        assert config.reduction_method == "pca"
        assert config.n_components == 10

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating PreprocessingConfig with custom values."""
        config = PreprocessingConfig(
            normalization_method="robust",
            missing_data_strategy="median",
            remove_outliers=True,
            outlier_method="zscore",
            outlier_threshold=2.5,
            feature_selection=True,
            feature_selection_method="correlation",
            min_variance=0.05,
            reduce_dimensions=True,
            reduction_method="ica",
            n_components=5,
        )

        assert config.normalization_method == "robust"
        assert config.missing_data_strategy == "median"
        assert config.remove_outliers is True
        assert config.outlier_method == "zscore"
        assert config.outlier_threshold == 2.5
        assert config.feature_selection is True
        assert config.feature_selection_method == "correlation"
        assert config.min_variance == 0.05
        assert config.reduce_dimensions is True
        assert config.reduction_method == "ica"
        assert config.n_components == 5


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return np.random.randn(100, 5) * 10 + 100

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.randn(50) * 10 + 100,
                "feature2": np.random.randn(50) * 5 + 50,
                "feature3": np.random.randn(50) * 2 + 10,
            }
        )

    @pytest.fixture
    def sample_dict_data(self):
        """Create sample dictionary data for testing."""

        class MockVoxelData:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
                self.laser_power = np.random.randn() * 10 + 100
                self.density = np.random.randn() * 0.1 + 0.9

        return {(i, j, k): MockVoxelData(i, j, k) for i in range(5) for j in range(5) for k in range(5)}

    @pytest.mark.unit
    def test_preprocessor_creation_default(self):
        """Test creating DataPreprocessor with default config."""
        preprocessor = DataPreprocessor()

        assert preprocessor.config is not None
        assert preprocessor.scaler is None
        assert preprocessor.feature_selector is None
        assert preprocessor.is_fitted is False

    @pytest.mark.unit
    def test_preprocessor_creation_custom_config(self):
        """Test creating DataPreprocessor with custom config."""
        config = PreprocessingConfig(normalization_method="robust")
        preprocessor = DataPreprocessor(config=config)

        assert preprocessor.config.normalization_method == "robust"

    @pytest.mark.unit
    def test_fit(self, sample_data):
        """Test fitting the preprocessor."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_data)

        assert preprocessor.is_fitted is True
        assert preprocessor.scaler is not None

    @pytest.mark.unit
    def test_fit_with_dataframe(self, sample_dataframe):
        """Test fitting with pandas DataFrame."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_dataframe)

        assert preprocessor.is_fitted is True

    @pytest.mark.unit
    def test_fit_with_dict_data(self, sample_dict_data):
        """Test fitting with dictionary data."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_dict_data)

        assert preprocessor.is_fitted is True
        assert hasattr(preprocessor, "feature_names")

    @pytest.mark.unit
    def test_transform_before_fit(self, sample_data):
        """Test that transform raises error if not fitted."""
        preprocessor = DataPreprocessor()

        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(sample_data)

    @pytest.mark.unit
    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        preprocessor = DataPreprocessor()
        transformed = preprocessor.fit_transform(sample_data)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape[0] == sample_data.shape[0]
        assert preprocessor.is_fitted is True

    @pytest.mark.unit
    def test_fit_transform_with_metadata(self, sample_data):
        """Test fit_transform with metadata."""
        preprocessor = DataPreprocessor()
        transformed, metadata = preprocessor.fit_transform(sample_data, return_metadata=True)

        assert isinstance(transformed, np.ndarray)
        assert isinstance(metadata, dict)
        assert "scaler" in metadata
        assert "n_features" in metadata
        assert "n_samples" in metadata

    @pytest.mark.unit
    def test_transform(self, sample_data):
        """Test transform method."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_data)
        transformed = preprocessor.transform(sample_data)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == sample_data.shape

    @pytest.mark.unit
    def test_normalization_standard(self, sample_data):
        """Test standard normalization."""
        preprocessor = DataPreprocessor(PreprocessingConfig(normalization_method="standard"))
        preprocessor.fit(sample_data)
        transformed = preprocessor.transform(sample_data)

        # Standardized data should have mean ~0 and std ~1
        assert np.allclose(np.mean(transformed, axis=0), 0, atol=0.1)
        assert np.allclose(np.std(transformed, axis=0), 1, atol=0.1)

    @pytest.mark.unit
    def test_normalization_robust(self, sample_data):
        """Test robust normalization."""
        preprocessor = DataPreprocessor(PreprocessingConfig(normalization_method="robust"))
        preprocessor.fit(sample_data)
        transformed = preprocessor.transform(sample_data)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == sample_data.shape

    @pytest.mark.unit
    def test_normalization_minmax(self, sample_data):
        """Test minmax normalization."""
        preprocessor = DataPreprocessor(PreprocessingConfig(normalization_method="minmax"))
        preprocessor.fit(sample_data)
        transformed = preprocessor.transform(sample_data)

        # MinMax scaled data should be in [0, 1] range
        assert np.all(transformed >= 0)
        assert np.all(transformed <= 1)

    @pytest.mark.unit
    def test_normalization_none(self, sample_data):
        """Test no normalization."""
        preprocessor = DataPreprocessor(PreprocessingConfig(normalization_method="none"))
        preprocessor.fit(sample_data)
        transformed = preprocessor.transform(sample_data)

        # Should be same as original (within numerical precision)
        assert np.allclose(transformed, sample_data)

    @pytest.mark.unit
    def test_missing_data_mean(self):
        """Test handling missing data with mean strategy."""
        data = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
        preprocessor = DataPreprocessor(PreprocessingConfig(missing_data_strategy="mean"))
        preprocessor.fit(data)
        transformed = preprocessor.transform(data)

        # NaN should be replaced with mean
        assert not np.isnan(transformed).any()

    @pytest.mark.unit
    def test_missing_data_median(self):
        """Test handling missing data with median strategy."""
        data = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
        preprocessor = DataPreprocessor(PreprocessingConfig(missing_data_strategy="median"))
        preprocessor.fit(data)
        transformed = preprocessor.transform(data)

        assert not np.isnan(transformed).any()

    @pytest.mark.unit
    def test_missing_data_drop(self):
        """Test handling missing data with drop strategy."""
        data = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
        preprocessor = DataPreprocessor(PreprocessingConfig(missing_data_strategy="drop"))
        preprocessor.fit(data)
        transformed = preprocessor.transform(data)

        # Rows with NaN should be dropped
        assert not np.isnan(transformed).any()
        assert transformed.shape[0] < data.shape[0]

    @pytest.mark.unit
    def test_missing_data_zero(self):
        """Test handling missing data with zero strategy."""
        data = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
        preprocessor = DataPreprocessor(PreprocessingConfig(missing_data_strategy="zero"))
        preprocessor.fit(data)
        transformed = preprocessor.transform(data)

        # NaN should be replaced with 0
        assert not np.isnan(transformed).any()
        assert np.any(transformed == 0)

    @pytest.mark.unit
    def test_remove_outliers_iqr(self, sample_data):
        """Test removing outliers using IQR method."""
        # Add some outliers
        data_with_outliers = sample_data.copy()
        data_with_outliers[0] = [1000, 1000, 1000, 1000, 1000]

        preprocessor = DataPreprocessor(PreprocessingConfig(remove_outliers=True, outlier_method="iqr"))
        preprocessor.fit(data_with_outliers)
        transformed = preprocessor.transform(data_with_outliers)

        # Outliers should be removed
        assert transformed.shape[0] <= data_with_outliers.shape[0]

    @pytest.mark.unit
    def test_remove_outliers_zscore(self, sample_data):
        """Test removing outliers using Z-score method."""
        # Add some outliers
        data_with_outliers = sample_data.copy()
        data_with_outliers[0] = [1000, 1000, 1000, 1000, 1000]

        preprocessor = DataPreprocessor(PreprocessingConfig(remove_outliers=True, outlier_method="zscore"))
        preprocessor.fit(data_with_outliers)
        transformed = preprocessor.transform(data_with_outliers)

        # Outliers should be removed
        assert transformed.shape[0] <= data_with_outliers.shape[0]

    @pytest.mark.unit
    def test_dict_to_array(self, sample_dict_data):
        """Test converting dictionary to array."""
        preprocessor = DataPreprocessor()
        array_data = preprocessor._to_array(sample_dict_data)

        assert isinstance(array_data, np.ndarray)
        assert array_data.shape[0] == len(sample_dict_data)
        assert hasattr(preprocessor, "feature_names")


class TestExtractFeaturesFromFusedData:
    """Test suite for extract_features_from_fused_data function."""

    @pytest.fixture
    def sample_fused_data(self):
        """Create sample fused data."""

        class MockVoxelData:
            def __init__(self, idx):
                self.laser_power = 100.0 + idx[0]
                self.density = 0.9 + idx[1] * 0.01
                self.temperature = 200.0 + idx[2]

        return {(i, j, k): MockVoxelData((i, j, k)) for i in range(3) for j in range(3) for k in range(3)}

    @pytest.mark.unit
    def test_extract_features(self, sample_fused_data):
        """Test extracting features from fused data."""
        features, indices, feature_names = extract_features_from_fused_data(sample_fused_data)

        assert isinstance(features, np.ndarray)
        assert len(indices) == len(sample_fused_data)
        assert len(feature_names) > 0
        assert features.shape[0] == len(sample_fused_data)
        assert features.shape[1] == len(feature_names)

    @pytest.mark.unit
    def test_extract_features_empty(self):
        """Test extracting features from empty data."""
        features, indices, feature_names = extract_features_from_fused_data({})

        assert isinstance(features, np.ndarray)
        assert len(features) == 0
        assert len(indices) == 0
        assert len(feature_names) == 0

    @pytest.mark.unit
    def test_extract_features_single_item(self):
        """Test extracting features from single item."""

        class MockVoxelData:
            def __init__(self):
                self.laser_power = 100.0
                self.density = 0.9

        fused_data = {(0, 0, 0): MockVoxelData()}
        features, indices, feature_names = extract_features_from_fused_data(fused_data)

        assert features.shape[0] == 1
        assert len(indices) == 1
        assert len(feature_names) == 2
