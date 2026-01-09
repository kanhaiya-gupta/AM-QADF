"""
Unit tests for sensor analysis.

Tests for SensorAnalysisConfig, SensorAnalysisResult, SensorAnalyzer, ISPMAnalyzer, and CTSensorAnalyzer.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.analytics.process_analysis.sensor_analysis import (
    SensorAnalysisConfig,
    SensorAnalysisResult,
    SensorAnalyzer,
    ISPMAnalyzer,
    CTSensorAnalyzer,
)


class TestSensorAnalysisConfig:
    """Test suite for SensorAnalysisConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating SensorAnalysisConfig with default values."""
        config = SensorAnalysisConfig()

        assert config.sampling_rate == 1000.0
        assert config.filter_type == "butterworth"
        assert config.filter_order == 4
        assert config.cutoff_frequency == 100.0
        assert config.anomaly_threshold == 3.0
        assert config.window_size == 100
        assert config.confidence_level == 0.95

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating SensorAnalysisConfig with custom values."""
        config = SensorAnalysisConfig(
            sampling_rate=2000.0,
            filter_type="chebyshev",
            filter_order=6,
            cutoff_frequency=200.0,
            anomaly_threshold=2.5,
            window_size=200,
        )

        assert config.sampling_rate == 2000.0
        assert config.filter_type == "chebyshev"
        assert config.filter_order == 6
        assert config.cutoff_frequency == 200.0
        assert config.anomaly_threshold == 2.5
        assert config.window_size == 200


class TestSensorAnalysisResult:
    """Test suite for SensorAnalysisResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating SensorAnalysisResult."""
        sensor_data = pd.DataFrame({"sensor1": [1.0, 2.0, 3.0]})
        processed_data = pd.DataFrame({"sensor1": [1.1, 2.1, 3.1]})

        result = SensorAnalysisResult(
            success=True,
            method="test_method",
            sensor_data=sensor_data,
            processed_data=processed_data,
            anomaly_detection={"anomalies": [0]},
            signal_statistics={"mean": 2.0, "std": 1.0},
            analysis_time=1.5,
        )

        assert result.success is True
        assert result.method == "test_method"
        assert len(result.sensor_data) == 3
        assert len(result.processed_data) == 3
        assert len(result.anomaly_detection) > 0
        assert result.analysis_time == 1.5


class TestSensorAnalyzer:
    """Test suite for SensorAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a SensorAnalyzer instance."""
        return SensorAnalyzer()

    @pytest.fixture
    def sample_sensor_data(self):
        """Create sample sensor data for testing."""
        np.random.seed(42)
        n_samples = 1000
        data = pd.DataFrame(
            {
                "sensor1": np.random.randn(n_samples) + 10.0,
                "sensor2": np.random.randn(n_samples) + 20.0,
                "sensor3": np.random.randn(n_samples) + 30.0,
                "timestamp": np.arange(n_samples) / 1000.0,
            }
        )
        return data

    @pytest.mark.unit
    def test_analyzer_creation_default(self):
        """Test creating SensorAnalyzer with default config."""
        analyzer = SensorAnalyzer()

        assert analyzer.config is not None
        assert analyzer.analysis_cache == {}

    @pytest.mark.unit
    def test_analyzer_creation_custom(self):
        """Test creating SensorAnalyzer with custom config."""
        config = SensorAnalysisConfig(sampling_rate=2000.0, filter_order=6)
        analyzer = SensorAnalyzer(config)

        assert analyzer.config.sampling_rate == 2000.0
        assert analyzer.config.filter_order == 6

    @pytest.mark.unit
    def test_analyze_sensor_data(self, analyzer, sample_sensor_data):
        """Test analyzing sensor data."""
        result = analyzer.analyze_sensor_data(sample_sensor_data, sensor_columns=["sensor1", "sensor2", "sensor3"])

        assert isinstance(result, SensorAnalysisResult)
        assert result.success is True
        assert result.method == "SensorAnalysis"
        assert len(result.processed_data) > 0
        assert len(result.signal_statistics) > 0

    @pytest.mark.unit
    def test_analyze_sensor_data_all_columns(self, analyzer, sample_sensor_data):
        """Test analyzing sensor data with all columns."""
        result = analyzer.analyze_sensor_data(sample_sensor_data)

        assert isinstance(result, SensorAnalysisResult)
        assert result.success is True


class TestISPMAnalyzer:
    """Test suite for ISPMAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create an ISPMAnalyzer instance."""
        return ISPMAnalyzer()

    @pytest.fixture
    def sample_ispm_data(self):
        """Create sample ISPM sensor data for testing."""
        np.random.seed(42)
        n_samples = 1000
        data = pd.DataFrame(
            {
                "temperature": np.random.randn(n_samples) + 1000.0,
                "pressure": np.random.randn(n_samples) + 1.0,
                "timestamp": np.arange(n_samples) / 1000.0,
            }
        )
        return data

    @pytest.mark.unit
    def test_analyzer_creation(self, analyzer):
        """Test creating ISPMAnalyzer."""
        assert analyzer is not None
        assert analyzer.config is not None

    @pytest.mark.unit
    def test_analyze_ispm_data(self, analyzer, sample_ispm_data):
        """Test analyzing ISPM sensor data."""
        result = analyzer.analyze_ispm_data(sample_ispm_data)

        assert isinstance(result, SensorAnalysisResult)
        assert result.success is True


class TestCTSensorAnalyzer:
    """Test suite for CTSensorAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a CTSensorAnalyzer instance."""
        return CTSensorAnalyzer()

    @pytest.fixture
    def sample_ct_data(self):
        """Create sample CT sensor data for testing."""
        np.random.seed(42)
        n_samples = 1000
        data = pd.DataFrame(
            {
                "density": np.random.rand(n_samples),
                "porosity": np.random.rand(n_samples),
                "timestamp": np.arange(n_samples) / 1000.0,
            }
        )
        return data

    @pytest.mark.unit
    def test_analyzer_creation(self, analyzer):
        """Test creating CTSensorAnalyzer."""
        assert analyzer is not None
        assert analyzer.config is not None

    @pytest.mark.unit
    def test_analyze_ct_data(self, analyzer, sample_ct_data):
        """Test analyzing CT sensor data."""
        result = analyzer.analyze_ct_data(sample_ct_data)

        assert isinstance(result, SensorAnalysisResult)
        assert result.success is True
