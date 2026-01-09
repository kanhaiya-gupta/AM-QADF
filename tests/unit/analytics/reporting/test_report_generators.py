"""
Unit tests for report generators.

Tests for ReportConfig, ReportResult, AnalysisReportGenerator, and SensitivityReportGenerator.
"""

import pytest
import os
import tempfile
import shutil
from am_qadf.analytics.reporting.report_generators import (
    ReportConfig,
    ReportResult,
    AnalysisReportGenerator,
    SensitivityReportGenerator,
)


class TestReportConfig:
    """Test suite for ReportConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating ReportConfig with default values."""
        config = ReportConfig()

        assert config.report_format == "html"
        assert config.include_plots is True
        assert config.include_tables is True
        assert config.include_statistics is True
        assert config.output_directory == "reports"
        assert config.filename_prefix == "pbf_analytics_report"
        assert config.include_summary is True
        assert config.include_details is True
        assert config.include_recommendations is True

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating ReportConfig with custom values."""
        config = ReportConfig(
            report_format="markdown",
            include_plots=False,
            include_tables=True,
            include_statistics=False,
            output_directory="custom_reports",
            filename_prefix="custom_report",
            include_summary=False,
            include_details=True,
            include_recommendations=False,
        )

        assert config.report_format == "markdown"
        assert config.include_plots is False
        assert config.include_tables is True
        assert config.include_statistics is False
        assert config.output_directory == "custom_reports"
        assert config.filename_prefix == "custom_report"
        assert config.include_summary is False
        assert config.include_details is True
        assert config.include_recommendations is False


class TestReportResult:
    """Test suite for ReportResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating ReportResult."""
        result = ReportResult(
            success=True,
            report_type="Comprehensive",
            report_path="/path/to/report.html",
            report_size=1024,
            generation_time=1.5,
        )

        assert result.success is True
        assert result.report_type == "Comprehensive"
        assert result.report_path == "/path/to/report.html"
        assert result.report_size == 1024
        assert result.generation_time == 1.5
        assert result.error_message is None

    @pytest.mark.unit
    def test_result_creation_with_error(self):
        """Test creating ReportResult with error."""
        result = ReportResult(
            success=False,
            report_type="Comprehensive",
            report_path="",
            report_size=0,
            generation_time=0.0,
            error_message="Test error",
        )

        assert result.success is False
        assert result.error_message == "Test error"


class TestAnalysisReportGenerator:
    """Test suite for AnalysisReportGenerator class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def generator(self, temp_dir):
        """Create an AnalysisReportGenerator instance."""
        config = ReportConfig(output_directory=temp_dir)
        return AnalysisReportGenerator(config)

    @pytest.fixture
    def sample_analytics_results(self):
        """Create sample analytics results for testing."""
        return {
            "sensitivity_analysis": {"sobol_indices": {"power": 0.6, "velocity": 0.4}},
            "statistical_analysis": {"mean": 0.8, "std": 0.1},
        }

    @pytest.mark.unit
    def test_generator_creation_default(self):
        """Test creating AnalysisReportGenerator with default config."""
        generator = AnalysisReportGenerator()

        assert generator.config is not None
        assert generator.config.report_format == "html"

    @pytest.mark.unit
    def test_generator_creation_custom(self, temp_dir):
        """Test creating AnalysisReportGenerator with custom config."""
        config = ReportConfig(report_format="markdown", output_directory=temp_dir)
        generator = AnalysisReportGenerator(config)

        assert generator.config.report_format == "markdown"
        assert generator.config.output_directory == temp_dir

    @pytest.mark.unit
    def test_generate_comprehensive_report_html(self, generator, sample_analytics_results):
        """Test generating comprehensive report in HTML format."""
        result = generator.generate_comprehensive_report(sample_analytics_results, report_title="Test Report")

        assert isinstance(result, ReportResult)
        assert result.success is True
        assert result.report_type == "Comprehensive"
        assert os.path.exists(result.report_path)
        assert result.report_size > 0

    @pytest.mark.unit
    def test_generate_comprehensive_report_markdown(self, temp_dir, sample_analytics_results):
        """Test generating comprehensive report in Markdown format."""
        config = ReportConfig(report_format="markdown", output_directory=temp_dir)
        generator = AnalysisReportGenerator(config)

        result = generator.generate_comprehensive_report(sample_analytics_results, report_title="Test Report")

        assert isinstance(result, ReportResult)
        assert result.success is True
        assert os.path.exists(result.report_path)
        assert result.report_path.endswith(".markdown") or result.report_path.endswith(".md")

    @pytest.mark.unit
    def test_generate_comprehensive_report_json(self, temp_dir, sample_analytics_results):
        """Test generating comprehensive report in JSON format."""
        config = ReportConfig(report_format="json", output_directory=temp_dir)
        generator = AnalysisReportGenerator(config)

        result = generator.generate_comprehensive_report(sample_analytics_results, report_title="Test Report")

        assert isinstance(result, ReportResult)
        assert result.success is True
        assert os.path.exists(result.report_path)
        assert result.report_path.endswith(".json")

    @pytest.mark.unit
    def test_generate_sensitivity_report(self, generator):
        """Test generating sensitivity report."""
        sensitivity_results = {
            "sobol_analysis": {"power": 0.6, "velocity": 0.4},
            "morris_analysis": {"power": 0.5, "velocity": 0.3},
        }

        result = generator.generate_sensitivity_report(sensitivity_results, report_title="Sensitivity Report")

        assert isinstance(result, ReportResult)
        assert result.success is True
        assert result.report_type == "Sensitivity"
        assert os.path.exists(result.report_path)

    @pytest.mark.unit
    def test_generate_sensitivity_report_markdown(self, temp_dir):
        """Test generating sensitivity report in Markdown format."""
        config = ReportConfig(report_format="markdown", output_directory=temp_dir)
        generator = AnalysisReportGenerator(config)

        sensitivity_results = {"sobol_analysis": {"power": 0.6}}

        result = generator.generate_sensitivity_report(sensitivity_results)

        assert isinstance(result, ReportResult)
        assert result.success is True


class TestSensitivityReportGenerator:
    """Test suite for SensitivityReportGenerator class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def generator(self, temp_dir):
        """Create a SensitivityReportGenerator instance."""
        config = ReportConfig(output_directory=temp_dir)
        return SensitivityReportGenerator(config)

    @pytest.mark.unit
    def test_generator_creation(self, generator):
        """Test creating SensitivityReportGenerator."""
        assert generator is not None
        assert generator.config is not None

    @pytest.mark.unit
    def test_generate_sensitivity_report(self, generator):
        """Test generating sensitivity report."""
        sensitivity_results = {"sobol_analysis": {"power": 0.6, "velocity": 0.4}}

        result = generator.generate_sensitivity_report(sensitivity_results)

        assert isinstance(result, ReportResult)
        assert result.success is True
        assert result.report_type == "Sensitivity"
