"""
Unit tests for documentation.

Tests for DocumentationConfig, DocumentationResult, AnalysisDocumentation, and APIDocumentation.
"""

import pytest
import os
import tempfile
import shutil
from am_qadf.analytics.reporting.documentation import (
    DocumentationConfig,
    DocumentationResult,
    AnalysisDocumentation,
    APIDocumentation,
)


class TestDocumentationConfig:
    """Test suite for DocumentationConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating DocumentationConfig with default values."""
        config = DocumentationConfig()

        assert config.doc_format == "markdown"
        assert config.include_examples is True
        assert config.include_api_reference is True
        assert config.output_directory == "docs"
        assert config.filename_prefix == "pbf_analytics_docs"
        assert config.include_installation is True
        assert config.include_quick_start is True
        assert config.include_tutorials is True

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating DocumentationConfig with custom values."""
        config = DocumentationConfig(
            doc_format="html",
            include_examples=False,
            include_api_reference=True,
            output_directory="custom_docs",
            filename_prefix="custom_docs",
            include_installation=False,
            include_quick_start=True,
            include_tutorials=False,
        )

        assert config.doc_format == "html"
        assert config.include_examples is False
        assert config.include_api_reference is True
        assert config.output_directory == "custom_docs"
        assert config.filename_prefix == "custom_docs"
        assert config.include_installation is False
        assert config.include_quick_start is True
        assert config.include_tutorials is False


class TestDocumentationResult:
    """Test suite for DocumentationResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating DocumentationResult."""
        result = DocumentationResult(
            success=True,
            doc_type="UserGuide",
            doc_path="/path/to/docs.md",
            doc_size=2048,
            generation_time=3.0,
        )

        assert result.success is True
        assert result.doc_type == "UserGuide"
        assert result.doc_path == "/path/to/docs.md"
        assert result.doc_size == 2048
        assert result.generation_time == 3.0
        assert result.error_message is None

    @pytest.mark.unit
    def test_result_creation_with_error(self):
        """Test creating DocumentationResult with error."""
        result = DocumentationResult(
            success=False,
            doc_type="UserGuide",
            doc_path="",
            doc_size=0,
            generation_time=0.0,
            error_message="Test error",
        )

        assert result.success is False
        assert result.error_message == "Test error"


class TestAnalysisDocumentation:
    """Test suite for AnalysisDocumentation class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def documentation(self, temp_dir):
        """Create an AnalysisDocumentation instance."""
        config = DocumentationConfig(output_directory=temp_dir)
        return AnalysisDocumentation(config)

    @pytest.mark.unit
    def test_documentation_creation_default(self):
        """Test creating AnalysisDocumentation with default config."""
        documentation = AnalysisDocumentation()

        assert documentation.config is not None
        assert documentation.config.doc_format == "markdown"

    @pytest.mark.unit
    def test_documentation_creation_custom(self, temp_dir):
        """Test creating AnalysisDocumentation with custom config."""
        config = DocumentationConfig(doc_format="html", output_directory=temp_dir)
        documentation = AnalysisDocumentation(config)

        assert documentation.config.doc_format == "html"
        assert documentation.config.output_directory == temp_dir

    @pytest.mark.unit
    def test_generate_user_guide_markdown(self, documentation):
        """Test generating user guide in Markdown format."""
        result = documentation.generate_user_guide(doc_title="Test User Guide")

        assert isinstance(result, DocumentationResult)
        assert result.success is True
        assert result.doc_type == "UserGuide"
        assert os.path.exists(result.doc_path)
        assert result.doc_size > 0

    @pytest.mark.unit
    def test_generate_user_guide_html(self, temp_dir):
        """Test generating user guide in HTML format."""
        config = DocumentationConfig(doc_format="html", output_directory=temp_dir)
        documentation = AnalysisDocumentation(config)

        result = documentation.generate_user_guide(doc_title="Test User Guide")

        assert isinstance(result, DocumentationResult)
        assert result.success is True
        assert os.path.exists(result.doc_path)
        assert result.doc_path.endswith(".html")

    @pytest.mark.unit
    def test_generate_api_documentation_markdown(self, documentation):
        """Test generating API documentation in Markdown format."""
        result = documentation.generate_api_documentation(doc_title="Test API Documentation")

        assert isinstance(result, DocumentationResult)
        assert result.success is True
        assert result.doc_type == "APIDocumentation"
        assert os.path.exists(result.doc_path)
        assert result.doc_size > 0

    @pytest.mark.unit
    def test_generate_api_documentation_html(self, temp_dir):
        """Test generating API documentation in HTML format."""
        config = DocumentationConfig(doc_format="html", output_directory=temp_dir)
        documentation = AnalysisDocumentation(config)

        result = documentation.generate_api_documentation(doc_title="Test API Documentation")

        assert isinstance(result, DocumentationResult)
        assert result.success is True
        assert os.path.exists(result.doc_path)
        assert result.doc_path.endswith(".html")


class TestAPIDocumentation:
    """Test suite for APIDocumentation class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def documentation(self, temp_dir):
        """Create an APIDocumentation instance."""
        config = DocumentationConfig(output_directory=temp_dir)
        return APIDocumentation(config)

    @pytest.mark.unit
    def test_documentation_creation(self, documentation):
        """Test creating APIDocumentation."""
        assert documentation is not None
        assert documentation.config is not None

    @pytest.mark.unit
    def test_generate_api_documentation(self, documentation):
        """Test generating API documentation."""
        result = documentation.generate_api_documentation(doc_title="Test API Documentation")

        assert isinstance(result, DocumentationResult)
        assert result.success is True
        assert result.doc_type == "APIDocumentation"
