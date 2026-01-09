"""
Unit tests for BuildMetadataClient.

Tests for build metadata query functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock
from am_qadf.query.build_metadata_client import (
    BuildMetadataClient,
    ComponentInfo,
    BuildStyleInfo,
)
from am_qadf.query.base_query_client import QueryResult


class TestComponentInfo:
    """Test suite for ComponentInfo dataclass."""

    @pytest.mark.unit
    def test_component_info_creation(self):
        """Test creating ComponentInfo."""
        component = ComponentInfo(
            component_id="comp_123",
            name="Test Component",
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            layer_count=100,
            metadata={"material": "Ti6Al4V"},
        )

        assert component.component_id == "comp_123"
        assert component.name == "Test Component"
        assert component.bbox_min == (0.0, 0.0, 0.0)
        assert component.bbox_max == (10.0, 10.0, 10.0)
        assert component.layer_count == 100
        assert component.metadata["material"] == "Ti6Al4V"


class TestBuildStyleInfo:
    """Test suite for BuildStyleInfo dataclass."""

    @pytest.mark.unit
    def test_build_style_info_creation(self):
        """Test creating BuildStyleInfo."""
        build_style = BuildStyleInfo(
            build_style_id=1,
            laser_power=200.0,
            laser_speed=100.0,
            energy_density=2.0,
            layer_indices=[0, 1, 2, 3],
        )

        assert build_style.build_style_id == 1
        assert build_style.laser_power == 200.0
        assert build_style.laser_speed == 100.0
        assert build_style.energy_density == 2.0
        assert build_style.layer_indices == [0, 1, 2, 3]


class TestBuildMetadataClient:
    """Test suite for BuildMetadataClient."""

    @pytest.mark.unit
    def test_build_metadata_client_creation(self):
        """Test creating BuildMetadataClient."""
        client = BuildMetadataClient()

        assert client.data_source == "build_metadata"
        assert client.generated_layers == []
        assert client.generated_models == []
        assert client.generated_build_styles == {}
        assert client.component_id is None

    @pytest.mark.unit
    def test_build_metadata_client_creation_with_component_id(self):
        """Test creating BuildMetadataClient with component ID."""
        client = BuildMetadataClient(component_id="comp_123")

        assert client.component_id == "comp_123"

    @pytest.mark.unit
    def test_build_metadata_client_creation_with_layers(self):
        """Test creating BuildMetadataClient with generated layers."""
        mock_layers = [Mock(), Mock()]
        client = BuildMetadataClient(generated_layers=mock_layers)

        assert len(client.generated_layers) == 2

    @pytest.mark.unit
    def test_build_metadata_client_query_empty(self):
        """Test querying with empty data."""
        client = BuildMetadataClient()

        result = client.query()

        assert isinstance(result, QueryResult)
        assert len(result.points) == 0

    @pytest.mark.unit
    def test_build_metadata_client_get_components(self):
        """Test getting component information."""
        client = BuildMetadataClient()

        # Mock component data
        client._components = {
            "comp_1": ComponentInfo(
                component_id="comp_1",
                name="Component 1",
                bbox_min=(0.0, 0.0, 0.0),
                bbox_max=(10.0, 10.0, 10.0),
                layer_count=50,
                metadata={},
            )
        }

        components = client.get_components()

        assert len(components) == 1
        assert "comp_1" in components
        assert isinstance(components["comp_1"], ComponentInfo)

    @pytest.mark.unit
    def test_build_metadata_client_get_build_styles(self):
        """Test getting build style information."""
        client = BuildMetadataClient()

        # Mock build style data
        client._build_styles = {
            1: BuildStyleInfo(
                build_style_id=1,
                laser_power=200.0,
                laser_speed=100.0,
                energy_density=2.0,
                layer_indices=[0, 1, 2],
            )
        }

        build_styles = client.get_build_styles()

        assert len(build_styles) == 1
        assert 1 in build_styles
        assert isinstance(build_styles[1], BuildStyleInfo)
