"""
Bridge tests: I/O (OpenVDBReader, VDBWriter, ParaViewExporter) via am_qadf_native.

Aligned with docs/Tests/Test_New_Plans.md (tests/integration/bridge/).
"""

import os
import pytest


@pytest.mark.integration
@pytest.mark.bridge
class TestIoBridge:
    """Python → C++ I/O API."""

    def test_vdb_writer_write_and_openvdb_reader_load_all(self, native_module):
        """VDBWriter.write then OpenVDBReader.load_all_grids round-trip."""
        UniformVoxelGrid = native_module.UniformVoxelGrid
        VDBWriter = native_module.io.VDBWriter
        OpenVDBReader = native_module.io.OpenVDBReader
        grid = UniformVoxelGrid(1.0, 0.0, 0.0, 0.0)
        grid.set_signal_name("signal")
        grid.add_point_at_voxel(0, 0, 0, 5.0)
        g = grid.get_grid()
        path = "_bridge_io_temp.vdb"
        try:
            writer = VDBWriter()
            writer.write(g, path)
            assert os.path.isfile(path)
            reader = OpenVDBReader()
            grids = reader.load_all_grids(path)
            assert grids is not None
            assert len(grids) >= 1
        finally:
            if os.path.isfile(path):
                os.remove(path)

    def test_openvdb_reader_load_grid_by_name(self, native_module):
        """OpenVDBReader.load_grid_by_name after write with name."""
        UniformVoxelGrid = native_module.UniformVoxelGrid
        VDBWriter = native_module.io.VDBWriter
        OpenVDBReader = native_module.io.OpenVDBReader
        grid = UniformVoxelGrid(1.0)
        grid.set_signal_name("named_grid")
        grid.add_point_at_voxel(0, 0, 0, 42.0)
        g = grid.get_grid()
        path = "_bridge_io_named.vdb"
        try:
            writer = VDBWriter()
            writer.write(g, path)
            reader = OpenVDBReader()
            loaded = reader.load_grid_by_name(path, "named_grid")
            assert loaded is not None
        finally:
            if os.path.isfile(path):
                os.remove(path)

    def test_paraview_exporter_export(self, native_module):
        """ParaViewExporter.export_to_paraview writes file."""
        if not hasattr(native_module.io, "ParaViewExporter"):
            pytest.skip("ParaViewExporter not exposed")
        UniformVoxelGrid = native_module.UniformVoxelGrid
        ParaViewExporter = native_module.io.ParaViewExporter
        grid = UniformVoxelGrid(1.0)
        grid.add_point_at_voxel(0, 0, 0, 1.0)
        path = "_bridge_io_temp.vti"
        try:
            exporter = ParaViewExporter()
            exporter.export_to_paraview(grid.get_grid(), path)
            assert os.path.isfile(path)
        finally:
            if os.path.isfile(path):
                os.remove(path)
