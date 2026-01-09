"""
Unit tests for VoxelGridStorage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from am_qadf.voxel_domain.voxel_storage import VoxelGridStorage

try:
    from am_qadf.voxelization.voxel_grid import VoxelGrid
    from am_qadf.core.entities import VoxelData

    VOXELIZATION_AVAILABLE = True
except ImportError:
    VOXELIZATION_AVAILABLE = False


class MockMongoClient:
    """Mock MongoDB client for testing."""

    def __init__(self):
        self.connected = True
        self.db = Mock()
        self._collections = {}
        self._files = {}
        self._file_counter = 0

    def is_connected(self):
        """Check if client is connected."""
        return self.connected

    def get_collection(self, name):
        """Get a collection."""
        if name not in self._collections:
            self._collections[name] = MockCollection()
        return self._collections[name]

    def store_file(self, data, filename, metadata=None):
        """Store a file in GridFS (mock)."""
        file_id = f"file_{self._file_counter}"
        self._file_counter += 1
        self._files[file_id] = {
            "data": data,
            "filename": filename,
            "metadata": metadata or {},
        }
        return file_id

    def get_file(self, file_id):
        """Get a file from GridFS (mock)."""
        return self._files.get(file_id, {}).get("data")

    def retrieve_file(self, file_id):
        """Retrieve a file from GridFS (mock)."""
        return self._files.get(file_id, {}).get("data")

    def delete_file(self, file_id):
        """Delete a file from GridFS (mock)."""
        if file_id in self._files:
            del self._files[file_id]


class MockCollection:
    """Mock MongoDB collection for testing."""

    def __init__(self):
        self._documents = []  # Store as list to allow searching by ObjectId
        self._counter = 0

    def find_one(self, query):
        """Find one document."""
        if "_id" in query:
            # Handle ObjectId conversion
            from bson import ObjectId

            doc_id = query["_id"]
            doc_id_str = str(doc_id) if isinstance(doc_id, ObjectId) else str(doc_id)
            # Search through all documents
            for doc in self._documents:
                if "_id" in doc:
                    if str(doc["_id"]) == doc_id_str:
                        return doc
        elif "model_id" in query and "grid_name" in query:
            for doc in self._documents:
                if doc.get("model_id") == query["model_id"] and doc.get("grid_name") == query["grid_name"]:
                    return doc
        return None

    def find(self, query, projection=None):
        """Find documents."""
        results = []
        for doc in self._documents:
            match = True
            if "model_id" in query and doc.get("model_id") != query["model_id"]:
                match = False
            if match:
                # Apply projection
                if projection:
                    filtered_doc = {k: v for k, v in doc.items() if k in projection or k == "_id"}
                    results.append(filtered_doc)
                else:
                    results.append(doc)
        return MockCursor(results)

    def insert_one(self, document):
        """Insert a document."""
        from bson import ObjectId
        import random

        # Generate a valid ObjectId string (24 hex characters)
        # Use counter + random to ensure uniqueness
        hex_part = f"{self._counter:012x}"  # 12 hex chars from counter
        random_part = "".join(random.choices("0123456789abcdef", k=12))  # 12 random hex chars
        doc_id_str = hex_part + random_part
        self._counter += 1
        # Create ObjectId from string
        doc_id = ObjectId(doc_id_str)
        document["_id"] = doc_id
        # Store document in list
        self._documents.append(document)
        return MockInsertResult(doc_id)

    def update_one(self, filter_query, update_query):
        """Update a document."""
        doc = self.find_one(filter_query)
        if doc:
            if "$set" in update_query:
                doc.update(update_query["$set"])
            return MockUpdateResult(1)
        return MockUpdateResult(0)

    def delete_one(self, filter_query):
        """Delete a document."""
        from bson import ObjectId

        doc_id_obj = filter_query.get("_id")
        if doc_id_obj:
            doc_id_str = str(doc_id_obj) if isinstance(doc_id_obj, ObjectId) else str(doc_id_obj)
            # Find and remove document
            for i, doc in enumerate(self._documents):
                if "_id" in doc and str(doc["_id"]) == doc_id_str:
                    del self._documents[i]
                    return MockDeleteResult(1)
        return MockDeleteResult(0)


class MockCursor:
    """Mock MongoDB cursor for testing."""

    def __init__(self, documents):
        self._documents = documents
        self._sorted = False

    def sort(self, field, direction):
        """Sort cursor."""
        self._sorted = True
        return self

    def limit(self, n):
        """Limit cursor."""
        self._documents = self._documents[:n]
        return self

    def __iter__(self):
        return iter(self._documents)


class MockInsertResult:
    """Mock insert result."""

    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class MockUpdateResult:
    """Mock update result."""

    def __init__(self, modified_count):
        self.modified_count = modified_count


class MockDeleteResult:
    """Mock delete result."""

    def __init__(self, deleted_count):
        self.deleted_count = deleted_count


class MockVoxelGrid:
    """Mock VoxelGrid for testing."""

    def __init__(self):
        self.bbox_min = np.array([0.0, 0.0, 0.0])
        self.bbox_max = np.array([10.0, 10.0, 10.0])
        self.resolution = 1.0
        self.dims = np.array([10, 10, 10])
        self.aggregation = "mean"
        self.available_signals = {"power", "velocity"}
        self.voxels = {}

        # Create some mock voxels
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    voxel = Mock()
                    voxel.signals = {
                        "power": float(i + j + k),
                        "velocity": float(i * j * k),
                    }
                    self.voxels[(i, j, k)] = voxel

    def get_signal_array(self, signal_name, default=0.0):
        """Return mock signal array."""
        return np.random.rand(*self.dims) * 100.0


@pytest.mark.skipif(not VOXELIZATION_AVAILABLE, reason="Voxelization module not available")
class TestVoxelGridStorage:
    """Test cases for VoxelGridStorage."""

    @pytest.fixture
    def mongo_client(self):
        """Create a mock MongoDB client."""
        client = MockMongoClient()
        client.db = Mock()
        client.db.__getitem__ = lambda self, name: MockCollection()
        return client

    @pytest.fixture
    def storage(self, mongo_client):
        """Create a VoxelGridStorage instance."""
        return VoxelGridStorage(mongo_client)

    @pytest.fixture
    def voxel_grid(self):
        """Create a mock voxel grid."""
        return MockVoxelGrid()

    @pytest.mark.unit
    def test_initialization(self, mongo_client):
        """Test storage initialization."""
        storage = VoxelGridStorage(mongo_client)
        assert storage.mongo_client == mongo_client
        assert storage.collection_name == "voxel_grids"
        assert storage.gridfs_bucket == "voxel_grid_data"

    @pytest.mark.unit
    def test_save_voxel_grid_new(self, storage, voxel_grid):
        """Test saving a new voxel grid."""
        grid_id = storage.save_voxel_grid(
            model_id="test_model",
            grid_name="test_grid",
            voxel_grid=voxel_grid,
            description="Test grid",
            tags=["test", "demo"],
        )
        assert grid_id is not None
        assert isinstance(grid_id, str)

    @pytest.mark.unit
    def test_save_voxel_grid_update(self, storage, voxel_grid):
        """Test updating an existing voxel grid."""
        # Save first time
        grid_id = storage.save_voxel_grid(model_id="test_model", grid_name="test_grid", voxel_grid=voxel_grid)

        # Update
        updated_id = storage.save_voxel_grid(
            model_id="test_model",
            grid_name="test_grid",
            voxel_grid=voxel_grid,
            description="Updated description",
        )
        assert updated_id == grid_id

    @pytest.mark.unit
    def test_save_voxel_grid_not_connected(self, mongo_client):
        """Test saving when MongoDB is not connected."""
        mongo_client.connected = False
        storage = VoxelGridStorage(mongo_client)

        with pytest.raises(ConnectionError, match="MongoDB client not connected"):
            storage.save_voxel_grid(model_id="test_model", grid_name="test_grid", voxel_grid=MockVoxelGrid())

    @pytest.mark.unit
    def test_save_voxel_grid_stores_signals(self, storage, voxel_grid):
        """Test that signals are stored."""
        with patch.object(storage, "_store_signal_sparse") as mock_store:
            mock_store.return_value = "file_id_123"
            storage.save_voxel_grid(model_id="test_model", grid_name="test_grid", voxel_grid=voxel_grid)
            # Should be called for each signal
            assert mock_store.call_count == len(voxel_grid.available_signals)

    @pytest.mark.unit
    def test_save_voxel_grid_signal_fallback(self, storage, voxel_grid):
        """Test signal storage fallback to dense format."""
        # Mock sparse storage to fail
        with patch.object(storage, "_store_signal_sparse", side_effect=Exception("Sparse failed")):
            with patch.object(storage, "_store_signal_array") as mock_dense:
                mock_dense.return_value = "file_id_123"
                storage.save_voxel_grid(model_id="test_model", grid_name="test_grid", voxel_grid=voxel_grid)
                # Should fall back to dense storage
                assert mock_dense.called

    @pytest.mark.unit
    def test_load_voxel_grid(self, storage, voxel_grid):
        """Test loading a voxel grid."""
        # First save
        grid_id = storage.save_voxel_grid(model_id="test_model", grid_name="test_grid", voxel_grid=voxel_grid)

        # Then load
        loaded = storage.load_voxel_grid(grid_id)
        assert loaded is not None
        assert loaded["grid_id"] == grid_id
        assert loaded["model_id"] == "test_model"
        assert loaded["grid_name"] == "test_grid"

    @pytest.mark.unit
    def test_load_voxel_grid_not_found(self, storage):
        """Test loading non-existent grid."""
        loaded = storage.load_voxel_grid("nonexistent_id")
        assert loaded is None

    @pytest.mark.unit
    def test_load_voxel_grid_not_connected(self, mongo_client):
        """Test loading when MongoDB is not connected."""
        mongo_client.connected = False
        storage = VoxelGridStorage(mongo_client)

        with pytest.raises(ConnectionError, match="MongoDB client not connected"):
            storage.load_voxel_grid("test_id")

    @pytest.mark.unit
    def test_load_voxel_grid_loads_signals(self, storage, voxel_grid):
        """Test that signals are loaded."""
        grid_id = storage.save_voxel_grid(model_id="test_model", grid_name="test_grid", voxel_grid=voxel_grid)

        with patch.object(storage, "_load_signal_array") as mock_load:
            mock_load.return_value = np.array([1, 2, 3])
            loaded = storage.load_voxel_grid(grid_id)
            # Should be called for each signal
            assert mock_load.call_count == len(voxel_grid.available_signals)
            assert "signal_arrays" in loaded

    @pytest.mark.unit
    def test_list_grids(self, storage, voxel_grid):
        """Test listing grids."""
        # Save a few grids
        storage.save_voxel_grid(model_id="model1", grid_name="grid1", voxel_grid=voxel_grid)
        storage.save_voxel_grid(model_id="model2", grid_name="grid2", voxel_grid=voxel_grid)

        grids = storage.list_grids()
        assert len(grids) >= 2

    @pytest.mark.unit
    def test_list_grids_by_model(self, storage, voxel_grid):
        """Test listing grids filtered by model."""
        storage.save_voxel_grid(model_id="model1", grid_name="grid1", voxel_grid=voxel_grid)
        storage.save_voxel_grid(model_id="model2", grid_name="grid2", voxel_grid=voxel_grid)

        grids = storage.list_grids(model_id="model1")
        assert all(g["model_id"] == "model1" for g in grids)

    @pytest.mark.unit
    def test_list_grids_limit(self, storage, voxel_grid):
        """Test listing grids with limit."""
        # Save multiple grids
        for i in range(10):
            storage.save_voxel_grid(model_id="test_model", grid_name=f"grid_{i}", voxel_grid=voxel_grid)

        grids = storage.list_grids(limit=5)
        assert len(grids) <= 5

    @pytest.mark.unit
    def test_list_grids_not_connected(self, mongo_client):
        """Test listing when MongoDB is not connected."""
        mongo_client.connected = False
        storage = VoxelGridStorage(mongo_client)

        with pytest.raises(ConnectionError, match="MongoDB client not connected"):
            storage.list_grids()

    @pytest.mark.unit
    def test_delete_grid(self, storage, voxel_grid):
        """Test deleting a grid."""
        grid_id = storage.save_voxel_grid(model_id="test_model", grid_name="test_grid", voxel_grid=voxel_grid)

        result = storage.delete_grid(grid_id)
        assert result == True

        # Verify it's deleted
        loaded = storage.load_voxel_grid(grid_id)
        assert loaded is None

    @pytest.mark.unit
    def test_delete_grid_not_found(self, storage):
        """Test deleting non-existent grid."""
        result = storage.delete_grid("nonexistent_id")
        assert result == False

    @pytest.mark.unit
    def test_delete_grid_not_connected(self, mongo_client):
        """Test deleting when MongoDB is not connected."""
        mongo_client.connected = False
        storage = VoxelGridStorage(mongo_client)

        with pytest.raises(ConnectionError, match="MongoDB client not connected"):
            storage.delete_grid("test_id")

    @pytest.mark.unit
    def test_delete_grid_deletes_files(self, storage, voxel_grid):
        """Test that GridFS files are deleted when grid is deleted."""
        grid_id = storage.save_voxel_grid(model_id="test_model", grid_name="test_grid", voxel_grid=voxel_grid)

        with patch.object(storage, "_delete_gridfs_file") as mock_delete:
            storage.delete_grid(grid_id)
            # Should delete signal files
            assert mock_delete.called

    @pytest.mark.unit
    def test_extract_metadata(self, storage, voxel_grid):
        """Test metadata extraction."""
        metadata = storage._extract_metadata(voxel_grid)
        assert "bbox_min" in metadata
        assert "bbox_max" in metadata
        assert "resolution" in metadata
        assert "dims" in metadata
        assert "aggregation" in metadata
        assert "grid_type" in metadata

    @pytest.mark.unit
    def test_store_signal_sparse(self, storage, voxel_grid):
        """Test sparse signal storage."""
        file_id = storage._store_signal_sparse(grid_id="test_grid", signal_name="power", voxel_grid=voxel_grid)
        assert file_id is not None
        assert isinstance(file_id, str)

    @pytest.mark.unit
    def test_store_signal_sparse_no_data(self, storage, voxel_grid):
        """Test sparse storage when no data exists."""
        # Create grid with no voxels
        empty_grid = MockVoxelGrid()
        empty_grid.voxels = {}

        with pytest.raises(ValueError, match="No data found for signal"):
            storage._store_signal_sparse(grid_id="test_grid", signal_name="power", voxel_grid=empty_grid)

    @pytest.mark.unit
    def test_store_signal_array(self, storage):
        """Test dense signal array storage."""
        signal_array = np.random.rand(10, 10, 10) * 100.0
        file_id = storage._store_signal_array(grid_id="test_grid", signal_name="power", signal_array=signal_array)
        assert file_id is not None

    @pytest.mark.unit
    def test_load_signal_array_sparse(self, storage, voxel_grid):
        """Test loading sparse signal array."""
        # Store sparse
        file_id = storage._store_signal_sparse(grid_id="test_grid", signal_name="power", voxel_grid=voxel_grid)

        # Load
        signal_array = storage._load_signal_array(file_id)
        assert signal_array is not None
        assert isinstance(signal_array, np.ndarray)
        assert signal_array.shape == tuple(voxel_grid.dims)

    @pytest.mark.unit
    def test_load_signal_array_dense(self, storage):
        """Test loading dense signal array."""
        signal_array = np.random.rand(10, 10, 10) * 100.0
        file_id = storage._store_signal_array(grid_id="test_grid", signal_name="power", signal_array=signal_array)

        loaded = storage._load_signal_array(file_id)
        assert loaded is not None
        assert np.array_equal(loaded, signal_array)

    @pytest.mark.unit
    def test_load_signal_array_not_found(self, storage):
        """Test loading non-existent signal array."""
        with pytest.raises(ValueError, match="File not found"):
            storage._load_signal_array("nonexistent_file_id")

    @pytest.mark.unit
    def test_store_voxel_data(self, storage, voxel_grid):
        """Test voxel data storage."""
        voxel_data = {
            "0_0_0": {"power": 10.0, "velocity": 5.0},
            "1_1_1": {"power": 20.0, "velocity": 10.0},
        }
        file_id = storage._store_voxel_data("test_grid", voxel_data)
        assert file_id is not None

    @pytest.mark.unit
    def test_load_voxel_data(self, storage, voxel_grid):
        """Test voxel data loading."""
        voxel_data = {
            "0_0_0": {"power": 10.0, "velocity": 5.0},
            "1_1_1": {"power": 20.0, "velocity": 10.0},
        }
        file_id = storage._store_voxel_data("test_grid", voxel_data)

        loaded = storage._load_voxel_data(file_id)
        assert loaded == voxel_data

    @pytest.mark.unit
    def test_load_voxel_data_not_found(self, storage):
        """Test loading non-existent voxel data."""
        with pytest.raises(ValueError, match="File not found"):
            storage._load_voxel_data("nonexistent_file_id")

    @pytest.mark.unit
    def test_delete_gridfs_file(self, storage, voxel_grid):
        """Test deleting GridFS file."""
        file_id = storage._store_signal_sparse(grid_id="test_grid", signal_name="power", voxel_grid=voxel_grid)

        # Should not raise error
        storage._delete_gridfs_file(file_id)

    @pytest.mark.unit
    def test_save_voxel_grid_metadata(self, storage, voxel_grid):
        """Test that metadata is saved correctly."""
        grid_id = storage.save_voxel_grid(
            model_id="test_model",
            grid_name="test_grid",
            voxel_grid=voxel_grid,
            description="Test description",
            tags=["tag1", "tag2"],
        )

        loaded = storage.load_voxel_grid(grid_id)
        assert loaded["description"] == "Test description"
        assert loaded["tags"] == ["tag1", "tag2"]
        assert "created_at" in loaded
        assert "updated_at" in loaded
