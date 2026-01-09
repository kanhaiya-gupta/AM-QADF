"""
STL Model Query Client

Query client for STL models stored in MongoDB data warehouse.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Handle both relative import (when used as package) and direct import (when loaded directly)
try:
    from .base_query_client import BaseQueryClient
    from .query_utils import (
        build_model_query,
        get_bounding_box_from_doc,
        extract_coordinate_system,
    )
except ImportError:
    # Fallback for direct module loading
    try:
        from base_query_client import BaseQueryClient
        from query_utils import (
            build_model_query,
            get_bounding_box_from_doc,
            extract_coordinate_system,
        )
    except ImportError:
        import sys
        from pathlib import Path

        current_file = Path(__file__).resolve()
        base_path = current_file.parent / "base_query_client.py"
        utils_path = current_file.parent / "query_utils.py"

        if base_path.exists():
            import importlib.util

            spec = importlib.util.spec_from_file_location("base_query_client", base_path)
            base_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(base_module)
            sys.modules["base_query_client"] = base_module
            BaseQueryClient = base_module.BaseQueryClient

        if utils_path.exists():
            import importlib.util

            spec = importlib.util.spec_from_file_location("query_utils", utils_path)
            utils_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils_module)
            sys.modules["query_utils"] = utils_module
            build_model_query = utils_module.build_model_query
            get_bounding_box_from_doc = utils_module.get_bounding_box_from_doc
            extract_coordinate_system = utils_module.extract_coordinate_system
        else:
            raise ImportError("Could not import required modules")


class STLModelClient(BaseQueryClient):
    """
    Query client for STL models from MongoDB data warehouse.

    Provides methods to query STL model metadata, retrieve file paths,
    and access coordinate system information.
    """

    def __init__(self, mongo_client=None, data_source: Optional[str] = None):
        """
        Initialize STL model query client.

        Args:
            mongo_client: MongoDBClient instance (if None, will create one)
            data_source: Optional identifier for the data source
        """
        super().__init__(data_source)
        self.mongo_client = mongo_client
        self.collection_name = "stl_models"
        self._available_signals = []  # STL models don't have signals

    def _get_collection(self):
        """Get MongoDB collection."""
        if self.mongo_client is None:
            raise RuntimeError("MongoDB client not initialized. Call set_mongo_client() first.")
        if not self.mongo_client.is_connected():
            raise ConnectionError("MongoDB client not connected")
        return self.mongo_client.get_collection(self.collection_name)

    def set_mongo_client(self, mongo_client):
        """Set MongoDB client instance."""
        self.mongo_client = mongo_client

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get STL model by model_id.

        Args:
            model_id: Model UUID

        Returns:
            Model document or None if not found
        """
        collection = self._get_collection()
        doc = collection.find_one({"model_id": model_id})
        return doc

    def get_model_by_name(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get STL model by model_name.

        Args:
            model_name: Model name

        Returns:
            Model document or None if not found
        """
        collection = self._get_collection()
        doc = collection.find_one({"model_name": model_name})
        return doc

    def list_models(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all STL models with optional filters.

        Args:
            filters: Optional MongoDB query filters
            limit: Maximum number of results
            skip: Number of results to skip

        Returns:
            List of model documents
        """
        collection = self._get_collection()
        query = filters or {}

        cursor = collection.find(query)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)

        return list(cursor)

    def get_model_bounding_box(self, model_id: str) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Get bounding box for a model.

        Args:
            model_id: Model UUID

        Returns:
            Tuple of (bbox_min, bbox_max) or None if not found
        """
        doc = self.get_model(model_id)
        if doc is None:
            return None

        return get_bounding_box_from_doc(doc)

    def get_coordinate_system(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get coordinate system information for a model.

        Args:
            model_id: Model UUID

        Returns:
            Coordinate system dictionary or None
        """
        doc = self.get_model(model_id)
        if doc is None:
            return None

        return extract_coordinate_system(doc)

    def load_stl_file(self, model_id: str) -> Optional[Path]:
        """
        Get STL file path for a model.

        Args:
            model_id: Model UUID

        Returns:
            Path to STL file or None if not found
        """
        doc = self.get_model(model_id)
        if doc is None:
            return None

        file_path = doc.get("file_path")
        if file_path:
            path = Path(file_path)
            if path.exists():
                return path

        return None

    def get_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a model.

        Args:
            model_id: Model UUID

        Returns:
            Metadata dictionary or None
        """
        doc = self.get_model(model_id)
        if doc is None:
            return None

        return doc.get("metadata", {})

    def query(
        self,
        spatial: Optional[Any] = None,  # SpatialQuery type
        temporal: Optional[Any] = None,  # TemporalQuery type
        signal_types: Optional[List[Any]] = None,  # SignalType enum
    ) -> Any:  # QueryResult type
        """
        Query STL models (implements BaseQueryClient interface).

        Note: STL models don't have signals, so this returns metadata only.

        Args:
            spatial: Spatial query (not used for STL models)
            temporal: Temporal query (not used for STL models)
            signal_types: Signal types (not used for STL models)

        Returns:
            QueryResult with empty points/signals but metadata
        """
        # STL models don't have point data or signals
        # This method exists for interface compatibility
        from .base_query_client import QueryResult

        models = self.list_models()
        return QueryResult(points=[], signals={}, metadata={"models": models, "count": len(models)})

    def get_available_signals(self) -> List[Any]:  # List[SignalType]
        """Get available signal types (STL models have no signals)."""
        return []

    def get_bounding_box(
        self, component_id: Optional[str] = None
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get bounding box (implements BaseQueryClient interface).

        Args:
            component_id: Model ID (for compatibility with BaseQueryClient)

        Returns:
            Tuple of (bbox_min, bbox_max)
        """
        if component_id:
            bbox = self.get_bounding_box(component_id)
            if bbox:
                return bbox

        # Return default bounding box if not found
        return ((0.0, 0.0, 0.0), (100.0, 100.0, 100.0))
