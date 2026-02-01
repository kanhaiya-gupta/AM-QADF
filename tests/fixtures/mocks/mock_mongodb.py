"""
Mock MongoDB client and utilities for testing.
"""

from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any, Optional
import numpy as np


class MockInsertResult:
    """Mock MongoDB insert result."""

    def __init__(self, inserted_id: str):
        self.inserted_id = inserted_id


class MockUpdateResult:
    """Mock MongoDB update result."""

    def __init__(self, matched_count: int = 1, modified_count: int = 1):
        self.matched_count = matched_count
        self.modified_count = modified_count


class MockDeleteResult:
    """Mock MongoDB delete result."""

    def __init__(self, deleted_count: int = 1):
        self.deleted_count = deleted_count


class MockCursor:
    """Mock MongoDB cursor."""

    def __init__(self, documents: List[Dict[str, Any]]):
        self._documents = documents
        self._index = 0

    def __iter__(self):
        return iter(self._documents)

    def __next__(self):
        if self._index >= len(self._documents):
            raise StopIteration
        doc = self._documents[self._index]
        self._index += 1
        return doc

    def __len__(self):
        return len(self._documents)

    def to_list(self, length: Optional[int] = None):
        """Convert cursor to list."""
        if length is None:
            return self._documents
        return self._documents[:length]

    def limit(self, limit: int):
        """Limit cursor results."""
        return MockCursor(self._documents[:limit])

    def skip(self, skip: int):
        """Skip cursor results."""
        return MockCursor(self._documents[skip:])

    def sort(self, field: str, direction: int = 1):
        """Sort cursor results."""
        # Simple sort implementation
        reverse = direction == -1
        sorted_docs = sorted(self._documents, key=lambda x: x.get(field, 0), reverse=reverse)
        return MockCursor(sorted_docs)


class MockCollection:
    """Mock MongoDB collection."""

    def __init__(self, name: str = "test_collection"):
        self.name = name
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._counter = 0
        self._files: Dict[str, bytes] = {}  # For GridFS simulation

    def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find one document."""
        if "_id" in query:
            doc_id = str(query["_id"])
            return self._documents.get(doc_id)
        elif "model_id" in query and "grid_name" in query:
            for doc in self._documents.values():
                if doc.get("model_id") == query["model_id"] and doc.get("grid_name") == query["grid_name"]:
                    return doc
        elif "model_id" in query:
            for doc in self._documents.values():
                if doc.get("model_id") == query["model_id"]:
                    return doc
        return None

    def find(self, query: Dict[str, Any] = None, projection: Optional[Dict[str, int]] = None) -> MockCursor:
        """Find documents."""
        if query is None:
            query = {}

        results = []
        for doc in self._documents.values():
            match = True

            # Simple query matching
            for key, value in query.items():
                if key == "_id":
                    if str(doc.get("_id")) != str(value):
                        match = False
                        break
                elif doc.get(key) != value:
                    match = False
                    break

            if match:
                # Apply projection
                if projection:
                    filtered_doc = {k: v for k, v in doc.items() if k in projection or k == "_id"}
                    results.append(filtered_doc)
                else:
                    results.append(doc)

        return MockCursor(results)

    def insert_one(self, document: Dict[str, Any]) -> MockInsertResult:
        """Insert a document."""
        doc_id = f"doc_{self._counter}"
        self._counter += 1
        document["_id"] = doc_id
        self._documents[doc_id] = document
        return MockInsertResult(doc_id)

    def insert_many(self, documents: List[Dict[str, Any]]) -> Mock:
        """Insert many documents."""
        inserted_ids = []
        for doc in documents:
            result = self.insert_one(doc)
            inserted_ids.append(result.inserted_id)

        mock_result = Mock()
        mock_result.inserted_ids = inserted_ids
        return mock_result

    def update_one(self, filter_query: Dict[str, Any], update_query: Dict[str, Any]) -> MockUpdateResult:
        """Update a document."""
        doc = self.find_one(filter_query)
        if doc:
            if "$set" in update_query:
                doc.update(update_query["$set"])
            return MockUpdateResult(matched_count=1, modified_count=1)
        return MockUpdateResult(matched_count=0, modified_count=0)

    def delete_one(self, filter_query: Dict[str, Any]) -> MockDeleteResult:
        """Delete a document."""
        doc = self.find_one(filter_query)
        if doc and "_id" in doc:
            doc_id = str(doc["_id"])
            if doc_id in self._documents:
                del self._documents[doc_id]
                return MockDeleteResult(deleted_count=1)
        return MockDeleteResult(deleted_count=0)

    def count_documents(self, query: Dict[str, Any] = None) -> int:
        """Count documents."""
        if query is None:
            return len(self._documents)
        cursor = self.find(query)
        return len(cursor)

    def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate documents (simplified)."""
        # Simple aggregation - just return all documents for now
        return list(self.find())


class MockMongoClient:
    """Mock MongoDB client for testing."""

    def __init__(self):
        self.connected = True
        self._collections: Dict[str, MockCollection] = {}
        self._files: Dict[str, Dict[str, Any]] = {}  # GridFS files
        self._file_counter = 0

        # Create mock db object
        self.db = MagicMock()
        self.db.__getitem__ = lambda name: self.get_collection(name)

    def get_collection(self, name: str) -> MockCollection:
        """Get a collection."""
        if name not in self._collections:
            self._collections[name] = MockCollection(name)
        return self._collections[name]

    def store_file(
        self,
        data: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        bucket_name: Optional[str] = None,
    ) -> str:
        """Store a file in GridFS (mock). Accepts bucket_name for OpenVDB/GridFS API compatibility."""
        file_id = f"file_{self._file_counter}"
        self._file_counter += 1
        self._files[file_id] = {
            "data": data,
            "filename": filename,
            "metadata": metadata or {},
        }
        return file_id

    def retrieve_file(self, file_id: str) -> Optional[bytes]:
        """Retrieve a file from GridFS (mock)."""
        return self._files.get(file_id, {}).get("data")

    def get_file(self, file_id: str, bucket_name: Optional[str] = None) -> Optional[bytes]:
        """Get a file from GridFS (mock). Alias for retrieve_file; bucket_name accepted for API compatibility."""
        return self.retrieve_file(file_id)

    def delete_file(self, file_id: str) -> bool:
        """Delete a file from GridFS (mock)."""
        if file_id in self._files:
            del self._files[file_id]
            return True
        return False

    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self.connected

    def close(self):
        """Close the connection."""
        self.connected = False
