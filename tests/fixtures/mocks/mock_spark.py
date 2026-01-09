"""
Mock Spark session and utilities for testing.
"""

from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any, Optional
import numpy as np


class MockDataFrame:
    """Mock Spark DataFrame."""

    def __init__(self, data: List[Dict[str, Any]] = None, schema: Dict[str, str] = None):
        self._data = data or []
        self._schema = schema or {}
        self._filters = []
        self._selects = []

    def select(self, *cols):
        """Select columns."""
        self._selects.extend(cols)
        return self

    def filter(self, condition):
        """Filter rows."""
        self._filters.append(condition)
        return self

    def where(self, condition):
        """Where clause (alias for filter)."""
        return self.filter(condition)

    def groupBy(self, *cols):
        """Group by columns."""
        mock_grouped = Mock()
        mock_grouped.agg = Mock(return_value=self)
        return mock_grouped

    def agg(self, *exprs):
        """Aggregate."""
        return self

    def join(self, other, on=None, how="inner"):
        """Join with another DataFrame."""
        return self

    def union(self, other):
        """Union with another DataFrame."""
        return self

    def distinct(self):
        """Get distinct rows."""
        return self

    def orderBy(self, *cols):
        """Order by columns."""
        return self

    def limit(self, num):
        """Limit number of rows."""
        return self

    def collect(self):
        """Collect results."""
        return [Mock(row) for row in self._data]

    def toPandas(self):
        """Convert to Pandas DataFrame."""
        try:
            import pandas as pd

            return pd.DataFrame(self._data)
        except ImportError:
            return None

    def show(self, n=20, truncate=True):
        """Show DataFrame."""
        print(f"MockDataFrame with {len(self._data)} rows")

    def count(self):
        """Count rows."""
        return len(self._data)

    def cache(self):
        """Cache DataFrame."""
        return self

    def persist(self, storage_level=None):
        """Persist DataFrame."""
        return self

    def unpersist(self):
        """Unpersist DataFrame."""
        return self


class MockSparkContext:
    """Mock Spark Context."""

    def __init__(self):
        self.appName = "test_app"
        self.master = "local[*]"
        self._conf = {}

    def setConf(self, key: str, value: str):
        """Set configuration."""
        self._conf[key] = value

    def getConf(self, key: str, defaultValue: str = None):
        """Get configuration."""
        return self._conf.get(key, defaultValue)

    def parallelize(self, data, numSlices=None):
        """Parallelize data."""
        return Mock()

    def stop(self):
        """Stop Spark context."""
        pass


class MockSparkSession:
    """Mock Spark session for testing."""

    def __init__(self):
        self.sparkContext = MockSparkContext()
        self.conf = Mock()
        self.conf.set = Mock()
        self.conf.get = Mock(return_value=None)
        self._dataframes: Dict[str, MockDataFrame] = {}

    def createDataFrame(self, data, schema=None, samplingRatio=None, verifySchema=True):
        """Create a DataFrame."""
        if isinstance(data, list):
            return MockDataFrame(data=data, schema=schema)
        elif isinstance(data, np.ndarray):
            # Convert numpy array to list of dicts
            if data.ndim == 2:
                columns = [f"col_{i}" for i in range(data.shape[1])]
                data_list = [dict(zip(columns, row)) for row in data]
                return MockDataFrame(data=data_list, schema={col: "float" for col in columns})
        return MockDataFrame()

    def read(self):
        """Create a DataFrameReader."""
        mock_reader = Mock()
        mock_reader.format = Mock(return_value=mock_reader)
        mock_reader.option = Mock(return_value=mock_reader)
        mock_reader.options = Mock(return_value=mock_reader)
        mock_reader.load = Mock(return_value=MockDataFrame())
        return mock_reader

    def sql(self, sqlQuery: str):
        """Execute SQL query."""
        return MockDataFrame()

    def table(self, tableName: str):
        """Read a table."""
        return self._dataframes.get(tableName, MockDataFrame())

    def stop(self):
        """Stop Spark session."""
        pass

    def getActiveSession(self):
        """Get active session."""
        return self

    @staticmethod
    def builder():
        """Create a SparkSession builder."""
        mock_builder = Mock()
        mock_builder.appName = Mock(return_value=mock_builder)
        mock_builder.master = Mock(return_value=mock_builder)
        mock_builder.config = Mock(return_value=mock_builder)
        mock_builder.getOrCreate = Mock(return_value=MockSparkSession())
        return mock_builder


def create_mock_spark_session():
    """Create a mock Spark session."""
    return MockSparkSession()
