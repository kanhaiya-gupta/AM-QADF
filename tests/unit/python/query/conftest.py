"""
Query unit test fixtures and helpers.

No credentials or env file: mock_mongodb_client uses fixed non-secret strings only.
Tests that call the C++ MongoDBQueryClient skip when MongoDB is not available or not authenticated.
"""

import pytest


@pytest.fixture
def query_or_skip_mongodb():
    """
    Fixture that returns a helper: call client.query(...); skip test if MongoDB auth/connection fails.

    Use in unit tests that hit the real C++ MongoDBQueryClient. When MongoDB is not running
    or requires authentication (fixture uses unauthenticated URI), the test skips instead of failing.
    No credentials or env file in repo.
    """

    def _query_or_skip_mongodb(client, spatial, temporal=None, signal_types=None, **kwargs):
        try:
            return client.query(
                spatial=spatial,
                temporal=temporal,
                signal_types=signal_types,
                **kwargs,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if "authentication" in msg or "connection" in msg or "generic server error" in msg:
                pytest.skip(
                    "MongoDB not available or not authenticated (C++ client); "
                    "run with MongoDB to execute this test."
                )
            raise

    return _query_or_skip_mongodb
