"""
Pytest configuration: automatically clear compilation caches between tests
to prevent memory accumulation in long test suites.
"""

import pytest


@pytest.fixture(autouse=True)
def clear_compilation_caches():
    """Clear all Qrisp LRU compilation caches after each test."""
    yield
    from qrisp._cache_config import clear_all_caches
    clear_all_caches()
