"""
Pytest configuration and fixtures for ha_redis tests.

Provides:
- Mock Redis clients and connection pools
- Configuration fixtures for various scenarios
- Async test support
- Integration test markers and CLI options
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator

from ha_redis import RedisConfig, RedisClient, SentinelTopology


# ============================================================================
# Pytest Hooks for Integration Tests
# ============================================================================

def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires running Redis instance)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires running Redis)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running (e.g., failover tests)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --run-integration is provided."""
    if config.getoption("--run-integration"):
        # --run-integration given: do not skip integration tests
        return
    
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


# ============================================================================
# Async Event Loop Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def default_config() -> RedisConfig:
    """Create a default Redis configuration."""
    return RedisConfig()


@pytest.fixture
def custom_config() -> RedisConfig:
    """Create a customized Redis configuration."""
    return RedisConfig(
        host="redis.example.com",
        port=6380,
        password="secret123",
        db=1,
        max_connections=100,
        socket_timeout=10.0,
        socket_connect_timeout=5.0,
        health_check_interval=60,
        ssl=False,
        retry_attempts=5,
        retry_base_delay=0.2,
    )


@pytest.fixture
def ssl_config() -> RedisConfig:
    """Create an SSL-enabled Redis configuration."""
    return RedisConfig(
        host="redis-secure.example.com",
        port=6379,
        password="secure_password",
        ssl=True,
    )


@pytest.fixture
def sentinel_config() -> RedisConfig:
    """Create a Sentinel (HA) Redis configuration."""
    return RedisConfig(
        use_sentinel=True,
        sentinel_hosts=[
            ("sentinel1.example.com", 26379),
            ("sentinel2.example.com", 26379),
            ("sentinel3.example.com", 26379),
        ],
        sentinel_master_name="mymaster",
        password="sentinel_password",
        max_connections=100,
    )


@pytest.fixture
def minimal_sentinel_config() -> RedisConfig:
    """Create a minimal Sentinel configuration with one host."""
    return RedisConfig(
        use_sentinel=True,
        sentinel_hosts=[("localhost", 26379)],
        sentinel_master_name="testmaster",
    )


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_redis_client() -> AsyncMock:
    """Create a mock async Redis client."""
    mock = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.get = AsyncMock(return_value="value")
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.exists = AsyncMock(return_value=1)
    mock.expire = AsyncMock(return_value=True)
    mock.ttl = AsyncMock(return_value=3600)
    mock.incr = AsyncMock(return_value=1)
    mock.decr = AsyncMock(return_value=0)
    mock.hget = AsyncMock(return_value="hash_value")
    mock.hset = AsyncMock(return_value=1)
    mock.hgetall = AsyncMock(return_value={"field": "value"})
    mock.lpush = AsyncMock(return_value=1)
    mock.rpush = AsyncMock(return_value=1)
    mock.lpop = AsyncMock(return_value="item")
    mock.rpop = AsyncMock(return_value="item")
    mock.lrange = AsyncMock(return_value=["item1", "item2"])
    mock.sadd = AsyncMock(return_value=1)
    mock.smembers = AsyncMock(return_value={"member1", "member2"})
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_connection_pool() -> AsyncMock:
    """Create a mock Redis connection pool."""
    mock = AsyncMock()
    mock.disconnect = AsyncMock()
    return mock


@pytest.fixture
def mock_sentinel() -> AsyncMock:
    """Create a mock Redis Sentinel."""
    mock = AsyncMock()
    mock.master_for = MagicMock(return_value=AsyncMock())
    mock.slave_for = MagicMock(return_value=AsyncMock())
    mock.discover_master = AsyncMock(return_value=("redis-master", 6379))
    mock.discover_slaves = AsyncMock(return_value=[("redis-replica-1", 6379)])
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_sentinel_topology() -> SentinelTopology:
    """Create a mock SentinelTopology for testing."""
    return SentinelTopology(
        master_name="mymaster",
        master_address=("redis-master", 6379),
        replica_addresses=[("redis-replica-1", 6379), ("redis-replica-2", 6379)],
        sentinel_count=3,
        is_healthy=True,
        quorum=2,
    )


# ============================================================================
# Client Fixtures
# ============================================================================

@pytest.fixture
def redis_client(default_config) -> RedisClient:
    """Create a RedisClient instance with default config."""
    return RedisClient(default_config)


@pytest.fixture
def redis_client_with_custom_config(custom_config) -> RedisClient:
    """Create a RedisClient instance with custom config."""
    return RedisClient(custom_config)


@pytest.fixture
def sentinel_client(sentinel_config) -> RedisClient:
    """Create a RedisClient instance configured for Sentinel."""
    return RedisClient(sentinel_config)


@pytest.fixture
async def mocked_redis_client(
    default_config,
    mock_redis_client,
    mock_connection_pool,
) -> AsyncGenerator[RedisClient, None]:
    """
    Create a RedisClient with mocked internals for testing.
    
    Yields a client that uses mock Redis and pool objects.
    """
    client = RedisClient(default_config)
    client._client = mock_redis_client
    client._pool = mock_connection_pool
    client._initialized = True
    
    yield client
    
    # Cleanup
    await client.close()


@pytest.fixture
async def mocked_sentinel_client(
    sentinel_config,
    mock_redis_client,
    mock_sentinel,
) -> AsyncGenerator[RedisClient, None]:
    """
    Create a Sentinel-configured RedisClient with mocked internals.
    
    Yields a client that uses mock Sentinel and Redis objects.
    """
    client = RedisClient(sentinel_config)
    client._client = mock_redis_client
    client._sentinel = mock_sentinel
    client._initialized = True
    
    yield client
    
    # Cleanup
    await client.close()


# ============================================================================
# Exception Fixtures
# ============================================================================

@pytest.fixture
def connection_error():
    """Create a Redis ConnectionError."""
    from redis.exceptions import ConnectionError
    return ConnectionError("Connection refused")


@pytest.fixture
def timeout_error():
    """Create a Redis TimeoutError."""
    from redis.exceptions import TimeoutError
    return TimeoutError("Operation timed out")


@pytest.fixture
def busy_loading_error():
    """Create a Redis BusyLoadingError."""
    from redis.exceptions import BusyLoadingError
    return BusyLoadingError("Redis is loading the dataset in memory")


@pytest.fixture
def redis_error():
    """Create a general RedisError."""
    from redis.exceptions import RedisError
    return RedisError("Generic Redis error")


# ============================================================================
# Logger Fixtures
# ============================================================================

@pytest.fixture
def mock_logger() -> MagicMock:
    """Create a mock logger for testing log output."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.debug = MagicMock()
    return logger


@pytest.fixture
def redis_client_with_mock_logger(default_config, mock_logger) -> RedisClient:
    """Create a RedisClient with a mock logger."""
    return RedisClient(default_config, logger=mock_logger)
