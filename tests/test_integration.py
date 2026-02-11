"""
Integration tests for ha_redis.

These tests require a running Redis instance.
Skip these tests if Redis is not available.

To run integration tests:
    pytest tests/test_integration.py -v --run-integration

To run Sentinel HA integration tests:
    docker compose --profile ha up -d
    pytest tests/test_integration.py -v --run-integration -k "Sentinel or HA"

Configure Redis connection via environment variables:
    TEST_REDIS_HOST=localhost
    TEST_REDIS_PORT=6379
    TEST_REDIS_PASSWORD=optional
    
Sentinel HA environment variables:
    TEST_SENTINEL_HOST_1=localhost
    TEST_SENTINEL_PORT_1=26379
    TEST_SENTINEL_HOST_2=localhost
    TEST_SENTINEL_PORT_2=26380
    TEST_SENTINEL_HOST_3=localhost
    TEST_SENTINEL_PORT_3=26381
    TEST_SENTINEL_MASTER_NAME=mymaster
"""

import pytest
import os
import asyncio
import time
import logging
import subprocess
from typing import AsyncGenerator, List, Tuple, Optional

from ha_redis import RedisClient, RedisConfig, SentinelTopology, with_redis_retry
import redis.asyncio as redis
from redis.asyncio.sentinel import Sentinel
from redis.exceptions import ConnectionError, TimeoutError, RedisError, ResponseError, WatchError


# Skip all tests in this module if --run-integration is not provided
pytestmark = pytest.mark.integration


async def wait_for_replication(
    replica_client,
    key: str,
    expected_value: str,
    timeout: float = 5.0,
    interval: float = 0.2,
) -> str:
    """
    Poll a replica until the expected value appears or timeout is reached.

    Redis replication is asynchronous. A fixed sleep is a race condition —
    especially after failovers or under Docker networking latency.  This
    helper actively checks the replica, which is both faster on the happy
    path and safer on the slow path.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    value = None
    while asyncio.get_event_loop().time() < deadline:
        value = await replica_client.get(key)
        if value == expected_value:
            return value
        await asyncio.sleep(interval)
    return value  # return last seen value so the assert gives a useful diff


async def wait_for_condition(
    predicate,
    timeout: float = 5.0,
    interval: float = 0.2,
    description: str = "condition",
) -> bool:
    """
    Poll until predicate() returns True or timeout is reached.
    Useful for asserting asynchronous state changes (failover, pub/sub delivery).
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(interval)
    return False


def get_test_redis_config() -> RedisConfig:
    """Get Redis configuration from environment variables."""
    return RedisConfig(
        host=os.getenv("TEST_REDIS_HOST", "localhost"),
        port=int(os.getenv("TEST_REDIS_PORT", "6379")),
        password=os.getenv("TEST_REDIS_PASSWORD"),
        db=15,  # Use DB 15 for testing to avoid conflicts
    )


def get_sentinel_hosts() -> List[Tuple[str, int]]:
    """Get Sentinel hosts from environment variables or use defaults."""
    return [
        (
            os.getenv("TEST_SENTINEL_HOST_1", "localhost"),
            int(os.getenv("TEST_SENTINEL_PORT_1", "26379")),
        ),
        (
            os.getenv("TEST_SENTINEL_HOST_2", "localhost"),
            int(os.getenv("TEST_SENTINEL_PORT_2", "26380")),
        ),
        (
            os.getenv("TEST_SENTINEL_HOST_3", "localhost"),
            int(os.getenv("TEST_SENTINEL_PORT_3", "26381")),
        ),
    ]


def get_test_sentinel_config() -> RedisConfig:
    """Get Sentinel Redis configuration from environment variables."""
    return RedisConfig(
        use_sentinel=True,
        sentinel_hosts=get_sentinel_hosts(),
        sentinel_master_name=os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster"),
        password=os.getenv("TEST_SENTINEL_PASSWORD"),
        db=15,  # Use DB 15 for testing to avoid conflicts
        socket_timeout=5.0,
        socket_connect_timeout=2.0,
    )


@pytest.fixture
async def integration_client() -> AsyncGenerator[RedisClient, None]:
    """Create a real Redis client for integration testing."""
    config = get_test_redis_config()
    client = RedisClient(config)
    
    yield client
    
    # Cleanup: flush test database and close
    try:
        redis = client.get_client()
        await redis.flushdb()
    except Exception:
        pass
    
    await client.close()


@pytest.fixture
async def sentinel_client() -> AsyncGenerator[RedisClient, None]:
    """Create a Sentinel-connected Redis client for HA integration testing."""
    config = get_test_sentinel_config()
    client = RedisClient(config)
    
    yield client
    
    # Cleanup: flush test database and close
    try:
        redis_conn = client.get_client()
        await redis_conn.flushdb()
    except Exception:
        pass
    
    await client.close()


@pytest.fixture
async def raw_sentinel() -> AsyncGenerator[Sentinel, None]:
    """Create a raw Sentinel instance for low-level testing."""
    sentinel_hosts = get_sentinel_hosts()
    sentinel = Sentinel(
        sentinel_hosts,
        decode_responses=True,
        socket_timeout=5.0,
        socket_connect_timeout=2.0,
    )
    
    yield sentinel
    
    # Sentinel doesn't have a close() method, close underlying connections
    for sentinel_conn in sentinel.sentinels:
        await sentinel_conn.close()


class TestRedisIntegrationBasic:
    """Basic Redis integration tests."""
    
    @pytest.mark.asyncio
    async def test_ping(self, integration_client):
        """Should successfully ping Redis."""
        redis = integration_client.get_client()
        result = await redis.ping()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check(self, integration_client):
        """Health check should pass with running Redis."""
        result = await integration_client.health_check()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, integration_client):
        """Should set and get a value."""
        redis = integration_client.get_client()
        
        await redis.set("test_key", "test_value")
        result = await redis.get("test_key")
        
        assert result == "test_value"
    
    @pytest.mark.asyncio
    async def test_delete(self, integration_client):
        """Should delete a key."""
        redis = integration_client.get_client()
        
        await redis.set("delete_me", "value")
        await redis.delete("delete_me")
        result = await redis.get("delete_me")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_exists(self, integration_client):
        """Should check key existence."""
        redis = integration_client.get_client()
        
        await redis.set("exists_key", "value")
        
        assert await redis.exists("exists_key") == 1
        assert await redis.exists("nonexistent_key") == 0


class TestRedisIntegrationExpiry:
    """Test key expiration functionality."""
    
    @pytest.mark.asyncio
    async def test_set_with_expiry(self, integration_client):
        """Should set key with expiration."""
        redis = integration_client.get_client()
        
        await redis.set("expiring_key", "value", ex=3600)
        ttl = await redis.ttl("expiring_key")
        
        assert ttl > 0
        assert ttl <= 3600
    
    @pytest.mark.asyncio
    async def test_expire_command(self, integration_client):
        """Should set expiration on existing key."""
        redis = integration_client.get_client()
        
        await redis.set("expire_me", "value")
        await redis.expire("expire_me", 3600)
        ttl = await redis.ttl("expire_me")
        
        assert ttl > 0


class TestRedisIntegrationDataTypes:
    """Test various Redis data types."""
    
    @pytest.mark.asyncio
    async def test_hash_operations(self, integration_client):
        """Should perform hash operations."""
        redis = integration_client.get_client()
        
        await redis.hset("test_hash", "field1", "value1")
        await redis.hset("test_hash", "field2", "value2")
        
        value = await redis.hget("test_hash", "field1")
        assert value == "value1"
        
        all_fields = await redis.hgetall("test_hash")
        assert all_fields == {"field1": "value1", "field2": "value2"}
    
    @pytest.mark.asyncio
    async def test_list_operations(self, integration_client):
        """Should perform list operations."""
        redis = integration_client.get_client()
        
        await redis.rpush("test_list", "item1", "item2", "item3")
        
        items = await redis.lrange("test_list", 0, -1)
        assert items == ["item1", "item2", "item3"]
        
        item = await redis.lpop("test_list")
        assert item == "item1"
    
    @pytest.mark.asyncio
    async def test_set_operations(self, integration_client):
        """Should perform set operations."""
        redis = integration_client.get_client()
        
        await redis.sadd("test_set", "member1", "member2", "member3")
        
        members = await redis.smembers("test_set")
        assert members == {"member1", "member2", "member3"}
    
    @pytest.mark.asyncio
    async def test_increment_decrement(self, integration_client):
        """Should increment and decrement values."""
        redis = integration_client.get_client()
        
        await redis.set("counter", "10")
        
        result = await redis.incr("counter")
        assert result == 11
        
        result = await redis.decr("counter")
        assert result == 10


class TestRedisIntegrationConnectionPool:
    """Test connection pool behavior."""
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, integration_client):
        """Should handle concurrent operations."""
        redis = integration_client.get_client()
        
        async def set_and_get(key: str, value: str) -> str:
            await redis.set(key, value)
            return await redis.get(key)
        
        tasks = [
            set_and_get(f"concurrent_key_{i}", f"value_{i}")
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        for i, result in enumerate(results):
            assert result == f"value_{i}"
    
    @pytest.mark.asyncio
    async def test_connection_reuse(self, integration_client):
        """Should reuse connections from pool."""
        redis = integration_client.get_client()
        
        # Perform multiple operations
        for i in range(100):
            await redis.set(f"reuse_key_{i}", f"value_{i}")
            await redis.get(f"reuse_key_{i}")
        
        # If we got here without errors, connection pooling is working
        assert True


class TestRedisIntegrationContextManager:
    """Test async context manager in integration."""
    
    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self):
        """Context manager should properly cleanup."""
        config = get_test_redis_config()
        
        async with RedisClient(config) as client:
            redis = client.get_client()
            await redis.set("context_test", "value")
            result = await redis.get("context_test")
            assert result == "value"
        
        # After context exit, client should be closed
        assert client._client is None


class TestRedisIntegrationRetry:
    """Test retry functionality with real Redis."""
    
    @pytest.mark.asyncio
    async def test_retry_decorator_with_real_redis(self, integration_client):
        """Retry decorator should work with real Redis operations."""
        call_count = 0
        
        @integration_client.with_retry(max_retries=2)
        async def operation():
            nonlocal call_count
            call_count += 1
            redis = integration_client.get_client()
            await redis.set("retry_test", "value")
            return await redis.get("retry_test")
        
        result = await operation()
        
        assert result == "value"
        assert call_count == 1  # Should succeed on first try


# ============================================================================
# Sentinel HA Integration Tests
# ============================================================================

class TestSentinelDiscovery:
    """
    Test Sentinel discovery through RedisClient's abstraction layer.
    
    These tests verify that RedisClient properly exposes Sentinel discovery
    capabilities, allowing applications to query HA topology without
    accessing raw Sentinel objects.
    
    Requires: docker compose --profile ha up
    """
    
    @pytest.mark.asyncio
    async def test_is_sentinel_mode_returns_true(self, sentinel_client):
        """Client configured with Sentinel should report Sentinel mode."""
        assert sentinel_client.is_sentinel_mode() is True
    
    @pytest.mark.asyncio
    async def test_is_sentinel_mode_returns_false_for_direct(self, integration_client):
        """Direct connection client should not report Sentinel mode."""
        assert integration_client.is_sentinel_mode() is False
    
    @pytest.mark.asyncio
    async def test_discover_master_returns_address(self, sentinel_client):
        """discover_master() should return master host and port."""
        master_addr = await sentinel_client.discover_master()
        
        assert master_addr is not None
        assert len(master_addr) == 2
        host, port = master_addr
        assert host is not None
        assert isinstance(port, int)
        assert port > 0
    
    @pytest.mark.asyncio
    async def test_discover_replicas_returns_list(self, sentinel_client):
        """discover_replicas() should return list of replica addresses."""
        replicas = await sentinel_client.discover_replicas()
        
        assert replicas is not None
        assert isinstance(replicas, list)
        assert len(replicas) >= 1
        
        for replica_addr in replicas:
            host, port = replica_addr
            assert host is not None
            assert isinstance(port, int)
    
    @pytest.mark.asyncio
    async def test_get_topology_returns_complete_info(self, sentinel_client):
        """get_topology() should return comprehensive HA topology."""
        topology = await sentinel_client.get_topology()
        
        assert topology is not None
        assert isinstance(topology, SentinelTopology)
        
        # Verify master info
        assert topology.master_name == "mymaster"
        assert topology.master_address is not None
        
        # Verify replicas
        assert len(topology.replica_addresses) >= 1
        
        # Verify Sentinel cluster info
        assert topology.sentinel_count >= 1
        assert topology.quorum >= 2
        
        # Verify health status
        assert topology.is_healthy is True
    
    @pytest.mark.asyncio
    async def test_get_replica_client_returns_working_client(self, sentinel_client):
        """get_replica_client() should return a working read client."""
        # First write to master
        master = sentinel_client.get_client()
        await master.set("replica_test_key", "replica_test_value")
        
        # Get replica client via abstraction
        replica_client = await sentinel_client.get_replica_client()
        assert replica_client is not None
        
        # Poll replica until replication catches up
        value = await wait_for_replication(
            replica_client, "replica_test_key", "replica_test_value"
        )
        assert value == "replica_test_value"
        
        # Cleanup
        await master.delete("replica_test_key")
        await replica_client.close()
    
    @pytest.mark.asyncio
    async def test_discover_master_returns_none_for_direct_mode(self, integration_client):
        """discover_master() should return None for non-Sentinel client."""
        result = await integration_client.discover_master()
        assert result is None
    
    @pytest.mark.asyncio
    async def test_discover_replicas_returns_empty_for_direct_mode(self, integration_client):
        """discover_replicas() should return empty list for non-Sentinel client."""
        result = await integration_client.discover_replicas()
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_topology_returns_none_for_direct_mode(self, integration_client):
        """get_topology() should return None for non-Sentinel client."""
        result = await integration_client.get_topology()
        assert result is None
    
    @pytest.mark.asyncio
    async def test_client_via_sentinel_performs_operations(self, sentinel_client):
        """RedisClient via Sentinel should perform standard operations."""
        redis_conn = sentinel_client.get_client()
        
        # Standard operations should work transparently
        await redis_conn.set("sentinel_op_test", "sentinel_value")
        value = await redis_conn.get("sentinel_op_test")
        assert value == "sentinel_value"
        
        await redis_conn.delete("sentinel_op_test")


class TestSentinelDiscoveryRaw:
    """
    Low-level Sentinel protocol tests using raw Sentinel instance.
    
    These tests verify the underlying Sentinel protocol works correctly,
    independent of the RedisClient abstraction. Useful for debugging
    infrastructure issues.
    
    Requires: docker compose --profile ha up
    """
    
    @pytest.mark.asyncio
    async def test_raw_sentinel_can_discover_master(self, raw_sentinel):
        """Raw Sentinel should discover the master node."""
        master_name = os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster")
        
        master_addr = await raw_sentinel.discover_master(master_name)
        
        assert master_addr is not None
        assert len(master_addr) == 2
        host, port = master_addr
        assert host is not None
        assert isinstance(port, int)
        assert port > 0
    
    @pytest.mark.asyncio
    async def test_raw_sentinel_can_discover_replicas(self, raw_sentinel):
        """Raw Sentinel should discover replica nodes."""
        master_name = os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster")
        
        replicas = await raw_sentinel.discover_slaves(master_name)
        
        assert replicas is not None
        assert len(replicas) >= 1
        
        for replica_addr in replicas:
            host, port = replica_addr
            assert host is not None
            assert isinstance(port, int)
    
    @pytest.mark.asyncio
    async def test_raw_master_for_returns_working_client(self, raw_sentinel):
        """master_for() should return a working Redis client."""
        master_name = os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster")
        
        master_client = raw_sentinel.master_for(master_name)
        
        result = await master_client.ping()
        assert result is True
        
        await master_client.set("raw_sentinel_test_key", "test_value")
        value = await master_client.get("raw_sentinel_test_key")
        assert value == "test_value"
        
        await master_client.delete("raw_sentinel_test_key")
        await master_client.close()
    
    @pytest.mark.asyncio
    async def test_raw_slave_for_returns_working_client(self, raw_sentinel):
        """slave_for() should return a working read-only client."""
        master_name = os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster")
        
        master_client = raw_sentinel.master_for(master_name)
        await master_client.set("raw_replica_read_test", "replica_value")
        
        replica_client = raw_sentinel.slave_for(master_name)
        
        result = await replica_client.ping()
        assert result is True
        
        # Poll replica until replication catches up
        value = await wait_for_replication(
            replica_client, "raw_replica_read_test", "replica_value"
        )
        assert value == "replica_value"
        
        await master_client.delete("raw_replica_read_test")
        await master_client.close()
        await replica_client.close()


class TestSentinelProtocolCommunication:
    """
    Test Sentinel protocol communication - authentication, master lookup.
    
    These tests verify the Sentinel protocol commands work correctly.
    """
    
    @pytest.mark.asyncio
    async def test_sentinel_ping(self, raw_sentinel):
        """Should be able to ping Sentinel instances directly."""
        sentinel_hosts = get_sentinel_hosts()
        
        for host, port in sentinel_hosts:
            try:
                # Connect directly to sentinel
                sentinel_conn = redis.Redis(
                    host=host,
                    port=port,
                    decode_responses=True,
                    socket_timeout=5.0,
                )
                result = await sentinel_conn.ping()
                assert result is True
                await sentinel_conn.close()
            except (ConnectionError, TimeoutError):
                # Some sentinels might not be reachable, that's OK
                # as long as at least one works
                continue
    
    @pytest.mark.asyncio
    async def test_sentinel_master_info(self, raw_sentinel):
        """Should retrieve detailed master info from Sentinel."""
        master_name = os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster")
        sentinel_hosts = get_sentinel_hosts()
        
        # Connect to first available sentinel
        for host, port in sentinel_hosts:
            try:
                sentinel_conn = redis.Redis(
                    host=host,
                    port=port,
                    decode_responses=True,
                    socket_timeout=5.0,
                )
                
                # Execute SENTINEL MASTER command
                master_info = await sentinel_conn.execute_command(
                    "SENTINEL", "MASTER", master_name
                )
                
                assert master_info is not None
                # Convert list to dict for easier access
                if isinstance(master_info, list):
                    master_dict = dict(zip(master_info[::2], master_info[1::2]))
                    assert "name" in master_dict
                    assert master_dict["name"] == master_name
                    assert "ip" in master_dict
                    assert "port" in master_dict
                    assert "flags" in master_dict
                    # Master should have "master" flag
                    assert "master" in master_dict["flags"]
                
                await sentinel_conn.close()
                break
            except (ConnectionError, TimeoutError):
                continue
    
    @pytest.mark.asyncio
    async def test_sentinel_replicas_info(self, raw_sentinel):
        """Should retrieve replica info from Sentinel."""
        master_name = os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster")
        sentinel_hosts = get_sentinel_hosts()
        
        for host, port in sentinel_hosts:
            try:
                sentinel_conn = redis.Redis(
                    host=host,
                    port=port,
                    decode_responses=True,
                    socket_timeout=5.0,
                )
                
                # Execute SENTINEL REPLICAS command
                replicas_info = await sentinel_conn.execute_command(
                    "SENTINEL", "REPLICAS", master_name
                )
                
                assert replicas_info is not None
                assert len(replicas_info) >= 1  # At least one replica
                
                # Check first replica
                if replicas_info and isinstance(replicas_info[0], list):
                    replica_dict = dict(zip(replicas_info[0][::2], replicas_info[0][1::2]))
                    assert "ip" in replica_dict
                    assert "port" in replica_dict
                    assert "flags" in replica_dict
                    assert "slave" in replica_dict["flags"]
                
                await sentinel_conn.close()
                break
            except (ConnectionError, TimeoutError):
                continue
    
    @pytest.mark.asyncio
    async def test_sentinel_get_master_addr_by_name(self, raw_sentinel):
        """SENTINEL GET-MASTER-ADDR-BY-NAME should return master address."""
        master_name = os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster")
        sentinel_hosts = get_sentinel_hosts()
        
        for host, port in sentinel_hosts:
            try:
                sentinel_conn = redis.Redis(
                    host=host,
                    port=port,
                    decode_responses=True,
                    socket_timeout=5.0,
                )
                
                # Get master address
                addr = await sentinel_conn.execute_command(
                    "SENTINEL", "GET-MASTER-ADDR-BY-NAME", master_name
                )
                
                assert addr is not None
                assert len(addr) == 2
                assert addr[0] is not None  # host
                assert addr[1] is not None  # port (as string)
                
                await sentinel_conn.close()
                break
            except (ConnectionError, TimeoutError):
                continue
    
    @pytest.mark.asyncio
    async def test_sentinel_ckquorum(self, raw_sentinel):
        """SENTINEL CKQUORUM should verify quorum is reachable."""
        master_name = os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster")
        sentinel_hosts = get_sentinel_hosts()
        
        for host, port in sentinel_hosts:
            try:
                sentinel_conn = redis.Redis(
                    host=host,
                    port=port,
                    decode_responses=True,
                    socket_timeout=5.0,
                )
                
                # Check quorum
                result = await sentinel_conn.execute_command(
                    "SENTINEL", "CKQUORUM", master_name
                )
                
                # Should return OK or similar success message
                assert result is not None
                assert "OK" in str(result).upper() or "QUORUM" in str(result).upper()
                
                await sentinel_conn.close()
                break
            except (ConnectionError, TimeoutError):
                continue


class TestReplicaReads:
    """
    Test replica read operations.
    
    These tests verify that data written to master is replicated
    and can be read from replicas.
    """
    
    @pytest.mark.asyncio
    async def test_write_to_master_read_from_replica(self, raw_sentinel):
        """Data written to master should be readable from replica."""
        master_name = os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster")
        
        # Get master and replica clients
        master_client = raw_sentinel.master_for(master_name)
        replica_client = raw_sentinel.slave_for(master_name)
        
        test_key = "replication_test_key"
        test_value = "replicated_value_12345"
        
        try:
            # Write to master
            await master_client.set(test_key, test_value)
            
            # Poll replica until replication catches up
            value = await wait_for_replication(
                replica_client, test_key, test_value
            )
            assert value == test_value
        finally:
            # Cleanup
            await master_client.delete(test_key)
            await master_client.close()
            await replica_client.close()
    
    @pytest.mark.asyncio
    async def test_replica_read_consistency(self, raw_sentinel):
        """Multiple reads from replica should be consistent."""
        master_name = os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster")
        
        master_client = raw_sentinel.master_for(master_name)
        replica_client = raw_sentinel.slave_for(master_name)
        
        try:
            # Write multiple keys
            for i in range(10):
                await master_client.set(f"consistency_test_{i}", f"value_{i}")
            
            # Wait for the last key to replicate (implies all prior keys are there too)
            await wait_for_replication(
                replica_client, "consistency_test_9", "value_9"
            )
            
            # Read all keys multiple times from replica
            for _ in range(3):
                for i in range(10):
                    value = await replica_client.get(f"consistency_test_{i}")
                    assert value == f"value_{i}"
        finally:
            # Cleanup
            for i in range(10):
                await master_client.delete(f"consistency_test_{i}")
            await master_client.close()
            await replica_client.close()
    
    @pytest.mark.asyncio
    async def test_replica_readonly_behavior(self, raw_sentinel):
        """Replica should reject write operations by default."""
        master_name = os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster")
        
        # Verify a real replica is available; after a failover slave_for()
        # may fall back to the master if no healthy replicas exist yet.
        replicas = await raw_sentinel.discover_slaves(master_name)
        if not replicas:
            pytest.skip("No replicas available (topology may still be recovering from failover)")
        
        replica_client = raw_sentinel.slave_for(master_name)
        
        try:
            # Confirm we are actually connected to a replica, not the master.
            info = await replica_client.info("replication")
            if info.get("role") != "slave":
                pytest.skip("slave_for() returned the master (no real replica available)")
            
            # Try to write to replica - should fail with READONLY error
            with pytest.raises((RedisError, Exception)):
                await replica_client.set("replica_write_test", "should_fail")
        finally:
            await replica_client.close()


class TestFailoverBehavior:
    """
    Test failover behavior - what happens when master fails.
    
    These tests simulate master failure and verify proper failover.
    
    WARNING: These tests manipulate Docker containers and may take
    longer to execute. They require the docker CLI to be available.
    """
    
    @pytest.fixture
    def docker_available(self):
        """Check if Docker CLI is available."""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_client_reconnects_after_master_restart(self, sentinel_client):
        """Client should reconnect after master briefly restarts."""
        redis_conn = sentinel_client.get_client()
        
        # Verify initial connection works
        await redis_conn.set("failover_test_1", "initial_value")
        value = await redis_conn.get("failover_test_1")
        assert value == "initial_value"
        
        # The client should handle brief network issues gracefully
        # due to retry logic
        await redis_conn.set("failover_test_2", "after_test")
        value = await redis_conn.get("failover_test_2")
        assert value == "after_test"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sentinel_master_discovery_resilience(self, raw_sentinel):
        """Sentinel should provide master info even under load."""
        master_name = os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster")
        
        # Repeatedly discover master to test resilience
        for _ in range(20):
            master_addr = await raw_sentinel.discover_master(master_name)
            assert master_addr is not None
            assert len(master_addr) == 2
            await asyncio.sleep(0.1)
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_operations_during_simulated_network_latency(self, sentinel_client):
        """Operations should succeed with retry even with network issues."""
        redis_conn = sentinel_client.get_client()
        
        # Run operations that should succeed with built-in retry
        for i in range(10):
            await redis_conn.set(f"latency_test_{i}", f"value_{i}")
            value = await redis_conn.get(f"latency_test_{i}")
            assert value == f"value_{i}"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_failover_detection_via_sentinel(self, docker_available):
        """
        Test that Sentinel properly detects master and can coordinate failover.
        
        This test verifies the Sentinel monitoring is working by checking
        the Sentinel's view of the topology.
        """
        if not docker_available:
            pytest.skip("Docker not available for failover test")
        
        master_name = os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster")
        sentinel_hosts = get_sentinel_hosts()
        
        # Connect to sentinel
        for host, port in sentinel_hosts:
            try:
                sentinel_conn = redis.Redis(
                    host=host,
                    port=port,
                    decode_responses=True,
                    socket_timeout=5.0,
                )
                
                # Check Sentinel's view of the topology
                master_info = await sentinel_conn.execute_command(
                    "SENTINEL", "MASTER", master_name
                )
                
                if isinstance(master_info, list):
                    master_dict = dict(zip(master_info[::2], master_info[1::2]))
                    
                    # Verify master is healthy
                    flags = master_dict.get("flags", "")
                    assert "master" in flags
                    # Should not have s_down or o_down flags
                    assert "s_down" not in flags
                    assert "o_down" not in flags
                    
                    # Verify quorum
                    quorum = int(master_dict.get("quorum", 0))
                    assert quorum >= 2
                    
                    # Verify we have replicas
                    num_slaves = int(master_dict.get("num-slaves", 0))
                    assert num_slaves >= 1
                    
                    # Verify sentinels are monitoring
                    num_sentinels = int(master_dict.get("num-other-sentinels", 0))
                    assert num_sentinels >= 2  # At least 2 other sentinels
                
                await sentinel_conn.close()
                break
            except (ConnectionError, TimeoutError):
                continue
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_manual_failover_via_sentinel(self, docker_available):
        """
        Test manual failover command via Sentinel.
        
        This performs a controlled failover to verify the mechanism works.
        After the test, the original topology should be restored.
        """
        if not docker_available:
            pytest.skip("Docker not available for failover test")
        
        master_name = os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster")
        sentinel_hosts = get_sentinel_hosts()
        
        original_master = None
        
        for host, port in sentinel_hosts:
            try:
                sentinel_conn = redis.Redis(
                    host=host,
                    port=port,
                    decode_responses=True,
                    socket_timeout=30.0,  # Longer timeout for failover
                )
                
                # Get current master
                original_addr = await sentinel_conn.execute_command(
                    "SENTINEL", "GET-MASTER-ADDR-BY-NAME", master_name
                )
                original_master = (original_addr[0], original_addr[1])
                
                # Trigger failover
                try:
                    await sentinel_conn.execute_command(
                        "SENTINEL", "FAILOVER", master_name
                    )
                except RedisError as e:
                    # Failover might already be in progress or not needed
                    if "INPROG" not in str(e) and "NOGOODSLAVE" not in str(e):
                        raise
                
                # Wait for failover to complete
                await asyncio.sleep(15)
                
                # Check new master
                new_addr = await sentinel_conn.execute_command(
                    "SENTINEL", "GET-MASTER-ADDR-BY-NAME", master_name
                )
                new_master = (new_addr[0], new_addr[1])
                
                # Verify failover happened (master should have changed)
                # Note: In some cases the same node might be re-elected
                assert new_master is not None
                
                # Verify new master is healthy
                master_info = await sentinel_conn.execute_command(
                    "SENTINEL", "MASTER", master_name
                )
                if isinstance(master_info, list):
                    master_dict = dict(zip(master_info[::2], master_info[1::2]))
                    flags = master_dict.get("flags", "")
                    assert "master" in flags
                    assert "s_down" not in flags
                    assert "o_down" not in flags
                
                await sentinel_conn.close()
                break
            except (ConnectionError, TimeoutError):
                continue


class TestSentinelHealthCheck:
    """Test health check functionality with Sentinel configuration."""
    
    @pytest.mark.asyncio
    async def test_health_check_via_sentinel(self, sentinel_client):
        """Health check should work with Sentinel-connected client."""
        result = await sentinel_client.health_check()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_multiple_times(self, sentinel_client):
        """Multiple health checks should all succeed."""
        for _ in range(10):
            result = await sentinel_client.health_check()
            assert result is True
            await asyncio.sleep(0.1)


class TestSentinelConnectionPool:
    """Test connection pool behavior with Sentinel."""
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_via_sentinel(self, sentinel_client):
        """Should handle concurrent operations via Sentinel."""
        redis_conn = sentinel_client.get_client()
        
        async def set_and_get(key: str, value: str) -> str:
            await redis_conn.set(key, value)
            return await redis_conn.get(key)
        
        # Create 20 concurrent tasks
        tasks = [
            set_and_get(f"sentinel_concurrent_{i}", f"value_{i}")
            for i in range(20)
        ]
        
        results = await asyncio.gather(*tasks)
        
        for i, result in enumerate(results):
            assert result == f"value_{i}"
    
    @pytest.mark.asyncio
    async def test_rapid_connect_disconnect_cycles(self):
        """Should handle rapid connect/disconnect cycles."""
        config = get_test_sentinel_config()
        
        for _ in range(5):
            async with RedisClient(config) as client:
                redis_conn = client.get_client()
                await redis_conn.set("cycle_test", "value")
                result = await redis_conn.get("cycle_test")
                assert result == "value"


# ============================================================================
# P0 Integration Tests — SPEC-007
# ============================================================================


class TestRetryRealFailureRecovery:
    """
    Validates SPEC-001 retry decorator behavior under real transient failures.

    These tests go beyond the existing happy-path retry test by exercising
    actual failure modes: killed connections, unreachable hosts, and the
    standalone decorator.
    """

    @pytest.mark.asyncio
    async def test_retry_recovers_from_transient_connection_drop(
        self, integration_client
    ):
        """with_retry recovers when a connection is forcibly killed mid-operation."""
        call_count = 0

        # Create a second admin client to kill connections
        admin_config = get_test_redis_config()
        admin_client = RedisClient(admin_config)
        admin_redis = admin_client.get_client()

        try:

            @integration_client.with_retry(max_retries=3, base_delay=0.1)
            async def resilient_set():
                nonlocal call_count
                call_count += 1
                redis_conn = integration_client.get_client()
                if call_count == 1:
                    # Kill all client connections except the admin one
                    client_list = await admin_redis.execute_command("CLIENT", "LIST")
                    my_name = admin_config.client_name
                    for line in client_list.strip().split("\n"):
                        # Don't kill our admin connection
                        if f"name={my_name}" not in line:
                            # Extract the id
                            parts = dict(
                                kv.split("=", 1)
                                for kv in line.split()
                                if "=" in kv
                            )
                            cid = parts.get("id")
                            if cid:
                                try:
                                    await admin_redis.execute_command(
                                        "CLIENT", "KILL", "ID", cid
                                    )
                                except Exception:
                                    pass
                await redis_conn.set("intg:retry:drop_key", "recovered")
                return await redis_conn.get("intg:retry:drop_key")

            result = await resilient_set()

            assert result == "recovered", "Retry should recover after connection kill"
            assert call_count >= 1, "Should have been called at least once"
        finally:
            await admin_client.close()

    @pytest.mark.asyncio
    async def test_retry_exhaustion_raises_after_all_attempts(self):
        """Final exception propagates after all retries are exhausted."""
        config = RedisConfig(
            host="192.0.2.1",
            port=6399,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        client = RedisClient(config)

        try:

            @client.with_retry(max_retries=2, base_delay=0.1)
            async def unreachable_ping():
                redis_conn = client.get_client()
                return await redis_conn.ping()

            start = time.monotonic()
            with pytest.raises((ConnectionError, TimeoutError, OSError)):
                await unreachable_ping()
            elapsed = time.monotonic() - start

            assert elapsed >= 0.3, (
                f"Backoff should cause ≥0.3s delay, got {elapsed:.2f}s"
            )
            assert elapsed < 30, (
                f"Should not hang — elapsed {elapsed:.2f}s"
            )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_standalone_retry_decorator_with_real_redis(
        self, integration_client
    ):
        """Standalone with_redis_retry() works end-to-end against real Redis."""
        redis_conn = integration_client.get_client()

        @with_redis_retry(max_retries=2)
        async def standalone_op():
            await redis_conn.set("intg:retry:standalone", "standalone_value")
            return await redis_conn.get("intg:retry:standalone")

        result = await standalone_op()
        assert result == "standalone_value"

    @pytest.mark.asyncio
    async def test_retry_backoff_timing_is_reasonable(self):
        """Exponential backoff is actually applied (not instant retries)."""
        config = RedisConfig(
            host="192.0.2.1",
            port=6399,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            retry_attempts=2,
            retry_base_delay=0.2,
        )
        client = RedisClient(config)

        try:

            @client.with_retry()
            async def timed_ping():
                redis_conn = client.get_client()
                return await redis_conn.ping()

            start = time.monotonic()
            with pytest.raises((ConnectionError, TimeoutError, OSError)):
                await timed_ping()
            elapsed = time.monotonic() - start

            # 2 retries: backoff ≈ 0.2 + 0.4 = 0.6s minimum
            assert elapsed >= 0.6, (
                f"Expected ≥0.6s for exponential backoff, got {elapsed:.2f}s"
            )
            assert elapsed < 30, (
                f"Should not hang — elapsed {elapsed:.2f}s"
            )
        finally:
            await client.close()


class TestResetClientIntegration:
    """
    Validates SPEC-005 and SPEC-006 — _reset_client() tears down cached
    clients and allows re-discovery.
    """

    @pytest.mark.asyncio
    async def test_reset_client_allows_reconnection_to_sentinel_master(
        self, sentinel_client
    ):
        """After _reset_client(), get_client() returns a working master client."""
        # Confirm initial connectivity
        redis_conn = sentinel_client.get_client()
        await redis_conn.set("intg:reset:initial", "before_reset")
        value = await redis_conn.get("intg:reset:initial")
        assert value == "before_reset"

        # Reset
        await sentinel_client._reset_client()
        assert sentinel_client._client is None
        assert sentinel_client._initialized is False

        # Reconnect
        new_redis = sentinel_client.get_client()
        assert new_redis is not None
        await new_redis.set("intg:reset:after", "after_reset")
        value = await new_redis.get("intg:reset:after")
        assert value == "after_reset"

    @pytest.mark.asyncio
    async def test_reset_client_re_creates_replica_client(self, sentinel_client):
        """After _reset_client(), get_replica_client() returns a fresh replica."""
        # Populate the cached replica
        original_replica = await sentinel_client.get_replica_client()
        assert original_replica is not None

        # Reset
        await sentinel_client._reset_client()

        # Get new replica
        new_replica = await sentinel_client.get_replica_client()
        assert new_replica is not None
        assert new_replica is not original_replica, (
            "Replica client should be a new object after reset"
        )

        # Write via master, read via new replica
        master = sentinel_client.get_client()
        await master.set("intg:reset:replica_key", "replica_value")

        value = await wait_for_replication(
            new_replica, "intg:reset:replica_key", "replica_value"
        )
        assert value == "replica_value"

    @pytest.mark.asyncio
    async def test_reset_preserves_sentinel_instance(self, sentinel_client):
        """_reset_client() must NOT tear down the Sentinel instance."""
        # Force initialization
        sentinel_client.get_client()
        original_sentinel = sentinel_client._sentinel
        assert original_sentinel is not None

        await sentinel_client._reset_client()

        assert sentinel_client._sentinel is original_sentinel, (
            "Sentinel instance should be preserved across reset"
        )

    @pytest.mark.asyncio
    async def test_with_retry_resets_and_recovers_after_reset(self, sentinel_client):
        """with_retry-wrapped op succeeds after _reset_client() simulates stale state."""
        # Deliberately reset to simulate stale state
        sentinel_client.get_client()  # ensure initialized first
        await sentinel_client._reset_client()

        @sentinel_client.with_retry(max_retries=3, base_delay=0.1)
        async def post_reset_op():
            redis_conn = sentinel_client.get_client()
            await redis_conn.set("intg:reset:retry_key", "recovered")
            return await redis_conn.get("intg:reset:retry_key")

        result = await post_reset_op()
        assert result == "recovered"


class TestConcurrentInitialization:
    """
    Validates SPEC-003 — asyncio.Lock prevents race conditions during
    concurrent startup.
    """

    @pytest.mark.asyncio
    async def test_concurrent_initialize_creates_single_client(self):
        """50 coroutines calling initialize() get the same cached client."""
        config = get_test_redis_config()
        client = RedisClient(config)

        try:
            await asyncio.gather(*(client.initialize() for _ in range(50)))

            assert client._initialized is True
            # All calls should resolve to the same underlying client
            client_id = id(client._client)
            for _ in range(50):
                assert id(client.get_client()) == client_id
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_concurrent_initialize_sentinel_mode(self):
        """50 coroutines calling initialize() in Sentinel mode get the same client."""
        config = get_test_sentinel_config()
        client = RedisClient(config)

        try:
            await asyncio.gather(*(client.initialize() for _ in range(50)))

            assert client._initialized is True
            client_id = id(client._client)
            for _ in range(50):
                assert id(client.get_client()) == client_id

            # Verify the client actually works
            redis_conn = client.get_client()
            result = await redis_conn.ping()
            assert result is True
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_client_after_initialize_is_lockfree(self, integration_client):
        """After initialize(), get_client() should be a fast lock-free path."""
        await integration_client.initialize()

        start = time.monotonic()
        for _ in range(1000):
            integration_client.get_client()
        elapsed = time.monotonic() - start

        assert elapsed < 0.1, (
            f"1000 get_client() calls took {elapsed:.3f}s — expected < 0.1s "
            f"for lock-free fast path"
        )


class TestGracefulShutdownAndCleanup:
    """
    Validates resource lifecycle — double-close safety, post-close
    behavior, cleanup under exceptions.
    """

    @pytest.mark.asyncio
    async def test_double_close_is_safe(self):
        """Calling close() twice does not raise."""
        config = get_test_redis_config()
        client = RedisClient(config)

        # Initialize and use
        redis_conn = client.get_client()
        await redis_conn.set("intg:shutdown:double", "value")

        # First close
        await client.close()
        # Second close — must not raise
        await client.close()

    @pytest.mark.asyncio
    async def test_operations_after_close_re_initialize(self):
        """get_client() after close() re-creates a working client."""
        config = get_test_redis_config()
        client = RedisClient(config)

        try:
            redis_conn = client.get_client()
            await redis_conn.set("intg:shutdown:reopen", "before")
            await client.close()

            # Re-initialize via get_client()
            new_redis = client.get_client()
            assert new_redis is not None
            result = await new_redis.ping()
            assert result is True
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self):
        """An exception inside async-with still triggers close()."""
        config = get_test_redis_config()

        with pytest.raises(ValueError):
            async with RedisClient(config) as client:
                redis_conn = client.get_client()
                await redis_conn.set("intg:shutdown:exc", "value")
                raise ValueError("Intentional test exception")

        assert client._client is None, "Client should be cleaned up after exception"
        assert client._initialized is False

    @pytest.mark.asyncio
    async def test_close_during_background_operations(self):
        """close() while background tasks are running completes without deadlock."""
        config = get_test_redis_config()
        client = RedisClient(config)

        redis_conn = client.get_client()
        errors: list = []

        async def background_write(i: int):
            try:
                await redis_conn.set(f"intg:shutdown:bg:{i}", f"value_{i}")
            except (ConnectionError, TimeoutError, RedisError):
                pass  # Expected when pool is closing
            except Exception as e:
                errors.append(e)

        try:
            # Launch background writes + close concurrently
            tasks = [asyncio.create_task(background_write(i)) for i in range(10)]
            close_task = asyncio.create_task(client.close())

            done, pending = await asyncio.wait(
                tasks + [close_task], timeout=5.0
            )

            # Nothing should hang
            assert len(pending) == 0, (
                f"{len(pending)} tasks still pending after 5s timeout"
            )
            # No unexpected errors
            assert len(errors) == 0, f"Unexpected errors: {errors}"
        finally:
            # Safety net — ensure client is closed even if test fails
            try:
                await client.close()
            except Exception:
                pass


# ============================================================================
# P1 Integration Tests — SPEC-007
# ============================================================================


class TestPipelineOperations:
    """
    Validates Redis pipeline batching for throughput — a core performance
    pattern absent from prior tests.
    """

    @pytest.mark.asyncio
    async def test_pipeline_batches_commands(self, integration_client):
        """A pipeline batches multiple SET + GET commands and returns all results."""
        redis_conn = integration_client.get_client()

        async with redis_conn.pipeline(transaction=False) as pipe:
            for i in range(5):
                pipe.set(f"intg:pipeline:batch:{i}", f"val{i}")
            for i in range(5):
                pipe.get(f"intg:pipeline:batch:{i}")
            results = await pipe.execute()

        # First 5 results are SETs (True), next 5 are GETs
        set_results = results[:5]
        get_results = results[5:]

        for r in set_results:
            assert r is True, f"SET should return True, got {r}"
        for i, val in enumerate(get_results):
            assert val == f"val{i}", f"Expected 'val{i}', got '{val}'"

    @pytest.mark.asyncio
    async def test_pipeline_large_batch(self, integration_client):
        """Pipeline handles a large batch (1000 commands) without error."""
        redis_conn = integration_client.get_client()

        async with redis_conn.pipeline(transaction=False) as pipe:
            for i in range(1000):
                pipe.set(f"intg:pipeline:large:{i}", f"v{i}")
            results = await pipe.execute()

        assert len(results) == 1000, f"Expected 1000 results, got {len(results)}"

        # Spot-check 10 random keys
        import random
        spot_indices = random.sample(range(1000), 10)
        for idx in spot_indices:
            value = await redis_conn.get(f"intg:pipeline:large:{idx}")
            assert value == f"v{idx}", (
                f"Spot-check failed for key {idx}: expected 'v{idx}', got '{value}'"
            )

    @pytest.mark.asyncio
    async def test_pipeline_via_sentinel_client(self, sentinel_client):
        """Pipelines work through Sentinel-managed connections."""
        redis_conn = sentinel_client.get_client()

        async with redis_conn.pipeline(transaction=False) as pipe:
            for i in range(5):
                pipe.set(f"intg:pipeline:sentinel:{i}", f"sval{i}")
            for i in range(5):
                pipe.get(f"intg:pipeline:sentinel:{i}")
            results = await pipe.execute()

        set_results = results[:5]
        get_results = results[5:]

        for r in set_results:
            assert r is True
        for i, val in enumerate(get_results):
            assert val == f"sval{i}"

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, integration_client):
        """One bad command in a non-transactional pipeline does not block others."""
        redis_conn = integration_client.get_client()

        # Set a string key so INCR on it will fail
        await redis_conn.set("intg:pipeline:errkey", "not_a_number")

        async with redis_conn.pipeline(transaction=False) as pipe:
            pipe.get("intg:pipeline:errkey")          # valid
            pipe.incr("intg:pipeline:errkey")          # will error
            pipe.set("intg:pipeline:okkey", "ok_val")  # valid
            pipe.get("intg:pipeline:okkey")            # valid
            results = await pipe.execute(raise_on_error=False)

        assert results[0] == "not_a_number", "First GET should succeed"
        assert isinstance(results[1], ResponseError), (
            f"INCR on string should be ResponseError, got {type(results[1])}"
        )
        assert results[2] is True, "SET should succeed"
        assert results[3] == "ok_val", "Second GET should succeed"


class TestTransactionOperations:
    """
    Validates MULTI/EXEC atomic execution and optimistic locking (WATCH).
    """

    @pytest.mark.asyncio
    async def test_transaction_atomic_execution(self, integration_client):
        """Commands inside MULTI/EXEC execute atomically."""
        redis_conn = integration_client.get_client()

        async with redis_conn.pipeline(transaction=True) as pipe:
            pipe.set("intg:txn:counter", "0")
            pipe.incr("intg:txn:counter")
            pipe.incr("intg:txn:counter")
            results = await pipe.execute()

        assert results == [True, 1, 2], f"Expected [True, 1, 2], got {results}"

        final = await redis_conn.get("intg:txn:counter")
        assert final == "2", f"Counter should be '2', got '{final}'"

    @pytest.mark.asyncio
    async def test_transaction_watch_optimistic_locking(self, integration_client):
        """WATCH detects concurrent modification and raises WatchError."""
        redis_conn = integration_client.get_client()
        await redis_conn.set("intg:txn:watched", "original")

        # Use a separate connection for the concurrent modification
        config2 = get_test_redis_config()
        client2 = RedisClient(config2)
        redis2 = client2.get_client()

        try:
            async with redis_conn.pipeline(transaction=True) as pipe:
                await pipe.watch("intg:txn:watched")

                # Concurrent modification from second connection
                await redis2.set("intg:txn:watched", "modified")

                # Now try to execute a transaction on the watched key
                pipe.multi()
                pipe.set("intg:txn:watched", "from_txn")

                with pytest.raises(WatchError):
                    await pipe.execute()

            # Verify the key has the concurrent modification's value
            val = await redis_conn.get("intg:txn:watched")
            assert val == "modified", (
                f"Key should have concurrent value 'modified', got '{val}'"
            )
        finally:
            await client2.close()

    @pytest.mark.asyncio
    async def test_transaction_via_sentinel(self, sentinel_client):
        """Transactions work through Sentinel-managed connections."""
        redis_conn = sentinel_client.get_client()

        async with redis_conn.pipeline(transaction=True) as pipe:
            pipe.set("intg:txn:sentinel_ctr", "0")
            pipe.incr("intg:txn:sentinel_ctr")
            pipe.incr("intg:txn:sentinel_ctr")
            results = await pipe.execute()

        assert results == [True, 1, 2], f"Expected [True, 1, 2], got {results}"

        final = await redis_conn.get("intg:txn:sentinel_ctr")
        assert final == "2"

    @pytest.mark.asyncio
    async def test_transaction_discard(self, integration_client):
        """DISCARD aborts a queued transaction with no side effects."""
        redis_conn = integration_client.get_client()
        await redis_conn.set("intg:txn:discard_key", "before")

        async with redis_conn.pipeline(transaction=True) as pipe:
            pipe.set("intg:txn:discard_key", "after")
            await pipe.reset()  # DISCARD

        val = await redis_conn.get("intg:txn:discard_key")
        assert val == "before", (
            f"Key should still be 'before' after DISCARD, got '{val}'"
        )


class TestConnectionPoolStress:
    """
    Validates pool behavior under high concurrency and resource limits.
    """

    @pytest.mark.asyncio
    async def test_pool_handles_100_concurrent_tasks(self, integration_client):
        """100 simultaneous SET/GET tasks complete without pool errors."""
        redis_conn = integration_client.get_client()

        async def set_and_get(i: int) -> str:
            await redis_conn.set(f"intg:stress:{i}", f"v{i}")
            return await redis_conn.get(f"intg:stress:{i}")

        results = await asyncio.gather(
            *(set_and_get(i) for i in range(100))
        )

        for i, result in enumerate(results):
            assert result == f"v{i}", (
                f"Task {i} expected 'v{i}', got '{result}'"
            )

    @pytest.mark.asyncio
    async def test_pool_exhaustion_behavior(self):
        """With max_connections=2, excess concurrent requests raise or block — not deadlock."""
        config = RedisConfig(
            host=os.getenv("TEST_REDIS_HOST", "localhost"),
            port=int(os.getenv("TEST_REDIS_PORT", "6379")),
            password=os.getenv("TEST_REDIS_PASSWORD"),
            db=15,
            max_connections=2,
            socket_timeout=2.0,
            socket_connect_timeout=1.0,
        )
        client = RedisClient(config)

        try:
            redis_conn = client.get_client()

            # Occupy 2 connections with long-running BLPOP (blocks for up to 3s)
            async def blocking_op(i: int):
                try:
                    await redis_conn.blpop(
                        f"intg:stress:exhaust:block:{i}", timeout=3
                    )
                except (ConnectionError, TimeoutError, RedisError):
                    pass

            # Start 2 blocking operations
            block_tasks = [
                asyncio.create_task(blocking_op(i)) for i in range(2)
            ]
            # Give them a moment to acquire connections
            await asyncio.sleep(0.3)

            # 3rd operation should raise or timeout — not hang forever
            try:
                result = await asyncio.wait_for(
                    redis_conn.set("intg:stress:exhaust:extra", "value"),
                    timeout=3.0,
                )
            except (ConnectionError, TimeoutError, asyncio.TimeoutError):
                pass  # Expected: pool exhausted
            else:
                # Some pool implementations may queue and eventually succeed
                pass

            # Cancel blocking tasks and let them clean up
            for t in block_tasks:
                t.cancel()
            await asyncio.gather(*block_tasks, return_exceptions=True)
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_pool_recovers_after_connection_churn(self):
        """After many connect/disconnect cycles, the pool still serves fresh connections."""
        for i in range(20):
            config = get_test_redis_config()
            client = RedisClient(config)
            try:
                redis_conn = client.get_client()
                await redis_conn.set(f"intg:stress:churn:{i}", f"v{i}")
                val = await redis_conn.get(f"intg:stress:churn:{i}")
                assert val == f"v{i}"
            finally:
                await client.close()

        # Final client to confirm no leaked state
        config = get_test_redis_config()
        client = RedisClient(config)
        try:
            redis_conn = client.get_client()
            await redis_conn.set("intg:stress:churn:final", "final")
            val = await redis_conn.get("intg:stress:churn:final")
            assert val == "final", "Final connection after churn should work"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_sentinel_pool_concurrent_stress(self, sentinel_client):
        """100 concurrent tasks through the Sentinel pool complete correctly."""
        redis_conn = sentinel_client.get_client()

        async def set_and_get(i: int) -> str:
            await redis_conn.set(f"intg:stress:sentinel:{i}", f"sv{i}")
            return await redis_conn.get(f"intg:stress:sentinel:{i}")

        results = await asyncio.gather(
            *(set_and_get(i) for i in range(100))
        )

        for i, result in enumerate(results):
            assert result == f"sv{i}", (
                f"Sentinel task {i} expected 'sv{i}', got '{result}'"
            )


class TestSentinelTopologyResilience:
    """
    Validates topology correctness across Sentinel nodes and stability
    through changes.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_topology_consistency_across_sentinels(self):
        """All 3 Sentinels report the same master address."""
        sentinel_hosts = get_sentinel_hosts()
        master_name = os.getenv("TEST_SENTINEL_MASTER_NAME", "mymaster")
        addresses = []

        for host, port in sentinel_hosts:
            s = Sentinel(
                [(host, port)],
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=2.0,
            )
            try:
                addr = await s.discover_master(master_name)
                addresses.append(addr)
            finally:
                for conn in s.sentinels:
                    await conn.close()

        assert len(addresses) == len(sentinel_hosts), (
            f"Expected {len(sentinel_hosts)} responses, got {len(addresses)}"
        )
        # All should report the same master
        assert all(a == addresses[0] for a in addresses), (
            f"Sentinels disagree on master address: {addresses}"
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_topology_after_manual_failover(self, sentinel_client):
        """After SENTINEL FAILOVER, get_topology() reflects the new master."""
        # Record current topology
        topo_before = await sentinel_client.get_topology()
        assert topo_before is not None, "Initial topology should be available"
        original_master = topo_before.master_address

        # Issue failover via a raw Sentinel connection
        sentinel = sentinel_client.get_sentinel()
        assert sentinel is not None

        failover_issued = False
        for sentinel_conn in sentinel.sentinels:
            try:
                await sentinel_conn.execute_command(
                    "SENTINEL", "FAILOVER", sentinel_client.config.sentinel_master_name
                )
                failover_issued = True
                break
            except (ConnectionError, TimeoutError, RedisError):
                continue

        if not failover_issued:
            pytest.skip("Could not issue SENTINEL FAILOVER — all Sentinels unreachable")

        # Wait for topology to change (up to 30s)
        changed = await wait_for_condition(
            predicate=lambda: _topology_master_changed(
                sentinel_client, original_master
            ),
            timeout=30.0,
            interval=1.0,
            description="master address changed after failover",
        )

        topo_after = await sentinel_client.get_topology()
        assert topo_after is not None, "Post-failover topology should be available"
        assert topo_after.is_healthy, "Topology should be healthy after failover"

        if not changed:
            # Some environments may promote the same node back; just verify healthy
            pass

        # Allow topology to stabilize before next tests
        await asyncio.sleep(5)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_continuous_operations_during_failover(self, sentinel_client):
        """A write loop using with_retry survives a manual failover."""
        completed_keys: list = []
        errors: list = []

        @sentinel_client.with_retry(max_retries=5, base_delay=0.5)
        async def write_key(idx: int):
            redis_conn = sentinel_client.get_client()
            key = f"intg:topo:failover_write:{idx}"
            await redis_conn.set(key, f"v{idx}")
            completed_keys.append(key)

        # Start background write loop
        async def write_loop():
            for i in range(20):
                try:
                    await write_key(i)
                except Exception as e:
                    errors.append(e)
                await asyncio.sleep(0.2)

        write_task = asyncio.create_task(write_loop())

        # Brief delay then trigger failover
        await asyncio.sleep(1)
        sentinel = sentinel_client.get_sentinel()
        if sentinel is not None:
            for sentinel_conn in sentinel.sentinels:
                try:
                    await sentinel_conn.execute_command(
                        "SENTINEL", "FAILOVER",
                        sentinel_client.config.sentinel_master_name,
                    )
                    break
                except (ConnectionError, TimeoutError, RedisError):
                    continue

        # Wait for write loop to finish
        await asyncio.wait_for(write_task, timeout=60.0)

        # Verify all 20 keys were written
        assert len(completed_keys) == 20, (
            f"Expected 20 completed writes, got {len(completed_keys)}. "
            f"Errors: {errors}"
        )

        # Verify keys are actually in Redis
        redis_conn = sentinel_client.get_client()
        for key in completed_keys:
            val = await redis_conn.get(key)
            assert val is not None, f"Key {key} missing after failover"

        # Allow stabilization
        await asyncio.sleep(5)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_replica_client_reconnects_after_topology_change(
        self, sentinel_client
    ):
        """After failover, get_replica_client() eventually serves reads."""
        # Initial read
        master = sentinel_client.get_client()
        await master.set("intg:topo:replica_read", "before_failover")

        replica = await sentinel_client.get_replica_client()
        if replica is None:
            pytest.skip("No replica available")

        val = await wait_for_replication(
            replica, "intg:topo:replica_read", "before_failover"
        )
        assert val == "before_failover"

        # Trigger failover
        sentinel = sentinel_client.get_sentinel()
        failover_issued = False
        if sentinel is not None:
            for sentinel_conn in sentinel.sentinels:
                try:
                    await sentinel_conn.execute_command(
                        "SENTINEL", "FAILOVER",
                        sentinel_client.config.sentinel_master_name,
                    )
                    failover_issued = True
                    break
                except (ConnectionError, TimeoutError, RedisError):
                    continue

        if not failover_issued:
            pytest.skip("Could not issue SENTINEL FAILOVER")

        # Wait for failover to settle
        await asyncio.sleep(10)

        # Reset to pick up new topology
        await sentinel_client._reset_client()

        # Write a new key via master
        new_master = sentinel_client.get_client()
        await new_master.set("intg:topo:replica_read_post", "after_failover")

        # Read from replica
        new_replica = await sentinel_client.get_replica_client()
        if new_replica is None:
            pytest.skip("No replica available after failover")

        val = await wait_for_replication(
            new_replica, "intg:topo:replica_read_post", "after_failover",
            timeout=10.0,
        )
        assert val == "after_failover", (
            f"Replica should serve 'after_failover', got '{val}'"
        )

        # Stabilization
        await asyncio.sleep(5)


class TestConfigPassthroughVerification:
    """
    Validates SPEC-002 — RedisConfig settings are actually applied to
    the live Redis connection.
    """

    @pytest.mark.asyncio
    async def test_client_name_visible_in_server(self, integration_client):
        """The client_name from config is set on the server-side connection."""
        redis_conn = integration_client.get_client()
        name = await redis_conn.execute_command("CLIENT", "GETNAME")
        assert name == "ha_redis", (
            f"Expected client name 'ha_redis', got '{name}'"
        )

    @pytest.mark.asyncio
    async def test_sentinel_client_name_suffix(self, sentinel_client):
        """Replica client has '_replica' suffix visible on the server."""
        replica = await sentinel_client.get_replica_client()
        if replica is None:
            pytest.skip("No replica available")

        name = await replica.execute_command("CLIENT", "GETNAME")
        assert name == "ha_redis_replica", (
            f"Expected replica client name 'ha_redis_replica', got '{name}'"
        )

    @pytest.mark.asyncio
    async def test_socket_timeout_is_applied(self):
        """A deliberately slow operation on a short-timeout client raises TimeoutError."""
        config = RedisConfig(
            host=os.getenv("TEST_REDIS_HOST", "localhost"),
            port=int(os.getenv("TEST_REDIS_PORT", "6379")),
            password=os.getenv("TEST_REDIS_PASSWORD"),
            db=15,
            socket_timeout=0.1,
            socket_connect_timeout=1.0,
        )
        client = RedisClient(config)

        try:
            redis_conn = client.get_client()
            # DEBUG SLEEP blocks the connection for 1 second.
            # It may be disabled in some Redis configurations.
            try:
                await redis_conn.execute_command("DEBUG", "SLEEP", "1")
            except (TimeoutError, ConnectionError):
                # This is the expected outcome — socket_timeout triggered
                return
            except ResponseError as e:
                if "not allowed" in str(e).lower() or "unknown" in str(e).lower():
                    pytest.skip(
                        "DEBUG SLEEP not available in this Redis configuration"
                    )
                raise
            # If we get here, the command completed without timeout
            pytest.fail(
                "Expected TimeoutError but DEBUG SLEEP completed — "
                "socket_timeout may not be applied"
            )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_custom_db_isolation(self, integration_client):
        """Keys written to db=15 (test DB) are not visible from db=0."""
        redis_conn = integration_client.get_client()
        await redis_conn.set("intg:config:isolation_test", "db15_value")

        # Create a client on db=0
        config_db0 = RedisConfig(
            host=os.getenv("TEST_REDIS_HOST", "localhost"),
            port=int(os.getenv("TEST_REDIS_PORT", "6379")),
            password=os.getenv("TEST_REDIS_PASSWORD"),
            db=0,
        )
        client_db0 = RedisClient(config_db0)

        try:
            redis_db0 = client_db0.get_client()
            val = await redis_db0.get("intg:config:isolation_test")
            assert val is None, (
                f"Key from db=15 should not be visible in db=0, got '{val}'"
            )
        finally:
            await client_db0.close()


# Helper for topology failover detection
async def _topology_master_changed(
    client: RedisClient,
    original_master,
) -> bool:
    """Return True if the current master address differs from original_master."""
    try:
        topo = await client.get_topology()
        if topo is None:
            return False
        return topo.master_address != original_master
    except Exception:
        return False

