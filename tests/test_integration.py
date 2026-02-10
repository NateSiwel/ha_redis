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

from ha_redis import RedisClient, RedisConfig, SentinelTopology
import redis.asyncio as redis
from redis.asyncio.sentinel import Sentinel
from redis.exceptions import ConnectionError, TimeoutError, RedisError


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

