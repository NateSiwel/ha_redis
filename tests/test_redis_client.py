"""
Tests for RedisClient class.

Covers:
- Initialization
- Client creation (direct and Sentinel modes)
- Connection pool creation
- Health checks
- Resource cleanup
- Context manager support
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import logging

from ha_redis import RedisClient, RedisConfig


class TestRedisClientInitialization:
    """Test RedisClient initialization."""
    
    def test_init_with_default_config(self, redis_client):
        """Client should initialize with default config."""
        assert redis_client.config is not None
        assert redis_client.config.host == "localhost"
    
    def test_init_with_custom_config(self, redis_client_with_custom_config):
        """Client should initialize with custom config."""
        assert redis_client_with_custom_config.config.host == "redis.example.com"
    
    def test_init_without_config(self):
        """Client should use default config when none provided."""
        client = RedisClient()
        assert client.config.host == "localhost"
        assert client.config.port == 6379
    
    def test_init_with_custom_logger(self, default_config, mock_logger):
        """Client should use provided logger."""
        client = RedisClient(default_config, logger=mock_logger)
        assert client.logger is mock_logger
    
    def test_init_creates_default_logger(self, redis_client):
        """Client should create logger if none provided."""
        assert redis_client.logger is not None
        assert isinstance(redis_client.logger, logging.Logger)
    
    def test_init_state(self, redis_client):
        """Initial state should be uninitialized."""
        assert redis_client._pool is None
        assert redis_client._client is None
        assert redis_client._sentinel is None
        assert redis_client._initialized is False


class TestRedisClientDirectMode:
    """Test RedisClient in direct connection mode."""
    
    @patch('ha_redis.redis.ConnectionPool.from_url')
    @patch('ha_redis.redis.Redis')
    def test_get_client_creates_pool(
        self,
        mock_redis_class,
        mock_pool_from_url,
        redis_client,
    ):
        """get_client should create connection pool on first call."""
        mock_pool = MagicMock()
        mock_pool_from_url.return_value = mock_pool
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        client = redis_client.get_client()
        
        mock_pool_from_url.assert_called_once()
        assert redis_client._pool is mock_pool
    
    @patch('ha_redis.redis.ConnectionPool.from_url')
    @patch('ha_redis.redis.Redis')
    def test_get_client_returns_redis_instance(
        self,
        mock_redis_class,
        mock_pool_from_url,
        redis_client,
    ):
        """get_client should return Redis instance."""
        mock_pool = MagicMock()
        mock_pool_from_url.return_value = mock_pool
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        client = redis_client.get_client()
        
        assert client is mock_client
    
    @patch('ha_redis.redis.ConnectionPool.from_url')
    @patch('ha_redis.redis.Redis')
    def test_get_client_reuses_existing(
        self,
        mock_redis_class,
        mock_pool_from_url,
        redis_client,
    ):
        """get_client should reuse existing client."""
        mock_pool = MagicMock()
        mock_pool_from_url.return_value = mock_pool
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        client1 = redis_client.get_client()
        client2 = redis_client.get_client()
        
        assert client1 is client2
        # Pool should only be created once
        assert mock_pool_from_url.call_count == 1
    
    @patch('ha_redis.redis.ConnectionPool.from_url')
    @patch('ha_redis.redis.Redis')
    def test_get_client_sets_initialized(
        self,
        mock_redis_class,
        mock_pool_from_url,
        redis_client,
    ):
        """get_client should set _initialized to True."""
        mock_pool_from_url.return_value = MagicMock()
        mock_redis_class.return_value = MagicMock()
        
        redis_client.get_client()
        
        assert redis_client._initialized is True


class TestRedisClientSentinelMode:
    """Test RedisClient in Sentinel (HA) mode."""
    
    @patch('ha_redis.Sentinel')
    def test_get_client_creates_sentinel(
        self,
        mock_sentinel_class,
        sentinel_client,
    ):
        """get_client should create Sentinel in HA mode."""
        mock_sentinel = MagicMock()
        mock_sentinel.master_for.return_value = MagicMock()
        mock_sentinel_class.return_value = mock_sentinel
        
        sentinel_client.get_client()
        
        mock_sentinel_class.assert_called_once()
        assert sentinel_client._sentinel is mock_sentinel
    
    @patch('ha_redis.Sentinel')
    def test_get_client_uses_master_for(
        self,
        mock_sentinel_class,
        sentinel_client,
    ):
        """get_client should call master_for with correct name."""
        mock_sentinel = MagicMock()
        mock_master = MagicMock()
        mock_sentinel.master_for.return_value = mock_master
        mock_sentinel_class.return_value = mock_sentinel
        
        client = sentinel_client.get_client()
        
        mock_sentinel.master_for.assert_called_once_with(
            "mymaster",
            db=0,
            ssl=False,
            password="sentinel_password",
            socket_timeout=5.0,
            socket_connect_timeout=2.0,
            max_connections=100,
            health_check_interval=30,
        )
        assert client is mock_master
    
    @patch('ha_redis.Sentinel')
    def test_sentinel_config_passed_correctly(
        self,
        mock_sentinel_class,
        sentinel_client,
    ):
        """Sentinel should be configured with correct hosts."""
        mock_sentinel = MagicMock()
        mock_sentinel.master_for.return_value = MagicMock()
        mock_sentinel_class.return_value = mock_sentinel
        
        sentinel_client.get_client()
        
        call_args = mock_sentinel_class.call_args
        hosts = call_args[0][0]
        assert len(hosts) == 3
        assert hosts[0] == ("sentinel1.example.com", 26379)

    @patch('ha_redis.Sentinel')
    def test_get_client_sentinel_custom_pool_config(
        self,
        mock_sentinel_class,
    ):
        """Custom RedisConfig pool values should propagate through master_for()."""
        config = RedisConfig(
            use_sentinel=True,
            sentinel_hosts=[("sentinel1", 26379)],
            sentinel_master_name="mymaster",
            password="mypass",
            db=2,
            ssl=True,
            max_connections=200,
            socket_timeout=15.0,
            socket_connect_timeout=8.0,
            health_check_interval=120,
        )
        client = RedisClient(config)

        mock_sentinel_instance = MagicMock()
        mock_sentinel_instance.master_for.return_value = MagicMock()
        mock_sentinel_class.return_value = mock_sentinel_instance

        client.get_client()

        mock_sentinel_instance.master_for.assert_called_once_with(
            "mymaster",
            db=2,
            ssl=True,
            password="mypass",
            socket_timeout=15.0,
            socket_connect_timeout=8.0,
            max_connections=200,
            health_check_interval=120,
        )

    @patch('ha_redis.Sentinel')
    def test_get_replica_client_forwards_pool_config(
        self,
        mock_sentinel_class,
    ):
        """slave_for() should receive all pool config kwargs."""
        config = RedisConfig(
            use_sentinel=True,
            sentinel_hosts=[("sentinel1", 26379)],
            sentinel_master_name="mymaster",
            password="replicapass",
            db=3,
            ssl=True,
            max_connections=75,
            socket_timeout=7.0,
            socket_connect_timeout=3.0,
            health_check_interval=45,
        )
        client = RedisClient(config)

        mock_sentinel_instance = MagicMock()
        mock_sentinel_instance.slave_for.return_value = MagicMock()
        mock_sentinel_class.return_value = mock_sentinel_instance

        import asyncio
        asyncio.get_event_loop().run_until_complete(client.get_replica_client())

        mock_sentinel_instance.slave_for.assert_called_once_with(
            "mymaster",
            db=3,
            ssl=True,
            password="replicapass",
            socket_timeout=7.0,
            socket_connect_timeout=3.0,
            max_connections=75,
            health_check_interval=45,
        )

    @patch('ha_redis.Sentinel')
    def test_sentinel_protocol_timeouts_configurable(
        self,
        mock_sentinel_class,
    ):
        """_create_sentinel() should use sentinel_socket_timeout and sentinel_socket_connect_timeout."""
        config = RedisConfig(
            use_sentinel=True,
            sentinel_hosts=[("sentinel1", 26379)],
            sentinel_master_name="mymaster",
            sentinel_socket_timeout=2.5,
            sentinel_socket_connect_timeout=1.5,
        )
        client = RedisClient(config)

        mock_sentinel_instance = MagicMock()
        mock_sentinel_instance.master_for.return_value = MagicMock()
        mock_sentinel_class.return_value = mock_sentinel_instance

        client.get_client()

        call_kwargs = mock_sentinel_class.call_args[1]
        assert call_kwargs["socket_timeout"] == 2.5
        assert call_kwargs["socket_connect_timeout"] == 1.5
        # max_connections should NOT be passed to Sentinel()
        assert "max_connections" not in call_kwargs

    @patch('ha_redis.Sentinel')
    def test_sentinel_password_forwarded_to_data_client(
        self,
        mock_sentinel_class,
    ):
        """password should be forwarded to both master_for() and slave_for()."""
        config = RedisConfig(
            use_sentinel=True,
            sentinel_hosts=[("sentinel1", 26379)],
            sentinel_master_name="mymaster",
            password="data_password",
        )
        client = RedisClient(config)

        mock_sentinel_instance = MagicMock()
        mock_sentinel_instance.master_for.return_value = MagicMock()
        mock_sentinel_instance.slave_for.return_value = MagicMock()
        mock_sentinel_class.return_value = mock_sentinel_instance

        # Verify master_for receives password
        client.get_client()
        master_call_kwargs = mock_sentinel_instance.master_for.call_args[1]
        assert master_call_kwargs["password"] == "data_password"

        # Verify slave_for receives password
        import asyncio
        asyncio.get_event_loop().run_until_complete(client.get_replica_client())
        slave_call_kwargs = mock_sentinel_instance.slave_for.call_args[1]
        assert slave_call_kwargs["password"] == "data_password"


class TestRedisClientHealthCheck:
    """Test health check functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, mocked_redis_client):
        """Health check should return True when Redis responds."""
        result = await mocked_redis_client.health_check()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_connection_error(
        self,
        mocked_redis_client,
        connection_error,
    ):
        """Health check should return False on ConnectionError."""
        mocked_redis_client._client.ping = AsyncMock(side_effect=connection_error)
        
        result = await mocked_redis_client.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_timeout_error(
        self,
        mocked_redis_client,
        timeout_error,
    ):
        """Health check should return False on TimeoutError."""
        mocked_redis_client._client.ping = AsyncMock(side_effect=timeout_error)
        
        result = await mocked_redis_client.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_busy_loading_error(
        self,
        mocked_redis_client,
        busy_loading_error,
    ):
        """Health check should return False on BusyLoadingError."""
        mocked_redis_client._client.ping = AsyncMock(side_effect=busy_loading_error)
        
        result = await mocked_redis_client.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_generic_redis_error(
        self,
        mocked_redis_client,
        redis_error,
    ):
        """Health check should return False on generic RedisError."""
        mocked_redis_client._client.ping = AsyncMock(side_effect=redis_error)
        
        result = await mocked_redis_client.health_check()
        
        assert result is False


class TestRedisClientClose:
    """Test resource cleanup."""
    
    @pytest.mark.asyncio
    async def test_close_client(self, mocked_redis_client):
        """close should close the Redis client."""
        await mocked_redis_client.close()
        
        assert mocked_redis_client._client is None
    
    @pytest.mark.asyncio
    async def test_close_pool(self, mocked_redis_client):
        """close should disconnect the connection pool."""
        await mocked_redis_client.close()
        
        assert mocked_redis_client._pool is None
    
    @pytest.mark.asyncio
    async def test_close_sentinel(self, mocked_sentinel_client):
        """close should close Sentinel connections."""
        await mocked_sentinel_client.close()
        
        assert mocked_sentinel_client._sentinel is None
    
    @pytest.mark.asyncio
    async def test_close_resets_initialized(self, mocked_redis_client):
        """close should reset _initialized flag."""
        await mocked_redis_client.close()
        
        assert mocked_redis_client._initialized is False
    
    @pytest.mark.asyncio
    async def test_close_idempotent(self, redis_client):
        """close should be safe to call multiple times."""
        await redis_client.close()
        await redis_client.close()
        
        assert redis_client._client is None


class TestRedisClientContextManager:
    """Test async context manager support."""
    
    @pytest.mark.asyncio
    async def test_context_manager_enter(self, default_config):
        """__aenter__ should return the client."""
        client = RedisClient(default_config)
        
        async with client as ctx:
            assert ctx is client
    
    @pytest.mark.asyncio
    async def test_context_manager_exit_closes(self, default_config):
        """__aexit__ should close resources."""
        client = RedisClient(default_config)
        client._client = AsyncMock()
        client._pool = AsyncMock()
        
        async with client:
            pass
        
        assert client._client is None
        assert client._pool is None
    
    @pytest.mark.asyncio
    async def test_context_manager_exit_on_exception(self, default_config):
        """__aexit__ should close resources even on exception."""
        client = RedisClient(default_config)
        client._client = AsyncMock()
        client._pool = AsyncMock()
        
        with pytest.raises(ValueError):
            async with client:
                raise ValueError("Test error")
        
        assert client._client is None


class TestRedisClientLogging:
    """Test logging behavior."""
    
    @patch('ha_redis.redis.ConnectionPool.from_url')
    @patch('ha_redis.redis.Redis')
    def test_pool_creation_logs_info(
        self,
        mock_redis_class,
        mock_pool_from_url,
        redis_client_with_mock_logger,
        mock_logger,
    ):
        """Pool creation should log info message."""
        mock_pool_from_url.return_value = MagicMock()
        mock_redis_class.return_value = MagicMock()
        
        redis_client_with_mock_logger.get_client()
        
        mock_logger.info.assert_called()
    
    @pytest.mark.asyncio
    async def test_close_logs_info(
        self,
        redis_client_with_mock_logger,
        mock_logger,
    ):
        """close should log info message."""
        await redis_client_with_mock_logger.close()
        
        mock_logger.info.assert_called()
