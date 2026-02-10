"""
Redis Client Module

Provides a resilient, connection-pooled Redis client with:
- Automatic retry with exponential backoff
- Connection health checks
- Support for both direct and Sentinel (HA) modes

"""

import logging
import asyncio
import random
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable, Any
from functools import wraps
import redis.asyncio as redis
from redis.asyncio.sentinel import Sentinel
from redis.exceptions import (
    ConnectionError,
    TimeoutError,
    BusyLoadingError,
    ReadOnlyError,
    RedisError
)

RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    ReadOnlyError,
    BusyLoadingError,
)


@dataclass
class SentinelTopology:
    """
    Represents the current Sentinel topology state.
    
    Provides a snapshot of the HA cluster's master, replicas, and Sentinel nodes.
    Useful for monitoring, debugging, and health reporting.
    
    Attributes:
        master_name: Name of the monitored master
        master_address: Tuple of (host, port) for current master
        replica_addresses: List of (host, port) tuples for replicas
        sentinel_count: Number of Sentinel nodes monitoring the master
        is_healthy: Whether the master is in a healthy state
        quorum: Required number of Sentinels for failover consensus
    """
    master_name: str
    master_address: Optional[Tuple[str, int]] = None
    replica_addresses: List[Tuple[str, int]] = field(default_factory=list)
    sentinel_count: int = 0
    is_healthy: bool = False
    quorum: int = 2


@dataclass
class RedisConfig:
    """
    Configuration for Redis client.
    
    Supports both direct connection and Sentinel modes.
    
    Attributes:
        host: Redis server hostname (for direct mode)
        port: Redis server port (for direct mode)
        password: Optional Redis password
        db: Redis database number
        max_connections: Maximum pool connections
        socket_timeout: Timeout for socket operations
        socket_connect_timeout: Timeout for establishing connections
        health_check_interval: Interval for connection health checks
        use_sentinel: Whether to use Sentinel for HA
        sentinel_hosts: List of (host, port) tuples for Sentinel nodes
        sentinel_master_name: Name of the Sentinel master
        ssl: Whether to use SSL/TLS
        retry_attempts: Number of retry attempts for operations
        retry_base_delay: Base delay for exponential backoff
        sentinel_password: Optional password for Sentinel nodes themselves
                           (separate from data node password)
    """
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 2.0
    health_check_interval: int = 30
    use_sentinel: bool = False
    sentinel_hosts: List[Tuple[str, int]] = field(default_factory=list)
    sentinel_master_name: str = "mymaster"
    ssl: bool = False
    retry_attempts: int = 3
    retry_base_delay: float = 0.1
    # Sentinel discovery protocol timeouts (intentionally short)
    sentinel_socket_timeout: float = 1.0
    sentinel_socket_connect_timeout: float = 1.0
    sentinel_password: Optional[str] = None
    client_name: str = "ha_redis"

    @property
    def url(self) -> str:
        """Generate Redis URL from configuration."""
        scheme = "rediss" if self.ssl else "redis"
        auth = ""
        if self.password:
            encoded_password = urllib.parse.quote(self.password)
            auth = f":{encoded_password}@"
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"


class RedisClient:
    """
    A resilient, async Redis client with connection pooling and HA support.
    
    Features:
    - Automatic retry with exponential backoff
    - Connection health checks
    - Support for both direct and Sentinel (HA) modes
    - Context manager support for proper resource cleanup
    
    Recommended usage (async context manager)::
    
        # Direct connection
        async with RedisClient(RedisConfig(host="localhost")) as client:
            redis = client.get_client()
            await redis.set("key", "value")
        
        # Sentinel (HA) mode
        config = RedisConfig(
            use_sentinel=True,
            sentinel_hosts=[("sentinel1", 26379), ("sentinel2", 26379)],
            sentinel_master_name="mymaster",
        )
        async with RedisClient(config) as client:
            redis = client.get_client()
            await redis.set("key", "value")
    
    Without a context manager, call ``initialize()`` before concurrent use::
    
        client = RedisClient(config)
        await client.initialize()  # acquires lock, safe for concurrency
        # get_client() is now a lock-free fast path
        redis = client.get_client()
        ...
        await client.close()  # clean up when done
    """
    
    def __init__(
        self,
        config: Optional[RedisConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Redis client.
        
        Args:
            config: Redis configuration. Uses defaults if not provided.
            logger: Custom logger instance. Creates one if not provided.
        """
        self.config = config or RedisConfig()
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._replica_client: Optional[redis.Redis] = None
        self._sentinel: Optional[Sentinel] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    def _create_pool(self) -> redis.ConnectionPool:
        """
        Create a connection pool with production-ready settings.
        
        Note: Internal command retries are disabled to avoid multiplicative
        retry behavior when combined with the with_retry decorator.
        Use with_retry for application-level retry logic.
        
        Returns:
            Configured Redis connection pool
        """
        pool = redis.ConnectionPool.from_url(
            self.config.url,
            decode_responses=True,
            max_connections=self.config.max_connections,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            retry_on_timeout=False,
            health_check_interval=self.config.health_check_interval,
            client_name=self.config.client_name,
        )
        
        self.logger.info(
            f"Redis pool created: {self.config.host}:{self.config.port} "
            f"(max_connections={self.config.max_connections})"
        )
        
        return pool
    
    def _create_sentinel(self) -> Sentinel:
        """
        Create a Sentinel instance for HA mode.
        
        Note: Internal command retries are disabled to avoid multiplicative
        retry behavior when combined with the with_retry decorator.
        Use with_retry for application-level retry logic.
        
        Returns:
            Configured Sentinel instance
        """
        # Build sentinel_kwargs for Sentinel node connections (authentication + TLS)
        sentinel_kwargs = {}
        if self.config.sentinel_password is not None:
            sentinel_kwargs["password"] = self.config.sentinel_password
        if self.config.ssl:
            sentinel_kwargs["ssl"] = True
        
        # Identify our client to Sentinel nodes
        sentinel_kwargs["client_name"] = f"{self.config.client_name}_sentinel"

        sentinel = Sentinel(
            self.config.sentinel_hosts,
            decode_responses=True,
            socket_timeout=self.config.sentinel_socket_timeout,
            socket_connect_timeout=self.config.sentinel_socket_connect_timeout,
            retry_on_timeout=False,
            password=self.config.password,
            ssl=self.config.ssl,
            **({"sentinel_kwargs": sentinel_kwargs} if sentinel_kwargs else {}),
        )
        
        self.logger.info(
            f"Redis Sentinel initialized with hosts: {self.config.sentinel_hosts} "
            f"(master={self.config.sentinel_master_name}, "
            f"max_connections={self.config.max_connections})"
        )
        
        return sentinel
    
    async def initialize(self) -> None:
        """
        Pre-initialize the Redis client and connection pool.

        Safe to call concurrently from multiple async tasks. Uses a lock
        to ensure initialization happens exactly once.

        This is called automatically when using the async context manager.
        For manual instantiation, call this method during application
        startup before issuing concurrent operations.

        Example:
            client = RedisClient(config)
            await client.initialize()
            # Now safe to call get_client() from any task
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return
            self.get_client()   # Synchronous — runs fully under the lock

    def get_client(self) -> redis.Redis:
        """
        Returns a resilient async Redis client instance.
        
        The client uses a shared connection pool and includes:
        - Automatic retry with exponential backoff
        - Connection health checks
        - Timeout protection
        
        Concurrency note:
            This method is synchronous and does not acquire ``_init_lock``.
            For concurrent startup, call ``await initialize()`` (or use the
            ``async with`` context manager) **before** issuing parallel
            operations. After initialization, this method is a lock-free
            fast-path that simply returns the cached client.
        
        Returns:
            Async Redis client instance
        """
        if self._client is not None:
            return self._client
        
        if self.config.use_sentinel:
            # High Availability mode: Use Sentinel for automatic failover
            if self._sentinel is None:
                self._sentinel = self._create_sentinel()
            
            self._client = self._sentinel.master_for(
                self.config.sentinel_master_name,
                db=self.config.db,
                ssl=self.config.ssl,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                max_connections=self.config.max_connections,
                health_check_interval=self.config.health_check_interval,
                client_name=self.config.client_name,
            )
            self.logger.info(
                f"Connected to Sentinel master: {self.config.sentinel_master_name}"
            )
        else:
            # Direct connection mode
            if self._pool is None:
                self._pool = self._create_pool()
            
            self._client = redis.Redis(connection_pool=self._pool)
        
        self._initialized = True
        return self._client
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the Redis connection.
        
        Returns:
            True if Redis is healthy and responding, False otherwise
        """
        try:
            client = self.get_client()
            result = await client.ping()
            return result is True
        except (ConnectionError, TimeoutError, BusyLoadingError) as e:
            self.logger.error(f"Redis health check failed: {type(e).__name__}: {e}")
            return False
        except RedisError as e:
            self.logger.error(f"Redis error during health check: {e}")
            return False
    
    async def close(self) -> None:
        """
        Gracefully close the Redis connection pool.
        
        Call this during application shutdown.
        Handles both Sentinel mode and direct connection mode cleanup.
        """
        async with self._init_lock:
            if self._client is not None:
                await self._client.close()
                self._client = None
            
            if self._replica_client is not None:
                await self._replica_client.close()
                self._replica_client = None
            
            if self._pool is not None:
                await self._pool.disconnect()
                self._pool = None
            
            if self._sentinel is not None:
                # Sentinel doesn't have a close() method, but we can close
                # the underlying sentinel connections
                for sentinel_conn in self._sentinel.sentinels:
                    await sentinel_conn.close()
                self._sentinel = None
                self.logger.info("Redis Sentinel connections closed")
            
            self._initialized = False
            self.logger.info("Redis connection pool closed")
    
    async def _reset_client(self) -> None:
        """
        Reset the cached clients to force re-discovery via Sentinel.

        This is a defensive fallback for edge cases where redis-py's built-in
        SentinelManagedConnection failover handling cannot recover (e.g.,
        delayed Sentinel convergence, pool state corruption).

        Both the master and replica clients are torn down. The Sentinel
        instance is preserved to avoid redundant discovery connections.

        Thread-safety: Acquires _init_lock to prevent races with
        initialize(), get_replica_client(), and close().
        """
        async with self._init_lock:
            if self._client is not None:
                try:
                    await self._client.close()
                except Exception:
                    pass
                self._client = None
            if self._replica_client is not None:
                try:
                    await self._replica_client.close()
                except Exception:
                    pass
                self._replica_client = None
            self._initialized = False
            self.logger.info("Client reset — will re-discover on next operation")
    
    # =========================================================================
    # Sentinel-specific Operations
    # =========================================================================
    
    def is_sentinel_mode(self) -> bool:
        """
        Check if the client is configured for Sentinel HA mode.
        
        Returns:
            True if using Sentinel, False for direct connection
        """
        return self.config.use_sentinel
    
    def get_sentinel(self) -> Optional[Sentinel]:
        """
        Get the underlying Sentinel instance.
        
        Returns:
            Sentinel instance if in Sentinel mode, None otherwise
        """
        if not self.config.use_sentinel:
            return None
        
        if self._sentinel is None:
            self._sentinel = self._create_sentinel()
        
        return self._sentinel
    
    async def discover_master(self) -> Optional[Tuple[str, int]]:
        """
        Discover the current master address via Sentinel.
        
        Returns:
            Tuple of (host, port) for the master, or None if not in Sentinel mode
            or discovery fails.
        """
        if not self.config.use_sentinel:
            self.logger.warning("discover_master() called but not in Sentinel mode")
            return None
        
        try:
            sentinel = self.get_sentinel()
            master_addr = await sentinel.discover_master(self.config.sentinel_master_name)
            return master_addr
        except (ConnectionError, TimeoutError, RedisError) as e:
            self.logger.error(f"Failed to discover master: {type(e).__name__}: {e}")
            return None
    
    async def discover_replicas(self) -> List[Tuple[str, int]]:
        """
        Discover replica addresses via Sentinel.
        
        Returns:
            List of (host, port) tuples for replicas. Empty list if not in
            Sentinel mode or discovery fails.
        """
        if not self.config.use_sentinel:
            self.logger.warning("discover_replicas() called but not in Sentinel mode")
            return []
        
        try:
            sentinel = self.get_sentinel()
            replicas = await sentinel.discover_slaves(self.config.sentinel_master_name)
            return replicas if replicas else []
        except (ConnectionError, TimeoutError, RedisError) as e:
            self.logger.error(f"Failed to discover replicas: {type(e).__name__}: {e}")
            return []
    
    async def get_topology(self) -> Optional[SentinelTopology]:
        """
        Get a snapshot of the current Sentinel HA topology.
        
        Provides comprehensive information about the master, replicas,
        and Sentinel cluster health. Useful for monitoring and debugging.
        
        This method is optimized to query a single Sentinel for all topology
        information, minimizing network round-trips by:
        - Reusing existing Sentinel connections from the connection pool
        - Batching SENTINEL MASTER and SENTINEL REPLICAS commands in a pipeline
        
        Returns:
            SentinelTopology object with current state, or None if not
            in Sentinel mode or query fails.
        """
        if not self.config.use_sentinel:
            self.logger.warning("get_topology() called but not in Sentinel mode")
            return None
        
        # Ensure Sentinel is initialized
        sentinel = self.get_sentinel()
        if sentinel is None:
            return None
        
        # Try each existing Sentinel connection until one responds successfully
        for sentinel_conn in sentinel.sentinels:
            try:
                # Use pipeline to batch both SENTINEL commands into a single round-trip
                pipe = sentinel_conn.pipeline(transaction=False)
                pipe.execute_command(
                    "SENTINEL", "MASTER", self.config.sentinel_master_name
                )
                pipe.execute_command(
                    "SENTINEL", "REPLICAS", self.config.sentinel_master_name
                )
                results = await pipe.execute()
                
                master_info = results[0]
                replicas_info = results[1]
                
                # Helper to decode bytes to str for consistent key lookup
                def decode_value(v):
                    return v.decode("utf-8") if isinstance(v, bytes) else v
                
                # Parse master info
                master_addr = None
                is_healthy = False
                quorum = 2
                sentinel_count = 0
                
                if isinstance(master_info, list):
                    # Decode all keys and values for consistent access
                    info_dict = {
                        decode_value(k): decode_value(v)
                        for k, v in zip(master_info[::2], master_info[1::2])
                    }
                    
                    # Extract master address
                    master_host = info_dict.get("ip")
                    master_port = info_dict.get("port")
                    if master_host and master_port:
                        master_addr = (master_host, int(master_port))
                    
                    # Extract health and cluster info
                    flags_str = info_dict.get("flags", "")
                    flags = flags_str.split(",") if isinstance(flags_str, str) else []
                    
                    is_healthy = (
                        "master" in flags and 
                        "s_down" not in flags and 
                        "o_down" not in flags
                    )
                    quorum = int(info_dict.get("quorum", 2))
                    sentinel_count = int(info_dict.get("num-other-sentinels", 0)) + 1
                
                # Parse replica addresses
                replica_addrs = []
                if isinstance(replicas_info, list):
                    for replica in replicas_info:
                        if isinstance(replica, list):
                            replica_dict = {
                                decode_value(k): decode_value(v)
                                for k, v in zip(replica[::2], replica[1::2])
                            }
                            replica_host = replica_dict.get("ip")
                            replica_port = replica_dict.get("port")
                            if replica_host and replica_port:
                                replica_addrs.append((replica_host, int(replica_port)))
                
                return SentinelTopology(
                    master_name=self.config.sentinel_master_name,
                    master_address=master_addr,
                    replica_addresses=replica_addrs,
                    sentinel_count=sentinel_count,
                    is_healthy=is_healthy,
                    quorum=quorum,
                )
                    
            except (ConnectionError, TimeoutError) as e:
                self.logger.warning(
                    f"Failed to query Sentinel: {type(e).__name__}: {e}"
                )
                continue
            except RedisError as e:
                self.logger.error(f"Failed to get topology: {type(e).__name__}: {e}")
                return None
        
        self.logger.error("Failed to get topology: all Sentinels unreachable")
        return None
    
    async def get_replica_client(self) -> Optional[redis.Redis]:
        """
        Get a Redis client connected to a replica for read operations.
        
        Use this for read-scaling by offloading reads to replicas.
        The returned client should only be used for read operations.
        
        The client is cached and reused across calls. Call close() to
        release the connection pool when done.

        Tip: Wrap replica operations with ``client.with_retry()`` for
        automatic failover recovery during topology changes.
        
        Returns:
            Redis client connected to a replica, or None if not in
            Sentinel mode or no replicas available.
        """
        if not self.config.use_sentinel:
            self.logger.warning("get_replica_client() called but not in Sentinel mode")
            return None
        
        if self._replica_client is not None:
            return self._replica_client
        
        async with self._init_lock:
            # Double-checked locking
            if self._replica_client is not None:
                return self._replica_client

            try:
                sentinel = self.get_sentinel()
                self._replica_client = sentinel.slave_for(
                    self.config.sentinel_master_name,
                    db=self.config.db,
                    ssl=self.config.ssl,
                    password=self.config.password,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.socket_connect_timeout,
                    max_connections=self.config.max_connections,
                    health_check_interval=self.config.health_check_interval,
                    client_name=f"{self.config.client_name}_replica",
                )
                return self._replica_client
            except (ConnectionError, TimeoutError, RedisError) as e:
                self.logger.error(
                    f"Failed to get replica client: {type(e).__name__}: {e}"
                )
                return None

    async def __aenter__(self) -> "RedisClient":
        """Async context manager entry — initializes client."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        await self.close()
    
    def with_retry(
        self,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None
    ) -> Callable:
        """
        Decorator for adding retry logic to Redis operations.
        
        Use this for high-level business logic that spans multiple Redis
        operations or requires application-level retry guarantees.
        
        Note: This decorator provides the sole layer of retry logic.
        Internal redis-py retries are disabled to avoid multiplicative
        retry behavior.
        
        Args:
            max_retries: Maximum number of retry attempts (defaults to config)
            base_delay: Base delay in seconds (defaults to config)
        
        Returns:
            Decorator function
        
        Example:
            @client.with_retry(max_retries=3)
            async def critical_operation():
                redis = client.get_client()
                await redis.set("key", "value")
        """
        retries = max_retries if max_retries is not None else self.config.retry_attempts
        delay = base_delay if base_delay is not None else self.config.retry_base_delay
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                last_error = None
                
                for attempt in range(retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except RETRYABLE_EXCEPTIONS as e:
                        last_error = e
                        if attempt < retries:
                            # On failover-indicating errors in Sentinel mode,
                            # reset the client to force master re-discovery.
                            # This complements redis-py's built-in
                            # SentinelManagedConnection failover handling as
                            # a defensive fallback for edge cases.
                            if self.config.use_sentinel and isinstance(
                                e, (ReadOnlyError, ConnectionError)
                            ):
                                self.logger.warning(
                                    f"Failover-related error in Sentinel mode "
                                    f"(attempt {attempt + 1}/{retries + 1}): "
                                    f"{type(e).__name__}: {e} — resetting client"
                                )
                                await self._reset_client()

                            # Exponential backoff with jitter to prevent thundering herds
                            backoff = delay * (2 ** attempt)
                            jitter = random.uniform(0, 0.1 * backoff)
                            sleep_time = backoff + jitter
                            
                            self.logger.warning(
                                f"Redis operation failed (attempt {attempt + 1}/{retries + 1}), "
                                f"retrying in {sleep_time:.2f}s: {e}"
                            )
                            await asyncio.sleep(sleep_time)
                        else:
                            self.logger.error(
                                f"Redis operation failed after {retries + 1} attempts: {e}"
                            )
                
                raise last_error
            
            return wrapper
        return decorator


def with_redis_retry(max_retries: int = 3, base_delay: float = 0.1) -> Callable:
    """
    Standalone decorator for adding retry logic to Redis operations.
    
    Use this for high-level business logic that spans multiple Redis
    operations or requires application-level retry guarantees.
    
    Note: This decorator provides the sole layer of retry logic.
    This standalone decorator does not have access to a RedisClient
    instance and cannot reset the client on failover. For Sentinel (HA)
    mode, prefer the instance method decorator: client.with_retry().
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)
    
    Returns:
        Decorator function
    
    Example:
        @with_redis_retry(max_retries=3)
        async def critical_operation():
            ...
    """
    logger = logging.getLogger("RedisRetry")
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e:
                    last_error = e
                    if attempt < max_retries:
                        # Exponential backoff with jitter
                        backoff = base_delay * (2 ** attempt)
                        jitter = random.uniform(0, 0.1 * backoff)
                        delay = backoff + jitter
                        
                        logger.warning(
                            f"Redis operation failed (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {delay:.2f}s: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Redis operation failed after {max_retries + 1} attempts: {e}"
                        )
            
            raise last_error
        
        return wrapper
    return decorator
