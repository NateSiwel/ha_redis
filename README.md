# ha_redis

A resilient, high-availability async Redis client library for Python with connection pooling, automatic retry logic, and Redis Sentinel support.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Redis 7+](https://img.shields.io/badge/redis-7+-red.svg)](https://redis.io/)

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Architecture](#architecture)

## Features

- **Automatic Retry** - Exponential backoff for transient failures with configurable attempts
- **Connection Pooling** - Efficient connection management sized for high concurrency (C10k ready)
- **High Availability** - Redis Sentinel support for automatic failover and leader election
- **Health Checks** - Built-in connection health monitoring with configurable intervals
- **Timeout Protection** - Prevents hung connections from blocking operations
- **SSL/TLS Support** - Secure connections out of the box
- **Async/Await** - Fully asynchronous with context manager support
- **Battle-tested** - Comprehensive test suite with unit and integration tests
- **Zero Dependencies** - Only requires `redis-py` (async) library

## Project Structure

```
ha_redis/
├── ha_redis.py           # Core library (RedisClient, RedisConfig, decorators)
├── config_sample.py      # Pydantic settings example for env var configuration
├── docker-compose.yml    # Docker setup for standalone and HA Redis
├── pytest.ini            # Pytest configuration
├── README.md             # This documentation
└── tests/
    ├── conftest.py           # Test fixtures and pytest hooks
    ├── test_config_sample.py # Config sample tests
    ├── test_integration.py   # Integration tests (requires Redis)
    ├── test_redis_client.py  # RedisClient unit tests
    ├── test_redis_config.py  # RedisConfig unit tests
    └── test_retry_decorator.py # Retry decorator tests
```

## Installation

### Requirements

- Python 3.9 or higher
- Redis server 6.0+ (Redis 7+ recommended)

### Install Dependencies

```bash
pip install redis
```

### Add to Your Project

Copy `ha_redis.py` to your project, or install directly:

```bash
# Option 1: Copy the file
curl -O https://raw.githubusercontent.com/your-repo/ha_redis/main/ha_redis.py

# Option 2: Add to your project structure
cp ha_redis.py /path/to/your/project/
```

### Optional: Pydantic Settings Integration

For environment variable configuration (see [config_sample.py](config_sample.py)):

```bash
pip install pydantic-settings
```

## Quick Start

### Basic Usage (Direct Connection)

```python
import asyncio
from ha_redis import RedisClient, RedisConfig

async def main():
    # Create configuration
    config = RedisConfig(
        host="localhost",
        port=6379,
        password="your_password"  # Optional
    )
    
    # Create client and use it
    async with RedisClient(config) as client:
        redis = client.get_client()
        
        # Basic operations
        await redis.set("key", "value")
        value = await redis.get("key")
        print(f"Retrieved: {value}")

asyncio.run(main())
```

### High Availability with Sentinel

```python
import asyncio
from ha_redis import RedisClient, RedisConfig

async def main():
    # Configure for Sentinel mode
    config = RedisConfig(
        use_sentinel=True,
        sentinel_hosts=[
            ("sentinel1.example.com", 26379),
            ("sentinel2.example.com", 26379),
            ("sentinel3.example.com", 26379),
        ],
        sentinel_master_name="mymaster",
        password="your_password"  # Optional
    )
    
    async with RedisClient(config) as client:
        redis = client.get_client()
        await redis.set("ha_key", "ha_value")

asyncio.run(main())
```

### Common Redis Operations

```python
import asyncio
from ha_redis import RedisClient, RedisConfig

async def redis_examples():
    async with RedisClient(RedisConfig()) as client:
        redis = client.get_client()
        
        # ─── String Operations ───
        await redis.set("user:1:name", "Alice")
        name = await redis.get("user:1:name")
        
        # Set with expiration (seconds)
        await redis.set("session:abc", "data", ex=3600)
        
        # Increment/Decrement
        await redis.set("counter", "0")
        await redis.incr("counter")      # 1
        await redis.incrby("counter", 5) # 6
        
        # ─── Hash Operations ───
        await redis.hset("user:1", mapping={
            "name": "Alice",
            "email": "alice@example.com",
            "age": "30"
        })
        user = await redis.hgetall("user:1")
        email = await redis.hget("user:1", "email")
        
        # ─── List Operations ───
        await redis.rpush("queue:jobs", "job1", "job2", "job3")
        job = await redis.lpop("queue:jobs")  # FIFO
        all_jobs = await redis.lrange("queue:jobs", 0, -1)
        
        # ─── Set Operations ───
        await redis.sadd("tags:article:1", "python", "redis", "async")
        tags = await redis.smembers("tags:article:1")
        has_python = await redis.sismember("tags:article:1", "python")
        
        # ─── Sorted Set Operations ───
        await redis.zadd("leaderboard", {"alice": 100, "bob": 85, "charlie": 95})
        top_players = await redis.zrevrange("leaderboard", 0, 2, withscores=True)
        
        # ─── Key Management ───
        exists = await redis.exists("user:1:name")
        await redis.expire("session:abc", 1800)  # Update TTL
        ttl = await redis.ttl("session:abc")
        await redis.delete("temp:key")

asyncio.run(redis_examples())
```

## Configuration

### RedisConfig Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str` | `"localhost"` | Redis server hostname (direct mode) |
| `port` | `int` | `6379` | Redis server port (direct mode) |
| `password` | `str \| None` | `None` | Redis authentication password |
| `db` | `int` | `0` | Redis database number |
| `max_connections` | `int` | `50` | Maximum pool connections |
| `socket_timeout` | `float` | `5.0` | Timeout for socket operations (seconds) |
| `socket_connect_timeout` | `float` | `2.0` | Timeout for establishing connections (seconds) |
| `health_check_interval` | `int` | `30` | Interval for connection health checks (seconds) |
| `use_sentinel` | `bool` | `False` | Enable Sentinel mode for HA |
| `sentinel_hosts` | `List[Tuple[str, int]]` | `[]` | List of Sentinel (host, port) tuples |
| `sentinel_master_name` | `str` | `"mymaster"` | Name of the Sentinel master |
| `ssl` | `bool` | `False` | Enable SSL/TLS connections |
| `retry_attempts` | `int` | `3` | Number of retry attempts |
| `retry_base_delay` | `float` | `0.1` | Base delay for exponential backoff (seconds) |

### Configuration Examples

#### Minimal Configuration
```python
config = RedisConfig()  # Uses localhost:6379
```

#### Production Direct Connection
```python
config = RedisConfig(
    host="redis.production.com",
    port=6379,
    password="secure_password",
    db=0,
    max_connections=100,
    socket_timeout=10.0,
    ssl=True
)
```

#### High Availability Configuration
```python
config = RedisConfig(
    use_sentinel=True,
    sentinel_hosts=[
        ("sentinel1.prod.com", 26379),
        ("sentinel2.prod.com", 26379),
        ("sentinel3.prod.com", 26379),
    ],
    sentinel_master_name="redis-master",
    password="secure_password",
    max_connections=100
)
```

## Architecture

### Connection Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        Application                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       RedisClient                           │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │   Config    │  │ Retry Logic  │  │   Health Check    │   │
│  │  (timeout,  │  │ (exponential │  │  (periodic ping)  │   │
│  │   pool)     │  │   backoff)   │  │                   │   │
│  └─────────────┘  └──────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│    Direct Connection    │     │     Sentinel Mode (HA)      │
│                         │     │                             │
│  ┌───────────────────┐  │     │  ┌───────────────────────┐  │
│  │  Connection Pool  │  │     │  │   Sentinel Cluster    │  │
│  │   (max_conn=50)   │  │     │  │   (3+ nodes)          │  │
│  └─────────┬─────────┘  │     │  └───────────┬───────────┘  │
│            │            │     │              │              │
│            ▼            │     │              ▼              │
│  ┌───────────────────┐  │     │  ┌───────────────────────┐  │
│  │   Redis Server    │  │     │  │  Master Discovery     │  │
│  └───────────────────┘  │     │  └───────────┬───────────┘  │
└─────────────────────────┘     │              │              │
                                │              ▼              │
                                │  ┌───────────────────────┐  │
                                │  │ Master ←→ Replica(s)  │  │
                                │  └───────────────────────┘  │
                                └─────────────────────────────┘
```

### Retry Strategy

The library implements exponential backoff for transient failures:

```
Attempt 1: Immediate
Attempt 2: base_delay × 2⁰ = 0.1s
Attempt 3: base_delay × 2¹ = 0.2s
Attempt 4: base_delay × 2² = 0.4s
... (capped at 2 seconds)
```

Retried exceptions:
- `ConnectionError` - Network connectivity issues
- `TimeoutError` - Operation timeouts
- `BusyLoadingError` - Redis loading dataset

### Connection Pool Sizing

The default pool size of 50 connections is designed for high-concurrency applications:

```python
# Rule of thumb for sizing
max_connections = (CPU_CORES × 2) + 10  # Buffer for burst traffic

# For a 4-core machine
config = RedisConfig(max_connections=4 * 2 + 10)  # 18 connections
```