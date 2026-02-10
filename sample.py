"""
ha_redis — Comprehensive Sample
================================

Connects to the Sentinel HA topology defined in docker-compose.yml
and exercises every feature of the ha_redis library.

Prerequisites:
    docker compose --profile ha up -d
    pip install redis

Run:
    python sample.py
"""

import asyncio
import logging
import time

from ha_redis import (
    RedisClient,
    RedisConfig,
    SentinelTopology,
    with_redis_retry,
)

# ─────────────────────────────────────────────────────────────
# Logging — see retry attempts, Sentinel discovery, pool events
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sample")


# ─────────────────────────────────────────────────────────────
# 1. Configuration — matches docker-compose.yml HA profile
# ─────────────────────────────────────────────────────────────
SENTINEL_CONFIG = RedisConfig(
    use_sentinel=True,
    sentinel_hosts=[
        ("localhost", 26379),   # redis-sentinel-1
        ("localhost", 26380),   # redis-sentinel-2
        ("localhost", 26381),   # redis-sentinel-3
    ],
    sentinel_master_name="mymaster",
    # No password — docker-compose topology has none
    password=None,
    sentinel_password=None,
    db=0,
    max_connections=20,
    socket_timeout=5.0,
    socket_connect_timeout=2.0,
    health_check_interval=30,
    retry_attempts=3,
    retry_base_delay=0.1,
    client_name="ha_redis_sample",
)


def banner(title: str) -> None:
    width = 60
    log.info("")
    log.info("=" * width)
    log.info(f"  {title}")
    log.info("=" * width)


# ─────────────────────────────────────────────────────────────
# 2. Sentinel Topology Inspection
# ─────────────────────────────────────────────────────────────
async def demo_topology(client: RedisClient) -> None:
    banner("Sentinel Topology Inspection")

    # 2a. Confirm we are in Sentinel mode
    log.info(f"Sentinel mode enabled: {client.is_sentinel_mode()}")

    # 2b. Discover master address
    master = await client.discover_master()
    log.info(f"Current master address: {master}")

    # 2c. Discover replica addresses
    replicas = await client.discover_replicas()
    log.info(f"Replica addresses: {replicas}")

    # 2d. Full topology snapshot
    topo: SentinelTopology | None = await client.get_topology()
    if topo:
        log.info(f"  master_name      : {topo.master_name}")
        log.info(f"  master_address   : {topo.master_address}")
        log.info(f"  replica_addresses: {topo.replica_addresses}")
        log.info(f"  sentinel_count   : {topo.sentinel_count}")
        log.info(f"  quorum           : {topo.quorum}")
        log.info(f"  is_healthy       : {topo.is_healthy}")

    # 2e. Access the underlying Sentinel object (advanced)
    sentinel_obj = client.get_sentinel()
    log.info(f"Sentinel object type: {type(sentinel_obj).__name__}")


# ─────────────────────────────────────────────────────────────
# 3. Health Check
# ─────────────────────────────────────────────────────────────
async def demo_health_check(client: RedisClient) -> None:
    banner("Health Check")
    healthy = await client.health_check()
    log.info(f"Redis is healthy: {healthy}")


# ─────────────────────────────────────────────────────────────
# 4. Core Redis Data-Structure Operations
# ─────────────────────────────────────────────────────────────
async def demo_data_operations(client: RedisClient) -> None:
    banner("Core Data-Structure Operations")

    r = client.get_client()

    # ── Strings ──────────────────────────────────────────────
    log.info("--- Strings ---")
    await r.set("sample:greeting", "Hello from ha_redis!")
    value = await r.get("sample:greeting")
    log.info(f"  GET sample:greeting  → {value}")

    # Set with TTL
    await r.set("sample:temp", "expires-soon", ex=60)
    ttl = await r.ttl("sample:temp")
    log.info(f"  TTL sample:temp      → {ttl}s")

    # Atomic increment
    await r.set("sample:counter", 0)
    await r.incr("sample:counter")
    await r.incrby("sample:counter", 10)
    counter = await r.get("sample:counter")
    log.info(f"  Counter after incr   → {counter}")

    # MSET / MGET
    await r.mset({"sample:a": "1", "sample:b": "2", "sample:c": "3"})
    vals = await r.mget("sample:a", "sample:b", "sample:c")
    log.info(f"  MGET [a, b, c]       → {vals}")

    # ── Hashes ───────────────────────────────────────────────
    log.info("--- Hashes ---")
    await r.hset("sample:user:1", mapping={
        "name": "Alice",
        "email": "alice@example.com",
        "score": "42",
    })
    user = await r.hgetall("sample:user:1")
    log.info(f"  HGETALL user:1       → {user}")

    email = await r.hget("sample:user:1", "email")
    log.info(f"  HGET email           → {email}")

    await r.hincrby("sample:user:1", "score", 8)
    score = await r.hget("sample:user:1", "score")
    log.info(f"  HINCRBY score +8     → {score}")

    # ── Lists ────────────────────────────────────────────────
    log.info("--- Lists ---")
    await r.delete("sample:queue")
    await r.rpush("sample:queue", "job-1", "job-2", "job-3")
    length = await r.llen("sample:queue")
    log.info(f"  Queue length         → {length}")

    popped = await r.lpop("sample:queue")
    log.info(f"  LPOP (FIFO)          → {popped}")

    remaining = await r.lrange("sample:queue", 0, -1)
    log.info(f"  Remaining jobs       → {remaining}")

    # ── Sets ─────────────────────────────────────────────────
    log.info("--- Sets ---")
    await r.delete("sample:tags")
    await r.sadd("sample:tags", "python", "redis", "async", "sentinel")
    members = await r.smembers("sample:tags")
    log.info(f"  SMEMBERS tags        → {members}")

    is_member = await r.sismember("sample:tags", "python")
    log.info(f"  SISMEMBER 'python'   → {is_member}")

    card = await r.scard("sample:tags")
    log.info(f"  SCARD                → {card}")

    # ── Sorted Sets ──────────────────────────────────────────
    log.info("--- Sorted Sets ---")
    await r.delete("sample:leaderboard")
    await r.zadd("sample:leaderboard", {
        "alice": 100,
        "bob": 85,
        "charlie": 95,
        "diana": 110,
    })
    top3 = await r.zrevrange("sample:leaderboard", 0, 2, withscores=True)
    log.info(f"  Top 3 players        → {top3}")

    rank = await r.zrevrank("sample:leaderboard", "charlie")
    log.info(f"  Charlie's rank       → {rank}")

    zscore = await r.zscore("sample:leaderboard", "alice")
    log.info(f"  Alice's score        → {zscore}")


# ─────────────────────────────────────────────────────────────
# 5. Pipeline (batched commands in a single round-trip)
# ─────────────────────────────────────────────────────────────
async def demo_pipeline(client: RedisClient) -> None:
    banner("Pipeline — Batched Commands")

    r = client.get_client()

    async with r.pipeline(transaction=False) as pipe:
        pipe.set("sample:pipe:x", "10")
        pipe.set("sample:pipe:y", "20")
        pipe.set("sample:pipe:z", "30")
        pipe.get("sample:pipe:x")
        pipe.get("sample:pipe:y")
        pipe.get("sample:pipe:z")
        results = await pipe.execute()

    log.info(f"  Pipeline results: {results}")


# ─────────────────────────────────────────────────────────────
# 6. Transactions (MULTI/EXEC via pipeline)
# ─────────────────────────────────────────────────────────────
async def demo_transaction(client: RedisClient) -> None:
    banner("Transaction (MULTI / EXEC)")

    r = client.get_client()

    # Atomic transfer between two counters
    await r.set("sample:acct:A", "100")
    await r.set("sample:acct:B", "50")

    async with r.pipeline(transaction=True) as pipe:
        pipe.decrby("sample:acct:A", 25)
        pipe.incrby("sample:acct:B", 25)
        results = await pipe.execute()

    a = await r.get("sample:acct:A")
    b = await r.get("sample:acct:B")
    log.info(f"  After transfer  A={a}  B={b}  (results={results})")


# ─────────────────────────────────────────────────────────────
# 7. Pub/Sub Messaging
# ─────────────────────────────────────────────────────────────
async def demo_pubsub(client: RedisClient) -> None:
    banner("Pub/Sub Messaging")

    r = client.get_client()
    channel = "sample:notifications"

    pubsub = r.pubsub()
    await pubsub.subscribe(channel)

    # Publish a few messages (small delay so subscriber can receive)
    for i in range(3):
        await r.publish(channel, f"event-{i}")

    # Read messages (first message is the subscribe confirmation)
    received = []
    for _ in range(4):  # 1 subscribe-ack + 3 messages
        msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=2.0)
        if msg:
            received.append(msg["data"])

    log.info(f"  Received messages: {received}")

    await pubsub.unsubscribe(channel)
    await pubsub.close()


# ─────────────────────────────────────────────────────────────
# 8. Key Expiry & Management
# ─────────────────────────────────────────────────────────────
async def demo_key_management(client: RedisClient) -> None:
    banner("Key Expiry & Management")

    r = client.get_client()

    await r.set("sample:volatile", "will-expire", ex=120)
    ttl = await r.ttl("sample:volatile")
    log.info(f"  TTL after SET ex=120       → {ttl}")

    await r.expire("sample:volatile", 60)
    ttl = await r.ttl("sample:volatile")
    log.info(f"  TTL after EXPIRE 60        → {ttl}")

    await r.persist("sample:volatile")
    ttl = await r.ttl("sample:volatile")
    log.info(f"  TTL after PERSIST          → {ttl}  (-1 = no expiry)")

    exists = await r.exists("sample:volatile")
    log.info(f"  EXISTS sample:volatile     → {exists}")

    key_type = await r.type("sample:user:1")
    log.info(f"  TYPE sample:user:1         → {key_type}")

    # SCAN for keys matching a pattern
    keys = []
    async for key in r.scan_iter(match="sample:*", count=50):
        keys.append(key)
    log.info(f"  SCAN sample:* found {len(keys)} keys")


# ─────────────────────────────────────────────────────────────
# 9. Retry Decorator — Instance Method (@client.with_retry)
# ─────────────────────────────────────────────────────────────
async def demo_retry_instance(client: RedisClient) -> None:
    banner("Retry Decorator — client.with_retry()")

    @client.with_retry(max_retries=3, base_delay=0.1)
    async def resilient_write():
        r = client.get_client()
        await r.set("sample:resilient", "wrote-with-retry")
        return await r.get("sample:resilient")

    result = await resilient_write()
    log.info(f"  Resilient write returned: {result}")


# ─────────────────────────────────────────────────────────────
# 10. Retry Decorator — Standalone (@with_redis_retry)
# ─────────────────────────────────────────────────────────────
async def demo_retry_standalone(client: RedisClient) -> None:
    banner("Retry Decorator — @with_redis_retry (standalone)")

    @with_redis_retry(max_retries=3, base_delay=0.1)
    async def standalone_op():
        r = client.get_client()
        await r.set("sample:standalone", "standalone-retry-ok")
        return await r.get("sample:standalone")

    result = await standalone_op()
    log.info(f"  Standalone retry result: {result}")


# ─────────────────────────────────────────────────────────────
# 11. Replica Read-Scaling
# ─────────────────────────────────────────────────────────────
async def demo_replica_reads(client: RedisClient) -> None:
    banner("Replica Client — Read Scaling")

    # Write via master
    master = client.get_client()
    await master.set("sample:replicated", "read-me-from-replica")

    # Short delay for replication propagation
    await asyncio.sleep(0.5)

    # Read via replica
    replica = await client.get_replica_client()
    if replica:
        value = await replica.get("sample:replicated")
        log.info(f"  Read from replica: {value}")

        info = await replica.info("replication")
        log.info(f"  Replica role: {info.get('role', 'unknown')}")
    else:
        log.warning("  No replica client available — skipping")


# ─────────────────────────────────────────────────────────────
# 12. Concurrent Operations (connection pool exercised)
# ─────────────────────────────────────────────────────────────
async def demo_concurrency(client: RedisClient) -> None:
    banner("Concurrent Operations (Pool Stress)")

    r = client.get_client()
    num_tasks = 50

    async def worker(task_id: int) -> str:
        key = f"sample:concurrent:{task_id}"
        await r.set(key, f"value-{task_id}", ex=30)
        return await r.get(key)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(worker(i) for i in range(num_tasks)))
    elapsed = time.perf_counter() - t0

    log.info(f"  {num_tasks} concurrent SET+GET completed in {elapsed:.3f}s")
    log.info(f"  Sample results: {results[:5]} ...")


# ─────────────────────────────────────────────────────────────
# 13. Lua Scripting
# ─────────────────────────────────────────────────────────────
async def demo_lua_script(client: RedisClient) -> None:
    banner("Lua Scripting (server-side atomics)")

    r = client.get_client()

    # Atomic compare-and-swap: set key only if current value matches
    cas_script = """
    local current = redis.call('GET', KEYS[1])
    if current == ARGV[1] then
        redis.call('SET', KEYS[1], ARGV[2])
        return 1
    end
    return 0
    """

    await r.set("sample:cas", "old-value")

    # Should succeed — current value matches
    result = await r.eval(cas_script, 1, "sample:cas", "old-value", "new-value")
    log.info(f"  CAS (old→new) result: {result}  (1=swapped)")

    # Should fail — current value is now 'new-value', not 'old-value'
    result = await r.eval(cas_script, 1, "sample:cas", "old-value", "another")
    log.info(f"  CAS (old→another) result: {result}  (0=no swap)")

    final = await r.get("sample:cas")
    log.info(f"  Final value: {final}")


# ─────────────────────────────────────────────────────────────
# 14. Server Info & Diagnostics
# ─────────────────────────────────────────────────────────────
async def demo_server_info(client: RedisClient) -> None:
    banner("Server Info & Diagnostics")

    r = client.get_client()

    info = await r.info("server")
    log.info(f"  Redis version  : {info.get('redis_version', '?')}")
    log.info(f"  OS             : {info.get('os', '?')}")
    log.info(f"  TCP port       : {info.get('tcp_port', '?')}")
    log.info(f"  Uptime (sec)   : {info.get('uptime_in_seconds', '?')}")

    memory = await r.info("memory")
    log.info(f"  Used memory    : {memory.get('used_memory_human', '?')}")

    clients = await r.info("clients")
    log.info(f"  Connected      : {clients.get('connected_clients', '?')} clients")

    db_size = await r.dbsize()
    log.info(f"  DB size        : {db_size} keys")


# ─────────────────────────────────────────────────────────────
# 15. Cleanup — remove all sample keys
# ─────────────────────────────────────────────────────────────
async def cleanup(client: RedisClient) -> None:
    banner("Cleanup")

    r = client.get_client()

    keys = []
    async for key in r.scan_iter(match="sample:*", count=100):
        keys.append(key)

    if keys:
        deleted = await r.delete(*keys)
        log.info(f"  Deleted {deleted} sample:* keys")
    else:
        log.info("  Nothing to clean up")


# ─────────────────────────────────────────────────────────────
# Main — wire everything together
# ─────────────────────────────────────────────────────────────
async def main() -> None:
    log.info("Connecting to Redis via Sentinel HA topology …")
    log.info(f"Sentinels: {SENTINEL_CONFIG.sentinel_hosts}")
    log.info(f"Master name: {SENTINEL_CONFIG.sentinel_master_name}")

    async with RedisClient(SENTINEL_CONFIG) as client:
        await demo_topology(client)
        await demo_health_check(client)
        await demo_data_operations(client)
        await demo_pipeline(client)
        await demo_transaction(client)
        await demo_pubsub(client)
        await demo_key_management(client)
        await demo_retry_instance(client)
        await demo_retry_standalone(client)
        await demo_replica_reads(client)
        await demo_concurrency(client)
        await demo_lua_script(client)
        await demo_server_info(client)
        await cleanup(client)

    log.info("")
    log.info("All demos completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
