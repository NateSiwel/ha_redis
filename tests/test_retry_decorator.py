"""
Tests for retry decorators.

Covers:
- RedisClient.with_retry instance method decorator
- with_redis_retry standalone decorator
- Exponential backoff behavior
- Error handling and propagation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from redis.exceptions import ConnectionError, TimeoutError, ReadOnlyError, BusyLoadingError

from ha_redis import RedisClient, RedisConfig, with_redis_retry


class TestWithRetryDecorator:
    """Test RedisClient.with_retry instance method decorator."""
    
    @pytest.mark.asyncio
    async def test_successful_operation_no_retry(self, mocked_redis_client):
        """Successful operation should not retry."""
        call_count = 0
        
        @mocked_redis_client.with_retry()
        async def operation():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await operation()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self, mocked_redis_client):
        """Should retry on ConnectionError."""
        call_count = 0
        
        @mocked_redis_client.with_retry(max_retries=2, base_delay=0.01)
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"
        
        result = await operation()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_on_timeout_error(self, mocked_redis_client):
        """Should retry on TimeoutError."""
        call_count = 0
        
        @mocked_redis_client.with_retry(max_retries=2, base_delay=0.01)
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Operation timed out")
            return "success"
        
        result = await operation()
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, mocked_redis_client):
        """Should raise exception after max retries exhausted."""
        call_count = 0
        
        @mocked_redis_client.with_retry(max_retries=2, base_delay=0.01)
        async def operation():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent failure")
        
        with pytest.raises(ConnectionError):
            await operation()
        
        # Initial call + 2 retries = 3 total calls
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_uses_config_defaults(self, mocked_redis_client):
        """Should use config defaults when not specified."""
        mocked_redis_client.config.retry_attempts = 1
        mocked_redis_client.config.retry_base_delay = 0.01
        call_count = 0
        
        @mocked_redis_client.with_retry()
        async def operation():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Failure")
        
        with pytest.raises(ConnectionError):
            await operation()
        
        # Initial call + 1 retry = 2 total calls
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_preserves_function_return_value(self, mocked_redis_client):
        """Should preserve the decorated function's return value."""
        @mocked_redis_client.with_retry()
        async def operation():
            return {"key": "value", "count": 42}
        
        result = await operation()
        
        assert result == {"key": "value", "count": 42}
    
    @pytest.mark.asyncio
    async def test_preserves_function_args(self, mocked_redis_client):
        """Should pass arguments correctly to decorated function."""
        received_args = []
        received_kwargs = {}
        
        @mocked_redis_client.with_retry()
        async def operation(*args, **kwargs):
            received_args.extend(args)
            received_kwargs.update(kwargs)
            return "done"
        
        await operation("arg1", "arg2", key1="val1", key2="val2")
        
        assert received_args == ["arg1", "arg2"]
        assert received_kwargs == {"key1": "val1", "key2": "val2"}
    
    @pytest.mark.asyncio
    async def test_retry_on_readonly_error(self, mocked_redis_client):
        """Should retry on ReadOnlyError (failover scenario)."""
        call_count = 0

        @mocked_redis_client.with_retry(max_retries=2, base_delay=0.01)
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ReadOnlyError("READONLY You can't write against a read only replica.")
            return "success"

        result = await operation()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_busy_loading_error(self, mocked_redis_client):
        """Should retry on BusyLoadingError."""
        call_count = 0

        @mocked_redis_client.with_retry(max_retries=2, base_delay=0.01)
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise BusyLoadingError("Redis is loading the dataset in memory")
            return "success"

        result = await operation()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_exception_propagates(self, mocked_redis_client):
        """Non-retryable exceptions should propagate immediately."""
        call_count = 0
        
        @mocked_redis_client.with_retry(max_retries=3, base_delay=0.01)
        async def operation():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not a Redis error")
        
        with pytest.raises(ValueError):
            await operation()
        
        # Should fail on first call without retrying
        assert call_count == 1


class TestWithRedisRetryStandalone:
    """Test standalone with_redis_retry decorator."""
    
    @pytest.mark.asyncio
    async def test_successful_operation(self):
        """Successful operation should not retry."""
        call_count = 0
        
        @with_redis_retry(max_retries=3, base_delay=0.01)
        async def operation():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await operation()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Should retry on ConnectionError."""
        call_count = 0
        
        @with_redis_retry(max_retries=2, base_delay=0.01)
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"
        
        result = await operation()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_on_timeout_error(self):
        """Should retry on TimeoutError."""
        call_count = 0
        
        @with_redis_retry(max_retries=2, base_delay=0.01)
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Operation timed out")
            return "success"
        
        result = await operation()
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        """Should raise exception after max retries exhausted."""
        call_count = 0
        
        @with_redis_retry(max_retries=2, base_delay=0.01)
        async def operation():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent failure")
        
        with pytest.raises(ConnectionError):
            await operation()
        
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_on_readonly_error(self):
        """Should retry on ReadOnlyError (failover scenario)."""
        call_count = 0

        @with_redis_retry(max_retries=2, base_delay=0.01)
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ReadOnlyError("READONLY You can't write against a read only replica.")
            return "success"

        result = await operation()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_busy_loading_error(self):
        """Should retry on BusyLoadingError."""
        call_count = 0

        @with_redis_retry(max_retries=2, base_delay=0.01)
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise BusyLoadingError("Redis is loading the dataset in memory")
            return "success"

        result = await operation()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_default_parameters(self):
        """Should use default parameters when not specified."""
        @with_redis_retry()
        async def operation():
            return "success"
        
        result = await operation()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self):
        """Should preserve function name and docstring."""
        @with_redis_retry()
        async def my_function():
            """My docstring."""
            return "result"
        
        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestExponentialBackoff:
    """Test exponential backoff timing behavior."""
    
    @pytest.mark.asyncio
    async def test_backoff_increases_exponentially(self, mocked_redis_client):
        """Delay should increase exponentially between retries."""
        delays = []
        original_sleep = asyncio.sleep
        
        async def mock_sleep(delay):
            delays.append(delay)
            # Don't actually sleep in tests
        
        with patch('asyncio.sleep', mock_sleep):
            call_count = 0
            
            @mocked_redis_client.with_retry(max_retries=3, base_delay=0.1)
            async def operation():
                nonlocal call_count
                call_count += 1
                raise ConnectionError("Failure")
            
            with pytest.raises(ConnectionError):
                await operation()
        
        # Verify exponential backoff: 0.1, 0.2, 0.4 (plus jitter)
        assert len(delays) == 3
        assert 0.1 <= delays[0] <= 0.11
        assert 0.2 <= delays[1] <= 0.22
        assert 0.4 <= delays[2] <= 0.44
    
    @pytest.mark.asyncio
    async def test_no_sleep_on_success(self, mocked_redis_client):
        """Should not sleep when operation succeeds."""
        sleep_called = False
        
        async def mock_sleep(delay):
            nonlocal sleep_called
            sleep_called = True
        
        with patch('asyncio.sleep', mock_sleep):
            @mocked_redis_client.with_retry()
            async def operation():
                return "success"
            
            await operation()
        
        assert not sleep_called
    
    @pytest.mark.asyncio
    async def test_no_sleep_after_final_failure(self, mocked_redis_client):
        """Should not sleep after the final retry fails."""
        sleep_count = 0
        
        async def mock_sleep(delay):
            nonlocal sleep_count
            sleep_count += 1
        
        with patch('asyncio.sleep', mock_sleep):
            @mocked_redis_client.with_retry(max_retries=2, base_delay=0.01)
            async def operation():
                raise ConnectionError("Failure")
            
            with pytest.raises(ConnectionError):
                await operation()
        
        # 2 retries = 2 sleeps (not 3)
        assert sleep_count == 2


class TestRetryWithLogging:
    """Test logging behavior during retries."""
    
    @pytest.mark.asyncio
    async def test_logs_warning_on_retry(
        self,
        redis_client_with_mock_logger,
        mock_logger,
    ):
        """Should log warning on each retry."""
        redis_client_with_mock_logger.config.retry_attempts = 2
        redis_client_with_mock_logger.config.retry_base_delay = 0.01
        
        @redis_client_with_mock_logger.with_retry()
        async def operation():
            raise ConnectionError("Failure")
        
        with pytest.raises(ConnectionError):
            await operation()
        
        # Should have logged warnings for retries
        assert mock_logger.warning.called
    
    @pytest.mark.asyncio
    async def test_logs_error_on_final_failure(
        self,
        redis_client_with_mock_logger,
        mock_logger,
    ):
        """Should log error when all retries exhausted."""
        redis_client_with_mock_logger.config.retry_attempts = 1
        redis_client_with_mock_logger.config.retry_base_delay = 0.01
        
        @redis_client_with_mock_logger.with_retry()
        async def operation():
            raise ConnectionError("Failure")
        
        with pytest.raises(ConnectionError):
            await operation()
        
        # Should have logged final error
        assert mock_logger.error.called
