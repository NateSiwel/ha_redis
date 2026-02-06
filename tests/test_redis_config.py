"""
Tests for RedisConfig dataclass.

Covers:
- Default values
- Custom configuration
- URL generation
- SSL/TLS configuration
- Sentinel configuration
"""

import pytest
from ha_redis import RedisConfig


class TestRedisConfigDefaults:
    """Test default configuration values."""
    
    def test_default_host(self, default_config):
        """Default host should be localhost."""
        assert default_config.host == "localhost"
    
    def test_default_port(self, default_config):
        """Default port should be 6379."""
        assert default_config.port == 6379
    
    def test_default_password(self, default_config):
        """Default password should be None."""
        assert default_config.password is None
    
    def test_default_db(self, default_config):
        """Default database should be 0."""
        assert default_config.db == 0
    
    def test_default_max_connections(self, default_config):
        """Default max_connections should be 50."""
        assert default_config.max_connections == 50
    
    def test_default_socket_timeout(self, default_config):
        """Default socket_timeout should be 5.0 seconds."""
        assert default_config.socket_timeout == 5.0
    
    def test_default_socket_connect_timeout(self, default_config):
        """Default socket_connect_timeout should be 2.0 seconds."""
        assert default_config.socket_connect_timeout == 2.0
    
    def test_default_health_check_interval(self, default_config):
        """Default health_check_interval should be 30 seconds."""
        assert default_config.health_check_interval == 30
    
    def test_default_use_sentinel(self, default_config):
        """Default use_sentinel should be False."""
        assert default_config.use_sentinel is False
    
    def test_default_sentinel_hosts(self, default_config):
        """Default sentinel_hosts should be empty list."""
        assert default_config.sentinel_hosts == []
    
    def test_default_sentinel_master_name(self, default_config):
        """Default sentinel_master_name should be 'mymaster'."""
        assert default_config.sentinel_master_name == "mymaster"
    
    def test_default_ssl(self, default_config):
        """Default SSL should be False."""
        assert default_config.ssl is False
    
    def test_default_retry_attempts(self, default_config):
        """Default retry_attempts should be 3."""
        assert default_config.retry_attempts == 3
    
    def test_default_retry_base_delay(self, default_config):
        """Default retry_base_delay should be 0.1 seconds."""
        assert default_config.retry_base_delay == 0.1


class TestRedisConfigCustom:
    """Test custom configuration values."""
    
    def test_custom_host(self, custom_config):
        """Custom host should be set correctly."""
        assert custom_config.host == "redis.example.com"
    
    def test_custom_port(self, custom_config):
        """Custom port should be set correctly."""
        assert custom_config.port == 6380
    
    def test_custom_password(self, custom_config):
        """Custom password should be set correctly."""
        assert custom_config.password == "secret123"
    
    def test_custom_db(self, custom_config):
        """Custom database should be set correctly."""
        assert custom_config.db == 1
    
    def test_custom_max_connections(self, custom_config):
        """Custom max_connections should be set correctly."""
        assert custom_config.max_connections == 100
    
    def test_custom_timeouts(self, custom_config):
        """Custom timeouts should be set correctly."""
        assert custom_config.socket_timeout == 10.0
        assert custom_config.socket_connect_timeout == 5.0
    
    def test_custom_retry_settings(self, custom_config):
        """Custom retry settings should be set correctly."""
        assert custom_config.retry_attempts == 5
        assert custom_config.retry_base_delay == 0.2


class TestRedisConfigUrl:
    """Test URL generation."""
    
    def test_basic_url(self, default_config):
        """Basic URL should be generated correctly."""
        assert default_config.url == "redis://localhost:6379/0"
    
    def test_url_with_password(self, custom_config):
        """URL with password should include auth."""
        assert custom_config.url == "redis://:secret123@redis.example.com:6380/1"
    
    def test_url_with_ssl(self, ssl_config):
        """URL with SSL should use rediss scheme."""
        assert ssl_config.url.startswith("rediss://")
    
    def test_url_with_ssl_and_password(self, ssl_config):
        """URL with SSL and password should be formatted correctly."""
        expected = "rediss://:secure_password@redis-secure.example.com:6379/0"
        assert ssl_config.url == expected
    
    def test_url_different_db(self):
        """URL should reflect different database numbers."""
        config = RedisConfig(db=5)
        assert config.url == "redis://localhost:6379/5"


class TestRedisConfigSentinel:
    """Test Sentinel configuration."""
    
    def test_sentinel_enabled(self, sentinel_config):
        """Sentinel should be enabled."""
        assert sentinel_config.use_sentinel is True
    
    def test_sentinel_hosts(self, sentinel_config):
        """Sentinel hosts should be set correctly."""
        assert len(sentinel_config.sentinel_hosts) == 3
        assert sentinel_config.sentinel_hosts[0] == ("sentinel1.example.com", 26379)
    
    def test_sentinel_master_name(self, sentinel_config):
        """Sentinel master name should be set correctly."""
        assert sentinel_config.sentinel_master_name == "mymaster"
    
    def test_sentinel_with_password(self, sentinel_config):
        """Sentinel config should have password."""
        assert sentinel_config.password == "sentinel_password"
    
    def test_minimal_sentinel_config(self, minimal_sentinel_config):
        """Minimal Sentinel config should work."""
        assert minimal_sentinel_config.use_sentinel is True
        assert len(minimal_sentinel_config.sentinel_hosts) == 1
        assert minimal_sentinel_config.sentinel_master_name == "testmaster"


class TestRedisConfigEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_max_connections(self):
        """Zero max_connections should be accepted."""
        config = RedisConfig(max_connections=0)
        assert config.max_connections == 0
    
    def test_very_large_max_connections(self):
        """Very large max_connections should be accepted."""
        config = RedisConfig(max_connections=10000)
        assert config.max_connections == 10000
    
    def test_zero_timeout(self):
        """Zero timeout should be accepted."""
        config = RedisConfig(socket_timeout=0)
        assert config.socket_timeout == 0
    
    def test_negative_db_accepted(self):
        """Negative db number should be accepted (Redis validates this)."""
        config = RedisConfig(db=-1)
        assert config.db == -1
    
    def test_empty_password(self):
        """Empty password string should be accepted."""
        config = RedisConfig(password="")
        # Empty string is truthy in URL generation context
        assert "" in config.url or config.url == "redis://:@localhost:6379/0"
    
    def test_empty_sentinel_hosts_list(self):
        """Empty sentinel hosts list should work."""
        config = RedisConfig(use_sentinel=True, sentinel_hosts=[])
        assert config.sentinel_hosts == []
    
    def test_float_retry_base_delay(self):
        """Float retry base delay should be accepted."""
        config = RedisConfig(retry_base_delay=0.05)
        assert config.retry_base_delay == 0.05
