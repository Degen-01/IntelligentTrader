"""Advanced rate limiting with token bucket algorithm."""

import time
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    requests_per_second: float
    burst_capacity: int
    cooldown_period: float = 1.0

class TokenBucket:
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        async with self._lock:
            now = time.time()
            # Refill tokens based on time elapsed
            elapsed = now - self.last_refill
            self.tokens = min(
                self.config.burst_capacity,
                self.tokens + elapsed * self.config.requests_per_second
            )
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def wait_for_tokens(self, tokens: int = 1):
        while not await self.acquire(tokens):
            await asyncio.sleep(0.1)

class RateLimiter:
    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        self.configs: Dict[str, RateLimitConfig] = {
            'exchange_api': RateLimitConfig(10.0, 20),  # 10 req/sec, burst 20
            'data_feed': RateLimitConfig(100.0, 200),   # 100 req/sec, burst 200
            'order_placement': RateLimitConfig(5.0, 10), # 5 req/sec, burst 10
        }

    def get_bucket(self, endpoint: str) -> TokenBucket:
        if endpoint not in self.buckets:
            config = self.configs.get(endpoint, RateLimitConfig(1.0, 1))
            self.buckets[endpoint] = TokenBucket(config)
        return self.buckets[endpoint]

    async def acquire(self, endpoint: str, tokens: int = 1) -> bool:
        bucket = self.get_bucket(endpoint)
        return await bucket.acquire(tokens)

    async def wait_for_capacity(self, endpoint: str, tokens: int = 1):
        bucket = self.get_bucket(endpoint)
        await bucket.wait_for_tokens(tokens)

# Global rate limiter instance
rate_limiter = RateLimiter()

def rate_limit(endpoint: str, tokens: int = 1):
    """Decorator for rate limiting."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            await rate_limiter.wait_for_capacity(endpoint, tokens)
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(rate_limiter.wait_for_capacity(endpoint, tokens))
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
