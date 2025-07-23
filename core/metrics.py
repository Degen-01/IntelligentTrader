"""Prometheus metrics for monitoring."""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
from functools import wraps
from typing import Callable, Any

# Create custom registry
REGISTRY = CollectorRegistry()

# Define metrics
TRADES_TOTAL = Counter('trades_total', 'Total number of trades executed', ['status', 'symbol'], registry=REGISTRY)
TRADE_DURATION = Histogram('trade_duration_seconds', 'Time spent executing trades', ['symbol'], registry=REGISTRY)
PORTFOLIO_VALUE = Gauge('portfolio_value_usd', 'Current portfolio value in USD', registry=REGISTRY)
RISK_EXPOSURE = Gauge('risk_exposure_percent', 'Current risk exposure percentage', registry=REGISTRY)
API_CALLS_TOTAL = Counter('api_calls_total', 'Total API calls made', ['endpoint', 'status'], registry=REGISTRY)
DRAWDOWN_CURRENT = Gauge('drawdown_current_percent', 'Current drawdown percentage', registry=REGISTRY)
ORDERS_PENDING = Gauge('orders_pending_count', 'Number of pending orders', registry=REGISTRY)

def track_trade_execution(func: Callable) -> Callable:
    """Decorator to track trade execution metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        symbol = kwargs.get('symbol', 'unknown')
        
        try:
            result = func(*args, **kwargs)
            TRADES_TOTAL.labels(status='success', symbol=symbol).inc()
            return result
        except Exception as e:
            TRADES_TOTAL.labels(status='error', symbol=symbol).inc()
            raise
        finally:
            TRADE_DURATION.labels(symbol=symbol).observe(time.time() - start_time)
    
    return wrapper

def track_api_call(endpoint: str):
    """Decorator to track API calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)
                API_CALLS_TOTAL.labels(endpoint=endpoint, status='success').inc()
                return result
            except Exception as e:
                API_CALLS_TOTAL.labels(endpoint=endpoint, status='error').inc()
                raise
        return wrapper
    return decorator
