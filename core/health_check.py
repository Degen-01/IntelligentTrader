"""Health check and monitoring system."""

import asyncio
import time
from typing import Dict, List, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    name: str
    check_func: Callable[[], bool]
    timeout: float = 5.0
    critical: bool = True
    last_check: float = field(default_factory=time.time)
    last_status: HealthStatus = HealthStatus.HEALTHY
    failure_count: int = 0

@dataclass
class SystemHealth:
    status: HealthStatus
    checks: Dict[str, HealthStatus]
    timestamp: float
    uptime: float

class HealthMonitor:
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.start_time = time.time()
        self.is_running = False

    def register_check(self, 
                      name: str, 
                      check_func: Callable[[], bool],
                      timeout: float = 5.0,
                      critical: bool = True):
        """Register a health check."""
        self.checks[name] = HealthCheck(
            name=name,
            check_func=check_func,
            timeout=timeout,
            critical=critical
        )
        logger.info(f"Registered health check: {name}")

    async def run_check(self, check: HealthCheck) -> HealthStatus:
        """Run a single health check."""
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(check.check_func),
                timeout=check.timeout
            )
            
            if result:
                check.failure_count = 0
                check.last_status = HealthStatus.HEALTHY
            else:
                check.failure_count += 1
                check.last_status = HealthStatus.DEGRADED if check.failure_count < 3 else HealthStatus.UNHEALTHY
                
        except asyncio.TimeoutError:
            check.failure_count += 1
            check.last_status = HealthStatus.UNHEALTHY
            logger.error(f"Health check {check.name} timed out")
        except Exception as e:
            check.failure_count += 1
            check.last_status = HealthStatus.UNHEALTHY
            logger.error(f"Health check {check.name} failed: {e}")
        
        check.last_check = time.time()
        return check.last_status

    async def check_all(self) -> SystemHealth:
        """Run all health checks."""
        check_results = {}
        
        # Run all checks concurrently
        tasks = [self.run_check(check) for check in self.checks.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for check, result in zip(self.checks.values(), results):
            if isinstance(result, Exception):
                check_results[check.name] = HealthStatus.UNHEALTHY
            else:
                check_results[check.name] = result

        # Determine overall system health
        critical_checks = [
            status for name, status in check_results.items()
            if self.checks[name].critical
        ]
        
        if any(status == HealthStatus.UNHEALTHY for status in critical_checks):
            overall_status = HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in check_results.values()):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return SystemHealth(
            status=overall_status,
            checks=check_results,
            timestamp=time.time(),
            uptime=time.time() - self.start_time
        )

    async def start_monitoring(self, interval: float = 30.0):
        """Start continuous health monitoring."""
        self.is_running = True
        logger.info("Health monitoring started")
        
        while self.is_running:
            try:
                health = await self.check_all()
                logger.info(f"System health: {health.status.value}")
                
                if health.status == HealthStatus.UNHEALTHY:
                    logger.critical("System is unhealthy!")
                    # Trigger alerts here
                    
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
            
            await asyncio.sleep(interval)

    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_running = False
        logger.info("Health monitoring stopped")

# Global health monitor
health_monitor = HealthMonitor()

# Example health checks
def check_database_connection() -> bool:
    """Check if database is accessible."""
    # Implement your database check
    return True

def check_exchange_connectivity() -> bool:
    """Check if exchange API is accessible."""
    # Implement your exchange connectivity check
    return True

def check_memory_usage() -> bool:
    """Check if memory usage is within limits."""
    import psutil
    return psutil.virtual_memory().percent < 90

def check_disk_space() -> bool:
    """Check if disk space is sufficient."""
    import psutil
    return psutil.disk_usage('/').percent < 90

# Register default health checks
health_monitor.register_check("database", check_database_connection, critical=True)
health_monitor.register_check("exchange", check_exchange_connectivity, critical=True)
health_monitor.register_check("memory", check_memory_usage, critical=False)
health_monitor.register_check("disk", check_disk_space, critical=False)
