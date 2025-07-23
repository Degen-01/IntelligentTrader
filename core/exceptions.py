"""Custom exceptions for the trading system."""

class TradingSystemError(Exception):
    """Base exception for trading system."""
    pass

class RiskLimitExceededError(TradingSystemError):
    """Raised when risk limits are breached."""
    pass

class InsufficientFundsError(TradingSystemError):
    """Raised when insufficient funds for trade."""
    pass

class ExchangeConnectionError(TradingSystemError):
    """Raised when exchange connection fails."""
    pass

class OrderExecutionError(TradingSystemError):
    """Raised when order execution fails."""
    pass

class DataValidationError(TradingSystemError):
    """Raised when data validation fails."""
    pass
  
