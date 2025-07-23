"""Advanced risk management with portfolio-level controls."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from .core.exceptions import RiskLimitExceededError
from .core.metrics import RISK_EXPOSURE, DRAWDOWN_CURRENT

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskLimits:
    max_portfolio_risk: float = 0.02  # 2% of portfolio per trade
    max_daily_loss: float = 0.05      # 5% max daily loss
    max_drawdown: float = 0.15        # 15% max drawdown
    max_correlation: float = 0.7      # Max correlation between positions
    max_sector_exposure: float = 0.3  # Max 30% in any sector
    var_confidence: float = 0.95      # VaR confidence level
    max_leverage: float = 3.0         # Max 3x leverage

@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    sector: str
    beta: float = 1.0
    
    @property
    def market_value(self) -> float:
        return abs(self.size * self.current_price)
    
    @property
    def weight(self) -> float:
        return self.market_value

class AdvancedRiskManager:
    def __init__(self, 
                 initial_capital: float,
                 risk_limits: RiskLimits = None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_limits = risk_limits or RiskLimits()
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.peak_capital = initial_capital
        self.trade_history: List[Dict] = []
        self.correlation_matrix = pd.DataFrame()
        
    def update_positions(self, positions: Dict[str, Position]):
        """Update current positions."""
        self.positions = positions
        self._update_metrics()
    
    def _update_metrics(self):
        """Update risk metrics."""
        total_value = sum(pos.market_value for pos in self.positions.values())
        if total_value > 0:
            risk_exposure = (total_value / self.current_capital) * 100
            RISK_EXPOSURE.set(risk_exposure)
        
        current_drawdown = self.calculate_drawdown()
        DRAWDOWN_CURRENT.set(current_drawdown)
    
    def calculate_portfolio_var(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """Calculate portfolio Value at Risk."""
        if not self.positions:
            return 0.0
        
        # Simplified VaR calculation - in production, use historical simulation
        # or Monte Carlo methods with proper correlation matrices
        portfolio_volatility = self._calculate_portfolio_volatility()
        z_score = 1.645 if confidence == 0.95 else 2.33  # 95% or 99%
        
        var = self.current_capital * portfolio_volatility * z_score * np.sqrt(horizon)
        return var
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility."""
        if len(self.positions) < 2:
            return 0.02  # Default 2% volatility
        
        # Simplified calculation - use actual price data in production
        weights = np.array([pos.weight for pos in self.positions.values()])
        weights = weights / weights.sum()
        
        # Assume 20% individual volatility and 0.3 correlation
        individual_vol = 0.20
        correlation = 0.30
        
        portfolio_var = np.sum(weights**2) * individual_vol**2
        portfolio_var += 2 * correlation * individual_vol**2 * np.sum(
            np.outer(weights, weights) - np.diag(weights**2)
        )
        
        return np.sqrt(portfolio_var)
    
    def calculate_drawdown(self) -> float:
        """Calculate current drawdown percentage."""
        if self.peak_capital == 0:
            return 0.0
        return ((self.peak_capital - self.current_capital) / self.peak_capital) * 100
    
    def calculate_position_size(self, 
                              symbol: str,
                              entry_price: float,
                              stop_loss: float,
                              sector: str = "unknown") -> float:
        """Calculate optimal position size using Kelly Criterion and risk limits."""
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            raise ValueError("Entry price and stop loss cannot be equal")
        
        # Maximum risk amount based on portfolio risk limit
        max_risk_amount = self.current_capital * self.risk_limits.max_portfolio_risk
        
        # Basic position size
        base_position_size = max_risk_amount / risk_per_share
        
        # Apply sector concentration limits
        sector_exposure = self._calculate_sector_exposure(sector)
        if sector_exposure >= self.risk_limits.max_sector_exposure:
            logger.warning(f"Sector {sector} exposure at limit: {sector_exposure:.2%}")
            return 0.0
        
        # Apply correlation limits
        correlation_adjustment = self._calculate_correlation_adjustment(symbol)
        adjusted_size = base_position_size * correlation_adjustment
        
        # Apply leverage limits
        total_exposure = sum(pos.market_value for pos in self.positions.values())
        total_exposure += adjusted_size * entry_price
        
        if total_exposure > self.current_capital * self.risk_limits.max_leverage:
            leverage_adjustment = (self.current_capital * self.risk_limits.max_leverage - 
                                 sum(pos.market_value for pos in self.positions.values())) / (adjusted_size * entry_price)
            adjusted_size *= max(0, leverage_adjustment)
        
        return max(0, adjusted_size)
    
    def _calculate_sector_exposure(self, sector: str) -> float:
        """Calculate current exposure to a sector."""
        sector_value = sum(
            pos.market_value for pos in self.positions.values() 
            if pos.sector == sector
        )
        return sector_value / self.current_capital if self.current_capital > 0 else 0
    
    def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Adjust position size based on correlation with existing positions."""
        if not self.positions or symbol not in self.correlation_matrix.columns:
            return 1.0
        
        # Calculate weighted average correlation
        total_weight = 0
        weighted_correlation = 0
        
        for pos_symbol, position in self.positions.items():
            if pos_symbol in self.correlation_matrix.index:
                weight = position.market_value / self.current_capital
                correlation = abs(self.correlation_matrix.loc[pos_symbol, symbol])
                weighted_correlation += correlation * weight
                total_weight += weight
        
        if total_weight == 0:
            return 1.0
        
        avg_correlation = weighted_correlation / total_weight
        
        # Reduce position size if high correlation
        if avg_correlation > self.risk_limits.max_correlation:
            return max(0.1, 1.0 - (avg_correlation - self.risk_limits.max_correlation))
        
        return 1.0
    
    def check_risk_limits(self) -> Tuple[bool, List[str]]:
        """Check all risk limits and return violations."""
        violations = []
        
        # Check drawdown
        current_drawdown = self.calculate_drawdown()
        if current_drawdown > self.risk_limits.max_drawdown * 100:
            violations.append(f"Max drawdown exceeded: {current_drawdown:.2f}% > {self.risk_limits.max_drawdown*100:.2f}%")
        
        # Check daily loss
        daily_loss_pct = abs(self.daily_pnl / self.current_capital) if self.current_capital > 0 else 0
        if self.daily_pnl < 0 and daily_loss_pct > self.risk_limits.max_daily_loss:
            violations.append(f"Daily loss limit exceeded: {daily_loss_pct:.2%} > {self.risk_limits.max_daily_loss:.2%}")
        
        # Check VaR
        var = self.calculate_portfolio_var()
        var_pct = var / self.current_capital if self.current_capital > 0 else 0
        if var_pct > 0.1:  # 10% VaR limit
            violations.append(f"VaR limit exceeded: {var_pct:.2%} > 10%")
        
        # Check leverage
        total_exposure = sum(pos.market_value for pos in self.positions.values())
        leverage = total_exposure / self.current_capital if self.current_capital > 0 else 0
        if leverage > self.risk_limits.max_leverage:
            violations.append(f"Leverage limit exceeded: {leverage:.2f}x > {self.risk_limits.max_leverage:.2f}x")
        
        return len(violations) == 0, violations
    
    def get_risk_level(self) -> RiskLevel:
        """Determine current risk level."""
        is_compliant, violations = self.check_risk_limits()
        
        if not is_compliant:
            return RiskLevel.CRITICAL
        
        # Check warning levels (80% of limits)
        current_drawdown = self.calculate_drawdown()
        daily_loss_pct = abs(self.daily_pnl / self.current_capital) if self.current_capital > 0 else 0
        
        if (current_drawdown > self.risk_limits.max_drawdown * 80 or
            daily_loss_pct > self.risk_limits.max_daily_loss * 0.8):
            return RiskLevel.HIGH
        elif (current_drawdown > self.risk_limits.max_drawdown * 60 or
              daily_loss_pct > self.risk_limits.max_daily_loss * 0.6):
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def should_halt_trading(self) -> bool:
        """Determine if trading should be halted due to risk."""
        is_compliant, violations = self.check_risk_limits()
        
        if not is_compliant:
            logger.critical(f"Trading halted due to risk violations: {violations}")
            return True
        
        return False
    
    def update_correlation_matrix(self, correlation_data: pd.DataFrame):
        """Update correlation matrix with latest data."""
        self.correlation_matrix = correlation_data
        logger.info("Correlation matrix updated")
    
    def log_trade(self, trade_data: Dict):
        """Log trade for risk analysis."""
        self.trade_history.append({
            **trade_data,
            'timestamp': pd.Timestamp.now(),
            'portfolio_value': self.current_capital,
            'drawdown': self.calculate_drawdown()
        })
        
        # Keep only last 1000 trades
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
