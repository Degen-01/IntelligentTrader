"""Advanced backtesting engine for rigorous strategy validation."""

import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass, field
import uuid
import warnings

# Suppress pandas FutureWarnings for now if using older pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    trade_id: str
    symbol: str
    entry_time: datetime
    entry_price: float
    side: str # 'buy' or 'sell'
    size: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    status: str = 'OPEN' # 'OPEN', 'CLOSED', 'CANCELED'

@dataclass
class EquityCurve:
    timestamps: List[datetime] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

class BacktestEngine:
    def __init__(self,
                 initial_capital: float = 1_000_000,
                 commission_rate: float = 0.0005,  # 0.05% per trade
                 slippage_bps: float = 2,       # 2 basis points slippage
                 risk_manager: Any = None # Optional: Inject AdvancedRiskManager
                ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.risk_manager = risk_manager
        
        self.current_capital = initial_capital
        self.portfolio_value = initial_capital
        self.trades: List[Trade] = []
        self.open_trades: Dict[str, Trade] = {}
        self.equity_curve = EquityCurve()
        self.current_timestamp: Optional[datetime] = None
        
        if self.risk_manager:
            self.risk_manager.initial_capital = initial_capital
            self.risk_manager.current_capital = initial_capital

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to price."""
        if side == 'buy':
            return price * (1 + self.slippage_bps / 10000)
        elif side == 'sell':
            return price * (1 - self.slippage_bps / 10000)
        return price

    def _execute_order(self, symbol: str, side: str, price: float, size: float) -> Tuple[bool, Optional[Trade]]:
        """Simulate order execution."""
        executed_price = self._apply_slippage(price, side)
        commission = executed_price * size * self.commission_rate
        
        cost = executed_price * size
        if side == 'buy':
            if self.current_capital < cost + commission:
                logger.warning(f"Insufficient capital for {symbol} {side} order. Needed: {cost + commission}, Have: {self.current_capital}")
                return False, None
            self.current_capital -= (cost + commission)
            
            trade = Trade(
                trade_id=str(uuid.uuid4()),
                symbol=symbol,
                entry_time=self.current_timestamp,
                entry_price=executed_price,
                side=side,
                size=size,
            )
            self.open_trades[symbol] = trade
            self.trades.append(trade)
            logger.debug(f"Executed BUY {symbol} at {executed_price:.2f} size {size:.4f}. Remaining capital: {self.current_capital:.2f}")
            return True, trade
        elif side == 'sell':
            if symbol not in self.open_trades:
                logger.warning(f"Cannot SELL {symbol}: no open position.")
                return False, None
            
            open_trade = self.open_trades[symbol]
            if open_trade.side == 'buy': # Selling to close a long position
                if open_trade.size < size:
                    logger.warning(f"Attempting to sell more than open position for {symbol}. Open: {open_trade.size}, Attempted: {size}. Selling available size.")
                    size = open_trade.size
                
                # Calculate PnL for the closed portion
                pnl = (executed_price - open_trade.entry_price) * size
                self.current_capital += (cost - commission) + pnl
                
                # Update open trade or close it fully
                open_trade.size -= size
                if open_trade.size <= 1e-9: # Effectively zero
                    open_trade.exit_time = self.current_timestamp
                    open_trade.exit_price = executed_price
                    open_trade.pnl = pnl # This PnL is for the *entire* original trade.
                    open_trade.status = 'CLOSED'
                    del self.open_trades[symbol]
                    logger.debug(f"Closed BUY {symbol}. PnL: {pnl:.2f}. New capital: {self.current_capital:.2f}")
                else:
                    logger.debug(f"Partially closed BUY {symbol}. Remaining size: {open_trade.size:.4f}")

            # For short positions, this would be reversed
            return True, open_trade # Return the updated open trade

        return False, None

    def _update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update portfolio value including open positions."""
        market_value_of_open_positions = 0
        current_positions_for_risk_manager = {}
        for symbol, trade in self.open_trades.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                market_value = trade.size * current_price
                market_value_of_open_positions += market_value
                unrealized_pnl = (current_price - trade.entry_price) * trade.size * (1 if trade.side == 'buy' else -1)

                current_positions_for_risk_manager[symbol] = AdvancedRiskManager.Position(
                    symbol=symbol,
                    size=trade.size,
                    entry_price=trade.entry_price,
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    sector="crypto" # Placeholder, enrich with actual sector data
                )

        self.portfolio_value = self.current_capital + market_value_of_open_positions
        if self.risk_manager:
            self.risk_manager.current_capital = self.portfolio_value
            self.risk_manager.update_positions(current_positions_for_risk_manager)
            self.risk_manager.peak_capital = max(self.risk_manager.peak_capital, self.portfolio_value)


    async def run_backtest(self,
                           price_data: pd.DataFrame, # OHLCV data, indexed by datetime, columns for each symbol
                           strategy_executor: Callable[[pd.Series, Dict[str, float]], List[Dict]]
                          ) -> Dict[str, Any]:
        """
        Run the backtest.
        price_data: DataFrame with datetime index and 'close' (or other) prices for symbols.
        strategy_executor: A function that takes current market data (a row from price_data)
                           and returns a list of dictionaries, each representing an order:
                           [{'symbol': 'BTC/USD', 'side': 'buy', 'size': 0.01, 'price': 50000}]
        """
        if not isinstance(price_data.index, pd.DatetimeIndex):
            raise ValueError("price_data must have a DatetimeIndex.")
        
        price_data = price_data.sort_index()
        
        self.equity_curve.timestamps = []
        self.equity_curve.values = []
        self.trades = []
        self.open_trades = {}
        self.current_capital = self.initial_capital
        self.portfolio_value = self.initial_capital
        
        if self.risk_manager:
            self.risk_manager.current_capital = self.initial_capital
            self.risk_manager.peak_capital = self.initial_capital
            self.risk_manager.positions = {}
        
        logger.info(f"Starting backtest with initial capital: {self.initial_capital}")

        for i, (timestamp, row) in enumerate(price_data.iterrows()):
            self.current_timestamp = timestamp
            current_prices = row.to_dict() # {symbol: close_price, ...}
            
            # 1. Update portfolio value based on current prices
            self._update_portfolio_value(current_prices)
            self.equity_curve.timestamps.append(timestamp)
            self.equity_curve.values.append(self.portfolio_value)

            # 2. Check risk limits and potentially halt
            if self.risk_manager and self.risk_manager.should_halt_trading():
                logger.warning(f"Risk limits breached at {timestamp}. Halting backtest.")
                break
            
            # 3. Get orders from strategy
            try:
                orders = await asyncio.to_thread(strategy_executor, row, self.open_trades)
            except Exception as e:
                logger.error(f"Error in strategy executor at {timestamp}: {e}")
                continue # Skip this bar, continue to next

            # 4. Execute orders
            for order in orders:
                symbol = order.get('symbol')
                side = order.get('side')
                size = order.get('size')
                price = order.get('price') # The price the strategy *wants* to trade at
                
                if symbol and side and size is not None and price is not None:
                    success, trade = self._execute_order(symbol, side, price, size)
                    if not success:
                        logger.warning(f"Order failed for {symbol} {side} at {timestamp}")
                else:
                    logger.warning(f"Malformed order received from strategy: {order}")
            
            if i % 1000 == 0:
                logger.info(f"Backtest progress: {timestamp}. Portfolio value: {self.portfolio_value:.2f}")

        # Final update
        self._update_portfolio_value(price_data.iloc[-1].to_dict())
        self.equity_curve.timestamps.append(self.current_timestamp)
        self.equity_curve.values.append(self.portfolio_value)

        return self._generate_performance_report()

    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance metrics."""
        equity = np.array(self.equity_curve.values)
        if len(equity) < 2:
            return {"error": "Not enough data for performance analysis."}
        
        returns = pd.Series(equity).pct_change().dropna()
        
        total_return = (equity[-1] / equity[0] - 1) * 100
        
        # Max Drawdown
        peak_value = np.maximum.accumulate(equity)
        drawdowns = (peak_value - equity) / peak_value
        max_drawdown = np.max(drawdowns) * 100

        # Sharpe Ratio (assuming daily data for simplicity, annualize for hourly/minute data)
        # Annualization factor for hourly data: sqrt(252 * 24)
        annualization_factor = np.sqrt(252 * 24) # Assuming hourly data
        if returns.std() == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = returns.mean() / returns.std() * annualization_factor

        # Win Rate
        closed_trades = [t for t in self.trades if t.status == 'CLOSED' and t.pnl is not None]
        winning_trades = sum(1 for t in closed_trades if t.pnl > 0)
        losing_trades = sum(1 for t in closed_trades if t.pnl < 0)
        total_closed_trades = len(closed_trades)
        win_rate = (winning_trades / total_closed_trades) * 100 if total_closed_trades > 0 else 0

        # Average PnL per trade
        avg_pnl_per_trade = sum(t.pnl for t in closed_trades) / total_closed_trades if total_closed_trades > 0 else 0

        # Calmar Ratio (Annualized Return / Max Drawdown)
        annualized_return = ( (equity[-1] / equity[0])**(annualization_factor/len(returns)) - 1 ) * 100
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else float('inf')

        report = {
            "initial_capital": self.initial_capital,
            "final_capital": self.portfolio_value,
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "num_trades": len(self.trades),
            "num_closed_trades": total_closed_trades,
            "win_rate_pct": win_rate,
            "avg_pnl_per_trade": avg_pnl_per_trade,
            "calmar_ratio": calmar_ratio,
            "equity_curve": pd.DataFrame({'timestamp': self.equity_curve.timestamps, 'value': self.equity_curve.values}).set_index('timestamp'),
            "trades": self.trades # All trade objects for detailed analysis
        }
        logger.info(f"Backtest complete. Final portfolio value: {self.portfolio_value:.2f}, Total Return: {total_return:.2f}%")
        return report

