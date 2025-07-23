"""
Robust backtesting engine with realistic simulation features.
Includes slippage, latency simulation, position sizing, portfolio rebalancing,
and comprehensive performance metrics.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import uuid
import numpy as np
import math
import asyncio
from collections import deque

from intelligent_trader.strategies.base_strategy import BaseStrategy, MarketData, TradingSignal, SignalType
from intelligent_trader.brokers.base_broker import PaperBroker, Order, OrderType, OrderSide, OrderStatus, Account, Position
from intelligent_trader.risk_management.advanced_risk_manager import AdvancedRiskManager # Assuming this exists from prior steps
from intelligent_trader.core.exceptions import TradingSystemError # Assuming this exists from prior steps

logger = logging.getLogger(__name__)

class BacktestEngineError(TradingSystemError):
    """Custom exception for backtest engine failures."""
    pass

class BacktestEngine:
    def __init__(self,
                 historical_data: pd.DataFrame,
                 strategies: List[BaseStrategy],
                 initial_cash: float = 100_000.0,
                 commission_per_trade: float = 0.001,  # 0.1% commission per trade
                 slippage_bps: float = 1.0,           # 1 basis point (0.01%) slippage
                 latency_ms: int = 50,                # 50ms execution latency
                 risk_manager: AdvancedRiskManager = None):
        
        self.historical_data = historical_data.sort_values(by='timestamp').set_index('timestamp')
        self.strategies = strategies
        self.initial_cash = initial_cash
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps
        self.latency_ms = latency_ms
        self.risk_manager = risk_manager if risk_manager else AdvancedRiskManager()

        self.broker = PaperBroker(config={
            'initial_balance': initial_cash,
            'commission': commission_per_trade,
            'slippage': slippage_bps / 10000.0  # Convert bps to percentage
        })

        self.portfolio_history: List[Dict[str, Any]] = []
        self.trade_log: List[Dict[str, Any]] = []
        self.current_timestamp: Optional[datetime] = None
        self.market_data_buffer = deque(maxlen=200) # Buffer to store recent market data for strategies

        # Ensure historical data has required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in self.historical_data.columns for col in required_cols):
            raise BacktestEngineError(f"Historical data must contain: {', '.join(required_cols)}")
        
        # Initialize strategies
        for strategy in self.strategies:
            strategy.reset()

        logger.info(f"Backtest engine initialized with {len(strategies)} strategies.")

    async def _update_portfolio_history(self, timestamp: datetime):
        """Updates the portfolio history with current account and position data."""
        account = await self.broker.get_account_info()
        total_equity = account.balance + sum(pos.market_value for pos in account.positions.values())

        # Update broker's internal market prices for accurate unrealized PnL
        for symbol in account.positions:
            if symbol in self.market_data_buffer:
                latest_price = next((md.close for md in reversed(self.market_data_buffer) if md.symbol == symbol), None)
                if latest_price:
                    self.broker.update_market_price(symbol, latest_price)

        # Re-fetch account info after price update for accurate market_value
        account = await self.broker.get_account_info()
        total_equity = account.balance + sum(pos.market_value for pos in account.positions.values())

        self.portfolio_history.append({
            'timestamp': timestamp,
            'cash': account.balance,
            'total_equity': total_equity,
            'positions': {s: p.quantity * p.average_price for s, p in account.positions.items()} # Example: Value of current positions
        })

    async def _execute_trade(self, signal: TradingSignal):
        """Simulates trade execution including latency and slippage."""
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        # Calculate actual quantity based on risk manager and signal
        current_account = await self.broker.get_account_info()
        account_balance = current_account.balance

        # Ensure we have a valid price for position sizing
        if signal.price <= 0:
            logger.warning(f"Invalid price for signal {signal.symbol} at {signal.timestamp}. Cannot calculate position size.")
            return

        try:
            # Use strategy's own position sizing logic
            quantity_to_trade = signal.quantity # Use the quantity suggested by the strategy
            if quantity_to_trade == 0:
                quantity_to_trade = signal.strategy.calculate_position_size(
                    signal, 
                    account_balance, 
                    self.risk_manager.risk_per_trade_capital
                )
            
            if quantity_to_trade == 0:
                logger.debug(f"Calculated quantity to trade is zero for {signal.symbol}. Skipping trade.")
                return

            if not self.risk_manager.can_enter_position(
                symbol=signal.symbol,
                side=signal.signal.name.lower(),
                quantity=quantity_to_trade,
                price=signal.price,
                current_account_balance=account_balance,
                current_portfolio_value=current_account.buying_power # Using buying_power as a proxy for total portfolio value for risk
            ):
                logger.warning(f"Risk manager prevented trade for {signal.symbol}: {signal.signal.name} {quantity_to_trade} units at {signal.price}")
                return

            order_id = str(uuid.uuid4())
            broker_order = Order(
                id=order_id,
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.signal == SignalType.BUY else OrderSide.SELL,
                type=OrderType.MARKET,
                quantity=quantity_to_trade,
                price=signal.price # This will be adjusted by PaperBroker for slippage
            )
            
            await self.broker.place_order(broker_order)
            
            # Check order status
            status = await self.broker.get_order_status(order_id)
            if status == OrderStatus.FILLED:
                filled_order = self.broker.orders[order_id]
                fill_price = filled_order.average_fill_price
                commission = filled_order.filled_quantity * fill_price * self.broker.commission
                
                # Log the trade
                self.trade_log.append({
                    'timestamp': self.current_timestamp,
                    'symbol': signal.symbol,
                    'strategy': signal.strategy_name,
                    'signal_type': signal.signal.name,
                    'quantity': filled_order.filled_quantity,
                    'price': fill_price,
                    'commission': commission,
                    'order_id': order_id
                })
                
                # Update strategy performance (simplified PnL calculation for backtest)
                # This needs to be more sophisticated to track individual trade PnL
                # and update strategy's state correctly (e.g., position management)
                trade_pnl = 0.0 # Will be calculated based on closing positions
                
                # Update strategy's internal state on successful execution
                strategy_instance = next((s for s in self.strategies if s.name == signal.strategy_name), None)
                if strategy_instance:
                    # Update strategy's position tracking (highly simplified)
                    if signal.signal == SignalType.BUY:
                        strategy_instance.position_size += filled_order.filled_quantity
                        strategy_instance.entry_price = fill_price # Very basic, needs to handle multiple entries
                    elif signal.signal == SignalType.SELL:
                        # For selling, if we hold a position, calculate PnL
                        if strategy_instance.position_size > 0 and filled_order.filled_quantity <= strategy_instance.position_size:
                            trade_pnl = (fill_price - strategy_instance.entry_price) * filled_order.filled_quantity
                            strategy_instance.position_size -= filled_order.filled_quantity
                            if strategy_instance.position_size == 0:
                                strategy_instance.entry_price = 0.0 # Position closed
                        else: # Selling more than held or shorting (if allowed)
                            logger.warning("Attempted to sell more than held or shorting not fully supported in simple backtest PnL.")
                            # For now, assume this is closing part of a position
                            trade_pnl = (fill_price - strategy_instance.entry_price) * filled_order.filled_quantity
                            strategy_instance.position_size -= filled_order.filled_quantity
                    
                    strategy_instance.update_performance({
                        'pnl': trade_pnl,
                        'symbol': signal.symbol,
                        'quantity': filled_order.filled_quantity,
                        'side': signal.signal.name
                    })
                
                logger.info(f"Trade executed for {signal.symbol}: {signal.signal.name} {filled_order.filled_quantity} @ {fill_price:.2f}")
            else:
                logger.warning(f"Order {order_id} for {signal.symbol} was not filled. Status: {status.name}")

        except Exception as e:
            logger.error(f"Error executing trade for {signal.symbol}: {e}")
            raise BacktestEngineError(f"Trade execution failed: {e}")

    async def run(self):
        """Runs the backtest simulation."""
        logger.info("Starting backtest simulation...")
        await self.broker.connect()
        await self._update_portfolio_history(self.historical_data.index[0])

        current_market_data: Dict[str, List[MarketData]] = {}

        for timestamp, row in self.historical_data.iterrows():
            self.current_timestamp = timestamp
            symbol = row.get('symbol', 'UNKNOWN') # Assuming symbol is a column in historical_data
            
            # Prepare MarketData object
            market_data_point = MarketData(
                timestamp=timestamp,
                symbol=symbol,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            
            # Update broker's simulated market price
            self.broker.update_market_price(symbol, market_data_point.close)

            # Store in buffer for strategy lookback
            self.market_data_buffer.append(market_data_point)

            # Organize market data by symbol for strategies
            if symbol not in current_market_data:
                current_market_data[symbol] = []
            current_market_data[symbol].append(market_data_point) # Append new data point

            # Process signals from strategies
            for strategy in self.strategies:
                # Provide only relevant historical data for the strategy's symbol
                strategy_market_data = [md for md in self.market_data_buffer if md.symbol == symbol]
                
                signal = strategy.generate_signal(strategy_market_data)
                if signal:
                    signal.strategy = strategy # Attach strategy instance for position sizing
                    await self._execute_trade(signal)
            
            # Update portfolio history at regular intervals or end of day/bar
            # For simplicity, update at each data point in backtest
            await self._update_portfolio_history(timestamp)

        await self.broker.disconnect()
        logger.info("Backtest simulation finished.")
        self.generate_performance_report()

    def generate_performance_report(self):
        """Generates a comprehensive performance report."""
        if not self.portfolio_history:
            logger.warning("No portfolio history to generate report.")
            return

        portfolio_df = pd.DataFrame(self.portfolio_history).set_index('timestamp')
        portfolio_df['returns'] = portfolio_df['total_equity'].pct_change().fillna(0)

        total_return = (portfolio_df['total_equity'].iloc[-1] / portfolio_df['total_equity'].iloc[0]) - 1
        
        # Max Drawdown
        peak = portfolio_df['total_equity'].expanding(min_periods=1).max()
        drawdown = (portfolio_df['total_equity'] - peak) / peak
        max_drawdown = drawdown.min()

        # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        daily_returns = portfolio_df['returns']
        if daily_returns.std() == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * math.sqrt(252) # Annualized daily Sharpe

        # CAGR (Compound Annual Growth Rate)
        start_date = portfolio_df.index.min()
        end_date = portfolio_df.index.max()
        years = (end_date - start_date).days / 365.25
        if years <= 0:
            cagr = total_return # If less than a year, CAGR is total return
        else:
            cagr = ((portfolio_df['total_equity'].iloc[-1] / portfolio_df['total_equity'].iloc[0]) ** (1/years)) - 1
            
        logger.info("\n--- Backtest Performance Report ---")
        logger.info(f"Initial Capital: ${self.initial_cash:,.2f}")
        logger.info(f"Final Equity: ${portfolio_df['total_equity'].iloc[-1]:,.2f}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"CAGR: {cagr:.2%}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Number of Trades: {len(self.trade_log)}")
        
        # Strategy-specific performance
        for strategy in self.strategies:
            strat_perf = strategy.get_performance_metrics()
            logger.info(f"\n--- Strategy: {strategy.name} Performance ---")
            for metric, value in strat_perf.items():
                if isinstance(value, float):
                    logger.info(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
                else:
                    logger.info(f"  {metric.replace('_', ' ').title()}: {value}")

        logger.info("\n--- Trade Log (first 5 entries) ---")
        for trade in self.trade_log[:5]:
            logger.info(trade)
        if len(self.trade_log) > 5:
            logger.info("...")
