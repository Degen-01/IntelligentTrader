import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ActionType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

@dataclass
class TradingState:
    """Represents the current state of the trading environment"""
    market_data: np.ndarray
    portfolio_value: float
    cash_balance: float
    positions: Dict[str, int]
    unrealized_pnl: float
    realized_pnl: float
    transaction_costs: float
    step: int

class TradingEnv(gym.Env):
    """
    A comprehensive trading environment for reinforcement learning.
    Supports multiple assets, realistic transaction costs, and complex reward functions.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_cost_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        lookback_window: int = 20,
        max_position_size: float = 0.1,  # 10% of portfolio per position
        reward_scaling: float = 1.0,
        symbols: List[str] = None
    ):
        super().__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost_rate = transaction_cost_rate
        self.slippage_rate = slippage_rate
        self.lookback_window = lookback_window
        self.max_position_size = max_position_size
        self.reward_scaling = reward_scaling
        self.symbols = symbols or ['AAPL']  # Default to single asset
        
        # Validate data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Environment state
        self.current_step = 0
        self.max_steps = len(self.data) - self.lookback_window - 1
        
        # Portfolio state
        self.cash_balance = initial_balance
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.portfolio_history = []
        self.trade_history = []
        
        # Define action space: [Hold, Buy, Sell] for each symbol
        # For simplicity, we'll use discrete actions
        self.n_actions_per_symbol = 3
        self.action_space = spaces.MultiDiscrete([self.n_actions_per_symbol] * len(self.symbols))
        
        # Define observation space
        # Market features + portfolio features
        market_features = len(required_columns) * len(self.symbols)  # OHLCV per symbol
        portfolio_features = 3 + len(self.symbols)  # cash, portfolio_value, total_pnl + positions
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, market_features + portfolio_features),
            dtype=np.float32
        )
        
        logger.info(f"TradingEnv initialized with {len(self.symbols)} symbols, "
                   f"observation space: {self.observation_space.shape}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.cash_balance = self.initial_balance
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.portfolio_history = []
        self.trade_history = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        logger.debug(f"Environment reset. Initial portfolio value: {self._get_portfolio_value():.2f}")
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        if self.current_step >= self.max_steps:
            raise ValueError("Episode has ended. Call reset() to start a new episode.")
        
        # Execute actions for each symbol
        total_reward = 0.0
        current_prices = self._get_current_prices()
        
        for i, symbol in enumerate(self.symbols):
            symbol_action = ActionType(action[i])
            reward = self._execute_trade(symbol, symbol_action, current_prices[symbol])
            total_reward += reward
        
        # Update portfolio history
        portfolio_value = self._get_portfolio_value()
        self.portfolio_history.append({
            'step': self.current_step,
            'portfolio_value': portfolio_value,
            'cash_balance': self.cash_balance,
            'positions': self.positions.copy(),
            'timestamp': self.data.index[self.current_step]
        })
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = portfolio_value <= self.initial_balance * 0.1  # Stop if 90% loss
        
        observation = self._get_observation()
        info = self._get_info()
        
        # Scale reward
        scaled_reward = total_reward * self.reward_scaling
        
        return observation, scaled_reward, terminated, truncated, info

    def _execute_trade(self, symbol: str, action: ActionType, current_price: float) -> float:
        """Execute a trade for a specific symbol and return the immediate reward"""
        reward = 0.0
        
        if action == ActionType.BUY:
            # Calculate maximum shares we can buy
            available_cash = self.cash_balance * self.max_position_size
            max_shares = int(available_cash / (current_price * (1 + self.transaction_cost_rate + self.slippage_rate)))
            
            if max_shares > 0:
                # Execute buy order
                execution_price = current_price * (1 + self.slippage_rate)
                total_cost = max_shares * execution_price * (1 + self.transaction_cost_rate)
                
                if total_cost <= self.cash_balance:
                    self.cash_balance -= total_cost
                    self.positions[symbol] += max_shares
                    
                    # Record trade
                    self.trade_history.append({
                        'step': self.current_step,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': max_shares,
                        'price': execution_price,
                        'cost': total_cost,
                        'timestamp': self.data.index[self.current_step]
                    })
                    
                    logger.debug(f"Bought {max_shares} shares of {symbol} at {execution_price:.2f}")
                else:
                    reward -= 0.01  # Small penalty for invalid action
        
        elif action == ActionType.SELL:
            if self.positions[symbol] > 0:
                # Sell all shares of this symbol
                shares_to_sell = self.positions[symbol]
                execution_price = current_price * (1 - self.slippage_rate)
                total_revenue = shares_to_sell * execution_price * (1 - self.transaction_cost_rate)
                
                self.cash_balance += total_revenue
                self.positions[symbol] = 0
                
                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': execution_price,
                    'revenue': total_revenue,
                    'timestamp': self.data.index[self.current_step]
                })
                
                logger.debug(f"Sold {shares_to_sell} shares of {symbol} at {execution_price:.2f}")
            else:
                reward -= 0.01  # Small penalty for invalid action
        
        # ActionType.HOLD requires no action
        return reward

    def _get_observation(self) -> np.ndarray:
        """Get the current observation"""
        # Get market data for lookback window
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        market_data = self.data.iloc[start_idx:end_idx][['open', 'high', 'low', 'close', 'volume']].values
        
        # Pad if necessary
        if market_data.shape[0] < self.lookback_window:
            padding = np.zeros((self.lookback_window - market_data.shape[0], market_data.shape[1]))
            market_data = np.vstack([padding, market_data])
        
        # Normalize market data (simple min-max normalization)
        market_data_norm = self._normalize_market_data(market_data)
        
        # Get portfolio features
        portfolio_value = self._get_portfolio_value()
        portfolio_features = np.array([
            self.cash_balance / self.initial_balance,  # Normalized cash
            portfolio_value / self.initial_balance,    # Normalized portfolio value
            (portfolio_value - self.initial_balance) / self.initial_balance,  # Normalized PnL
        ] + [self.positions[symbol] for symbol in self.symbols])  # Position sizes
        
        # Repeat portfolio features for each timestep in lookback window
        portfolio_features_expanded = np.tile(portfolio_features, (self.lookback_window, 1))
        
        # Combine market and portfolio features
        observation = np.hstack([market_data_norm, portfolio_features_expanded])
        
        return observation.astype(np.float32)

    def _normalize_market_data(self, data: np.ndarray) -> np.ndarray:
        """Simple normalization of market data"""
        # Use percentage change for price data, log for volume
        normalized = np.zeros_like(data)
        
        for i in range(data.shape[1]):
            if i < 4:  # OHLC data
                if data[0, i] != 0:
                    normalized[:, i] = (data[:, i] - data[0, i]) / data[0, i]
            else:  # Volume data
                normalized[:, i] = np.log1p(data[:, i]) / 10  # Log normalization
        
        return normalized

    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols"""
        current_row = self.data.iloc[self.current_step]
        return {symbol: current_row['close'] for symbol in self.symbols}

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        current_prices = self._get_current_prices()
        positions_value = sum(
            self.positions[symbol] * current_prices[symbol] 
            for symbol in self.symbols
        )
        return self.cash_balance + positions_value

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state"""
        portfolio_value = self._get_portfolio_value()
        return {
            'step': self.current_step,
            'portfolio_value': portfolio_value,
            'cash_balance': self.cash_balance,
            'positions': self.positions.copy(),
            'total_return': (portfolio_value - self.initial_balance) / self.initial_balance,
            'num_trades': len(self.trade_history),
            'timestamp': self.data.index[self.current_step] if self.current_step < len(self.data) else None
        }

    def render(self, mode: str = 'human'):
        """Render the environment"""
        if mode == 'human':
            portfolio_value = self._get_portfolio_value()
            total_return = (portfolio_value - self.initial_balance) / self.initial_balance
            
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${portfolio_value:,.2f}")
            print(f"Cash Balance: ${self.cash_balance:,.2f}")
            print(f"Total Return: {total_return:.2%}")
            print(f"Positions: {self.positions}")
            print("-" * 50)

    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame"""
        return pd.DataFrame(self.portfolio_history)

    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        return pd.DataFrame(self.trade_history)
