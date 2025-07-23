"""
Abstract base class for all trading strategies
Provides a plugin framework for modular strategy development
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class TradingSignal:
    timestamp: datetime
    symbol: str
    signal: SignalType
    confidence: float
    price: float
    quantity: int
    strategy_name: str
    metadata: Dict[str, Any] = None

@dataclass
class MarketData:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        self.is_active = True
        self.position_size = 0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        self.current_equity = 10000.0  # Starting equity
        self.trade_history = []
        
    @abstractmethod
    def generate_signal(self, market_data: List[MarketData]) -> Optional[TradingSignal]:
        """Generate trading signal based on market data"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal, account_balance: float, risk_per_trade: float) -> int:
        """Calculate position size based on risk management rules"""
        pass
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """Update strategy performance metrics"""
        self.trade_count += 1
        pnl = trade_result.get('pnl', 0.0)
        self.realized_pnl += pnl
        
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
            
        self.current_equity += pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
            
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            
        self.trade_history.append(trade_result)
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics"""
        win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0.0
        avg_win = np.mean([t['pnl'] for t in self.trade_history if t['pnl'] > 0]) if self.win_count > 0 else 0.0
        avg_loss = np.mean([t['pnl'] for t in self.trade_history if t['pnl'] < 0]) if self.loss_count > 0 else 0.0
        profit_factor = abs(avg_win * self.win_count / (avg_loss * self.loss_count)) if avg_loss != 0 else float('inf')
        
        return {
            'total_trades': self.trade_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': self.realized_pnl,
            'max_drawdown': self.max_drawdown,
            'current_equity': self.current_equity,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def reset(self):
        """Reset strategy state"""
        self.position_size = 0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        self.current_equity = 10000.0
        self.trade_history = []

class AIStrategy(BaseStrategy):
    """AI-based trading strategy using machine learning models"""
    
    def __init__(self, name: str, model, feature_engineer, parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.model = model
        self.feature_engineer = feature_engineer
        self.lookback_period = parameters.get('lookback_period', 100)
        self.confidence_threshold = parameters.get('confidence_threshold', 0.6)
        
    def generate_signal(self, market_data: List[MarketData]) -> Optional[TradingSignal]:
        """Generate AI-based trading signal"""
        if len(market_data) < self.lookback_period:
            return None
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': md.timestamp,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume
            } for md in market_data])
            
            df.set_index('timestamp', inplace=True)
            
            # Generate features
            features_df = self.feature_engineer.calculate_features(df)
            latest_features = features_df.tail(1)
            
            if latest_features.empty:
                return None
                
            # Get prediction
            prediction = self.model.predict(latest_features)[0]
            probabilities = self.model.predict_proba(latest_features)[0]
            confidence = np.max(probabilities)
            
            if confidence < self.confidence_threshold:
                return None
                
            # Map prediction to signal
            signal_map = {0: SignalType.SELL, 1: SignalType.HOLD, 2: SignalType.BUY}
            signal = signal_map.get(prediction, SignalType.HOLD)
            
            if signal == SignalType.HOLD:
                return None
                
            latest_data = market_data[-1]
            
            return TradingSignal(
                timestamp=latest_data.timestamp,
                symbol=latest_data.symbol,
                signal=signal,
                confidence=confidence,
                price=latest_data.close,
                quantity=0,  # Will be calculated by position sizing
                strategy_name=self.name,
                metadata={'probabilities': probabilities.tolist()}
            )
            
        except Exception as e:
            logger.error(f"Error generating AI signal: {e}")
            return None
    
    def calculate_position_size(self, signal: TradingSignal, account_balance: float, risk_per_trade: float) -> int:
        """Calculate position size using Kelly Criterion with AI confidence"""
        base_risk = account_balance * risk_per_trade
        confidence_multiplier = signal.confidence
        adjusted_risk = base_risk * confidence_multiplier
        
        # Simple position sizing - can be enhanced with volatility adjustment
        position_value = adjusted_risk / 0.02  # Assuming 2% stop loss
        quantity = int(position_value / signal.price)
        
        return max(1, quantity)

class TechnicalStrategy(BaseStrategy):
    """Rule-based technical analysis strategy"""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.short_ma_period = parameters.get('short_ma_period', 10)
        self.long_ma_period = parameters.get('long_ma_period', 20)
        self.rsi_period = parameters.get('rsi_period', 14)
        self.rsi_oversold = parameters.get('rsi_oversold', 30)
        self.rsi_overbought = parameters.get('rsi_overbought', 70)
        
    def generate_signal(self, market_data: List[MarketData]) -> Optional[TradingSignal]:
        """Generate technical analysis signal"""
        if len(market_data) < max(self.long_ma_period, self.rsi_period):
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame([{
            'close': md.close,
            'high': md.high,
            'low': md.low
        } for md in market_data])
        
        # Calculate indicators
        df['short_ma'] = df['close'].rolling(self.short_ma_period).mean()
        df['long_ma'] = df['close'].rolling(self.long_ma_period).mean()
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signal = SignalType.HOLD
        confidence = 0.5
        
        # Moving average crossover + RSI confirmation
        if (latest['short_ma'] > latest['long_ma'] and 
            prev['short_ma'] <= prev['long_ma'] and 
            latest['rsi'] < self.rsi_overbought):
            signal = SignalType.BUY
            confidence = 0.8
        elif (latest['short_ma'] < latest['long_ma'] and 
              prev['short_ma'] >= prev['long_ma'] and 
              latest['rsi'] > self.rsi_oversold):
            signal = SignalType.SELL
            confidence = 0.8
            
        if signal == SignalType.HOLD:
            return None
            
        latest_data = market_data[-1]
        
        return TradingSignal(
            timestamp=latest_data.timestamp,
            symbol=latest_data.symbol,
            signal=signal,
            confidence=confidence,
            price=latest_data.close,
            quantity=0,
            strategy_name=self.name,
            metadata={
                'short_ma': latest['short_ma'],
                'long_ma': latest['long_ma'],
                'rsi': latest['rsi']
            }
        )
    
    def calculate_position_size(self, signal: TradingSignal, account_balance: float, risk_per_trade: float) -> int:
        """Calculate position size based on fixed risk percentage"""
        risk_amount = account_balance * risk_per_trade
        # Assuming 2% stop loss
        position_value = risk_amount / 0.02
        quantity = int(position_value / signal.price)
        return max(1, quantity)

class StrategyManager:
    """Manages multiple trading strategies"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategies: List[str] = []
        
    def register_strategy(self, strategy: BaseStrategy):
        """Register a new strategy"""
        self.strategies[strategy.name] = strategy
        if strategy.is_active:
            self.active_strategies.append(strategy.name)
        logger.info(f"Registered strategy: {strategy.name}")
        
    def deactivate_strategy(self, strategy_name: str):
        """Deactivate a strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].is_active = False
            if strategy_name in self.active_strategies:
                self.active_strategies.remove(strategy_name)
        
    def activate_strategy(self, strategy_name: str):
        """Activate a strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].is_active = True
            if strategy_name not in self.active_strategies:
                self.active_strategies.append(strategy_name)
                
    def get_signals(self, market_data: List[MarketData]) -> List[TradingSignal]:
        """Get signals from all active strategies"""
        signals = []
        for strategy_name in self.active_strategies:
            strategy = self.strategies[strategy_name]
            signal = strategy.generate_signal(market_data)
            if signal:
                signals.append(signal)
        return signals
        
    def get_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all strategies"""
        performance = {}
        for name, strategy in self.strategies.items():
            performance[name] = strategy.get_performance_metrics()
        return performance
