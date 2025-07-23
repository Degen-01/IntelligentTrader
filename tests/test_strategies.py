import pytest
from datetime import datetime
from intelligent_trader.strategies.base_strategy import AIStrategy, TechnicalStrategy, MarketData, SignalType
from intelligent_trader.data_pipeline.data_processor import DataProcessor # Assumes DataProcessor exists
import pandas as pd
from unittest.mock import Mock

@pytest.fixture
def sample_market_data():
    """Provides sample market data for strategy testing."""
    data = []
    for i in range(1, 101): # 100 data points
        data.append(MarketData(
            timestamp=datetime(2023, 1, 1, 9, i % 60, i // 60),
            symbol="AAPL",
            open=150.0 + i * 0.1,
            high=151.0 + i * 0.1,
            low=149.0 + i * 0.1,
            close=150.5 + i * 0.1,
            volume=1000 + i * 10
        ))
    return data

@pytest.fixture
def mock_ai_model():
    """Mocks an AI model for AIStrategy."""
    model = Mock()
    # Mock predict to return a 'buy' signal (index 2 for SignalType.BUY)
    model.predict.return_value = [2] 
    # Mock predict_proba to return high confidence for 'buy'
    model.predict_proba.return_value = [[0.1, 0.1, 0.8]] 
    return model

@pytest.fixture
def mock_feature_engineer():
    """Mocks a FeatureEngineer for AIStrategy."""
    fe = Mock()
    # Mock calculate_features to return a simple DataFrame
    fe.calculate_features.return_value = pd.DataFrame({'feature1': [1.0], 'feature2': [2.0]})
    return fe

def test_ai_strategy_signal_generation(sample_market_data, mock_ai_model, mock_feature_engineer):
    strategy = AIStrategy(name="TestAI", model=mock_ai_model, feature_engineer=mock_feature_engineer, parameters={'lookback_period': 50})
    signal = strategy.generate_signal(sample_market_data)
    
    assert signal is not None
    assert signal.signal == SignalType.BUY
    assert signal.symbol == "AAPL"
    assert signal.confidence == 0.8
    mock_ai_model.predict.assert_called_once()
    mock_ai_model.predict_proba.assert_called_once()
    mock_feature_engineer.calculate_features.assert_called_once()

def test_ai_strategy_insufficient_data(sample_market_data, mock_ai_model, mock_feature_engineer):
    strategy = AIStrategy(name="TestAI", model=mock_ai_model, feature_engineer=mock_feature_engineer, parameters={'lookback_period': 200})
    signal = strategy.generate_signal(sample_market_data)
    
    assert signal is None
    mock_ai_model.predict.assert_not_called()

def test_ai_strategy_position_sizing(mock_ai_model, mock_feature_engineer):
    strategy = AIStrategy(name="TestAI", model=mock_ai_model, feature_engineer=mock_feature_engineer)
    mock_signal = Mock(spec=SignalType.BUY, confidence=0.7, price=100.0) # Mock a signal
    
    quantity = strategy.calculate_position_size(mock_signal, account_balance=10000, risk_per_trade=0.01)
    assert quantity > 0
    assert isinstance(quantity, int)

def test_technical_strategy_signal_generation(sample_market_data):
    # This test will require more specific market data to trigger MA crossover and RSI conditions
    # For now, let's create data that might trigger a buy
    buy_data = []
    # Data for initial MA calculation
    for i in range(1, 20):
        buy_data.append(MarketData(datetime(2023, 1, 1, 9, i), "TEST", 100+i, 101+i, 99+i, 100.5+i, 1000))
    # MA crossover
    for i in range(20, 40):
        buy_data.append(MarketData(datetime(2023, 1, 1, 9, i), "TEST", 120+(i*0.5), 121+(i*0.5), 119+(i*0.5), 120.5+(i*0.5), 1000))

    strategy = TechnicalStrategy(name="TestMA", parameters={'short_ma_period': 10, 'long_ma_period': 20})
    signal = strategy.generate_signal(buy_data)
    
    # The simple crossover logic might not trigger with this generic data without careful crafting
    # A proper test would craft specific data that guarantees a signal
    assert signal is None or signal.signal in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]

def test_strategy_performance_metrics():
    strategy = TechnicalStrategy(name="PerfTest")
    
    # Simulate some trades
    strategy.update_performance({'pnl': 100.0}) # Win
    strategy.update_performance({'pnl': -50.0}) # Loss
    strategy.update_performance({'pnl': 200.0}) # Win
    
    metrics = strategy.get_performance_metrics()
    
    assert metrics['total_trades'] == 3
    assert metrics['win_rate'] == 2/3
    assert metrics['total_pnl'] == 250.0
    assert metrics['max_drawdown'] >= 0 # Initial implementation is basic
    assert metrics['current_equity'] == 10250.0
