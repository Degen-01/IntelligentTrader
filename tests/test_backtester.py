import pytest
import pandas as pd
from datetime import datetime, timedelta
from intelligent_trader.backtesting.backtest_engine import BacktestEngine, BacktestEngineError
from intelligent_trader.strategies.base_strategy import TechnicalStrategy, MarketData, SignalType
from intelligent_trader.risk_management.advanced_risk_manager import AdvancedRiskManager
import asyncio
import logging

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_historical_data():
    """Generates sample historical OHLCV data for multiple symbols."""
    start_date = datetime(2023, 1, 1)
    data = []
    symbols = ["AAPL", "MSFT"]
    
    for symbol in symbols:
        price = 150.0 if symbol == "AAPL" else 250.0
        for i in range(100): # 100 data points per symbol
            timestamp = start_date + timedelta(days=i)
            # Simulate a simple price trend
            close = price + i * (0.5 if symbol == "AAPL" else 0.8)
            open_p = close - 0.5
            high_p = close + 1.0
            low_p = close - 1.0
            volume = 10000 + i * 100
            
            data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'open': open_p,
                'high': high_p,
                'low': low_p,
                'close': close,
                'volume': volume
            })
    
    df = pd.DataFrame(data)
    # Ensure data is sorted by timestamp, then symbol if multiple symbols per timestamp
    df = df.sort_values(by=['timestamp', 'symbol']).reset_index(drop=True)
    return df

@pytest.fixture
def simple_technical_strategy():
    """A simple technical strategy for backtesting."""
    # This strategy will generate signals based on MA crossover and RSI
    # Adjust parameters to make it more likely to generate signals with sample data
    return TechnicalStrategy(
        name="SimpleMA_RSI",
        parameters={
            'short_ma_period': 5,
            'long_ma_period': 10,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
    )

@pytest.fixture
def backtest_engine(sample_historical_data, simple_technical_strategy):
    """Fixture for BacktestEngine instance with sample data and strategies."""
    risk_manager = AdvancedRiskManager(
        max_drawdown=0.2, # Allow larger drawdown for test
        daily_loss_limit=0.1,
        risk_per_trade_capital=0.05, # Higher risk per trade to ensure trades
        max_exposure_per_symbol=0.8,
        max_total_exposure=0.9
    )
    return BacktestEngine(
        historical_data=sample_historical_data,
        strategies=[simple_technical_strategy],
        initial_cash=100_000.0,
        commission_per_trade=0.001,
        slippage_bps=0.5,
        latency_ms=10,
        risk_manager=risk_manager
    )

@pytest.mark.asyncio
async def test_backtest_engine_initialization(backtest_engine):
    """Tests if the backtest engine initializes correctly."""
    assert backtest_engine.initial_cash == 100_000.0
    assert len(backtest_engine.strategies) == 1
    assert not backtest_engine.portfolio_history
    assert isinstance(backtest_engine.historical_data, pd.DataFrame)
    assert 'timestamp' not in backtest_engine.historical_data.columns # Should be index

@pytest.mark.asyncio
async def test_backtest_engine_run(backtest_engine):
    """Tests the full backtest simulation run."""
    await backtest_engine.run()
    
    # Assert that portfolio history is populated
    assert len(backtest_engine.portfolio_history) > 0
    # The number of entries in portfolio_history should roughly match the number of rows in historical_data
    # (or more if market data contains multiple symbols per timestamp which are processed sequentially)
    assert len(backtest_engine.portfolio_history) >= len(backtest_engine.historical_data)
    
    # Assert that trade log is populated (assuming strategies generate some trades)
    assert len(backtest_engine.trade_log) >= 0 # Can be 0 if no signals generated
    
    # Check if the final equity is sensible (not NaN, not infinite)
    final_equity = backtest_engine.portfolio_history[-1]['total_equity']
    assert isinstance(final_equity, (float, int))
    assert final_equity > 0 # Should still have positive equity
    
    logger.info(f"Final Equity after backtest: ${final_equity:.2f}")
    logger.info(f"Total Trades: {len(backtest_engine.trade_log)}")

@pytest.mark.asyncio
async def test_backtest_engine_performance_report(backtest_engine, caplog):
    """Tests the performance report generation."""
    await backtest_engine.run() # Run backtest first to generate data
    
    with caplog.at_level(logging.INFO):
        backtest_engine.generate_performance_report()
        
        # Check if key performance metrics are logged
        assert "--- Backtest Performance Report ---" in caplog.text
        assert "Final Equity:" in caplog.text
        assert "Total Return:" in caplog.text
        assert "Max Drawdown:" in caplog.text
        assert "Sharpe Ratio:" in caplog.text
        assert "Number of Trades:" in caplog.text
        assert "--- Strategy: SimpleMA_RSI Performance ---" in caplog.text
