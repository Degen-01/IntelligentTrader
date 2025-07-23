"""
Main entry point for the IntelligentTrader bot.
Orchestrates data loading, model training, backtesting, and live trading.
"""

import asyncio
import logging
import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

# Core components
from intelligent_trader.core.config import Config # Assuming config.py is available
from intelligent_trader.core.logger import setup_logging # Assuming logger.py is available
from intelligent_trader.core.exceptions import TradingSystemError

# Data pipeline
from intelligent_trader.data_pipeline.data_loader import DataLoader # Assuming data_loader.py for fetching data
from intelligent_trader.data_pipeline.data_processor import DataProcessor # Assuming data_processor.py is available
from intelligent_trader.data_pipeline.feature_engineer import FeatureEngineer # Assuming feature_engineer.py for live fe

# AI Engine & MLOps
from intelligent_trader.ai_engine import LiveAIEngine # Assuming live_ai_engine.py is available
from intelligent_trader.mlops.model_store import ModelStore
from intelligent_trader.mlops.model_evaluator import ModelEvaluator

# Backtesting
from intelligent_trader.backtesting.backtest_engine import BacktestEngine

# Strategies
from intelligent_trader.strategies.base_strategy import TechnicalStrategy, AIStrategy # Import concrete strategies

# Broker
from intelligent_trader.brokers.base_broker import PaperBroker, AlpacaBroker

# Risk Management
from intelligent_trader.risk_management.advanced_risk_manager import AdvancedRiskManager

# Monitoring
from intelligent_trader.monitoring.alerts import AlertManager # Assuming alerts.py is available

logger = logging.getLogger(__name__)

async def run_backtest_workflow(config: Config):
    """
    Executes the backtesting workflow.
    """
    logger.info("Starting backtesting workflow...")
    
    try:
        # 1. Load historical data
        data_loader = DataLoader(config)
        # For backtesting, assume a local CSV for simplicity or fetch from a known source
        # For production, this would involve more sophisticated data ingestion
        historical_data_path = config.get("BACKTEST_HISTORICAL_DATA_PATH", "data/sample_ohlcv.csv")
        
        # Create dummy historical data if not exists for testing purposes
        if not os.path.exists(historical_data_path):
            logger.warning(f"Sample historical data not found at {historical_data_path}. Generating dummy data.")
            dummy_data = []
            start_date = datetime(2022, 1, 1)
            for i in range(365):
                date = start_date + timedelta(days=i)
                price = 100 + np.sin(i / 10) * 10 + np.random.rand() * 2
                dummy_data.append({
                    'timestamp': date, 'symbol': 'TEST',
                    'open': price - 0.5, 'high': price + 1, 'low': price - 1,
                    'close': price, 'volume': 1000 + np.random.randint(0, 100)
                })
            pd.DataFrame(dummy_data).to_csv(historical_data_path, index=False)
            logger.info("Dummy historical data generated.")
        
        historical_data = pd.read_csv(historical_data_path, parse_dates=['timestamp'])
        
        # 2. Process data (feature engineering)
        data_processor = DataProcessor(config.get("DATA_PROCESSOR_CONFIG", {}))
        processed_data = data_processor.create_features(historical_data)
        logger.info(f"Processed historical data with {len(processed_data.columns)} features.")

        # 3. Initialize strategies
        # Example: Technical Strategy
        technical_strategy = TechnicalStrategy(name="MACrossoverRSI", parameters=config.get("STRATEGY_TECH_PARAMS", {}))
        
        # Example: AI Strategy (needs an AI model, will use placeholder for backtest)
        # For backtest, we might train a model on the fly or load a pre-trained one
        ai_model_name = "XGBoostModel" # Placeholder for AI Strategy
        model_store = ModelStore()
        
        # For demonstration, assume we have a simple AI strategy that just uses processed_data for signals.
        # In a real backtest, the AIStrategy would need to be integrated with a model trained specifically for the backtest period.
        # This part assumes that `AIStrategy` can work with a mock/dummy model or that model training is handled externally for backtesting.
        # For now, let's just use the technical strategy for a simpler backtest run.
        
        strategies_for_backtest = [technical_strategy]
        
        # 4. Initialize Risk Manager
        risk_manager = AdvancedRiskManager(
            max_drawdown=config.get("RISK_MAX_DRAWDOWN", 0.05),
            daily_loss_limit=config.get("RISK_DAILY_LOSS_LIMIT", 0.02),
            risk_per_trade_capital=config.get("RISK_PER_TRADE_CAPITAL", 0.01)
        )
        
        # 5. Initialize Backtest Engine
        backtest_engine = BacktestEngine(
            historical_data=processed_data, # Use processed data for backtest
            strategies=strategies_for_backtest,
            initial_cash=config.get("BACKTEST_INITIAL_CASH", 100000.0),
            commission_per_trade=config.get("BROKER_COMMISSION_PER_TRADE", 0.001),
            slippage_bps=config.get("BROKER_SLIPPAGE_BPS", 1.0),
            risk_manager=risk_manager
        )
        
        # 6. Run Backtest
        await backtest_engine.run()
        
        logger.info("Backtesting workflow completed.")

    except Exception as e:
        logger.critical(f"Backtesting workflow failed: {e}", exc_info=True)
        # Potentially send an alert
        alert_manager = AlertManager(config.get("ALERTS_CONFIG", {}))
        await alert_manager.send_alert("CRITICAL", f"Backtesting failed: {e}")

async def run_live_trading_workflow(config: Config):
    """
    Executes the live trading workflow.
    """
    logger.info("Starting live trading workflow...")

    try:
        # 1. Initialize Alerts
        alert_manager = AlertManager(config.get("ALERTS_CONFIG", {}))
        
        # 2. Initialize Broker (Alpaca for live, PaperBroker for dry run)
        broker_type = config.get("LIVE_BROKER_TYPE", "paper").lower()
        if broker_type == "alpaca":
            broker = AlpacaBroker(config.get("ALPACA_CONFIG", {}))
        elif broker_type == "paper":
            broker = PaperBroker(config.get("PAPER_BROKER_CONFIG", {}))
        else:
            raise TradingSystemError(f"Unknown broker type: {broker_type}")
        
        await broker.connect()
        logger.info(f"Connected to {broker_type} broker.")

        # 3. Initialize Risk Manager
        risk_manager = AdvancedRiskManager(
            max_drawdown=config.get("RISK_MAX_DRAWDOWN", 0.05),
            daily_loss_limit=config.get("RISK_DAILY_LOSS_LIMIT", 0.02),
            risk_per_trade_capital=config.get("RISK_PER_TRADE_CAPITAL", 0.01)
        )
        
        # 4. Initialize Data Loader (for live data feed, e.g., WebSocket)
        # Assuming LiveDataFeed uses a separate mechanism from DataLoader
        from intelligent_trader.data_pipeline.live_data_feed import LiveDataFeed # Needs to be defined
        live_data_feed = LiveDataFeed(config.get("LIVE_DATA_FEED_CONFIG", {}))
        
        # 5. Initialize Feature Engineer for live data
        feature_engineer = FeatureEngineer(config.get("FEATURE_ENGINEER_CONFIG", {}))

        # 6. Initialize Model Store and Evaluator for AI Engine
        model_store = ModelStore(base_path=config.get("MODEL_STORE_PATH", "models/model_store"))
        model_evaluator = ModelEvaluator(model_name=config.get("AI_MODEL_NAME", "XGBoostClassifier"), 
                                         model_type=config.get("AI_MODEL_TYPE", "classification"))

        # 7. Initialize AI Engine
        ai_engine = LiveAIEngine(
            model_store=model_store,
            model_evaluator=model_evaluator,
            symbols=config.get("TRADING_SYMBOLS", ["AAPL", "MSFT"]),
            model_update_interval_hours=config.get("MODEL_UPDATE_INTERVAL_HOURS", 24)
        )

        # 8. Initialize Strategies for live trading
        live_strategies = []
        if config.get("ENABLE_TECHNICAL_STRATEGY", False):
            live_strategies.append(TechnicalStrategy(name="LiveMACrossoverRSI", parameters=config.get("STRATEGY_TECH_PARAMS", {})))
        if config.get("ENABLE_AI_STRATEGY", True): # AI strategy usually central to such a bot
            # AIStrategy requires the AI Engine to generate signals
            live_strategies.append(AIStrategy(name="LiveAITrader", ai_engine=ai_engine, parameters=config.get("STRATEGY_AI_PARAMS", {})))
        
        if not live_strategies:
            raise TradingSystemError("No trading strategies enabled for live trading.")

        # 9. Start the live trading loop
        await ai_engine.run_live_trading_loop(
            live_data_feed=live_data_feed,
            feature_engineer=feature_engineer,
            broker=broker,
            risk_manager=risk_manager,
            alert_manager=alert_manager,
            strategies=live_strategies,
            signal_interval_seconds=config.get("LIVE_SIGNAL_INTERVAL_SECONDS", 60)
        )
        
        logger.info("Live trading workflow completed (or stopped).")

    except TradingSystemError as e:
        logger.critical(f"Live trading workflow halted due to configuration error: {e}", exc_info=True)
        alert_manager = AlertManager(config.get("ALERTS_CONFIG", {}))
        await alert_manager.send_alert("CRITICAL", f"Live trading configuration error: {e}")
    except Exception as e:
        logger.critical(f"Live trading workflow failed unexpectedly: {e}", exc_info=True)
        alert_manager = AlertManager(config.get("ALERTS_CONFIG", {}))
        await alert_manager.send_alert("CRITICAL", f"Live trading unexpected failure: {e}")
    finally:
        if 'broker' in locals() and broker.is_connected:
            await broker.disconnect()
            logger.info("Disconnected from broker.")

async def main():
    """Main function to load configuration and run selected workflow."""
    setup_logging() # Initialize global logging configuration
    
    config = Config()
    
    workflow_type = config.get("WORKFLOW_TYPE", "backtest").lower()

    if workflow_type == "backtest":
        await run_backtest_workflow(config)
    elif workflow_type == "live":
        await run_live_trading_workflow(config)
    else:
        logger.error(f"Unknown WORKFLOW_TYPE: {workflow_type}. Must be 'backtest' or 'live'.")

if __name__ == "__main__":
    asyncio.run(main())

