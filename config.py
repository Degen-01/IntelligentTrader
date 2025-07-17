import os
from datetime import timedelta

class Config:
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY') or 'demo'
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY') or ''
    BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY') or ''
    
    # Wallet Configuration
    ETHEREUM_RPC_URL = os.environ.get('ETHEREUM_RPC_URL') or 'https://mainnet.infura.io/v3/your-project-id'
    PRIVATE_KEY = os.environ.get('PRIVATE_KEY') or ''
    
    # Trading Configuration
    INITIAL_BALANCE = 10000.0
    MAX_POSITION_SIZE = 0.1  # 10% of portfolio per trade
    STOP_LOSS_PERCENTAGE = 0.05  # 5% stop loss
    TAKE_PROFIT_PERCENTAGE = 0.15  # 15% take profit
    
    # AI Configuration
    PREDICTION_INTERVAL = timedelta(minutes=5)
    CONFIDENCE_THRESHOLD = 0.7
    
    # Risk Management
    MAX_DAILY_LOSS = 0.02  # 2% max daily loss
    MAX_OPEN_POSITIONS = 5
    