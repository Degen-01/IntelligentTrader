"""
Production-Ready AI Engine for Live Trading
Supports real-time signal generation, model management, and live market integration
"""

import logging
import pandas as pd
import numpy as np
import joblib
import os
import asyncio
import aiohttp
import websockets
import json
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import queue
import sqlite3
from contextlib import asynccontextmanager

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Technical Analysis
import talib
import ta

# Custom imports
from ..core.exceptions import TradingSystemError, DataValidationError, ModelTrainingError
from ..core.metrics import MODEL_PREDICTIONS_TOTAL, MODEL_ACCURACY, TRADES_TOTAL
from ..core.security import decrypt_api_key
from ..advanced_risk_manager import AdvancedRiskManager
from ..monitoring.alerts import AlertManager

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Real-time market data structure"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None

@dataclass
class TradingSignal:
    """Trading signal with confidence and metadata"""
    timestamp: datetime
    symbol: str
    signal: int  # -1: sell, 0: hold, 1: buy
    confidence: float
    probability: float
    features_used: List[str]
    model_version: str

class LiveDataFeed:
    """Real-time market data feed handler"""
    
    def __init__(self, api_key: str, symbols: List[str]):
        self.api_key = api_key
        self.symbols = symbols
        self.data_queue = queue.Queue(maxsize=10000)
        self.websocket = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
    async def start_feed(self):
        """Start the live data feed"""
        self.running = True
        while self.running and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                await self._connect_websocket()
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                self.reconnect_attempts += 1
                await asyncio.sleep(min(2 ** self.reconnect_attempts, 30))
                
    async def _connect_websocket(self):
        """Connect to WebSocket feed"""
        uri = "wss://stream.binance.com:9443/ws/stream"
        
        async with websockets.connect(uri) as websocket:
            self.websocket = websocket
            self.reconnect_attempts = 0
            
            # Subscribe to streams
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{symbol.lower()}@ticker" for symbol in self.symbols],
                "id": 1
            }
            await websocket.send(json.dumps(subscribe_msg))
            
            async for message in websocket:
                if not self.running:
                    break
                    
                try:
                    data = json.loads(message)
                    if 'data' in data:
                        await self._process_market_data(data['data'])
                except Exception as e:
                    logger.error(f"Error processing market data: {e}")
                    
    async def _process_market_data(self, data: Dict):
        """Process incoming market data"""
        try:
            market_data = MarketData(
                timestamp=datetime.fromtimestamp(data['E'] / 1000),
                symbol=data['s'],
                open=float(data['o']),
                high=float(data['h']),
                low=float(data['l']),
                close=float(data['c']),
                volume=float(data['v']),
                bid=float(data['b']) if 'b' in data else None,
                ask=float(data['a']) if 'a' in data else None
            )
            
            if not self.data_queue.full():
                self.data_queue.put(market_data)
            else:
                # Remove oldest data if queue is full
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.put(market_data)
                except queue.Empty:
                    pass
                    
        except Exception as e:
            logger.error(f"Error creating market data object: {e}")
            
    def get_latest_data(self, max_items: int = 100) -> List[MarketData]:
        """Get latest market data"""
        data = []
        try:
            while len(data) < max_items and not self.data_queue.empty():
                data.append(self.data_queue.get_nowait())
        except queue.Empty:
            pass
        return data
        
    def stop_feed(self):
        """Stop the data feed"""
        self.running = False

class FeatureEngineer:
    """Real-time feature engineering for trading signals"""
    
    def __init__(self):
        self.feature_cache = {}
        self.lookback_periods = <span class="footnote-wrapper">[5](5)[10](10)[20](20)[50](50)[100](100)</span>
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators and features"""
        if len(df) < 100:
            logger.warning(f"Insufficient data for feature calculation: {len(df)} rows")
            return df
            
        try:
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['price_change'] = df['close'] - df['open']
            df['price_range'] = df['high'] - df['low']
            df['body_size'] = abs(df['close'] - df['open'])
            
            # Volume features
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['price_volume'] = df['close'] * df['volume']
            
            # Moving averages
            for period in <span class="footnote-wrapper">[5](5)[10](10)[20](20)[50](50)[100](100)</span>:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
                
            # Technical indicators using TA-Lib
            df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
            df['rsi_7'] = talib.RSI(df['close'].values, timeperiod=7)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].values)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic
            slowk, slowd = talib.STOCH(df['high'].values, df['low'].values, df['close'].values)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
            
            # ADX
            df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values)
            
            # Williams %R
            df['williams_r'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values)
            
            # Commodity Channel Index
            df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values)
            
            # Average True Range
            df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values)
            df['atr_ratio'] = df['atr'] / df['close']
            
            # Momentum indicators
            df['momentum_10'] = talib.MOM(df['close'].values, timeperiod=10)
            df['roc_10'] = talib.ROC(df['close'].values, timeperiod=10)
            
            # Volatility features
            for period in <span class="footnote-wrapper">[10](10)[20](20)[50](50)</span>:
                df[f'volatility_{period}'] = df['returns'].rolling(period).std()
                df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df[f'volatility_{period}'].rolling(50).mean()
                
            # Support and resistance levels
            df['resistance'] = df['high'].rolling(20).max()
            df['support'] = df['low'].rolling(20).min()
            df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
            df['support_distance'] = (df['close'] - df['support']) / df['close']
            
            # Market microstructure features
            if 'bid' in df.columns and 'ask' in df.columns:
                df['bid_ask_spread'] = df['ask'] - df['bid']
                df['mid_price'] = (df['bid'] + df['ask']) / 2
                df['spread_ratio'] = df['bid_ask_spread'] / df['mid_price']
                
            # Time-based features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = df['day_of_week'].isin(<span class="footnote-wrapper">[5](5)[6](6)</span>).astype(int)
            
            # Lagged features
            for lag in <span class="footnote-wrapper">[1](1)[2](2)[3](3)[5](5)</span>:
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'rsi_lag_{lag}'] = df['rsi_14'].shift(lag)
                
            # Rolling statistics
            for window in <span class="footnote-wrapper">[5](5)[10](10)[20](20)</span>:
                df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
                df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
                df[f'returns_skew_{window}'] = df['returns'].rolling(window).skew()
                df[f'returns_kurt_{window}'] = df['returns'].rolling(window).kurt()
                
            # Clean infinite and NaN values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(0)
            
            logger.info(f"Calculated {len(df.columns)} features for {len(df)} data points")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            raise FeatureEngineeringError(f"Feature calculation failed: {e}")

class ProductionAIModel(ABC):
    """Abstract base class for production AI models"""
    
    def __init__(self, model_name: str, model_params: Dict = None):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.is_trained = False
        self.training_metrics = {}
        
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
        
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass
        
    def preprocess_features(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """Preprocess features for model input"""
        if not self.feature_columns:
            self.feature_columns = X.columns.tolist()
            
        # Select only the features used during training
        X_selected = X[self.feature_columns].copy()
        
        # Handle missing values
        X_selected = X_selected.fillna(method='ffill').fillna(0)
        
        # Scale features
        if fit_scaler:
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X_selected)
        else:
            if self.scaler is None:
                raise ModelTrainingError("Scaler not fitted. Train model first.")
            X_scaled = self.scaler.transform(X_selected)
            
        return X_scaled
        
    def save_model(self, path: str):
        """Save model to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(path, f"{self.model_name}_model.joblib")
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(path, f"{self.model_name}_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'feature_columns': self.feature_columns,
            'model_params': self.model_params,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(path, f"{self.model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load model from disk"""
        # Load model
        model_path = os.path.join(path, f"{self.model_name}_model.joblib")
        self.model = joblib.load(model_path)
        
        # Load scaler
        scaler_path = os.path.join(path, f"{self.model_name}_scaler.joblib")
        self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = os.path.join(path, f"{self.model_name}_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        self.feature_columns = metadata['feature_columns']
        self.model_params = metadata['model_params']
        self.training_metrics = metadata['training_metrics']
        self.is_trained = metadata['is_trained']
        
        logger.info(f"Model loaded from {path}")

class XGBoostModel(ProductionAIModel):
    """Production XGBoost model for trading signals"""
    
    def __init__(self, model_name: str = "xgboost", model_params: Dict = None):
        default_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        if model_params:
            default_params.update(model_params)
        super().__init__(model_name, default_params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train XGBoost model"""
        logger.info(f"Training {self.model_name} model...")
        
        # Preprocess features
        X_train_scaled = self.preprocess_features(X_train, fit_scaler=True)
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.model_params)
        
        # Prepare validation data if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.preprocess_features(X_val, fit_scaler=False)
            eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
            
        # Train model
        self.model.fit(
            X_train_scaled, 
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=20,
            verbose=False
        )
        
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train_scaled)
        self.training_metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_f1': f1_score(y_train, train_pred, average='weighted')
        }
        
        if X_val is not None:
            val_pred = self.model.predict(X_val_scaled)
            self.training_metrics.update({
                'val_accuracy': accuracy_score(y_val, val_pred),
                'val_f1': f1_score(y_val, val_pred, average='weighted')
            })
            
        logger.info(f"Training complete. Metrics: {self.training_metrics}")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ModelTrainingError("Model not trained")
            
        X_scaled = self.preprocess_features(X, fit_scaler=False)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions"""
        if not self.is_trained:
            raise ModelTrainingError("Model not trained")
            
        X_scaled = self.preprocess_features(X, fit_scaler=False)
        return self.model.predict_proba(X_scaled)

class LSTMModel(ProductionAIModel):
    """Production LSTM model for trading signals"""
    
    def __init__(self, model_name: str = "lstm", model_params: Dict = None):
        default_params = {
            'sequence_length': 60,
            'lstm_units': <span class="footnote-wrapper">[50](50)[50](50)</span>,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10
        }
        if model_params:
            default_params.update(model_params)
        super().__init__(model_name, default_params)
        
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i, :-1])  # All features except target
            y.append(data[i, -1])  # Target variable
        return np.array(X), np.array(y)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train LSTM model"""
        logger.info(f"Training {self.model_name} model...")
        
        # Combine features and target
        train_data = X_train.copy()
        train_data['target'] = y_train
        
        # Preprocess
        train_scaled = self.preprocess_features(train_data, fit_scaler=True)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(train_scaled, self.model_params['sequence_length'])
        
        # Build model
        self.model = Sequential([
            LSTM(self.model_params['lstm_units']<span class="footnote-wrapper">[0](0)</span>, return_sequences=True, 
                 input_shape=(self.model_params['sequence_length'], X_seq.shape<span class="footnote-wrapper">[2](2)</span>)),
            Dropout(self.model_params['dropout_rate']),
            LSTM(self.model_params['lstm_units']<span class="footnote-wrapper">[1](1)</span>),
            Dropout(self.model_params['dropout_rate']),
            Dense(3, activation='softmax')  # 3 classes: sell, hold, buy
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.model_params['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=self.model_params['patience'], restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            val_data = X_val.copy()
            val_data['target'] = y_val
            val_scaled = self.preprocess_features(val_data, fit_scaler=False)
            X_val_seq, y_val_seq = self._create_sequences(val_scaled, self.model_params['sequence_length'])
            validation_data = (X_val_seq, y_val_seq)
            
        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            batch_size=self.model_params['batch_size'],
            epochs=self.model_params['epochs'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Store training metrics
        self.training_metrics = {
            'train_accuracy': float(history.history['accuracy'][-1]),
            'train_loss': float(history.history['loss'][-1])
        }
        
        if validation_data:
            self.training_metrics.update({
                'val_accuracy': float(history.history['val_accuracy'][-1]),
                'val_loss': float(history.history['val_loss'][-1])
            })
            
        logger.info(f"Training complete. Metrics: {self.training_metrics}")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ModelTrainingError("Model not trained")
            
        # Add dummy target for preprocessing
        X_with_target = X.copy()
        X_with_target['target'] = 0
        
        X_scaled = self.preprocess_features(X_with_target, fit_scaler=False)
        
        if len(X_scaled) < self.model_params['sequence_length']:
            logger.warning(f"Insufficient data for prediction: {len(X_scaled)} < {self.model_params['sequence_length']}")
            return np.array(<span class="footnote-wrapper">[1](1)</span>)  # Default to hold
            
        # Create sequence
        X_seq = X_scaled[-self.model_params['sequence_length']:, :-1].reshape(1, self.model_params['sequence_length'], -1)
        
        # Predict
        probabilities = self.model.predict(X_seq, verbose=0)
        return np.argmax(probabilities, axis=1)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions"""
        if not self.is_trained:
            raise ModelTrainingError("Model not trained")
            
        # Add dummy target for preprocessing
        X_with_target = X.copy()
        X_with_target['target'] = 0
        
        X_scaled = self.preprocess_features(X_with_target, fit_scaler=False)
        
        if len(X_scaled) < self.model_params['sequence_length']:
            logger.warning(f"Insufficient data for prediction: {len(X_scaled)} < {self.model_params['sequence_length']}")
            return np.array([[0.33, 0.34, 0.33]])  # Equal probabilities
            
        # Create sequence
        X_seq = X_scaled[-self.model_params['sequence_length']:, :-1].reshape(1, self.model_params['sequence_length'], -1)
        
        # Predict probabilities
        return self.model.predict(X_seq, verbose=0)

class LiveAIEngine:
    """Production AI Engine for live trading"""
    
    def __init__(self, 
                 symbols: List[str],
                 model_type: str = "xgboost",
                 model_params: Dict = None,
                 data_path: str = "data",
                 model_path: str = "models"):
        
        self.symbols = symbols
        self.model_type = model_type
        self.model_params = model_params or {}
        self.data_path = data_path
        self.model_path = model_path
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.risk_manager = AdvancedRiskManager()
        self.alert_manager = AlertManager()
        
        # Model management
        self.models = {}  # symbol -> model
        self.model_versions = {}
        self.active_models = {}
        
        # Data management
        self.data_feeds = {}  # symbol -> LiveDataFeed
        self.market_data_cache = {}  # symbol -> DataFrame
        self.signal_history = []
        
        # Performance tracking
        self.prediction_count = 0
        self.accuracy_tracker = {}
        
        # Database for persistence
        self._init_database()
        
        # Executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"LiveAIEngine initialized for symbols: {symbols}")
        
    def _init_database(self):
        """Initialize SQLite database for persistence"""
        os.makedirs(self.data_path, exist_ok=True)
        db_path = os.path.join(self.data_path, "trading_engine.db")
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    probability REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    features_used TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    prediction_count INTEGER NOT NULL
                )
            """)
            
    async def initialize_data_feeds(self, api_key: str):
        """Initialize live data feeds for all symbols"""
        for symbol in self.symbols:
            feed = LiveDataFeed(api_key, [symbol])
            self.data_feeds[symbol] = feed
            
            # Start feed in background
            asyncio.create_task(feed.start_feed())
            
        logger.info("Data feeds initialized")
        
    async def train_models(self, 
                          historical_data: Dict[str, pd.DataFrame],
                          target_column: str = 'signal',
                          validation_split: float = 0.2):
        """Train models for all symbols"""
        
        for symbol in self.symbols:
            if symbol not in historical_data:
                logger.warning(f"No historical data for {symbol}")
                continue
                
            try:
                await self._train_symbol_model(
                    symbol, 
                    historical_data[symbol], 
                    target_column, 
                    validation_split
                )
            except Exception as e:
                logger.error(f"Failed to train model for {symbol}: {e}")
                await self.alert_manager.send_alert(
                    f"Model training failed for {symbol}: {e}",
                    severity="ERROR"
                )
                
    async def _train_symbol_model(self, 
                                 symbol: str, 
                                 data: pd.DataFrame, 
                                 target_column: str,
                                 validation_split: float):
        """Train model for a specific symbol"""
        logger.info(f"Training model for {symbol}")
        
        # Feature engineering
        data_with_features = self.feature_engineer.calculate_features(data)
        
        # Prepare target variable
        if target_column not in data_with_features.columns:
            # Create target based on future returns
            data_with_features['future_return'] = data_with_features['close'].shift(-1) / data_with_features['close'] - 1
            data_with_features[target_column] = pd.cut(
                data_with_features['future_return'],
                bins=[-np.inf, -0.001, 0.001, np.inf],
                labels=<span class="footnote-wrapper">[0](0)[1](1)[2](2)  </span># sell, hold, buy
            ).astype(int)
            
        # Remove rows with NaN target
        data_clean = data_with_features.dropna(subset=[target_column])
            # Split data for training and validation using time-series split
            # Ensure enough data points for sequence-based models like LSTM
            if len(data_clean) < 2 * self.model_params.get('sequence_length', 60):
                logger.warning(f"Not enough data for time-series split for {symbol}. Skipping model training.")
                return

            X = data_clean.drop(columns=[target_column, 'future_return'], errors='ignore')
            y = data_clean[target_column]

            tscv = TimeSeriesSplit(n_splits=5) # Example, adjust as needed
            train_idx, val_idx = list(tscv.split(X))[4] # Take the last split for training and validation

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Initialize the specific model type
            model_class = None
            if self.model_type.lower() == "xgboost":
                model_class = XGBoostModel
            elif self.model_type.lower() == "lstm":
                model_class = LSTMModel
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            model_instance = model_class(
                model_name=f"{self.model_type}_{symbol}", 
                model_params=self.model_params
            )
            
            # Train the model
            await asyncio.to_thread(model_instance.train, X_train, y_train, X_val, y_val)
            
            # Evaluate model performance on validation set
            metrics = self._evaluate_model(model_instance, X_val, y_val)
            logger.info(f"Validation metrics for {symbol}: {metrics}")
            
            # Save the trained model and associated metadata
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = os.path.join(self.model_path, symbol, model_version)
            await asyncio.to_thread(model_instance.save_model, model_save_path)
            
            # Store model performance in database
            with sqlite3.connect(os.path.join(self.data_path, "trading_engine.db")) as conn:
                conn.execute(
                    "INSERT INTO model_performance (timestamp, symbol, model_version, accuracy, f1_score, prediction_count) VALUES (?, ?, ?, ?, ?, ?)",
                    (datetime.now().isoformat(), symbol, model_version, metrics['accuracy'], metrics['f1_weighted'], len(y_val))
                )
                conn.commit()
            
            # Set this newly trained model as the active model
            self.active_models[symbol] = model_instance
            self.model_versions[symbol] = model_version
            logger.info(f"Successfully trained and activated model {model_version} for {symbol}")

    def _evaluate_model(self, model: ProductionAIModel, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate the model and return key metrics"""
        X_test_scaled = model.preprocess_features(X_test, fit_scaler=False)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Log and store Prometheus metrics
        MODEL_ACCURACY.labels(model_name=model.model_name, metric_type='accuracy').set(accuracy)
        MODEL_ACCURACY.labels(model_name=model.model_name, metric_type='f1_weighted').set(f1_weighted)
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'classification_report': report
        }

    async def _load_latest_model(self, symbol: str) -> Optional[ProductionAIModel]:
        """Load the latest trained model for a symbol"""
        symbol_model_path = os.path.join(self.model_path, symbol)
        if not os.path.exists(symbol_model_path):
            logger.warning(f"No trained models found for {symbol} at {symbol_model_path}")
            return None
        
        # Find the latest model version
        model_versions = [d for d in os.listdir(symbol_model_path) if os.path.isdir(os.path.join(symbol_model_path, d))]
        if not model_versions:
            logger.warning(f"No model versions found for {symbol}")
            return None
        
        latest_version = max(model_versions)
        model_load_path = os.path.join(symbol_model_path, latest_version)

        model_class = None
        if self.model_type.lower() == "xgboost":
            model_class = XGBoostModel
        elif self.model_type.lower() == "lstm":
            model_class = LSTMModel
        else:
            logger.error(f"Unsupported model type: {self.model_type}")
            return None

        model_instance = model_class(model_name=f"{self.model_type}_{symbol}")
        try:
            await asyncio.to_thread(model_instance.load_model, model_load_path)
            self.active_models[symbol] = model_instance
            self.model_versions[symbol] = latest_version
            logger.info(f"Loaded latest model {latest_version} for {symbol}")
            return model_instance
        except Exception as e:
            logger.error(f"Error loading model for {symbol} from {model_load_path}: {e}")
            return None

    async def deploy_model_version(self, symbol: str, version: str):
        """Deploy a specific model version as active"""
        model_load_path = os.path.join(self.model_path, symbol, version)
        if not os.path.exists(model_load_path):
            raise FileNotFoundError(f"Model version {version} for {symbol} not found at {model_load_path}")

        model_class = None
        if self.model_type.lower() == "xgboost":
            model_class = XGBoostModel
        elif self.model_type.lower() == "lstm":
            model_class = LSTMModel
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        model_instance = model_class(model_name=f"{self.model_type}_{symbol}")
        try:
            await asyncio.to_thread(model_instance.load_model, model_load_path)
            self.active_models[symbol] = model_instance
            self.model_versions[symbol] = version
            logger.info(f"Deployed model version {version} for {symbol}")
            await self.alert_manager.send_alert(f"Model {symbol} deployed to version {version}", severity="INFO")
        except Exception as e:
            logger.error(f"Failed to deploy model version {version} for {symbol}: {e}")
            await self.alert_manager.send_alert(f"Failed to deploy model {symbol} version {version}: {e}", severity="ERROR")
            raise ModelTrainingError(f"Model deployment failed: {e}")

    async def rollback_model_version(self, symbol: str, num_versions_back: int = 1):
        """Rollback to a previous model version"""
        symbol_model_path = os.path.join(self.model_path, symbol)
        model_versions = sorted([d for d in os.listdir(symbol_model_path) if os.path.isdir(os.path.join(symbol_model_path, d))], reverse=True)
        
        if len(model_versions) <= num_versions_back:
            logger.warning(f"Not enough previous versions for rollback for {symbol}")
            await self.alert_manager.send_alert(f"Rollback failed for {symbol}: not enough previous versions", severity="WARNING")
            return

        target_version = model_versions[num_versions_back]
        logger.info(f"Attempting to rollback {symbol} to version {target_version}")
        await self.deploy_model_version(symbol, target_version)

    async def generate_signals_live(self, symbol: str) -> Optional[TradingSignal]:
        """Generate a trading signal for a symbol using the live market data"""
        if symbol not in self.active_models or not self.active_models[symbol].is_trained:
            logger.warning(f"No active or trained model for {symbol}. Cannot generate signal.")
            return None
        
        feed = self.data_feeds.get(symbol)
        if not feed:
            logger.error(f"No live data feed configured for {symbol}.")
            return None

        # Get latest data from feed (e.g., last 100 entries for feature calculation)
        raw_data = feed.get_latest_data(max_items=200) # Need enough data for lookbacks
        if not raw_data:
            logger.warning(f"No live data available for {symbol}.")
            return None

        # Convert to DataFrame
        data_df = pd.DataFrame([md.__dict__ for md in raw_data])
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        data_df = data_df.set_index('timestamp').sort_index()

        # Ensure OHLCV structure and numeric types
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data_df.columns for col in required_cols):
            logger.error(f"Missing required columns in live data for {symbol}: {required_cols}")
            return None
        data_df[required_cols] = data_df[required_cols].apply(pd.to_numeric, errors='coerce')
        
        # Feature engineering on the latest data
        try:
            features_df = self.feature_engineer.calculate_features(data_df)
            latest_features = features_df.tail(1)
            
            if latest_features.empty:
                logger.warning(f"Feature engineering resulted in empty dataframe for {symbol}.")
                return None
                
        except Exception as e:
            logger.error(f"Error during feature engineering for {symbol}: {e}")
            return None

        model = self.active_models[symbol]
        
        try:
            # Predict signal and probabilities
            signal_prediction = model.predict(latest_features)[0]
            signal_probabilities = model.predict_proba(latest_features)[0]
            
            # Map signal (0, 1, 2) to (-1, 0, 1) or other desired enum
            mapped_signal = {0: -1, 1: 0, 2: 1}.get(signal_prediction, 0) # Default to hold if unexpected
            confidence = np.max(signal_probabilities)
            
            # Update prediction count and accuracy if we have ground truth (not in live but for monitoring)
            self.prediction_count += 1
            MODEL_PREDICTIONS_TOTAL.labels(symbol=symbol, model_version=self.model_versions[symbol], signal=mapped_signal).inc()
            
            trading_signal = TradingSignal(
                timestamp=data_df.index[-1],
                symbol=symbol,
                signal=mapped_signal,
                confidence=confidence,
                probability=signal_probabilities[signal_prediction],
                features_used=model.feature_columns,
                model_version=self.model_versions[symbol]
            )
            self.signal_history.append(trading_signal)
            
            # Persist signal to database
            with sqlite3.connect(os.path.join(self.data_path, "trading_engine.db")) as conn:
                conn.execute(
                    "INSERT INTO signals (timestamp, symbol, signal, confidence, probability, model_version, features_used) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (trading_signal.timestamp.isoformat(), trading_signal.symbol, trading_signal.signal,
                     trading_signal.confidence, trading_signal.probability, trading_signal.model_version,
                     json.dumps(trading_signal.features_used))
                )
                conn.commit()
            
            logger.info(f"Generated signal for {symbol}: {trading_signal.signal} with confidence {trading_signal.confidence:.2f}")
            return trading_signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            await self.alert_manager.send_alert(f"Error generating signal for {symbol}: {e}", severity="ERROR")
            return None

    async def run_live_trading_loop(self, signal_interval_seconds: int = 60):
        """Main loop for live trading signal generation and processing"""
        logger.info("Starting live trading signal generation loop...")
        
        # Initial model loading for all symbols
        for symbol in self.symbols:
            await self._load_latest_model(symbol)
            if symbol not in self.active_models:
                logger.error(f"No model loaded for {symbol}. Cannot run live trading for this symbol.")
                await self.alert_manager.send_alert(f"No model loaded for {symbol}. Live trading halted for this symbol.", severity="CRITICAL")
                self.symbols.remove(symbol) # Remove from active symbols if no model

        while True:
            start_time = time.time()
            tasks = []
            for symbol in self.symbols:
                tasks.append(self.generate_signals_live(symbol))
            
            signals = await asyncio.gather(*tasks)
            
            for signal in signals:
                if signal:
                    # Here you would integrate with your execution engine
                    # Example: self.trading_strategy.process_signal(signal)
                    # And then trigger trade execution if risk manager allows
                    
                    # For demonstration, just log
                    logger.debug(f"Signal generated: {signal}")
                    
                    # Increment trades total (placeholder, should be actual trades)
                    TRADES_TOTAL.labels(symbol=signal.symbol, signal=signal.signal).inc()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            sleep_duration = max(0, signal_interval_seconds - elapsed_time)
            
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

class FeatureEngineeringError(TradingSystemError):
    """Custom exception for feature engineering failures."""
    pass
