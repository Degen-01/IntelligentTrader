import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from datetime import datetime, timedelta
import logging

class AITradingEngine:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.confidence_threshold = 0.7
        
    def prepare_features(self, data):
        """Prepare features for ML model"""
        df = data.copy()
        
        # Technical indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'] = self.calculate_macd(df['close'])
        df['bollinger_upper'], df['bollinger_lower'] = self.calculate_bollinger_bands(df['close'])
        
        # Price features
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # Remove NaN values
        df = df.dropna()
        
        feature_columns = ['sma_20', 'sma_50', 'rsi', 'macd', 'bollinger_upper', 
                          'bollinger_lower', 'price_change', 'volume_change', 
                          'high_low_ratio', 'volatility']
        
        return df[feature_columns]
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        return exp1 - exp2
    
    def calculate_bollinger_bands(self, prices, period=20):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, lower
    
    def train_model(self, historical_data):
        """Train the AI model"""
        try:
            features = self.prepare_features(historical_data)
            
            # Create target variable (future price movement)
            target = historical_data['close'].shift(-1).pct_change()
            target = target[features.index]
            
            # Remove NaN values
            valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
            features = features[valid_indices]
            target = target[valid_indices]
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train Random Forest model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(features_scaled, target)
            
            self.is_trained = True
            logging.info("AI model trained successfully")
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
    
    def predict_price_movement(self, current_data):
        """Predict future price movement"""
        if not self.is_trained:
            return None, 0
        
        try:
            features = self.prepare_features(current_data)
            if len(features) == 0:
                return None, 0
            
            latest_features = features.iloc[-1:].values
            features_scaled = self.scaler.transform(latest_features)
            
            prediction = self.model.predict(features_scaled)[0]
            
            # Calculate confidence based on feature importance
            feature_importance = self.model.feature_importances_
            confidence = np.mean(feature_importance)
            
            return prediction, confidence
            
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            return None, 0
    
    def generate_trading_signal(self, symbol, current_data):
        """Generate trading signal based on AI prediction"""
        prediction, confidence = self.predict_price_movement(current_data)
        
        if prediction is None or confidence < self.confidence_threshold:
            return {
                'action': 'HOLD',
                'confidence': confidence,
                'prediction': prediction,
                'reason': 'Low confidence or no prediction'
            }
        
        if prediction > 0.02:  # 2% upward movement predicted
            action = 'BUY'
            reason = f'AI predicts {prediction:.2%} upward movement'
        elif prediction < -0.02:  # 2% downward movement predicted
            action = 'SELL'
            reason = f'AI predicts {prediction:.2%} downward movement'
        else:
            action = 'HOLD'
            reason = 'Predicted movement too small'
        
        return {
            'action': action,
            'confidence': confidence,
            'prediction': prediction,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        