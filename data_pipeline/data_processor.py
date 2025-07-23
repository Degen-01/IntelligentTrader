"""Data processing, validation, and feature engineering for market data."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from ..core.exceptions import DataValidationError

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, required_columns: List[str] = ['open', 'high', 'low', 'close', 'volume']):
        self.required_columns = required_columns
        
    def validate_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """
        Validate OHLCV data for completeness and correctness.
        Checks for:
        - Datetime index
        - Required columns
        - Non-negative values for volume, open, high, low, close
        - High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Input is not a pandas DataFrame.")
            raise DataValidationError("Input is not a pandas DataFrame.")

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame index must be a DatetimeIndex.")
            raise DataValidationError("DataFrame index must be a DatetimeIndex.")

        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise DataValidationError(f"Missing required columns: {missing_cols}")
        
        # Check for non-negative values
        numeric_cols = df[self.required_columns].select_dtypes(include=np.number).columns
        if (df[numeric_cols] < 0).any().any():
            logger.error("Negative values found in OHLCV data.")
            raise DataValidationError("Negative values found in OHLCV data.")

        # Check OHLC relationships
        # This can be computationally intensive for large datasets; consider sampling or vectorized checks for production
        # For simplicity, keeping row-wise check here. Vectorized check example: (df['high'] < df['low']).any()
        for i, row in df.iterrows():
            if not (row['high'] >= row['low'] and
                    row['high'] >= row['open'] and
                    row['high'] >= row['close'] and
                    row['low'] <= row['open'] and
                    row['low'] <= row['close']):
                logger.error(f"Invalid OHLC relationship at {i}: {row.to_dict()}")
                raise DataValidationError(f"Invalid OHLC relationship at {i}")
        
        logger.info("OHLCV data validated successfully.")
        return True

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and outliers (basic)."""
        original_rows = len(df)
        df.dropna(subset=self.required_columns, inplace=True)
        if len(df) < original_rows:
            logger.warning(f"Removed {original_rows - len(df)} rows with missing OHLCV data.")
            
        # Optional: Outlier detection and capping/replacement
        # For a truly institutional grade, this would involve more sophisticated statistical methods
        # like Z-score, IQR, or machine learning based outlier detection.
        for col in ['volume', 'open', 'high', 'low', 'close']:
            if col in df.columns:
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=q1, upper=q99)
                logger.debug(f"Capped outliers for column {col} between {q1:.2f} and {q99:.2f}.")
                
        return df

    def add_technical_indicators(self, df: pd.DataFrame, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Add common technical indicators to the DataFrame.
        Requires 'TA-Lib' or custom implementations.
        For production, use a dedicated library like 'ta' (technical analysis).
        """
        try:
            import ta
        except ImportError:
            logger.warning("TA-Lib or 'ta' library not found. Skipping technical indicator generation. Install 'ta' (pip install ta) for this feature.")
            return df
        
        df_copy = df.copy() # Operate on a copy to avoid SettingWithCopyWarning

        if indicators is None:
            # Default set of indicators
            indicators = [
                'rsi', 'macd', 'bollinger', 'sma', 'ema', 'adx', 'stoch', 'volume_oscillator'
            ]
        
        # Ensure required columns are present for indicators
        # This is a simplified check; each indicator might have specific dependencies
        if not all(col in df_copy.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            logger.error("Missing OHLCV data for technical indicator calculation.")
            return df_copy
            
        if 'rsi' in indicators:
            df_copy['rsi'] = ta.momentum.RSIIndicator(df_copy['close'], window=14).rsi()
        
        if 'macd' in indicators:
            macd = ta.trend.MACD(df_copy['close'])
            df_copy['macd'] = macd.macd()
            df_copy['macd_signal'] = macd.macd_signal()
            df_copy['macd_diff'] = macd.macd_diff()

        if 'bollinger' in indicators:
            bollinger = ta.volatility.BollingerBands(df_copy['close'])
            df_copy['bollinger_hband'] = bollinger.bollinger_hband()
            df_copy['bollinger_lband'] = bollinger.bollinger_lband()
            df_copy['bollinger_mband'] = bollinger.bollinger_mband()
        
        if 'sma' in indicators:
            df_copy['sma_20'] = ta.trend.SMAIndicator(df_copy['close'], window=20).sma_indicator()
            df_copy['sma_50'] = ta.trend.SMAIndicator(df_copy['close'], window=50).sma_indicator()
        
        if 'ema' in indicators:
            df_copy['ema_20'] = ta.trend.EMAIndicator(df_copy['close'], window=20).ema_indicator()
            df_copy['ema_50'] = ta.trend.EMAIndicator(df_copy['close'], window=50).ema_indicator()
        
        if 'adx' in indicators:
            adx = ta.trend.ADXIndicator(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
            df_copy['adx'] = adx.adx()
            df_copy['adx_pos'] = adx.adx_pos()
            df_copy['adx_neg'] = adx.adx_neg()

        if 'stoch' in indicators:
            stoch = ta.momentum.StochasticOscillator(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
            df_copy['stoch_k'] = stoch.stoch()
            df_copy['stoch_d'] = stoch.stoch_signal()

        if 'volume_oscillator' in indicators:
            df_copy['volume_oscillator'] = ta.volume.VolumeOscillator(df_copy['volume']).volume_oscillator()

        logger.info(f"Added {len(indicators)} technical indicators.")
        return df_copy.fillna(method='ffill').fillna(method='bfill') # Fill NaNs created by indicators

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for AI model, e.g., price changes, volatility, etc.
        """
        df_copy = df.copy()

        # Simple Price Changes / Returns
        df_copy['daily_return'] = df_copy['close'].pct_change()
        df_copy['log_return'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
        
        # Volatility
        df_copy['volatility_5d'] = df_copy['log_return'].rolling(window=5).std() * np.sqrt(5)
        df_copy['volatility_20d'] = df_copy['log_return'].rolling(window=20).std() * np.sqrt(20)

        # Volume-based features
        df_copy['volume_change'] = df_copy['volume'].pct_change()
        df_copy['volume_sma_5'] = df_copy['volume'].rolling(window=5).mean()
        df_copy['volume_ratio'] = df_copy['volume'] / df_copy['volume_sma_5']

        # Price range features
        df_copy['high_low_range'] = (df_copy['high'] - df_copy['low']) / df_copy['close']
        df_copy['open_close_range'] = (df_copy['close'] - df_copy['open']) / df_copy['open']

        # Lagged features (important for time series models)
        for col in ['close', 'volume', 'rsi', 'macd']: # Example columns to lag
            if col in df_copy.columns:
                for lag in [1][2][3]:
                    df_copy[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)
        
        logger.info("Created additional features.")
        return df_copy.fillna(0) # Fill NaNs created by feature engineering, usually 0 or mean/median

    def preprocess_data(self, df: pd.DataFrame, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Orchestrates the entire data processing pipeline.
        """
        self.validate_ohlcv_data(df)
        processed_df = self.clean_data(df)
        processed_df = self.add_technical_indicators(processed_df, indicators)
        processed_df = self.create_features(processed_df)
        
        logger.info("Data preprocessing complete.")
        return processed_df
