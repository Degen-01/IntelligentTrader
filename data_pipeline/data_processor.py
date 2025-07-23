"""Data processing, validation, and feature engineering for market data."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
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
                'rsi', 'macd', 'bollinger', 'sma', 'ema', 'adx', 'stoch'
            ]
        
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

        logger.info(f"Added {len(indicators)} technical indicators.")
        return df_copy

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for AI model, e.g., price changes, volatility.
        """
        df
