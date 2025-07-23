"""
Advanced signal filtering system that only acts on the strongest, most reliable predictions.
Implements multiple filtering criteria including confidence, volatility, market conditions, and ensemble agreement.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from intelligent_trader.core.config import Config
from intelligent_trader.monitoring.alerts import AlertManager

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    """Enumeration for signal strength levels."""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

class MarketRegime(Enum):
    """Enumeration for market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class FilteredSignal:
    """Data class for filtered trading signals."""
    symbol: str
    signal: int  # -1, 0, 1 for sell, hold, buy
    confidence: float
    strength: SignalStrength
    raw_prediction_proba: np.ndarray
    filter_scores: Dict[str, float]
    market_regime: MarketRegime
    timestamp: datetime
    position_size_multiplier: float = 1.0
    risk_adjusted: bool = False
    ensemble_agreement: Optional[float] = None

class StrategySignalFilter:
    """
    Advanced signal filtering system for trading strategies.
    Filters signals based on multiple criteria to ensure only the strongest predictions are acted upon.
    """
    
    def __init__(self, config: Config, alert_manager: AlertManager):
        self.config = config
        self.alert_manager = alert_manager
        
        # Filter thresholds from config
        self.min_confidence_threshold = config.get("SIGNAL_MIN_CONFIDENCE", 0.65)
        self.strong_confidence_threshold = config.get("SIGNAL_STRONG_CONFIDENCE", 0.80)
        self.very_strong_confidence_threshold = config.get("SIGNAL_VERY_STRONG_CONFIDENCE", 0.90)
        
        # Market condition filters
        self.max_volatility_threshold = config.get("SIGNAL_MAX_VOLATILITY", 0.05)  # 5% daily volatility
        self.min_volume_threshold = config.get("SIGNAL_MIN_VOLUME_RATIO", 0.8)  # 80% of avg volume
        
        # Ensemble agreement thresholds
        self.min_ensemble_agreement = config.get("SIGNAL_MIN_ENSEMBLE_AGREEMENT", 0.7)
        
        # Technical indicator filters
        self.use_technical_filters = config.get("SIGNAL_USE_TECHNICAL_FILTERS", True)
        self.rsi_overbought = config.get("SIGNAL_RSI_OVERBOUGHT", 70)
        self.rsi_oversold = config.get("SIGNAL_RSI_OVERSOLD", 30)
        
        # Position sizing parameters
        self.base_position_size = config.get("SIGNAL_BASE_POSITION_SIZE", 0.02)  # 2% of portfolio
        self.max_position_size = config.get("SIGNAL_MAX_POSITION_SIZE", 0.10)   # 10% of portfolio
        
        logger.info("Initialized StrategySignalFilter with advanced filtering criteria.")

    def filter_signals(self, 
                      signals: List[Dict[str, Any]], 
                      market_data: Dict[str, pd.DataFrame],
                      ensemble_predictions: Optional[Dict[str, List[float]]] = None) -> List[FilteredSignal]:
        """
        Filters trading signals based on multiple criteria.
        
        Args:
            signals: List of raw trading signals with predictions
            market_data: Dictionary of market data by symbol
            ensemble_predictions: Optional ensemble model predictions for agreement calculation
            
        Returns:
            List of filtered signals that meet all criteria
        """
        filtered_signals = []
        
        for signal_data in signals:
            try:
                symbol = signal_data['symbol']
                raw_signal = signal_data['signal']
                confidence = signal_data['confidence']
                prediction_proba = signal_data.get('prediction_proba', np.array([0.5, 0.5]))
                
                # Get market data for this symbol
                symbol_data = market_data.get(symbol)
                if symbol_data is None or symbol_data.empty:
                    logger.warning(f"No market data available for {symbol}, skipping signal.")
                    continue
                
                # Apply all filters
                filter_results = self._apply_all_filters(
                    symbol, raw_signal, confidence, prediction_proba, 
                    symbol_data, ensemble_predictions
                )
                
                # Check if signal passes all filters
                if filter_results['passes_all_filters']:
                    filtered_signal = FilteredSignal(
                        symbol=symbol,
                        signal=filter_results['final_signal'],
                        confidence=confidence,
                        strength=filter_results['signal_strength'],
                        raw_prediction_proba=prediction_proba,
                        filter_scores=filter_results['filter_scores'],
                        market_regime=filter_results['market_regime'],
                        timestamp=datetime.now(),
                        position_size_multiplier=filter_results['position_size_multiplier'],
                        risk_adjusted=True,
                        ensemble_agreement=filter_results.get('ensemble_agreement')
                    )
                    
                    filtered_signals.append(filtered_signal)
                    
                    logger.info(f"Signal for {symbol} passed all filters. "
                              f"Strength: {filter_results['signal_strength'].name}, "
                              f"Confidence: {confidence:.3f}")
                else:
                    logger.debug(f"Signal for {symbol} filtered out. "
                               f"Failed filters: {filter_results['failed_filters']}")
            
            except Exception as e:
                logger.error(f"Error filtering signal for {signal_data.get('symbol', 'unknown')}: {e}")
                continue
        
        # Sort by signal strength and confidence
        filtered_signals.sort(key=lambda x: (x.strength.value, x.confidence), reverse=True)
        
        logger.info(f"Filtered {len(filtered_signals)} signals from {len(signals)} raw signals.")
        
        return filtered_signals

    def _apply_all_filters(self, 
                          symbol: str, 
                          signal: int, 
                          confidence: float,
                          prediction_proba: np.ndarray,
                          market_data: pd.DataFrame,
                          ensemble_predictions: Optional[Dict[str, List[float]]]) -> Dict[str, Any]:
        """
        Applies all filtering criteria to a single signal.
        
        Returns:
            Dictionary containing filter results and final decision
        """
        filter_scores = {}
        failed_filters = []
        
        # 1. Confidence Filter
        confidence_pass, confidence_score = self._confidence_filter(confidence)
        filter_scores['confidence'] = confidence_score
        if not confidence_pass:
            failed_filters.append('confidence')
        
        # 2. Market Volatility Filter
        volatility_pass, volatility_score = self._volatility_filter(market_data)
        filter_scores['volatility'] = volatility_score
        if not volatility_pass:
            failed_filters.append('volatility')
        
        # 3. Volume Filter
        volume_pass, volume_score = self._volume_filter(market_data)
        filter_scores['volume'] = volume_score
        if not volume_pass:
            failed_filters.append('volume')
        
        # 4. Technical Indicator Filter
        technical_pass, technical_score = self._technical_filter(market_data, signal)
        filter_scores['technical'] = technical_score
        if not technical_pass:
            failed_filters.append('technical')
        
        # 5. Market Regime Filter
        regime_pass, regime_score, market_regime = self._market_regime_filter(market_data, signal)
        filter_scores['market_regime'] = regime_score
        if not regime_pass:
            failed_filters.append('market_regime')
        
        # 6. Ensemble Agreement Filter (if available)
        ensemble_agreement = None
        if ensemble_predictions and symbol in ensemble_predictions:
            ensemble_pass, ensemble_score, ensemble_agreement = self._ensemble_agreement_filter(
                signal, ensemble_predictions[symbol]
            )
            filter_scores['ensemble'] = ensemble_score
            if not ensemble_pass:
                failed_filters.append('ensemble')
        else:
            ensemble_pass = True  # Skip if no ensemble data
        
        # 7. Risk-Adjusted Position Sizing
        position_size_multiplier = self._calculate_position_size_multiplier(
            confidence, filter_scores, market_regime
        )
        
        
        # Determine overall pass
        passes_all_filters = all([
            confidence_pass, volatility_pass, volume_pass, technical_pass,
            regime_pass, ensemble_pass
        ])

        # Determine signal strength based on confidence and filter passes
        signal_strength = self._determine_signal_strength(confidence, passes_all_filters, ensemble_agreement)

        return {
            "final_signal": signal if passes_all_filters else 0, # Return 0 (hold) if filters fail
            "confidence": confidence,
            "signal_strength": signal_strength,
            "passes_all_filters": passes_all_filters,
            "filter_scores": filter_scores,
            "failed_filters": failed_filters,
            "market_regime": market_regime,
            "ensemble_agreement": ensemble_agreement,
            "position_size_multiplier": position_size_multiplier
        }

    def _confidence_filter(self, confidence: float) -> Tuple[bool, float]:
        """Checks if signal confidence meets the minimum threshold."""
        return confidence >= self.min_confidence_threshold, confidence

    def _volatility_filter(self, market_data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Checks if current market volatility is within acceptable limits.
        Assumes market_data has a 'daily_volatility' column (e.g., 20-day rolling std of returns).
        """
        if 'daily_volatility' not in market_data.columns or market_data.empty:
            logger.warning("Daily volatility data not found. Skipping volatility filter.")
            return True, 1.0 # Pass by default if data is missing
        
        current_volatility = market_data['daily_volatility'].iloc[-1]
        
        passes = current_volatility <= self.max_volatility_threshold
        score = 1.0 - (current_volatility / self.max_volatility_threshold) if self.max_volatility_threshold > 0 else 0.0
        score = max(0.0, min(1.0, score)) # Normalize to 0-1
        
        if not passes:
            self.alert_manager.send_alert("INFO", f"Volatility filter failed. Current: {current_volatility:.4f}, Max: {self.max_volatility_threshold:.4f}")
        return passes, score

    def _volume_filter(self, market_data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Checks if current trading volume is sufficient.
        Assumes market_data has 'volume' and 'avg_volume_20d' columns.
        """
        if 'volume' not in market_data.columns or 'avg_volume_20d' not in market_data.columns or market_data.empty:
            logger.warning("Volume data or average volume data not found. Skipping volume filter.")
            return True, 1.0 # Pass by default if data is missing
        
        current_volume = market_data['volume'].iloc[-1]
        avg_volume = market_data['avg_volume_20d'].iloc[-1]
        
        if avg_volume == 0:
            return True, 1.0 # Avoid division by zero, assume passes if no avg volume
        
        volume_ratio = current_volume / avg_volume
        passes = volume_ratio >= self.min_volume_threshold
        score = volume_ratio / self.min_volume_threshold if self.min_volume_threshold > 0 else 0.0
        score = max(0.0, min(1.0, score)) # Normalize to 0-1
        
        if not passes:
            self.alert_manager.send_alert("INFO", f"Volume filter failed. Current: {current_volume}, Avg: {avg_volume}, Min Ratio: {self.min_volume_threshold:.2f}")
        return passes, score

    def _technical_filter(self, market_data: pd.DataFrame, signal: int) -> Tuple[bool, float]:
        """
        Applies technical indicator-based filters (e.g., RSI overbought/oversold).
        Assumes market_data has 'RSI' column.
        """
        if not self.use_technical_filters:
            return True, 1.0
        
        if 'RSI' not in market_data.columns or market_data.empty:
            logger.warning("RSI data not found. Skipping technical filter.")
            return True, 1.0 # Pass by default if data is missing
            
        current_rsi = market_data['RSI'].iloc[-1]
        
        passes = True
        score = 1.0
        
        if signal == 1 and current_rsi > self.rsi_overbought: # Buy signal but overbought
            passes = False
            score = 1.0 - ((current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)) # Scale penalty
            self.alert_manager.send_alert("INFO", f"Technical filter failed (RSI overbought for Buy). RSI: {current_rsi}")
        elif signal == -1 and current_rsi < self.rsi_oversold: # Sell signal but oversold
            passes = False
            score = 1.0 - ((self.rsi_oversold - current_rsi) / self.rsi_oversold) # Scale penalty
            self.alert_manager.send_alert("INFO", f"Technical filter failed (RSI oversold for Sell). RSI: {current_rsi}")
            
        score = max(0.0, min(1.0, score))
        return passes, score

    def _market_regime_filter(self, market_data: pd.DataFrame, signal: int) -> Tuple[bool, float, MarketRegime]:
        """
        Identifies current market regime and checks if signal is appropriate.
        Assumes market_data has 'SMA_50', 'SMA_200', 'ATR' columns.
        """
        if 'SMA_50' not in market_data.columns or 'SMA_200' not in market_data.columns or 'ATR' not in market_data.columns or market_data.empty:
            logger.warning("Moving average or ATR data not found. Skipping market regime filter.")
            return True, 1.0, MarketRegime.SIDEWAYS # Pass by default
        
        last_close = market_data['close'].iloc[-1]
        sma_50 = market_data['SMA_50'].iloc[-1]
        sma_200 = market_data['SMA_200'].iloc[-1]
        atr = market_data['ATR'].iloc[-1]
        
        # Determine regime
        regime = MarketRegime.SIDEWAYS
        if sma_50 > sma_200 and last_close > sma_50:
            regime = MarketRegime.TRENDING_UP
        elif sma_50 < sma_200 and last_close < sma_50:
            regime = MarketRegime.TRENDING_DOWN
            
        # Check volatility level
        avg_price = (market_data['high'].iloc[-1] + market_data['low'].iloc[-1]) / 2
        relative_atr = atr / avg_price if avg_price > 0 else 0
        
        if relative_atr > self.max_volatility_threshold * 1.5: # Use a higher threshold for regime volatility
            regime = MarketRegime.HIGH_VOLATILITY
        elif relative_atr < self.max_volatility_threshold / 2:
            regime = MarketRegime.LOW_VOLATILITY
            
        # Check if signal is appropriate for regime
        passes = True
        score = 1.0
        
        if signal == 1 and regime == MarketRegime.TRENDING_DOWN: # Buy in downtrend
            passes = False
            score = 0.2
            self.alert_manager.send_alert("INFO", f"Market regime filter failed (Buy in Downtrend). Regime: {regime.name}")
        elif signal == -1 and regime == MarketRegime.TRENDING_UP: # Sell in uptrend
            passes = False
            score = 0.2
            self.alert_manager.send_alert("INFO", f"Market regime filter failed (Sell in Uptrend). Regime: {regime.name}")
        elif regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.LOW_VOLATILITY]:
            # Consider being more cautious in extreme volatility
            if signal != 0: # If it's a buy/sell signal
                passes = False
                score = 0.5
                self.alert_manager.send_alert("INFO", f"Market regime filter failed (Extreme Volatility for Buy/Sell). Regime: {regime.name}")
        
        return passes, score, regime

    def _ensemble_agreement_filter(self, 
                                 signal: int, 
                                 ensemble_predictions_for_symbol: List[float]) -> Tuple[bool, float, float]:
        """
        Checks agreement among ensemble models for the given signal.
        Ensemble predictions are assumed to be probabilities for the 'buy' class (1).
        """
        if not ensemble_predictions_for_symbol:
            return True, 1.0, 0.0 # No ensemble, pass
        
        # Convert probabilities to discrete signals (-1, 0, 1) for agreement
        ensemble_signals = [1 if p > 0.55 else (-1 if p < 0.45 else 0) for p in ensemble_predictions_for_symbol] # Adjust thresholds for clarity
        
        agreement_count = ensemble_signals.count(signal)
        total_models = len(ensemble_signals)
        
        if total_models == 0:
            return True, 1.0, 0.0
            
        agreement_ratio = agreement_count / total_models
        
        passes = agreement_ratio >= self.min_ensemble_agreement
        score = agreement_ratio
        
        if not passes:
            self.alert_manager.send_alert("INFO", f"Ensemble agreement filter failed. Ratio: {agreement_ratio:.2f}, Min: {self.min_ensemble_agreement:.2f}")
        return passes, score, agreement_ratio

    def _determine_signal_strength(self, 
                                 confidence: float, 
                                 passes_all_filters: bool,
                                 ensemble_agreement: Optional[float] = None) -> SignalStrength:
        """
        Determines the overall strength of a signal.
        """
        if not passes_all_filters:
            return SignalStrength.VERY_WEAK # Cannot act on signals that fail filters
        
        if confidence >= self.very_strong_confidence_threshold and \
           (ensemble_agreement is None or ensemble_agreement >= 0.9):
            return SignalStrength.VERY_STRONG
        elif confidence >= self.strong_confidence_threshold and \
             (ensemble_agreement is None or ensemble_agreement >= 0.75):
            return SignalStrength.STRONG
        elif confidence >= self.min_confidence_threshold: # Passed all filters, so at least moderate
            return SignalStrength.MODERATE
        else: # Should not happen if passes_all_filters is true, but as fallback
            return SignalStrength.WEAK

    def _calculate_position_size_multiplier(self, 
                                          confidence: float, 
                                          filter_scores: Dict[str, float],
                                          market_regime: MarketRegime) -> float:
        """
        Calculates a multiplier for position sizing based on signal confidence,
        filter scores, and market regime.
        """
        # Base multiplier from confidence
        conf_mult = (confidence - self.min_confidence_threshold) / (1.0 - self.min_confidence_threshold)
        conf_mult = max(0.0, min(1.0, conf_mult))
        
        # Aggregate filter scores (simple average for now)
        avg_filter_score = np.mean(list(filter_scores.values())) if filter_scores else 1.0
        
        # Adjust based on market regime (be more conservative in high volatility)
        regime_adjustment = 1.0
        if market_regime == MarketRegime.HIGH_VOLATILITY:
            regime_adjustment = 0.7 # Reduce size in high volatility
        elif market_regime == MarketRegime.LOW_VOLATILITY:
            regime_adjustment = 0.9 # Slightly reduce if too quiet
        elif market_regime == MarketRegime.TRENDING_UP or market_regime == MarketRegime.TRENDING_DOWN:
            regime_adjustment = 1.1 # Slightly increase in strong trends
            
        # Combine factors
        multiplier = conf_mult * avg_filter_score * regime_adjustment
        
        # Scale to fit within base and max position size
        # This multiplier will be applied to a base position size defined elsewhere,
        # or can directly define a scaled percentage of portfolio.
        # For this module, let's keep it as a relative multiplier [0, 1.5] roughly.
        
        final_multiplier = max(0.0, min(1.5, multiplier)) # Cap for sanity
        return final_multiplier
