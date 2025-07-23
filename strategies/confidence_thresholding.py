"""
Module for applying confidence thresholds to trading signals,
allowing trades only when prediction confidence meets a specified level.
"""

import logging
from typing import Dict, Any, List, Optional
from intelligent_trader.core.config import Config
from intelligent_trader.monitoring.alerts import AlertManager
from intelligent_trader.strategies.strategy_signal_filter import FilteredSignal, SignalStrength

logger = logging.getLogger(__name__)

class ConfidenceThresholdingError(Exception):
    """Custom exception for confidence thresholding failures."""
    pass

class ConfidenceThresholding:
    """
    Applies confidence-based thresholds to FilteredSignals.
    Only signals exceeding a defined confidence level or signal strength will be acted upon.
    """

    def __init__(self, config: Config, alert_manager: AlertManager):
        self.config = config
        self.alert_manager = alert_manager
        
        # Default minimum confidence from config
        self.default_min_confidence = config.get("TRADE_MIN_CONFIDENCE_THRESHOLD", 0.70)
        self.trade_min_signal_strength = config.get("TRADE_MIN_SIGNAL_STRENGTH", SignalStrength.MODERATE.name)
        
        # Convert string to Enum member
        try:
            self.trade_min_signal_strength_enum = SignalStrength[self.trade_min_signal_strength.upper()]
        except KeyError:
            logger.error(f"Invalid signal strength configured: {self.trade_min_signal_strength}. Defaulting to MODERATE.")
            self.trade_min_signal_strength_enum = SignalStrength.MODERATE

        logger.info(f"Initialized ConfidenceThresholding with default min confidence: {self.default_min_confidence}, "
                    f"min signal strength: {self.trade_min_signal_strength_enum.name}.")

    def apply_threshold(self, 
                        signal: FilteredSignal, 
                        min_confidence: Optional[float] = None,
                        min_signal_strength: Optional[SignalStrength] = None) -> Optional[FilteredSignal]:
        """
        Applies a confidence threshold to a single filtered trading signal.
        
        Args:
            signal: The FilteredSignal object to evaluate.
            min_confidence: Optional, override the default minimum confidence for this call.
            min_signal_strength: Optional, override the default minimum signal strength for this call.
            
        Returns:
            The original FilteredSignal if it passes the threshold, otherwise None.
        """
        threshold = min_confidence if min_confidence is not None else self.default_min_confidence
        strength_threshold = min_signal_strength if min_signal_strength is not None else self.trade_min_signal_strength_enum

        if signal.signal == 0:
            logger.debug(f"Signal for {signal.symbol} is HOLD (0). No action required.")
            return None # No action for HOLD signals

        if signal.confidence < threshold:
            logger.info(f"Signal for {signal.symbol} with confidence {signal.confidence:.4f} is below threshold {threshold:.4f}. Not trading.")
            self.alert_manager.send_alert("INFO", f"Signal {signal.signal} for {signal.symbol} rejected due to low confidence ({signal.confidence:.2f} < {threshold:.2f}).")
            return None
        
        if signal.strength.value < strength_threshold.value:
            logger.info(f"Signal for {signal.symbol} with strength {signal.strength.name} is below required strength {strength_threshold.name}. Not trading.")
            self.alert_manager.send_alert("INFO", f"Signal {signal.signal} for {signal.symbol} rejected due to low strength ({signal.strength.name} < {strength_threshold.name}).")
            return None

        logger.info(f"Signal for {signal.symbol} passed confidence and strength thresholds. Confidence: {signal.confidence:.4f}, Strength: {signal.strength.name}.")
        return signal

    def apply_thresholds_to_list(self, 
                                 signals: List[FilteredSignal], 
                                 min_confidence: Optional[float] = None,
                                 min_signal_strength: Optional[SignalStrength] = None) -> List[FilteredSignal]:
        """
        Applies confidence thresholds to a list of filtered trading signals.
        
        Args:
            signals: A list of FilteredSignal objects.
            min_confidence: Optional, override the default minimum confidence for this call.
            min_signal_strength: Optional, override the default minimum signal strength for this call.
            
        Returns:
            A new list containing only the signals that pass the threshold.
        """
        passed_signals = []
        for signal in signals:
            try:
                processed_signal = self.apply_threshold(signal, min_confidence, min_signal_strength)
                if processed_signal:
                    passed_signals.append(processed_signal)
            except Exception as e:
                logger.error(f"Error processing signal for {signal.symbol}: {e}", exc_info=True)
                # Continue processing other signals even if one fails
        
        logger.info(f"Applied thresholds. {len(passed_signals)} out of {len(signals)} signals passed.")
        return passed_signals

    def get_dynamic_threshold(self, 
                              market_volatility_index: float, 
                              system_health_score: float) -> float:
        """
        Calculates a dynamic confidence threshold based on current market conditions and system health.
        
        Args:
            market_volatility_index: A normalized index of market volatility (e.g., VIX, or custom). Higher = more volatile.
            system_health_score: A score representing the overall health of the trading system (0 to 1). Higher = healthier.
            
        Returns:
            A dynamically adjusted minimum confidence threshold.
        """
        # Base threshold is the default
        dynamic_threshold = self.default_min_confidence
        
        # Adjust based on volatility: higher volatility -> higher threshold (be more selective)
        # Assuming market_volatility_index is scaled such that 0.5 is average, 1.0 is high.
        # Example: if volatility is 0.8 (high), add 0.05 to threshold. If 0.2 (low), subtract 0.02.
        volatility_adjustment = (market_volatility_index - 0.5) * 0.1 # Max +/- 0.05
        dynamic_threshold += volatility_adjustment
        
        # Adjust based on system health: lower health -> higher threshold (be more cautious)
        # Example: if health is 0.7 (good), no change. If 0.3 (poor), add 0.05.
        health_adjustment = (1.0 - system_health_score) * 0.1 # Max +0.1
        dynamic_threshold += health_adjustment
        
        # Ensure threshold stays within reasonable bounds (e.g., 0.5 to 0.95)
        dynamic_threshold = max(0.50, min(0.95, dynamic_threshold))
        
        logger.info(f"Calculated dynamic confidence threshold: {dynamic_threshold:.4f} "
                    f"(Vol: {market_volatility_index:.2f}, Health: {system_health_score:.2f})")
        return dynamic_threshold

