# intelligent_trader.py

import pandas as pd
import numpy as np

class IntelligentTrader:
    def __init__(self, short_window=5, long_window=20, volatility_threshold=0.02):
        self.short_window = short_window
        self.long_window = long_window
        self.volatility_threshold = volatility_threshold

    def generate_signal(self, prices):
        if not prices or len(prices) < self.long_window:
            return "hold"  # Not enough data

        df = pd.DataFrame(prices, columns=["price"])

        # Calculate moving averages
        df["short_ma"] = df["price"].rolling(window=self.short_window).mean()
        df["long_ma"] = df["price"].rolling(window=self.long_window).mean()

        # Calculate rolling volatility (standard deviation over last N periods)
        df["volatility"] = df["price"].pct_change().rolling(window=self.long_window).std()

        # Get the latest row
        latest = df.iloc[-1]

        # If volatility is high, hold to avoid risky trades
        if latest["volatility"] > self.volatility_threshold:
            return "hold"

        # Trading logic
        if latest["short_ma"] > latest["long_ma"]:
            return "buy"
        elif latest["short_ma"] < latest["long_ma"]:
            return "sell"
        else:
            return "hold"