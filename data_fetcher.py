import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

class DataFetcher:
    def __init__(self, config):
        self.config = config
        self.alpha_vantage_key = config.ALPHA_VANTAGE_API_KEY
        
    def get_live_price(self, symbol):
        """Get live price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return {
                    'symbol': symbol,
                    'price': float(data['Close'].iloc[-1]),
                    'change': float(data['Close'].iloc[-1] - data['Close'].iloc[-2]),
                    'volume': int(data['Volume'].iloc[-1]),
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logging.error(f"Error fetching live price for {symbol}: {e}")
        return None
    
    def get_historical_data(self, symbol, period="1mo", interval="1h"):
        """Get historical data for training"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                data.columns = data.columns.str.lower()
                data.reset_index(inplace=True)
                return data
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()
    
    def get_market_data(self, symbols):
        """Get market data for multiple symbols"""
        market_data = {}
        for symbol in symbols:
            price_data = self.get_live_price(symbol)
            if price_data:
                market_data[symbol] = price_data
        return market_data
    
    def get_crypto_prices(self):
        """Get cryptocurrency prices"""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': 'bitcoin,ethereum,cardano,polkadot,chainlink',
                'vs_currencies': 'usd',
                'include_24hr_change': 'true'
            }
            response = requests.get(url, params=params)
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching crypto prices: {e}")
        return {}
    
    def get_market_news(self):
        """Get latest market news"""
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'apikey': self.alpha_vantage_key,
                'limit': 10
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'feed' in data:
                return data['feed'][:5]  # Return top 5 news
        except Exception as e:
            logging.error(f"Error fetching news: {e}")
        return []
        