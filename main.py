from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import logging
from datetime import datetime

from config import Config
from ai_engine import AITradingEngine
from wallet_manager import WalletManager
from data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config.from_object(Config)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize components
config = Config()
ai_engine = AITradingEngine()
wallet_manager = WalletManager(config)
data_fetcher = DataFetcher(config)

# Trading symbols
SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']

# Global variables
trading_active = False
current_signals = {}

def autonomous_trading_loop():
    """Main autonomous trading loop"""
    global trading_active, current_signals
    
    while True:
        if trading_active:
            try:
                # Get market data
                market_data = data_fetcher.get_market_data(SYMBOLS)
                
                # Update wallet positions
                price_data = {symbol: data['price'] for symbol, data in market_data.items()}
                wallet_manager.update_position_prices(price_data)
                
                # Generate AI signals for each symbol
                for symbol in SYMBOLS:
                    historical_data = data_fetcher.get_historical_data(symbol)
                    
                    if not historical_data.empty:
                        # Train model if not trained
                        if not ai_engine.is_trained:
                            ai_engine.train_model(historical_data)
                        
                        # Generate signal
                        signal = ai_engine.generate_trading_signal(symbol, historical_data)
                        current_signals[symbol] = signal
                        
                        # Execute trade if signal is strong enough
                        if signal['action'] in ['BUY', 'SELL'] and signal['confidence'] > 0.8:
                            execute_autonomous_trade(symbol, signal, market_data[symbol]['price'])
                
                # Emit updates to frontend
                socketio.emit('market_update', {
                    'market_data': market_data,
                    'signals': current_signals,
                    'portfolio': wallet_manager.get_portfolio_performance(),
                    'balance': wallet_manager.get_balance()
                })
                
            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
        
        time.sleep(30)  # Update every 30 seconds

def execute_autonomous_trade(symbol, signal, current_price):
    """Execute autonomous trade based on AI signal"""
    try:
        balance_info = wallet_manager.get_balance()
        available_balance = balance_info['available_balance']
        
        if signal['action'] == 'BUY' and available_balance > 0:
            # Calculate position size (max 10% of portfolio)
            max_trade_value = available_balance * config.MAX_POSITION_SIZE
            quantity = int(max_trade_value / current_price)
            
            if quantity > 0:
                result = wallet_manager.execute_trade(symbol, 'BUY', quantity, current_price)
                if result['success']:
                    logging.info(f"Autonomous BUY: {quantity} shares of {symbol} at ${current_price}")
        
        elif signal['action'] == 'SELL' and symbol in wallet_manager.positions:
            # Sell entire position
            quantity = wallet_manager.positions[symbol]['quantity']
            result = wallet_manager.execute_trade(symbol, 'SELL', quantity, current_price)
            if result['success']:
                logging.info(f"Autonomous SELL: {quantity} shares of {symbol} at ${current_price}")
                
    except Exception as e:
        logging.error(f"Error executing autonomous trade: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_trading', methods=['POST'])
def start_trading():
    global trading_active
    trading_active = True
    return jsonify({'status': 'Trading started', 'active': trading_active})

@app.route('/api/stop_trading', methods=['POST'])
def stop_trading():
    global trading_active
    trading_active = False
    return jsonify({'status': 'Trading stopped', 'active': trading_active})

@app.route('/api/trading_status')
def trading_status():
    return jsonify({'active': trading_active})

@app.route('/api/portfolio')
def get_portfolio():
    return jsonify(wallet_manager.get_portfolio_performance())

@app.route('/api/balance')
def get_balance():
    return jsonify(wallet_manager.get_balance())

@app.route('/api/signals')
def get_signals():
    return jsonify(current_signals)

@app.route('/api/market_data')
def get_market_data():
    return jsonify(data_fetcher.get_market_data(SYMBOLS))

@app.route('/api/trade', methods=['POST'])
def manual_trade():
    data = request.json
    symbol = data.get('symbol')
    action = data.get('action')
    quantity = int(data.get('quantity', 0))
    
    # Get current price
    price_data = data_fetcher.get_live_price(symbol)
    if not price_data:
        return jsonify({'success': False, 'message': 'Could not fetch price'})
    
    current_price = price_data['price']
    result = wallet_manager.execute_trade(symbol, action, quantity, current_price)
    
    return jsonify(result)

@app.route('/api/news')
def get_news():
    return jsonify(data_fetcher.get_market_news())

@app.route('/api/crypto')
def get_crypto():
    return jsonify(data_fetcher.get_crypto_prices())

if __name__ == '__main__':
    # Start autonomous trading thread
    trading_thread = threading.Thread(target=autonomous_trading_loop, daemon=True)
    trading_thread.start()
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    