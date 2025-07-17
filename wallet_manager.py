from web3 import Web3
import ccxt
import logging
from datetime import datetime

class WalletManager:
    def __init__(self, config):
        self.config = config
        self.balance = config.INITIAL_BALANCE
        self.positions = {}
        self.trade_history = []
        
        # Initialize Web3 for Ethereum
        try:
            self.w3 = Web3(Web3.HTTPProvider(config.ETHEREUM_RPC_URL))
            self.eth_connected = self.w3.is_connected()
        except:
            self.eth_connected = False
            
        # Initialize Binance
        try:
            self.binance = ccxt.binance({
                'apiKey': config.BINANCE_API_KEY,
                'secret': config.BINANCE_SECRET_KEY,
                'sandbox': True,  # Use testnet
            })
            self.binance_connected = True
        except:
            self.binance_connected = False
    
    def get_balance(self):
        """Get current balance"""
        return {
            'total_balance': self.balance,
            'available_balance': self.balance - self.get_total_position_value(),
            'positions': self.positions,
            'eth_connected': self.eth_connected,
            'binance_connected': self.binance_connected
        }
    
    def get_total_position_value(self):
        """Calculate total value of open positions"""
        return sum(pos['quantity'] * pos['current_price'] for pos in self.positions.values())
    
    def execute_trade(self, symbol, action, quantity, price):
        """Execute a trade"""
        try:
            trade_value = quantity * price
            
            if action.upper() == 'BUY':
                if trade_value > self.balance:
                    return {'success': False, 'message': 'Insufficient balance'}
                
                self.balance -= trade_value
                
                if symbol in self.positions:
                    # Add to existing position
                    old_qty = self.positions[symbol]['quantity']
                    old_price = self.positions[symbol]['avg_price']
                    new_qty = old_qty + quantity
                    new_avg_price = ((old_qty * old_price) + (quantity * price)) / new_qty
                    
                    self.positions[symbol].update({
                        'quantity': new_qty,
                        'avg_price': new_avg_price,
                        'current_price': price
                    })
                else:
                    # New position
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'avg_price': price,
                        'current_price': price,
                        'timestamp': datetime.now().isoformat()
                    }
            
            elif action.upper() == 'SELL':
                if symbol not in self.positions or self.positions[symbol]['quantity'] < quantity:
                    return {'success': False, 'message': 'Insufficient position'}
                
                self.balance += trade_value
                self.positions[symbol]['quantity'] -= quantity
                
                if self.positions[symbol]['quantity'] == 0:
                    del self.positions[symbol]
            
            # Record trade
            trade_record = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'value': trade_value,
                'timestamp': datetime.now().isoformat()
            }
            self.trade_history.append(trade_record)
            
            return {'success': True, 'trade': trade_record}
            
        except Exception as e:
            logging.error(f"Error executing trade: {e}")
            return {'success': False, 'message': str(e)}
    
    def update_position_prices(self, price_data):
        """Update current prices for all positions"""
        for symbol in self.positions:
            if symbol in price_data:
                self.positions[symbol]['current_price'] = price_data[symbol]
    
    def get_portfolio_performance(self):
        """Calculate portfolio performance"""
        total_value = self.balance + self.get_total_position_value()
        total_return = total_value - self.config.INITIAL_BALANCE
        return_percentage = (total_return / self.config.INITIAL_BALANCE) * 100
        
        return {
            'total_value': total_value,
            'total_return': total_return,
            'return_percentage': return_percentage,
            'trade_count': len(self.trade_history)
        }
        