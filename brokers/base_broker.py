"""
Broker abstraction layer for multiple trading platforms
Supports paper trading, live trading, and multiple brokers
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio
import aiohttp
import time
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    timestamp: datetime = None
    broker_order_id: Optional[str] = None

@dataclass
class Position:
    symbol: str
    quantity: float
    average_price: float
    market_value: float
    unrealized_pnl: float
    side: str  # "long" or "short"

@dataclass
class Account:
    balance: float
    buying_power: float
    positions: Dict[str, Position]
    orders: Dict[str, Order]

class BaseBroker(ABC):
    """Abstract base class for all brokers"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_connected = False
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.account_balance = 0.0
        self.retry_attempts = 3
        self.retry_delay = 1.0
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker"""
        pass
        
    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker"""
        pass
        
    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """Place an order"""
        pass
        
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
        
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        pass
        
    @abstractmethod
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        pass
        
    @abstractmethod
    async def get_account_info(self) -> Account:
        """Get account information"""
        pass
        
    @abstractmethod
    async def get_market_price(self, symbol: str) -> float:
        """Get current market price"""
        pass
        
    async def retry_request(self, func, *args, **kwargs):
        """Retry failed requests with exponential backoff"""
        for attempt in range(self.retry_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise e
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)

class AlpacaBroker(BaseBroker):
    """Alpaca broker implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Alpaca", config)
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.base_url = config.get('base_url', 'https://paper-api.alpaca.markets')
        self.session = None
        
    async def connect(self) -> bool:
        """Connect to Alpaca"""
        try:
            self.session = aiohttp.ClientSession(
                headers={
                    'APCA-API-KEY-ID': self.api_key,
                    'APCA-API-SECRET-KEY': self.secret_key
                }
            )
            
            # Test connection
            async with self.session.get(f"{self.base_url}/v2/account") as response:
                if response.status == 200:
                    self.is_connected = True
                    logger.info("Connected to Alpaca")
                    return True
                else:
                    logger.error(f"Failed to connect to Alpaca: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error connecting to Alpaca: {e}")
            return False
            
    async def disconnect(self):
        """Disconnect from Alpaca"""
        if self.session:
            await self.session.close()
        self.is_connected = False
        
    async def place_order(self, order: Order) -> str:
        """Place order with Alpaca"""
        order_data = {
            'symbol': order.symbol,
            'qty': str(order.quantity),
            'side': order.side.value,
            'type': order.type.value,
            'time_in_force': 'day'
        }
        
        if order.type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            order_data['limit_price'] = str(order.price)
            
        if order.type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            order_data['stop_price'] = str(order.stop_price)
            
        try:
            async with self.session.post(
                f"{self.base_url}/v2/orders",
                json=order_data
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    order.broker_order_id = result['id']
                    order.timestamp = datetime.now()
                    self.orders[order.id] = order
                    logger.info(f"Order placed: {order.id}")
                    return order.id
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to place order: {error_text}")
                    raise Exception(f"Order placement failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with Alpaca"""
        if order_id not in self.orders:
            return False
            
        broker_order_id = self.orders[order_id].broker_order_id
        
        try:
            async with self.session.delete(
                f"{self.base_url}/v2/orders/{broker_order_id}"
            ) as response:
                if response.status == 204:
                    self.orders[order_id].status = OrderStatus.CANCELLED
                    logger.info(f"Order cancelled: {order_id}")
                    return True
                else:
                    logger.error(f"Failed to cancel order: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
            
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from Alpaca"""
        if order_id not in self.orders:
            return OrderStatus.REJECTED
            
        broker_order_id = self.orders[order_id].broker_order_id
        
        try:
            async with self.session.get(
                f"{self.base_url}/v2/orders/{broker_order_id}"
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    status_map = {
                        'new': OrderStatus.PENDING,
                        'partially_filled': OrderStatus.PARTIALLY_FILLED,
                        'filled': OrderStatus.FILLED,
                        'done_for_day': OrderStatus.CANCELLED,
                        'canceled': OrderStatus.CANCELLED,
                        'expired': OrderStatus.CANCELLED,
                        'replaced': OrderStatus.CANCELLED,
                        'pending_cancel': OrderStatus.PENDING,
                        'pending_replace': OrderStatus.PENDING,
                        'accepted': OrderStatus.PENDING,
                        'pending_new': OrderStatus.PENDING,
                        'accepted_for_bidding': OrderStatus.PENDING,
                        'stopped': OrderStatus.CANCELLED,
                        'rejected': OrderStatus.REJECTED,
                        'suspended': OrderStatus.CANCELLED,
                        'calculated': OrderStatus.PENDING
                    }
                    
                    alpaca_status = result.get('status', 'rejected')
                    status = status_map.get(alpaca_status, OrderStatus.REJECTED)
                    
                    # Update order
                    self.orders[order_id].status = status
                    self.orders[order_id].filled_quantity = float(result.get('filled_qty', 0))
                    if result.get('filled_avg_price'):
                        self.orders[order_id].average_fill_price = float(result['filled_avg_price'])
                        
                    return status
                else:
                    return OrderStatus.REJECTED
                    
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return OrderStatus.REJECTED
            
    async def get_positions(self) -> Dict[str, Position]:
        """Get positions from Alpaca"""
        try:
            async with self.session.get(f"{self.base_url}/v2/positions") as response:
                if response.status == 200:
                    positions_data = await response.json()
                    positions = {}
                    
                    for pos_data in positions_data:
                        symbol = pos_data['symbol']
                        position = Position(
                            symbol=symbol,
                            quantity=float(pos_data['qty']),
                            average_price=float(pos_data['avg_entry_price']),
                            market_value=float(pos_data['market_value']),
                            unrealized_pnl=float(pos_data['unrealized_pl']),
                            side='long' if float(pos_data['qty']) > 0 else 'short'
                        )
                        positions[symbol] = position
                        
                    self.positions = positions
                    return positions
                else:
                    logger.error(f"Failed to get positions: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
            
    async def get_account_info(self) -> Account:
        """Get account info from Alpaca"""
        try:
            async with self.session.get(f"{self.base_url}/v2/account") as response:
                if response.status == 200:
                    account_data = await response.json()
                    positions = await self.get_positions()
                    
                    account = Account(
                        balance=float(account_data['cash']),
                        buying_power=float(account_data['buying_power']),
                        positions=positions,
                        orders=self.orders
                    )
                    
                    self.account_balance = account.balance
                    return account
                else:
                    logger.error(f"Failed to get account info: {response.status}")
                    return Account(0.0, 0.0, {}, {})
                    
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return Account(0.0, 0.0, {}, {})
            
    async def get_market_price(self, symbol: str) -> float:
        """Get market price from Alpaca"""
        try:
            async with self.session.get(
                f"{self.base_url}/v2/stocks/{symbol}/quotes/latest"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    quote = data['quote']
                    return (float(quote['bid_price']) + float(quote['ask_price'])) / 2
                else:
                    logger.error(f"Failed to get market price: {response.status}")
                    return 0.0
                    
        except Exception as e:
            logger.error(f"Error getting market price: {e}")
            return 0.0

class PaperBroker(BaseBroker):
    """Paper trading broker for simulation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Paper", config)
        self.initial_balance = config.get('initial_balance', 100000.0)
        self.account_balance = self.initial_balance
        self.commission = config.get('commission', 1.0)
        self.slippage = config.get('slippage', 0.001)  # 0.1% slippage
        self.market_prices = {}
        
    async def connect(self) -> bool:
        """Connect to paper broker"""
        self.is_connected = True
        logger.info("Connected to Paper Broker")
        return True
        
    async def disconnect(self):
        """Disconnect from paper broker"""
        self.is_connected = False
        
    async def place_order(self, order: Order) -> str:
        """Simulate order placement"""
        # Simulate market price with slippage
        market_price = self.market_prices.get(order.symbol, order.price or 100.0)
        
        if order.type == OrderType.MARKET:
            # Apply slippage
            if order.side == OrderSide.BUY:
                fill_price = market_price * (1 + self.slippage)
            else:
                fill_price = market_price * (1 - self.slippage)
                
            # Simulate immediate fill
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = fill_price
            order.timestamp = datetime.now()
            
            # Update account balance
            total_cost = order.quantity * fill_price + self.commission
            if order.side == OrderSide.BUY:
                self.account_balance -= total_cost
            else:
                self.account_balance += total_cost - self.commission
                
            # Update positions
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                if order.side == OrderSide.BUY:
                    new_qty = pos.quantity + order.quantity
                    if new_qty != 0:
                        pos.average_price = ((pos.quantity * pos.average_price) + 
                                           (order.quantity * fill_price)) / new_qty
                    pos.quantity = new_qty
                else:
                    pos.quantity -= order.quantity
                    
                if pos.quantity == 0:
                    del self.positions[order.symbol]
            else:
                if order.side == OrderSide.BUY:
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=order.quantity,
                        average_price=fill_price,
                        market_value=order.quantity * market_price,
                        unrealized_pnl=0.0,
                        side='long'
                    )
                    
        else:
            # For limit orders, just mark as pending
            order.status = OrderStatus.PENDING
            order.timestamp = datetime.now()
            
        self.orders[order.id] = order
        logger.info(f"Paper order placed: {order.id} - {order.status}")
        return order.id
        
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel paper order"""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False
        
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get paper order status"""
        if order_id in self.orders:
            return self.orders[order_id].status
        return OrderStatus.REJECTED
        
    async def get_positions(self) -> Dict[str, Position]:
        """Get paper positions"""
        # Update market values
        for symbol, position in self.positions.items():
            market_price = self.market_prices.get(symbol, position.average_price)
            position.market_value = position.quantity * market_price
            position.unrealized_pnl = (market_price - position.average_price) * position.quantity
            
        return self.positions
        
    async def get_account_info(self) -> Account:
        """Get paper account info"""
        positions = await self.get_positions()
        return Account(
            balance=self.account_balance,
            buying_power=self.account_balance,
            positions=positions,
            orders=self.orders
        )
        
    async def get_market_price(self, symbol: str) -> float:
        """Get simulated market price"""
        return self.market_prices.get(symbol, 100.0)
        
    def update_market_price(self, symbol: str, price: float):
        """Update market price for simulation"""
        self.market_prices[symbol] = price
