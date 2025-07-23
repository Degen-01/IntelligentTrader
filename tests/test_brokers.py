import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from datetime import datetime
from intelligent_trader.brokers.base_broker import PaperBroker, AlpacaBroker, Order, OrderType, OrderSide, OrderStatus, Account, Position

@pytest.fixture
def paper_broker():
    """Fixture for PaperBroker instance."""
    return PaperBroker(config={'initial_balance': 100000.0, 'commission': 0.001, 'slippage': 0.0001})

@pytest.mark.asyncio
async def test_paper_broker_connect_disconnect(paper_broker):
    assert await paper_broker.connect() is True
    assert paper_broker.is_connected is True
    await paper_broker.disconnect()
    assert paper_broker.is_connected is False

@pytest.mark.asyncio
async def test_paper_broker_place_market_order(paper_broker):
    await paper_broker.connect()
    paper_broker.update_market_price("AAPL", 150.0)
    
    order = Order(id="test_order_1", symbol="AAPL", side=OrderSide.BUY, type=OrderType.MARKET, quantity=10)
    order_id = await paper_broker.place_order(order)
    
    assert order_id == "test_order_1"
    assert paper_broker.orders[order_id].status == OrderStatus.FILLED
    assert paper_broker.orders[order_id].filled_quantity == 10
    assert paper_broker.orders[order_id].average_fill_price > 150.0 # Due to slippage
    
    account = await paper_broker.get_account_info()
    assert account.balance < 100000.0 # Cash reduced
    assert "AAPL" in account.positions
    assert account.positions["AAPL"].quantity == 10

@pytest.mark.asyncio
async def test_paper_broker_cancel_order(paper_broker):
    await paper_broker.connect()
    order = Order(id="test_order_2", symbol="MSFT", side=OrderSide.BUY, type=OrderType.LIMIT, quantity=5, price=200.0)
    await paper_broker.place_order(order)
    
    status_before_cancel = await paper_broker.get_order_status("test_order_2")
    assert status_before_cancel == OrderStatus.PENDING
    
    cancelled = await paper_broker.cancel_order("test_order_2")
    assert cancelled is True
    
    status_after_cancel = await paper_broker.get_order_status("test_order_2")
    assert status_after_cancel == OrderStatus.CANCELLED

@pytest.mark.asyncio
@patch('aiohttp.ClientSession')
async def test_alpaca_broker_connect_success(mock_client_session, caplog):
    # Mocking Alpaca API responses
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {} # Empty JSON for account
    
    mock_client_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
    
    broker = AlpacaBroker(config={'api_key': 'fake_key', 'secret_key': 'fake_secret'})
    with caplog.at_level(caplog.INFO):
        connected = await broker.connect()
        assert connected is True
        assert broker.is_connected is True
        assert "Connected to Alpaca" in caplog.text

@pytest.mark.asyncio
@patch('
