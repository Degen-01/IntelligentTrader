import pytest
from intelligent_trader.risk_management.advanced_risk_manager import AdvancedRiskManager, RiskLevel, RiskLimitExceededError

@pytest.fixture
def risk_manager():
    """Fixture for a default AdvancedRiskManager instance."""
    rm = AdvancedRiskManager(
        max_drawdown=0.05,
        daily_loss_limit=0.02,
        risk_per_trade_capital=0.01,
        max_exposure_per_symbol=0.2,
        max_total_exposure=0.5
    )
    rm.update_metrics(current_portfolio_value=100000) # Set initial portfolio value
    return rm

def test_initial_risk_level(risk_manager):
    assert risk_manager.get_risk_level() == RiskLevel.LOW

def test_daily_loss_exceeded(risk_manager):
    # Simulate a loss that exceeds the daily_loss_limit
    initial_value = risk_manager.current_portfolio_value
    loss_amount = initial_value * (risk_manager.daily_loss_limit + 0.001)
    
    # Update portfolio value with loss
    risk_manager.update_metrics(current_portfolio_value=initial_value - loss_amount)
    assert risk_manager.get_risk_level() == RiskLevel.CRITICAL
    
    with pytest.raises(RiskLimitExceededError):
        risk_manager.check_daily_loss()

def test_can_enter_position_allowed(risk_manager):
    # This should be allowed under initial conditions
    assert risk_manager.can_enter_position(
        symbol="AAPL", side="buy", quantity=10, price=150,
        current_account_balance=100000, current_portfolio_value=100000
    ) is True

def test_can_enter_position_too_much_exposure(risk_manager):
    # Simulate high exposure to a symbol
    risk_manager.update_exposure("AAPL", 0.19) # Close to limit
    
    # Attempting to add more
    assert risk_manager.can_enter_position(
        symbol="AAPL", side="buy", quantity=1000, price=150, # This would push it over
        current_account_balance=100000, current_portfolio_value=100000
    ) is False

def test_can_enter_position_too_much_total_exposure(risk_manager):
    # Simulate high overall exposure
    risk_manager.current_total_exposure = risk_manager.max_total_exposure - 0.01
    
    # Attempting to add more
    assert risk_manager.can_enter_position(
        symbol="GOOG", side="buy", quantity=10, price=1000, # This would push it over
        current_account_balance=100000, current_portfolio_value=100000
    ) is False

def test_check_max_drawdown(risk_manager):
    # Initial state, no drawdown
    risk_manager.update_metrics(current_portfolio_value=100000)
    risk_manager.peak_equity = 100000 # Ensure peak is set
    
    # Simulate a drawdown that is within limits
    risk_manager.update_metrics(current_portfolio_value=98000)
    assert risk_manager.get_risk_level() == RiskLevel.LOW # 2% drawdown, still low
    
    # Simulate a drawdown that exceeds limit
    risk_manager.update_metrics(current_portfolio_value=94000) # 6% drawdown
    assert risk_manager.get_risk_level() == RiskLevel.CRITICAL
    with pytest.raises(RiskLimitExceededError):
        risk_manager.check_max_drawdown()

def test_reset_daily_metrics(risk_manager):
    # Simulate a loss
    risk_manager.update_metrics(current_portfolio_value=risk_manager.current_portfolio_value * 0.95)
    assert risk_manager.daily_loss > 0
    
    # Reset
    risk_manager.reset_daily_metrics()
    assert risk_manager.daily_loss == 0
    assert risk_manager.day_start_portfolio_value == risk_manager.current_portfolio_value

