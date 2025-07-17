let priceChart;

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    loadStocks();
    loadPortfolio();
    initializeChart();
    
    // Refresh data every 30 seconds
    setInterval(() => {
        loadStocks();
        loadPortfolio();
    }, 30000);
});

// Load stocks data
async function loadStocks() {
    try {
        const response = await fetch('/api/stocks');
        const stocks = await response.json();
        
        const tbody = document.getElementById('stocks-tbody');
        tbody.innerHTML = '';
        
        stocks.forEach(stock => {
            const row = document.createElement('tr');
            const changeClass = stock.change >= 0 ? 'positive' : 'negative';
            
            row.innerHTML = `
                <td><strong>${stock.symbol}</strong></td>
                <td>${stock.price}</td>
                <td class="${changeClass}">${stock.change}</td>
                <td class="${changeClass}">${stock.change_percent.toFixed(2)}%</td>
                <td>${stock.volume.toLocaleString()}</td>
                <td>${stock.market_cap}</td>
            `;
            
            tbody.appendChild(row);
        });
    } catch (error) {
        console.error('Error loading stocks:', error);
    }
}

// Load portfolio data
async function loadPortfolio() {
    try {
        const response = await fetch('/api/portfolio');
        const portfolio = await response.json();
        
        const portfolioList = document.getElementById('portfolio-list');
        portfolioList.innerHTML = '';
        
        portfolio.forEach(holding => {
            const div = document.createElement('div');
            const currentValue = holding.shares * holding.current_price;
            const totalCost = holding.shares * holding.avg_price;
            const gainLoss = currentValue - totalCost;
            const gainLossPercent = (gainLoss / totalCost) * 100;
            const gainLossClass = gainLoss >= 0 ? 'positive' : 'negative';
            
            div.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #eee;">
                    <div>
                        <strong>${holding.symbol}</strong><br>
                        <small>${holding.shares} shares @ ${holding.avg_price}</small>
                    </div>
                    <div style="text-align: right;">
                        <div>${currentValue.toFixed(2)}</div>
                        <div class="${gainLossClass}" style="font-size: 0.9rem;">
                            ${gainLoss >= 0 ? '+' : ''}${gainLoss.toFixed(2)} (${gainLossPercent.toFixed(2)}%)
                        </div>
                    </div>
                </div>
            `;
            
            portfolioList.appendChild(div);
        });
    } catch (error) {
        console.error('Error loading portfolio:', error);
    }
}

// Initialize price chart
async function initializeChart() {
    try {
        const response = await fetch('/api/chart/AAPL');
        const data = await response.json();
        
        const ctx = document.getElementById('priceChart').getContext('2d');
        
        priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [{
                    label: 'AAPL Price',
                    data: data.prices,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error initializing chart:', error);
    }
}

// Execute trade function
function executeTrade() {
    const symbol = document.getElementById('trade-symbol').value;
    const quantity = document.getElementById('trade-quantity').value;
    const action = document.getElementById('trade-action').value;
    
    if (!symbol || !quantity) {
        alert('Please fill in all fields');
        return;
    }
    
    // Simulate trade execution
    alert(`${action.toUpperCase()} order for ${quantity} shares of ${symbol} has been placed!`);
    
    // Clear form
    document.getElementById('trade-symbol').value = '';
    document.getElementById('trade-quantity').value = '';
}
