from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# Sample data for demonstration
def generate_sample_data():
    """Generate sample trading data"""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    data = []
    
    for symbol in symbols:
        price = np.random.uniform(100, 3000)
        change = np.random.uniform(-50, 50)
        change_percent = (change / price) * 100
        
        data.append({
            'symbol': symbol,
            'price': round(price, 2),
            'change': round(change, 2),
            'change_percent': round(change_percent, 2),
            'volume': np.random.randint(1000000, 50000000),
            'market_cap': f"${np.random.randint(100, 3000)}B"
        })
    
    return data

def generate_chart_data():
    """Generate sample chart data"""
    dates = []
    prices = []
    base_date = datetime.now() - timedelta(days=30)
    base_price = 150
    
    for i in range(30):
        dates.append((base_date + timedelta(days=i)).strftime('%Y-%m-%d'))
        base_price += np.random.uniform(-5, 5)
        prices.append(round(base_price, 2))
    
    return {'dates': dates, 'prices': prices}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stocks')
def get_stocks():
    return jsonify(generate_sample_data())

@app.route('/api/chart/<symbol>')
def get_chart_data(symbol):
    return jsonify(generate_chart_data())

@app.route('/api/portfolio')
def get_portfolio():
    portfolio = [
        {'symbol': 'AAPL', 'shares': 10, 'avg_price': 145.50, 'current_price': 150.25},
        {'symbol': 'GOOGL', 'shares': 5, 'avg_price': 2750.00, 'current_price': 2800.50},
        {'symbol': 'MSFT', 'shares': 15, 'avg_price': 280.00, 'current_price': 285.75}
    ]
    return jsonify(portfolio)

if __name__ == '__main__':
    app.run(debug=True)
    