from flask import Flask, request, jsonify
from intelligent_trader import IntelligentTrader

app = Flask(__name__)
trader = IntelligentTrader()

@app.route('/')
def home():
    return "🚀 IntelligentTrader API is live."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'prices' not in data:
        return jsonify({'error': 'Missing price data'}), 400

    prices = data['prices']
    try:
        signal = trader.generate_signal(prices)
        return jsonify({'signal': signal})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)