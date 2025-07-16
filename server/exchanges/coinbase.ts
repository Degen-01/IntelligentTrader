import ccxt from 'ccxt';
import axios from 'axios';
import crypto from 'crypto';

export class CoinbaseExchange {
  private client: ccxt.coinbaseadvanced;
  private apiKey: string;
  private secret: string;
  private passphrase: string;
  private baseURL: string;

  constructor(apiKey: string, secret: string, passphrase: string) {
    this.apiKey = apiKey;
    this.secret = secret;
    this.passphrase = passphrase;
    this.baseURL = 'https://api.exchange.coinbase.com';

    this.client = new ccxt.coinbaseadvanced({
      apiKey,
      secret,
      password: passphrase,
      sandbox: false,
      enableRateLimit: true,
    });
  }

  private createSignature(message: string, timestamp: string) {
    const hmac = crypto.createHmac('sha256', Buffer.from(this.secret, 'base64'));
    hmac.update(timestamp + 'GET' + message);
    return hmac.digest('base64');
  }

  async getBalance() {
    try {
      const balance = await this.client.fetchBalance();
      return Object.entries(balance.total)
        .filter(([_, amount]) => (amount as number) > 0)
        .map(([asset, amount]) => ({
          asset,
          free: balance.free[asset] || 0,
          locked: balance.used[asset] || 0,
          total: amount as number,
        }));
    } catch (error) {
      console.error('Error getting Coinbase balance:', error);
      throw error;
    }
  }

  async getPrice(symbol: string) {
    try {
      const ticker = await this.client.fetchTicker(symbol);
      return ticker.last;
    } catch (error) {
      console.error(`Error getting price for ${symbol}:`, error);
      throw error;
    }
  }

  async get24hrStats(symbol: string) {
    try {
      const ticker = await this.client.fetchTicker(symbol);
      return {
        symbol,
        price: ticker.last,
        change: ticker.change,
        changePercent: ticker.percentage,
        volume: ticker.baseVolume,
        high: ticker.high,
        low: ticker.low,
      };
    } catch (error) {
      console.error(`Error getting 24hr stats for ${symbol}:`, error);
      throw error;
    }
  }

  async getKlines(symbol: string, timeframe: string, limit: number = 100) {
    try {
      const candles = await this.client.fetchOHLCV(symbol, timeframe, undefined, limit);
      return candles.map(c => ({
        openTime: c[0],
        open: c[1],
        high: c[2],
        low: c[3],
        close: c[4],
        volume: c[5],
        closeTime: c[0] + this.getTimeframeMs(timeframe),
      }));
    } catch (error) {
      console.error(`Error getting klines for ${symbol}:`, error);
      throw error;
    }
  }

  private getTimeframeMs(timeframe: string): number {
    const timeframes: { [key: string]: number } = {
      '1m': 60000,
      '5m': 300000,
      '15m': 900000,
      '1h': 3600000,
      '4h': 14400000,
      '1d': 86400000,
    };
    return timeframes[timeframe] || 60000;
  }

  async placeOrder(params: {
    symbol: string;
    side: 'buy' | 'sell';
    type: 'market' | 'limit' | 'stop' | 'stop_limit';
    quantity: number;
    price?: number;
    stopPrice?: number;
    timeInForce?: 'GTC' | 'IOC' | 'FOK';
  }) {
    try {
      const order = await this.client.createOrder(
        params.symbol,
        params.type,
        params.side,
        params.quantity,
        params.price,
        undefined,
        {
          stopPrice: params.stopPrice,
          timeInForce: params.timeInForce,
        }
      );
      
      return {
        orderId: order.id,
        symbol: order.symbol,
        side: order.side,
        type: order.type,
        quantity: order.amount,
        price: order.price || 0,
        status: order.status,
        executedQuantity: order.filled,
        timestamp: order.timestamp,
      };
    } catch (error) {
      console.error('Error placing order:', error);
      throw error;
    }
  }

  async cancelOrder(symbol: string, orderId: string) {
    try {
      const result = await this.client.cancelOrder(orderId, symbol);
      return result;
    } catch (error) {
      console.error('Error cancelling order:', error);
      throw error;
    }
  }

  async getOrderBook(symbol: string, limit: number = 100) {
    try {
      const orderBook = await this.client.fetchOrderBook(symbol, limit);
      return {
        bids: orderBook.bids.map(b => ({
          price: b[0],
          quantity: b[1],
        })),
        asks: orderBook.asks.map(a => ({
          price: a[0],
          quantity: a[1],
        })),
      };
    } catch (error) {
      console.error(`Error getting order book for ${symbol}:`, error);
      throw error;
    }
  }

  async getMyTrades(symbol: string, limit: number = 100) {
    try {
      const trades = await this.client.fetchMyTrades(symbol, undefined, limit);
      return trades.map(t => ({
        id: t.id,
        orderId: t.order,
        symbol: t.symbol,
        side: t.side,
        quantity: t.amount,
        price: t.price,
        commission: t.fee?.cost || 0,
        commissionAsset: t.fee?.currency || '',
        time: t.timestamp,
      }));
    } catch (error) {
      console.error(`Error getting my trades for ${symbol}:`, error);
      throw error;
    }
  }

  // Advanced order types
  async placeTWAPOrder(params: {
    symbol: string;
    side: 'buy' | 'sell';
    quantity: number;
    timeWindow: number; // in minutes
    sliceSize: number;
  }) {
    const { symbol, side, quantity, timeWindow, sliceSize } = params;
    const totalSlices = Math.ceil(quantity / sliceSize);
    const intervalMs = (timeWindow * 60 * 1000) / totalSlices;
    
    const orders = [];
    
    for (let i = 0; i < totalSlices; i++) {
      const sliceQuantity = Math.min(sliceSize, quantity - (i * sliceSize));
      
      setTimeout(async () => {
        try {
          const order = await this.placeOrder({
            symbol,
            side,
            type: 'market',
            quantity: sliceQuantity,
          });
          orders.push(order);
        } catch (error) {
          console.error(`Error placing TWAP slice ${i + 1}:`, error);
        }
      }, i * intervalMs);
    }
    
    return { orderId: `twap_${Date.now()}`, totalSlices, orders };
  }

  async placeVWAPOrder(params: {
    symbol: string;
    side: 'buy' | 'sell';
    quantity: number;
    lookbackPeriod: number; // in candles
  }) {
    const { symbol, side, quantity, lookbackPeriod } = params;
    
    // Get historical data to calculate VWAP
    const klines = await this.getKlines(symbol, '1m', lookbackPeriod);
    
    let totalVolume = 0;
    let totalVolumePrice = 0;
    
    klines.forEach(k => {
      const typicalPrice = (k.high + k.low + k.close) / 3;
      totalVolume += k.volume;
      totalVolumePrice += typicalPrice * k.volume;
    });
    
    const vwap = totalVolumePrice / totalVolume;
    
    // Place limit order at VWAP price
    return await this.placeOrder({
      symbol,
      side,
      type: 'limit',
      quantity,
      price: vwap,
      timeInForce: 'GTC',
    });
  }
}

export default CoinbaseExchange;