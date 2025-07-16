import Binance from 'binance-api-node';
import ccxt from 'ccxt';

export class BinanceExchange {
  private client: any;
  private ccxtClient: ccxt.binance;

  constructor(apiKey: string, secret: string) {
    this.client = Binance({
      apiKey,
      apiSecret: secret,
      useServerTime: true,
      recvWindow: 5000,
    });

    this.ccxtClient = new ccxt.binance({
      apiKey,
      secret,
      sandbox: false,
      enableRateLimit: true,
    });
  }

  async getBalance() {
    try {
      const balance = await this.client.accountInfo();
      return balance.balances
        .filter((b: any) => parseFloat(b.free) > 0 || parseFloat(b.locked) > 0)
        .map((b: any) => ({
          asset: b.asset,
          free: parseFloat(b.free),
          locked: parseFloat(b.locked),
          total: parseFloat(b.free) + parseFloat(b.locked),
        }));
    } catch (error) {
      console.error('Error getting Binance balance:', error);
      throw error;
    }
  }

  async getPrice(symbol: string) {
    try {
      const ticker = await this.client.prices({ symbol });
      return parseFloat(ticker[symbol]);
    } catch (error) {
      console.error(`Error getting price for ${symbol}:`, error);
      throw error;
    }
  }

  async get24hrStats(symbol: string) {
    try {
      const stats = await this.client.dailyStats({ symbol });
      return {
        symbol,
        price: parseFloat(stats.lastPrice),
        change: parseFloat(stats.priceChange),
        changePercent: parseFloat(stats.priceChangePercent),
        volume: parseFloat(stats.volume),
        high: parseFloat(stats.highPrice),
        low: parseFloat(stats.lowPrice),
      };
    } catch (error) {
      console.error(`Error getting 24hr stats for ${symbol}:`, error);
      throw error;
    }
  }

  async getKlines(symbol: string, interval: string, limit: number = 100) {
    try {
      const klines = await this.client.candles({
        symbol,
        interval,
        limit,
      });
      
      return klines.map((k: any) => ({
        openTime: k.openTime,
        open: parseFloat(k.open),
        high: parseFloat(k.high),
        low: parseFloat(k.low),
        close: parseFloat(k.close),
        volume: parseFloat(k.volume),
        closeTime: k.closeTime,
      }));
    } catch (error) {
      console.error(`Error getting klines for ${symbol}:`, error);
      throw error;
    }
  }

  async placeOrder(params: {
    symbol: string;
    side: 'BUY' | 'SELL';
    type: 'MARKET' | 'LIMIT' | 'STOP_LOSS' | 'STOP_LOSS_LIMIT';
    quantity: number;
    price?: number;
    stopPrice?: number;
    timeInForce?: 'GTC' | 'IOC' | 'FOK';
  }) {
    try {
      const order = await this.client.order(params);
      return {
        orderId: order.orderId,
        symbol: order.symbol,
        side: order.side,
        type: order.type,
        quantity: parseFloat(order.origQty),
        price: parseFloat(order.price || '0'),
        status: order.status,
        executedQuantity: parseFloat(order.executedQty),
        cummulativeQuoteQty: parseFloat(order.cummulativeQuoteQty),
        transactionTime: order.transactTime,
      };
    } catch (error) {
      console.error('Error placing order:', error);
      throw error;
    }
  }

  async cancelOrder(symbol: string, orderId: string) {
    try {
      const result = await this.client.cancelOrder({
        symbol,
        orderId: parseInt(orderId),
      });
      return result;
    } catch (error) {
      console.error('Error cancelling order:', error);
      throw error;
    }
  }

  async getOrderBook(symbol: string, limit: number = 100) {
    try {
      const orderBook = await this.client.book({ symbol, limit });
      return {
        bids: orderBook.bids.map((b: any) => ({
          price: parseFloat(b.price),
          quantity: parseFloat(b.quantity),
        })),
        asks: orderBook.asks.map((a: any) => ({
          price: parseFloat(a.price),
          quantity: parseFloat(a.quantity),
        })),
      };
    } catch (error) {
      console.error(`Error getting order book for ${symbol}:`, error);
      throw error;
    }
  }

  async getMyTrades(symbol: string, limit: number = 100) {
    try {
      const trades = await this.client.myTrades({ symbol, limit });
      return trades.map((t: any) => ({
        id: t.id,
        orderId: t.orderId,
        symbol: t.symbol,
        side: t.isBuyer ? 'BUY' : 'SELL',
        quantity: parseFloat(t.qty),
        price: parseFloat(t.price),
        commission: parseFloat(t.commission),
        commissionAsset: t.commissionAsset,
        time: t.time,
      }));
    } catch (error) {
      console.error(`Error getting my trades for ${symbol}:`, error);
      throw error;
    }
  }

  // Advanced order types
  async placeTWAPOrder(params: {
    symbol: string;
    side: 'BUY' | 'SELL';
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
            type: 'MARKET',
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
    side: 'BUY' | 'SELL';
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
      type: 'LIMIT',
      quantity,
      price: vwap,
      timeInForce: 'GTC',
    });
  }
}

export default BinanceExchange;