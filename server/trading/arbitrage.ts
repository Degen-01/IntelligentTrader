import { BinanceExchange } from '../exchanges/binance';
import { CoinbaseExchange } from '../exchanges/coinbase';
import { storage } from '../storage-database';

interface ArbitrageOpportunity {
  symbol: string;
  buyExchange: string;
  sellExchange: string;
  buyPrice: number;
  sellPrice: number;
  spread: number;
  spreadPercent: number;
  volume: number;
  profitPotential: number;
  confidence: number;
  timestamp: Date;
}

interface ArbitrageConfig {
  minSpread: number;
  maxSlippage: number;
  minVolume: number;
  maxPositionSize: number;
  enabledExchanges: string[];
}

export class ArbitrageEngine {
  private exchanges: Map<string, BinanceExchange | CoinbaseExchange>;
  private config: ArbitrageConfig;
  private priceCache: Map<string, { price: number; timestamp: number }> = new Map();
  private isRunning: boolean = false;

  constructor(
    exchanges: Map<string, BinanceExchange | CoinbaseExchange>,
    config: ArbitrageConfig
  ) {
    this.exchanges = exchanges;
    this.config = config;
  }

  async start() {
    if (this.isRunning) return;
    
    this.isRunning = true;
    console.log('Starting arbitrage engine...');
    
    // Start price monitoring
    this.startPriceMonitoring();
    
    // Start opportunity scanning
    this.startOpportunityScanning();
  }

  async stop() {
    this.isRunning = false;
    console.log('Stopping arbitrage engine...');
  }

  private startPriceMonitoring() {
    setInterval(async () => {
      if (!this.isRunning) return;
      
      await this.updatePrices();
    }, 1000); // Update prices every second
  }

  private startOpportunityScanning() {
    setInterval(async () => {
      if (!this.isRunning) return;
      
      await this.scanForOpportunities();
    }, 5000); // Scan every 5 seconds
  }

  private async updatePrices() {
    const symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT'];
    
    for (const [exchangeName, exchange] of this.exchanges) {
      if (!this.config.enabledExchanges.includes(exchangeName)) continue;
      
      try {
        for (const symbol of symbols) {
          const price = await exchange.getPrice(symbol);
          const key = `${exchangeName}:${symbol}`;
          
          this.priceCache.set(key, {
            price,
            timestamp: Date.now()
          });
        }
      } catch (error) {
        console.error(`Error updating prices for ${exchangeName}:`, error);
      }
    }
  }

  private async scanForOpportunities() {
    const symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT'];
    const opportunities: ArbitrageOpportunity[] = [];
    
    for (const symbol of symbols) {
      const exchangePrices = this.getExchangePrices(symbol);
      
      if (exchangePrices.length < 2) continue;
      
      // Find price differences
      for (let i = 0; i < exchangePrices.length; i++) {
        for (let j = i + 1; j < exchangePrices.length; j++) {
          const exchange1 = exchangePrices[i];
          const exchange2 = exchangePrices[j];
          
          const opportunity = this.calculateOpportunity(symbol, exchange1, exchange2);
          
          if (opportunity && opportunity.spreadPercent >= this.config.minSpread) {
            opportunities.push(opportunity);
          }
        }
      }
    }
    
    // Sort by profit potential
    opportunities.sort((a, b) => b.profitPotential - a.profitPotential);
    
    // Store top opportunities
    for (const opportunity of opportunities.slice(0, 10)) {
      await this.storeOpportunity(opportunity);
    }
    
    // Execute if profitable enough
    for (const opportunity of opportunities.slice(0, 3)) {
      if (opportunity.confidence > 0.8 && opportunity.profitPotential > 50) {
        await this.executeArbitrage(opportunity);
      }
    }
  }

  private getExchangePrices(symbol: string): { exchange: string; price: number; timestamp: number }[] {
    const prices = [];
    
    for (const exchangeName of this.config.enabledExchanges) {
      const key = `${exchangeName}:${symbol}`;
      const cached = this.priceCache.get(key);
      
      if (cached && Date.now() - cached.timestamp < 10000) { // 10 second freshness
        prices.push({
          exchange: exchangeName,
          price: cached.price,
          timestamp: cached.timestamp
        });
      }
    }
    
    return prices;
  }

  private calculateOpportunity(
    symbol: string,
    exchange1: { exchange: string; price: number; timestamp: number },
    exchange2: { exchange: string; price: number; timestamp: number }
  ): ArbitrageOpportunity | null {
    const lowPrice = Math.min(exchange1.price, exchange2.price);
    const highPrice = Math.max(exchange1.price, exchange2.price);
    
    const buyExchange = exchange1.price < exchange2.price ? exchange1.exchange : exchange2.exchange;
    const sellExchange = exchange1.price > exchange2.price ? exchange1.exchange : exchange2.exchange;
    
    const spread = highPrice - lowPrice;
    const spreadPercent = (spread / lowPrice) * 100;
    
    // Estimate volume (simplified)
    const volume = Math.min(100000, 50000 / lowPrice); // Up to $50k or equivalent
    
    // Calculate profit potential (after fees)
    const buyFee = 0.001; // 0.1% typical fee
    const sellFee = 0.001;
    const transferFee = 0.0005; // Transfer fee
    
    const totalCost = lowPrice * (1 + buyFee + transferFee);
    const totalRevenue = highPrice * (1 - sellFee);
    const profit = totalRevenue - totalCost;
    const profitPotential = profit * volume;
    
    // Calculate confidence based on spread stability and volume
    const timeDiff = Math.abs(exchange1.timestamp - exchange2.timestamp);
    const timeConfidence = Math.max(0, 1 - timeDiff / 5000); // 5 second window
    const spreadConfidence = Math.min(1, spreadPercent / 2); // Higher spread = higher confidence
    const confidence = (timeConfidence + spreadConfidence) / 2;
    
    if (profitPotential <= 0) return null;
    
    return {
      symbol,
      buyExchange,
      sellExchange,
      buyPrice: lowPrice,
      sellPrice: highPrice,
      spread,
      spreadPercent,
      volume,
      profitPotential,
      confidence,
      timestamp: new Date()
    };
  }

  private async storeOpportunity(opportunity: ArbitrageOpportunity) {
    try {
      await storage.createAlert({
        userId: 'system',
        type: 'info',
        title: `Arbitrage Opportunity: ${opportunity.symbol}`,
        message: `Buy on ${opportunity.buyExchange} at $${opportunity.buyPrice.toFixed(2)}, sell on ${opportunity.sellExchange} at $${opportunity.sellPrice.toFixed(2)}. Spread: ${opportunity.spreadPercent.toFixed(2)}%`
      });
    } catch (error) {
      console.error('Error storing arbitrage opportunity:', error);
    }
  }

  private async executeArbitrage(opportunity: ArbitrageOpportunity) {
    console.log(`Executing arbitrage for ${opportunity.symbol}:`);
    console.log(`Buy ${opportunity.volume} on ${opportunity.buyExchange} at $${opportunity.buyPrice}`);
    console.log(`Sell ${opportunity.volume} on ${opportunity.sellExchange} at $${opportunity.sellPrice}`);
    console.log(`Expected profit: $${opportunity.profitPotential.toFixed(2)}`);
    
    try {
      const buyExchange = this.exchanges.get(opportunity.buyExchange);
      const sellExchange = this.exchanges.get(opportunity.sellExchange);
      
      if (!buyExchange || !sellExchange) {
        console.error('Exchange not found for arbitrage execution');
        return;
      }
      
      // Check balances
      const buyBalance = await buyExchange.getBalance();
      const sellBalance = await sellExchange.getBalance();
      
      const baseAsset = opportunity.symbol.split('/')[0];
      const quoteAsset = opportunity.symbol.split('/')[1];
      
      const usdtBalance = buyBalance.find(b => b.asset === quoteAsset);
      const assetBalance = sellBalance.find(b => b.asset === baseAsset);
      
      if (!usdtBalance || usdtBalance.free < opportunity.buyPrice * opportunity.volume) {
        console.log('Insufficient USDT balance for arbitrage');
        return;
      }
      
      if (!assetBalance || assetBalance.free < opportunity.volume) {
        console.log(`Insufficient ${baseAsset} balance for arbitrage`);
        return;
      }
      
      // Execute simultaneous trades
      const [buyOrder, sellOrder] = await Promise.all([
        buyExchange.placeOrder({
          symbol: opportunity.symbol,
          side: 'BUY',
          type: 'MARKET',
          quantity: opportunity.volume
        }),
        sellExchange.placeOrder({
          symbol: opportunity.symbol,
          side: 'SELL',
          type: 'MARKET',
          quantity: opportunity.volume
        })
      ]);
      
      console.log('Arbitrage executed successfully!');
      console.log('Buy order:', buyOrder);
      console.log('Sell order:', sellOrder);
      
      // Store the trade
      await storage.createTrade({
        userId: 'system',
        symbol: opportunity.symbol,
        side: 'arbitrage',
        amount: opportunity.volume.toString(),
        price: opportunity.buyPrice.toString(),
        pnl: opportunity.profitPotential.toString(),
        strategy: 'cross-exchange-arbitrage',
        orderType: 'market',
        exchange: `${opportunity.buyExchange}-${opportunity.sellExchange}`
      });
      
    } catch (error) {
      console.error('Error executing arbitrage:', error);
      
      await storage.createAlert({
        userId: 'system',
        type: 'danger',
        title: 'Arbitrage Execution Failed',
        message: `Failed to execute arbitrage for ${opportunity.symbol}: ${error.message}`
      });
    }
  }

  // Statistical arbitrage using mean reversion
  async detectStatisticalArbitrage(symbol: string, lookback: number = 100): Promise<{
    signal: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    zscore: number;
    meanPrice: number;
    currentPrice: number;
  }> {
    const marketData = await storage.getMarketData(symbol);
    
    if (marketData.length < lookback) {
      return {
        signal: 'HOLD',
        confidence: 0,
        zscore: 0,
        meanPrice: 0,
        currentPrice: 0
      };
    }
    
    const prices = marketData.slice(-lookback).map(m => parseFloat(m.price));
    const currentPrice = prices[prices.length - 1];
    
    // Calculate mean and standard deviation
    const meanPrice = prices.reduce((sum, p) => sum + p, 0) / prices.length;
    const variance = prices.reduce((sum, p) => sum + Math.pow(p - meanPrice, 2), 0) / prices.length;
    const stdDev = Math.sqrt(variance);
    
    // Calculate z-score
    const zscore = (currentPrice - meanPrice) / stdDev;
    
    // Generate signal based on z-score
    let signal: 'BUY' | 'SELL' | 'HOLD';
    let confidence: number;
    
    if (zscore > 2) {
      signal = 'SELL'; // Price is too high, expect reversion
      confidence = Math.min(1, Math.abs(zscore) / 3);
    } else if (zscore < -2) {
      signal = 'BUY'; // Price is too low, expect reversion
      confidence = Math.min(1, Math.abs(zscore) / 3);
    } else {
      signal = 'HOLD';
      confidence = 0;
    }
    
    return {
      signal,
      confidence,
      zscore,
      meanPrice,
      currentPrice
    };
  }

  // Triangular arbitrage detection
  async detectTriangularArbitrage(
    base: string,
    quote: string,
    intermediate: string
  ): Promise<{
    profitable: boolean;
    profit: number;
    path: string[];
    rates: number[];
  }> {
    try {
      // Get exchange rates for all three pairs
      const pair1 = `${base}/${intermediate}`;
      const pair2 = `${intermediate}/${quote}`;
      const pair3 = `${base}/${quote}`;
      
      const rate1 = await this.getAveragePrice(pair1);
      const rate2 = await this.getAveragePrice(pair2);
      const rate3 = await this.getAveragePrice(pair3);
      
      // Calculate cross rate
      const crossRate = rate1 * rate2;
      const directRate = rate3;
      
      // Check for arbitrage opportunity
      const profit = crossRate - directRate;
      const profitPercent = (profit / directRate) * 100;
      
      const profitable = Math.abs(profitPercent) > 0.1; // 0.1% threshold
      
      return {
        profitable,
        profit: profitPercent,
        path: [base, intermediate, quote, base],
        rates: [rate1, rate2, rate3]
      };
    } catch (error) {
      console.error('Error detecting triangular arbitrage:', error);
      return {
        profitable: false,
        profit: 0,
        path: [],
        rates: []
      };
    }
  }

  private async getAveragePrice(symbol: string): Promise<number> {
    const prices = [];
    
    for (const [exchangeName, exchange] of this.exchanges) {
      try {
        const price = await exchange.getPrice(symbol);
        prices.push(price);
      } catch (error) {
        // Skip if pair not available on exchange
        continue;
      }
    }
    
    if (prices.length === 0) {
      throw new Error(`No prices available for ${symbol}`);
    }
    
    return prices.reduce((sum, p) => sum + p, 0) / prices.length;
  }

  // Funding rate arbitrage
  async detectFundingArbitrage(symbol: string): Promise<{
    opportunity: boolean;
    fundingRate: number;
    spotPrice: number;
    futurePrice: number;
    basis: number;
    annualizedReturn: number;
  }> {
    try {
      // This would require futures data - simplified implementation
      const spotPrice = await this.getAveragePrice(symbol);
      
      // Mock funding rate calculation
      const fundingRate = (Math.random() - 0.5) * 0.002; // -0.1% to 0.1%
      const futurePrice = spotPrice * (1 + fundingRate);
      const basis = futurePrice - spotPrice;
      const annualizedReturn = (fundingRate * 365 * 3) * 100; // 3 funding periods per day
      
      return {
        opportunity: Math.abs(annualizedReturn) > 10, // 10% APY threshold
        fundingRate,
        spotPrice,
        futurePrice,
        basis,
        annualizedReturn
      };
    } catch (error) {
      console.error('Error detecting funding arbitrage:', error);
      return {
        opportunity: false,
        fundingRate: 0,
        spotPrice: 0,
        futurePrice: 0,
        basis: 0,
        annualizedReturn: 0
      };
    }
  }

  // Get current arbitrage opportunities
  async getOpportunities(): Promise<ArbitrageOpportunity[]> {
    const symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'];
    const opportunities: ArbitrageOpportunity[] = [];
    
    for (const symbol of symbols) {
      const exchangePrices = this.getExchangePrices(symbol);
      
      if (exchangePrices.length < 2) continue;
      
      for (let i = 0; i < exchangePrices.length; i++) {
        for (let j = i + 1; j < exchangePrices.length; j++) {
          const opportunity = this.calculateOpportunity(symbol, exchangePrices[i], exchangePrices[j]);
          
          if (opportunity && opportunity.spreadPercent >= 0.1) {
            opportunities.push(opportunity);
          }
        }
      }
    }
    
    return opportunities.sort((a, b) => b.profitPotential - a.profitPotential);
  }
}

export default ArbitrageEngine;