import { storage } from '../storage-database';
import { BinanceExchange } from '../exchanges/binance';
import { CoinbaseExchange } from '../exchanges/coinbase';
import { SignalGenerator } from './signalGenerator';
import { calculateTechnicalIndicators } from './indicators';
import { Telegraf } from 'telegraf';
import cron from 'node-cron';
import { EventEmitter } from 'events';

interface TradingConfig {
  userId: string;
  exchangeKeys: {
    binance?: { apiKey: string; secret: string };
    coinbase?: { apiKey: string; secret: string; passphrase: string };
  };
  telegramBotToken?: string;
  telegramChatId?: string;
  riskManagement: {
    maxPositionSize: number;
    maxDrawdown: number;
    stopLossPercent: number;
    takeProfitPercent: number;
  };
  strategies: string[];
  symbols: string[];
}

export class TradingEngine extends EventEmitter {
  private config: TradingConfig;
  private exchanges: Map<string, BinanceExchange | CoinbaseExchange>;
  private signalGenerator: SignalGenerator;
  private telegramBot?: Telegraf;
  private isRunning: boolean = false;
  private priceData: Map<string, number[]> = new Map();
  private volumeData: Map<string, number[]> = new Map();

  constructor(config: TradingConfig) {
    super();
    this.config = config;
    this.exchanges = new Map();
    this.signalGenerator = new SignalGenerator();
    this.initializeExchanges();
    this.initializeTelegram();
    this.setupEventListeners();
  }

  private initializeExchanges() {
    if (this.config.exchangeKeys.binance) {
      const binance = new BinanceExchange(
        this.config.exchangeKeys.binance.apiKey,
        this.config.exchangeKeys.binance.secret
      );
      this.exchanges.set('binance', binance);
    }

    if (this.config.exchangeKeys.coinbase) {
      const coinbase = new CoinbaseExchange(
        this.config.exchangeKeys.coinbase.apiKey,
        this.config.exchangeKeys.coinbase.secret,
        this.config.exchangeKeys.coinbase.passphrase
      );
      this.exchanges.set('coinbase', coinbase);
    }
  }

  private initializeTelegram() {
    if (this.config.telegramBotToken) {
      this.telegramBot = new Telegraf(this.config.telegramBotToken);
      this.telegramBot.start((ctx) => {
        ctx.reply('🤖 AI Trading Bot is now active!');
      });
      this.telegramBot.launch();
    }
  }

  private setupEventListeners() {
    this.on('signal', this.handleTradingSignal.bind(this));
    this.on('trade', this.handleTradeExecution.bind(this));
    this.on('risk', this.handleRiskAlert.bind(this));
  }

  async start() {
    if (this.isRunning) {
      console.log('Trading engine is already running');
      return;
    }

    this.isRunning = true;
    console.log('Starting AI Trading Engine...');

    // Start market data collection
    this.startMarketDataCollection();

    // Start signal generation
    this.startSignalGeneration();

    // Start risk monitoring
    this.startRiskMonitoring();

    // Update bot status
    await storage.updateBotStatus(this.config.userId, {
      isActive: true,
      currentStrategy: this.config.strategies[0] || 'Multi-Strategy',
      uptime: 0,
    });

    await this.sendTelegramMessage('🚀 AI Trading Bot started successfully!');
  }

  async stop() {
    if (!this.isRunning) {
      console.log('Trading engine is not running');
      return;
    }

    this.isRunning = false;
    console.log('Stopping AI Trading Engine...');

    // Close all open positions
    await this.closeAllPositions();

    // Update bot status
    await storage.updateBotStatus(this.config.userId, {
      isActive: false,
    });

    await this.sendTelegramMessage('🛑 AI Trading Bot stopped safely!');
  }

  private startMarketDataCollection() {
    // Collect market data every 30 seconds
    cron.schedule('*/30 * * * * *', async () => {
      if (!this.isRunning) return;

      try {
        await this.collectMarketData();
      } catch (error) {
        console.error('Error collecting market data:', error);
      }
    });
  }

  private startSignalGeneration() {
    // Generate signals every 5 minutes
    cron.schedule('*/5 * * * *', async () => {
      if (!this.isRunning) return;

      try {
        await this.generateSignals();
      } catch (error) {
        console.error('Error generating signals:', error);
      }
    });
  }

  private startRiskMonitoring() {
    // Monitor risk every minute
    cron.schedule('* * * * *', async () => {
      if (!this.isRunning) return;

      try {
        await this.monitorRisk();
      } catch (error) {
        console.error('Error monitoring risk:', error);
      }
    });
  }

  private async collectMarketData() {
    const binance = this.exchanges.get('binance') as BinanceExchange;
    if (!binance) return;

    for (const symbol of this.config.symbols) {
      try {
        const stats = await binance.get24hrStats(symbol);
        const klines = await binance.getKlines(symbol, '1m', 100);

        // Update price and volume data
        const prices = klines.map(k => k.close);
        const volumes = klines.map(k => k.volume);
        
        this.priceData.set(symbol, prices);
        this.volumeData.set(symbol, volumes);

        // Store in database
        await storage.updateMarketData(symbol, {
          price: stats.price.toString(),
          change24h: stats.change.toString(),
          changePercent24h: stats.changePercent.toString(),
          volume24h: stats.volume.toString(),
          high24h: stats.high.toString(),
          low24h: stats.low.toString(),
          exchange: 'binance',
        });
      } catch (error) {
        console.error(`Error collecting data for ${symbol}:`, error);
      }
    }
  }

  private async generateSignals() {
    for (const symbol of this.config.symbols) {
      try {
        const prices = this.priceData.get(symbol);
        const volumes = this.volumeData.get(symbol);

        if (!prices || !volumes || prices.length < 20) continue;

        // Generate trading signal
        const signal = await this.signalGenerator.generateSignal(
          symbol,
          prices,
          volumes,
          [] // TODO: Add news data
        );

        // Emit signal event
        this.emit('signal', { symbol, signal });

        // Store signal in database
        await storage.createAlert({
          userId: this.config.userId,
          type: signal.signal === 'BUY' ? 'success' : signal.signal === 'SELL' ? 'danger' : 'info',
          title: `${signal.signal} Signal for ${symbol}`,
          message: `${signal.reasoning} (Confidence: ${(signal.confidence * 100).toFixed(1)}%)`,
        });
      } catch (error) {
        console.error(`Error generating signal for ${symbol}:`, error);
      }
    }
  }

  private async handleTradingSignal(event: { symbol: string; signal: any }) {
    const { symbol, signal } = event;

    // Check if signal confidence is high enough
    if (signal.confidence < 0.7) {
      console.log(`Signal confidence too low for ${symbol}: ${signal.confidence}`);
      return;
    }

    // Check risk management
    if (!(await this.checkRiskManagement(symbol, signal))) {
      console.log(`Risk management check failed for ${symbol}`);
      return;
    }

    // Execute trade
    await this.executeTrade(symbol, signal);
  }

  private async executeTrade(symbol: string, signal: any) {
    const binance = this.exchanges.get('binance') as BinanceExchange;
    if (!binance) return;

    try {
      const balance = await binance.getBalance();
      const usdtBalance = balance.find(b => b.asset === 'USDT');
      
      if (!usdtBalance || usdtBalance.free < 10) {
        console.log('Insufficient USDT balance');
        return;
      }

      // Calculate position size
      const positionSize = Math.min(
        usdtBalance.free * (this.config.riskManagement.maxPositionSize / 100),
        usdtBalance.free * 0.1 // Max 10% per trade
      );

      const currentPrice = await binance.getPrice(symbol);
      const quantity = positionSize / currentPrice;

      // Place order
      const order = await binance.placeOrder({
        symbol,
        side: signal.signal === 'BUY' ? 'BUY' : 'SELL',
        type: 'MARKET',
        quantity,
      });

      // Store trade in database
      await storage.createTrade({
        userId: this.config.userId,
        symbol,
        side: signal.signal.toLowerCase(),
        amount: quantity.toString(),
        price: currentPrice.toString(),
        pnl: '0',
        strategy: signal.strategy,
        orderType: 'market',
        exchange: 'binance',
      });

      // Create position
      await storage.createPosition({
        userId: this.config.userId,
        symbol,
        side: signal.signal.toLowerCase(),
        size: quantity.toString(),
        entryPrice: currentPrice.toString(),
        currentPrice: currentPrice.toString(),
        pnl: '0',
        pnlPercent: '0',
        stopLoss: (currentPrice * (1 - this.config.riskManagement.stopLossPercent / 100)).toString(),
        takeProfit: (currentPrice * (1 + this.config.riskManagement.takeProfitPercent / 100)).toString(),
      });

      this.emit('trade', { symbol, signal, order });

      await this.sendTelegramMessage(
        `🎯 Trade Executed!\n` +
        `Symbol: ${symbol}\n` +
        `Action: ${signal.signal}\n` +
        `Quantity: ${quantity.toFixed(6)}\n` +
        `Price: $${currentPrice.toFixed(2)}\n` +
        `Confidence: ${(signal.confidence * 100).toFixed(1)}%`
      );
    } catch (error) {
      console.error('Error executing trade:', error);
      await this.sendTelegramMessage(`❌ Trade failed for ${symbol}: ${error.message}`);
    }
  }

  private async checkRiskManagement(symbol: string, signal: any): Promise<boolean> {
    // Check portfolio heat
    const riskMetrics = await storage.getRiskMetrics(this.config.userId);
    if (riskMetrics && parseFloat(riskMetrics.portfolioHeat) > 80) {
      return false;
    }

    // Check max drawdown
    if (riskMetrics && parseFloat(riskMetrics.maxDrawdown) > this.config.riskManagement.maxDrawdown) {
      return false;
    }

    // Check open positions
    const positions = await storage.getPositions(this.config.userId);
    const openPositions = positions.filter(p => p.isActive);
    
    if (openPositions.length >= 5) { // Max 5 open positions
      return false;
    }

    return true;
  }

  private async monitorRisk() {
    const positions = await storage.getPositions(this.config.userId);
    const openPositions = positions.filter(p => p.isActive);

    for (const position of openPositions) {
      try {
        const binance = this.exchanges.get('binance') as BinanceExchange;
        if (!binance) continue;

        const currentPrice = await binance.getPrice(position.symbol);
        const entryPrice = parseFloat(position.entryPrice);
        const pnl = (currentPrice - entryPrice) * parseFloat(position.size);
        const pnlPercent = ((currentPrice - entryPrice) / entryPrice) * 100;

        // Update position
        await storage.updatePosition(position.id, {
          currentPrice: currentPrice.toString(),
          pnl: pnl.toString(),
          pnlPercent: pnlPercent.toString(),
        });

        // Check stop loss
        if (position.stopLoss && currentPrice <= parseFloat(position.stopLoss)) {
          await this.closePosition(position.id, 'Stop Loss Triggered');
        }

        // Check take profit
        if (position.takeProfit && currentPrice >= parseFloat(position.takeProfit)) {
          await this.closePosition(position.id, 'Take Profit Triggered');
        }

        // Check if position is too old (24 hours)
        const positionAge = Date.now() - position.createdAt.getTime();
        if (positionAge > 24 * 60 * 60 * 1000) {
          await this.closePosition(position.id, 'Position Expired');
        }
      } catch (error) {
        console.error(`Error monitoring position ${position.id}:`, error);
      }
    }

    // Update risk metrics
    await this.updateRiskMetrics();
  }

  private async closePosition(positionId: number, reason: string) {
    const position = await storage.closePosition(positionId);
    
    await this.sendTelegramMessage(
      `🔄 Position Closed\n` +
      `Symbol: ${position.symbol}\n` +
      `Reason: ${reason}\n` +
      `P&L: ${parseFloat(position.pnl) > 0 ? '+' : ''}${parseFloat(position.pnl).toFixed(2)} USDT\n` +
      `P&L%: ${parseFloat(position.pnlPercent) > 0 ? '+' : ''}${parseFloat(position.pnlPercent).toFixed(2)}%`
    );
  }

  private async closeAllPositions() {
    const positions = await storage.getPositions(this.config.userId);
    const openPositions = positions.filter(p => p.isActive);

    for (const position of openPositions) {
      await this.closePosition(position.id, 'Bot Stopped');
    }
  }

  private async updateRiskMetrics() {
    const positions = await storage.getPositions(this.config.userId);
    const trades = await storage.getTrades(this.config.userId, 100);

    // Calculate metrics
    const totalPnl = positions.reduce((sum, p) => sum + parseFloat(p.pnl), 0);
    const winningTrades = trades.filter(t => parseFloat(t.pnl) > 0);
    const winRate = trades.length > 0 ? (winningTrades.length / trades.length) * 100 : 0;

    // Portfolio heat calculation
    const openPositions = positions.filter(p => p.isActive);
    const portfolioHeat = Math.min(100, openPositions.length * 20); // 20% per position

    // Update metrics
    await storage.updateRiskMetrics(this.config.userId, {
      portfolioHeat: portfolioHeat.toString(),
      maxDrawdown: Math.abs(Math.min(0, totalPnl)).toString(),
      var95: Math.abs(totalPnl * 0.05).toString(),
      riskScore: (portfolioHeat / 10).toString(),
    });
  }

  private async sendTelegramMessage(message: string) {
    if (this.telegramBot && this.config.telegramChatId) {
      try {
        await this.telegramBot.telegram.sendMessage(this.config.telegramChatId, message);
      } catch (error) {
        console.error('Error sending Telegram message:', error);
      }
    }
  }

  private async handleTradeExecution(event: any) {
    console.log('Trade executed:', event);
  }

  private async handleRiskAlert(event: any) {
    console.log('Risk alert:', event);
    await this.sendTelegramMessage(`⚠️ Risk Alert: ${event.message}`);
  }

  // Public methods for external control
  async updateConfig(newConfig: Partial<TradingConfig>) {
    this.config = { ...this.config, ...newConfig };
    
    // Reinitialize exchanges if keys changed
    if (newConfig.exchangeKeys) {
      this.exchanges.clear();
      this.initializeExchanges();
    }
  }

  async getStatus() {
    return {
      isRunning: this.isRunning,
      config: this.config,
      connectedExchanges: Array.from(this.exchanges.keys()),
      dataPoints: {
        priceData: this.priceData.size,
        volumeData: this.volumeData.size,
      },
    };
  }

  async forceSignalGeneration() {
    await this.generateSignals();
  }

  async emergencyStop() {
    await this.stop();
    await this.sendTelegramMessage('🚨 EMERGENCY STOP ACTIVATED!');
  }
}

export default TradingEngine;