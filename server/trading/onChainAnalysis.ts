import { storage } from '../storage-database';
import { ethers } from 'ethers';
import crypto from 'crypto';

interface OnChainMetrics {
  symbol: string;
  activeAddresses: number;
  transactionCount: number;
  transactionVolume: number;
  networkHashRate: number;
  difficulty: number;
  whaleMovements: WhaleMovement[];
  exchangeFlows: ExchangeFlow[];
  derivedSignals: OnChainSignal[];
  timestamp: Date;
}

interface WhaleMovement {
  address: string;
  amount: number;
  direction: 'inflow' | 'outflow';
  exchange?: string;
  timestamp: Date;
}

interface ExchangeFlow {
  exchange: string;
  inflow: number;
  outflow: number;
  netFlow: number;
  timestamp: Date;
}

interface OnChainSignal {
  indicator: string;
  value: number;
  signal: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  confidence: number;
  description: string;
}

export class OnChainAnalysis {
  private provider: ethers.JsonRpcProvider;
  private knownExchanges: Map<string, string> = new Map();
  private whaleAddresses: Set<string> = new Set();

  constructor(rpcUrl: string = 'https://mainnet.infura.io/v3/demo') {
    this.provider = new ethers.JsonRpcProvider(rpcUrl);
    this.initializeKnownAddresses();
  }

  private initializeKnownAddresses() {
    // Known exchange addresses (simplified)
    this.knownExchanges.set('0x28c6c06298d514db089934071355e5743bf21d60', 'Binance');
    this.knownExchanges.set('0x21a31ee1afc51d94c2efccaa2092ad1028285549', 'Binance');
    this.knownExchanges.set('0x564286362092d8e7936f0549571a803b203aaced', 'Binance');
    this.knownExchanges.set('0x0681d8db095565fe8a346fa0277bffde9c0edbbf', 'Binance');
    this.knownExchanges.set('0x4e9ce36e442e55ecd9025b9a6e0d88485d628a67', 'Binance');
    this.knownExchanges.set('0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be', 'Binance');
    this.knownExchanges.set('0xd551234ae421e3bcba99a0da6d736074f22192ff', 'Binance');
    this.knownExchanges.set('0x4976fb03c32e5b8cfe2b6ccb31c09ba78ebaba41', 'Binance');
    this.knownExchanges.set('0xa910f92acdaf488fa6ef02174fb86208ad7722ba', 'Binance');
    this.knownExchanges.set('0x1522900b6dafac587d499a862861c0869be6e428', 'Binance');
    this.knownExchanges.set('0x4fabb145d64652a948d72533023f6e7a623c7c53', 'Binance BUSD');
    this.knownExchanges.set('0x6cc5f688a315f3dc28a7781717a9a798a59fda7b', 'Coinbase');
    this.knownExchanges.set('0x503828976d22510aad0201ac7ec88293211d23da', 'Coinbase');
    this.knownExchanges.set('0xddfabcdc4d8ffc6d5beaf154f18b778f892a0740', 'Coinbase');
    this.knownExchanges.set('0x3cd751e6b0078be393132286c442345e5dc49699', 'Coinbase');
    this.knownExchanges.set('0xb5d85cbf7cb3ee0d56b3bb207d5fc4b82f43f511', 'Coinbase');
    this.knownExchanges.set('0xeb2629a2734e272bcc07bda959863f316f4bd4cf', 'Coinbase');
    this.knownExchanges.set('0x5041ed759dd4afc3a72b8192c143f72f4724081a', 'Kraken');
    this.knownExchanges.set('0x2910543af39aba0cd09dbb2d50200b3e800a63d2', 'Kraken');
    this.knownExchanges.set('0x0a869d79a7052c7f1b55a8ebabbea3420f0d1e13', 'Kraken');
    this.knownExchanges.set('0xe853c56864a2ebe4576a807d26fdc4a0ada51919', 'Kraken');
    this.knownExchanges.set('0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0', 'Kraken');

    // Known whale addresses (simplified)
    this.whaleAddresses.add('0x00000000219ab540356cbb839cbe05303d7705fa'); // Ethereum 2.0 Deposit Contract
    this.whaleAddresses.add('0x8315177ab297ba92a06054ce80a67ed4dbd7ed3a'); // Bitfinex
    this.whaleAddresses.add('0x876eabf441b2ee5b5b0554fd502a8e0600950cfa'); // Gemini
    this.whaleAddresses.add('0x1db3439a222c519ab44bb1144fc28167b4fa6ee6'); // Vitalik Buterin
    this.whaleAddresses.add('0xab5801a7d398351b8be11c439e05c5b3259aec9b'); // Vitalik Buterin
    this.whaleAddresses.add('0x9ee0a4e21bd95b1e4b8e7e9c7d8b7d6c4e5f2b6a'); // Example whale
  }

  async analyzeOnChainMetrics(symbol: string): Promise<OnChainMetrics> {
    try {
      const blockNumber = await this.provider.getBlockNumber();
      const block = await this.provider.getBlock(blockNumber);
      
      // Get recent transaction data
      const transactions = [];
      for (let i = 0; i < Math.min(10, block?.transactions?.length || 0); i++) {
        const tx = await this.provider.getTransaction(block.transactions[i]);
        if (tx) transactions.push(tx);
      }

      // Analyze whale movements
      const whaleMovements = await this.analyzeWhaleMovements(transactions);
      
      // Analyze exchange flows
      const exchangeFlows = await this.analyzeExchangeFlows(transactions);
      
      // Calculate metrics
      const metrics: OnChainMetrics = {
        symbol,
        activeAddresses: await this.calculateActiveAddresses(transactions),
        transactionCount: transactions.length,
        transactionVolume: this.calculateTransactionVolume(transactions),
        networkHashRate: await this.getNetworkHashRate(),
        difficulty: await this.getDifficulty(),
        whaleMovements,
        exchangeFlows,
        derivedSignals: await this.generateOnChainSignals(symbol, whaleMovements, exchangeFlows),
        timestamp: new Date()
      };

      // Store metrics in database
      await this.storeOnChainData(metrics);
      
      return metrics;
    } catch (error) {
      console.error('Error analyzing on-chain metrics:', error);
      throw error;
    }
  }

  private async analyzeWhaleMovements(transactions: any[]): Promise<WhaleMovement[]> {
    const movements: WhaleMovement[] = [];
    
    for (const tx of transactions) {
      if (!tx.from || !tx.to || !tx.value) continue;
      
      const value = parseFloat(ethers.formatEther(tx.value));
      
      // Check if it's a whale movement (>100 ETH)
      if (value > 100) {
        const isFromWhale = this.whaleAddresses.has(tx.from);
        const isToWhale = this.whaleAddresses.has(tx.to);
        const fromExchange = this.knownExchanges.get(tx.from);
        const toExchange = this.knownExchanges.get(tx.to);
        
        if (isFromWhale || isToWhale || fromExchange || toExchange) {
          movements.push({
            address: isFromWhale ? tx.from : tx.to,
            amount: value,
            direction: fromExchange ? 'outflow' : 'inflow',
            exchange: fromExchange || toExchange,
            timestamp: new Date()
          });
        }
      }
    }
    
    return movements;
  }

  private async analyzeExchangeFlows(transactions: any[]): Promise<ExchangeFlow[]> {
    const flows: Map<string, { inflow: number; outflow: number }> = new Map();
    
    for (const tx of transactions) {
      if (!tx.from || !tx.to || !tx.value) continue;
      
      const value = parseFloat(ethers.formatEther(tx.value));
      const fromExchange = this.knownExchanges.get(tx.from);
      const toExchange = this.knownExchanges.get(tx.to);
      
      if (fromExchange) {
        const current = flows.get(fromExchange) || { inflow: 0, outflow: 0 };
        current.outflow += value;
        flows.set(fromExchange, current);
      }
      
      if (toExchange) {
        const current = flows.get(toExchange) || { inflow: 0, outflow: 0 };
        current.inflow += value;
        flows.set(toExchange, current);
      }
    }
    
    return Array.from(flows.entries()).map(([exchange, flow]) => ({
      exchange,
      inflow: flow.inflow,
      outflow: flow.outflow,
      netFlow: flow.inflow - flow.outflow,
      timestamp: new Date()
    }));
  }

  private async calculateActiveAddresses(transactions: any[]): Promise<number> {
    const addresses = new Set<string>();
    
    for (const tx of transactions) {
      if (tx.from) addresses.add(tx.from);
      if (tx.to) addresses.add(tx.to);
    }
    
    return addresses.size;
  }

  private calculateTransactionVolume(transactions: any[]): number {
    return transactions.reduce((total, tx) => {
      if (tx.value) {
        return total + parseFloat(ethers.formatEther(tx.value));
      }
      return total;
    }, 0);
  }

  private async getNetworkHashRate(): Promise<number> {
    try {
      const block = await this.provider.getBlock('latest');
      // Simplified hash rate calculation
      return block?.difficulty ? Number(block.difficulty) / 1e12 : 0;
    } catch (error) {
      console.error('Error getting network hash rate:', error);
      return 0;
    }
  }

  private async getDifficulty(): Promise<number> {
    try {
      const block = await this.provider.getBlock('latest');
      return block?.difficulty ? Number(block.difficulty) : 0;
    } catch (error) {
      console.error('Error getting difficulty:', error);
      return 0;
    }
  }

  private async generateOnChainSignals(
    symbol: string,
    whaleMovements: WhaleMovement[],
    exchangeFlows: ExchangeFlow[]
  ): Promise<OnChainSignal[]> {
    const signals: OnChainSignal[] = [];
    
    // Whale movement signal
    const whaleOutflows = whaleMovements.filter(m => m.direction === 'outflow');
    const whaleInflows = whaleMovements.filter(m => m.direction === 'inflow');
    
    if (whaleOutflows.length > whaleInflows.length) {
      signals.push({
        indicator: 'Whale Movement',
        value: whaleOutflows.length - whaleInflows.length,
        signal: 'BEARISH',
        confidence: Math.min(0.8, (whaleOutflows.length - whaleInflows.length) / 10),
        description: 'Large holders are moving funds out of their wallets'
      });
    } else if (whaleInflows.length > whaleOutflows.length) {
      signals.push({
        indicator: 'Whale Movement',
        value: whaleInflows.length - whaleOutflows.length,
        signal: 'BULLISH',
        confidence: Math.min(0.8, (whaleInflows.length - whaleOutflows.length) / 10),
        description: 'Large holders are accumulating'
      });
    }
    
    // Exchange flow signal
    const totalNetFlow = exchangeFlows.reduce((sum, flow) => sum + flow.netFlow, 0);
    
    if (totalNetFlow > 0) {
      signals.push({
        indicator: 'Exchange Flow',
        value: totalNetFlow,
        signal: 'BEARISH',
        confidence: Math.min(0.7, Math.abs(totalNetFlow) / 1000),
        description: 'Net inflow to exchanges suggests selling pressure'
      });
    } else if (totalNetFlow < 0) {
      signals.push({
        indicator: 'Exchange Flow',
        value: Math.abs(totalNetFlow),
        signal: 'BULLISH',
        confidence: Math.min(0.7, Math.abs(totalNetFlow) / 1000),
        description: 'Net outflow from exchanges suggests accumulation'
      });
    }
    
    // Network activity signal
    const networkActivity = whaleMovements.length + exchangeFlows.length;
    
    if (networkActivity > 10) {
      signals.push({
        indicator: 'Network Activity',
        value: networkActivity,
        signal: 'BULLISH',
        confidence: Math.min(0.6, networkActivity / 20),
        description: 'High network activity suggests increased interest'
      });
    } else if (networkActivity < 3) {
      signals.push({
        indicator: 'Network Activity',
        value: networkActivity,
        signal: 'BEARISH',
        confidence: 0.4,
        description: 'Low network activity suggests decreased interest'
      });
    }
    
    return signals;
  }

  private async storeOnChainData(metrics: OnChainMetrics): Promise<void> {
    try {
      // Store in database (simplified)
      await storage.createAlert({
        userId: 'system',
        type: 'info',
        title: `On-Chain Analysis: ${metrics.symbol}`,
        message: `Active addresses: ${metrics.activeAddresses}, TX count: ${metrics.transactionCount}, Volume: ${metrics.transactionVolume.toFixed(2)} ETH`
      });
    } catch (error) {
      console.error('Error storing on-chain data:', error);
    }
  }

  // Advanced on-chain metrics
  async calculateNVT(symbol: string): Promise<number> {
    try {
      // Network Value to Transactions ratio
      const marketData = await storage.getMarketData(symbol);
      if (marketData.length === 0) return 0;
      
      const latestData = marketData[0];
      const marketCap = parseFloat(latestData.price) * 21000000; // Simplified for BTC
      const transactionVolume = parseFloat(latestData.volume24h);
      
      return transactionVolume > 0 ? marketCap / transactionVolume : 0;
    } catch (error) {
      console.error('Error calculating NVT:', error);
      return 0;
    }
  }

  async calculateRVT(symbol: string): Promise<number> {
    try {
      // Realized Value to Transactions ratio
      const nvt = await this.calculateNVT(symbol);
      const realizedCapMultiplier = 0.6; // Simplified ratio
      
      return nvt * realizedCapMultiplier;
    } catch (error) {
      console.error('Error calculating RVT:', error);
      return 0;
    }
  }

  async calculateMVRV(symbol: string): Promise<number> {
    try {
      // Market Value to Realized Value ratio
      const marketData = await storage.getMarketData(symbol);
      if (marketData.length === 0) return 1;
      
      const currentPrice = parseFloat(marketData[0].price);
      const realizedPrice = currentPrice * 0.8; // Simplified realized price
      
      return currentPrice / realizedPrice;
    } catch (error) {
      console.error('Error calculating MVRV:', error);
      return 1;
    }
  }

  async calculateSoprScore(symbol: string): Promise<number> {
    try {
      // Spent Output Profit Ratio
      const mvrv = await this.calculateMVRV(symbol);
      
      // Simplified SOPR calculation
      return mvrv > 1 ? Math.min(2, mvrv) : Math.max(0, mvrv);
    } catch (error) {
      console.error('Error calculating SOPR:', error);
      return 1;
    }
  }

  async detectUnusualActivity(symbol: string): Promise<{
    detected: boolean;
    type: string;
    severity: 'LOW' | 'MEDIUM' | 'HIGH';
    description: string;
    confidence: number;
  }> {
    try {
      const metrics = await this.analyzeOnChainMetrics(symbol);
      
      // Check for unusual whale activity
      const largeMovements = metrics.whaleMovements.filter(m => m.amount > 1000);
      
      if (largeMovements.length > 5) {
        return {
          detected: true,
          type: 'Whale Activity',
          severity: 'HIGH',
          description: `${largeMovements.length} large whale movements detected`,
          confidence: Math.min(0.9, largeMovements.length / 10)
        };
      }
      
      // Check for unusual exchange flows
      const totalFlow = metrics.exchangeFlows.reduce((sum, flow) => sum + Math.abs(flow.netFlow), 0);
      
      if (totalFlow > 10000) {
        return {
          detected: true,
          type: 'Exchange Flow',
          severity: 'MEDIUM',
          description: `Unusual exchange flow of ${totalFlow.toFixed(0)} ETH`,
          confidence: Math.min(0.8, totalFlow / 20000)
        };
      }
      
      // Check for network congestion
      if (metrics.transactionCount > 50) {
        return {
          detected: true,
          type: 'Network Congestion',
          severity: 'LOW',
          description: `High transaction count: ${metrics.transactionCount}`,
          confidence: Math.min(0.6, metrics.transactionCount / 100)
        };
      }
      
      return {
        detected: false,
        type: 'None',
        severity: 'LOW',
        description: 'No unusual activity detected',
        confidence: 0
      };
    } catch (error) {
      console.error('Error detecting unusual activity:', error);
      return {
        detected: false,
        type: 'Error',
        severity: 'LOW',
        description: 'Error analyzing activity',
        confidence: 0
      };
    }
  }

  // Sentiment analysis from on-chain data
  async generateSentimentFromOnChain(symbol: string): Promise<{
    sentiment: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
    score: number;
    confidence: number;
    factors: string[];
  }> {
    try {
      const metrics = await this.analyzeOnChainMetrics(symbol);
      const factors: string[] = [];
      let score = 0;
      let confidence = 0;
      
      // Analyze signals
      for (const signal of metrics.derivedSignals) {
        if (signal.signal === 'BULLISH') {
          score += signal.value * signal.confidence;
          factors.push(`Bullish ${signal.indicator}`);
        } else if (signal.signal === 'BEARISH') {
          score -= signal.value * signal.confidence;
          factors.push(`Bearish ${signal.indicator}`);
        }
        confidence += signal.confidence;
      }
      
      // Normalize
      confidence = Math.min(1, confidence / metrics.derivedSignals.length);
      score = score / Math.max(1, metrics.derivedSignals.length);
      
      let sentiment: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
      
      if (score > 0.3) {
        sentiment = 'POSITIVE';
      } else if (score < -0.3) {
        sentiment = 'NEGATIVE';
      } else {
        sentiment = 'NEUTRAL';
      }
      
      return {
        sentiment,
        score,
        confidence,
        factors
      };
    } catch (error) {
      console.error('Error generating sentiment from on-chain data:', error);
      return {
        sentiment: 'NEUTRAL',
        score: 0,
        confidence: 0,
        factors: []
      };
    }
  }
}

export default OnChainAnalysis;