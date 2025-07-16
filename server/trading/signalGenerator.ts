import { MLTradingModels } from './mlModels';
import { SentimentAnalysis } from './sentimentAnalysis';
import { OnChainAnalysis } from './onChainAnalysis';
import { ArbitrageEngine } from './arbitrage';
import { RiskManagement } from './riskManagement';
import { calculateTechnicalIndicators } from './indicators';
import { storage } from '../storage-database';

export interface TradingSignal {
  symbol: string;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  strategy: string;
  reasoning: string;
  technicalScore: number;
  fundamentalScore: number;
  sentimentScore: number;
  riskScore: number;
  priceTarget: number;
  stopLoss: number;
  timeframe: string;
  timestamp: Date;
  components: {
    technical: any;
    ml: any;
    sentiment: any;
    onchain: any;
    arbitrage: any;
    risk: any;
  };
}

export class SignalGenerator {
  private mlModels: MLTradingModels;
  private sentimentAnalysis: SentimentAnalysis;
  private onChainAnalysis: OnChainAnalysis;
  private riskManagement: RiskManagement;

  constructor() {
    this.mlModels = new MLTradingModels();
    this.sentimentAnalysis = new SentimentAnalysis(process.env.OPENAI_API_KEY);
    this.onChainAnalysis = new OnChainAnalysis(process.env.ETHEREUM_RPC_URL);
    this.riskManagement = new RiskManagement('system', 100000, {
      maxPositionSize: 10,
      maxDrawdown: 20,
      var95: 5000,
      var99: 8000,
      sharpeRatio: 1.5,
      portfolioHeat: 60,
      concentrationLimit: 30,
      correlationThreshold: 0.7,
      leverageRatio: 2,
      liquidityRatio: 0.8
    });
  }

  async generateSignal(
    symbol: string,
    priceData: number[],
    volumeData: number[],
    newsData: string[] = []
  ): Promise<TradingSignal> {
    try {
      // 1. Technical Analysis
      const technicalIndicators = calculateTechnicalIndicators(priceData, volumeData);
      const technicalScore = this.calculateTechnicalScore(technicalIndicators);

      // 2. Machine Learning Analysis
      const mlFeatures = await this.mlModels.extractFeatures(
        symbol,
        priceData,
        volumeData,
        0 // Will be filled by sentiment
      );
      const mlPrediction = await this.mlModels.predictSignal(mlFeatures);

      // 3. Sentiment Analysis
      const sentimentSignal = await this.sentimentAnalysis.analyzeSentiment(symbol);

      // 4. On-Chain Analysis
      const onChainMetrics = await this.onChainAnalysis.analyzeOnChainMetrics(symbol);
      const onChainSentiment = await this.onChainAnalysis.generateSentimentFromOnChain(symbol);

      // 5. Risk Assessment
      const riskMetrics = await this.riskManagement.monitorRealTimeRisk();

      // 6. Generate Combined Signal
      const combinedSignal = this.combineSignals({
        technical: { indicators: technicalIndicators, score: technicalScore },
        ml: mlPrediction,
        sentiment: sentimentSignal,
        onchain: { metrics: onChainMetrics, sentiment: onChainSentiment },
        risk: riskMetrics
      }, symbol, priceData);

      // 7. Calculate Price Targets
      const priceTargets = this.calculatePriceTargets(
        symbol,
        priceData,
        combinedSignal.signal,
        technicalIndicators
      );

      // 8. Generate Reasoning
      const reasoning = this.generateReasoning(
        combinedSignal,
        technicalIndicators,
        mlPrediction,
        sentimentSignal,
        onChainSentiment,
        riskMetrics
      );

      const signal: TradingSignal = {
        symbol,
        signal: combinedSignal.signal,
        confidence: combinedSignal.confidence,
        strategy: combinedSignal.strategy,
        reasoning,
        technicalScore,
        fundamentalScore: this.calculateFundamentalScore(onChainMetrics),
        sentimentScore: sentimentSignal.score,
        riskScore: riskMetrics.riskScore,
        priceTarget: priceTargets.target,
        stopLoss: priceTargets.stopLoss,
        timeframe: this.determineTimeframe(technicalIndicators, mlPrediction),
        timestamp: new Date(),
        components: {
          technical: { indicators: technicalIndicators, score: technicalScore },
          ml: mlPrediction,
          sentiment: sentimentSignal,
          onchain: { metrics: onChainMetrics, sentiment: onChainSentiment },
          arbitrage: null, // Would be filled by arbitrage engine
          risk: riskMetrics
        }
      };

      // Store signal in database
      await this.storeSignal(signal);

      return signal;
    } catch (error) {
      console.error('Error generating signal:', error);
      return this.getDefaultSignal(symbol);
    }
  }

  private calculateTechnicalScore(indicators: any): number {
    let score = 0;
    let components = 0;

    // RSI analysis
    if (indicators.rsi < 30) {
      score += 1; // Oversold - bullish
    } else if (indicators.rsi > 70) {
      score -= 1; // Overbought - bearish
    }
    components++;

    // MACD analysis
    if (indicators.macd.macd > indicators.macd.signal) {
      score += 1; // Bullish crossover
    } else {
      score -= 1; // Bearish crossover
    }
    components++;

    // Bollinger Bands analysis
    const currentPrice = indicators.bollinger.middle; // Simplified
    if (currentPrice < indicators.bollinger.lower) {
      score += 1; // Below lower band - bullish
    } else if (currentPrice > indicators.bollinger.upper) {
      score -= 1; // Above upper band - bearish
    }
    components++;

    // Moving Average analysis
    if (indicators.sma < indicators.ema) {
      score += 0.5; // EMA above SMA - bullish
    } else {
      score -= 0.5; // EMA below SMA - bearish
    }
    components++;

    // Volume analysis
    if (indicators.volume > indicators.sma) {
      score += 0.5; // High volume confirms trend
    }
    components++;

    return score / components; // Normalize to -1 to 1
  }

  private calculateFundamentalScore(onChainMetrics: any): number {
    let score = 0;
    let components = 0;

    // Network activity
    if (onChainMetrics.activeAddresses > 1000) {
      score += 1;
    } else if (onChainMetrics.activeAddresses < 500) {
      score -= 1;
    }
    components++;

    // Transaction volume
    if (onChainMetrics.transactionVolume > 10000) {
      score += 1;
    } else if (onChainMetrics.transactionVolume < 1000) {
      score -= 1;
    }
    components++;

    // Derived signals
    for (const signal of onChainMetrics.derivedSignals) {
      if (signal.signal === 'BULLISH') {
        score += signal.confidence;
      } else if (signal.signal === 'BEARISH') {
        score -= signal.confidence;
      }
      components++;
    }

    return components > 0 ? score / components : 0;
  }

  private combineSignals(
    components: any,
    symbol: string,
    priceData: number[]
  ): { signal: 'BUY' | 'SELL' | 'HOLD'; confidence: number; strategy: string } {
    // Weighted scoring system
    const weights = {
      technical: 0.25,
      ml: 0.30,
      sentiment: 0.20,
      onchain: 0.15,
      risk: 0.10
    };

    let totalScore = 0;
    let totalWeight = 0;

    // Technical score
    totalScore += components.technical.score * weights.technical;
    totalWeight += weights.technical;

    // ML prediction
    const mlScore = components.ml.signal === 'BUY' ? components.ml.confidence :
                    components.ml.signal === 'SELL' ? -components.ml.confidence : 0;
    totalScore += mlScore * weights.ml;
    totalWeight += weights.ml;

    // Sentiment score
    totalScore += components.sentiment.score * weights.sentiment;
    totalWeight += weights.sentiment;

    // On-chain score
    const onChainScore = components.onchain.sentiment.sentiment === 'POSITIVE' ? components.onchain.sentiment.score :
                         components.onchain.sentiment.sentiment === 'NEGATIVE' ? -components.onchain.sentiment.score : 0;
    totalScore += onChainScore * weights.onchain;
    totalWeight += weights.onchain;

    // Risk adjustment
    const riskAdjustment = 1 - (components.risk.riskScore / 100);
    totalScore *= riskAdjustment;

    // Normalize
    const finalScore = totalScore / totalWeight;
    const confidence = Math.abs(finalScore);

    // Generate signal
    let signal: 'BUY' | 'SELL' | 'HOLD';
    let strategy: string;

    if (finalScore > 0.3) {
      signal = 'BUY';
      strategy = this.getBuyStrategy(components);
    } else if (finalScore < -0.3) {
      signal = 'SELL';
      strategy = this.getSellStrategy(components);
    } else {
      signal = 'HOLD';
      strategy = 'Conservative - Mixed signals';
    }

    return {
      signal,
      confidence: Math.min(1, confidence),
      strategy
    };
  }

  private getBuyStrategy(components: any): string {
    const strategies = [];

    if (components.technical.score > 0.5) {
      strategies.push('Technical Breakout');
    }
    if (components.ml.confidence > 0.7) {
      strategies.push('AI High Confidence');
    }
    if (components.sentiment.score > 0.3) {
      strategies.push('Positive Sentiment');
    }
    if (components.onchain.sentiment.score > 0.3) {
      strategies.push('On-Chain Bullish');
    }

    return strategies.length > 0 ? strategies.join(' + ') : 'Multi-Factor Buy';
  }

  private getSellStrategy(components: any): string {
    const strategies = [];

    if (components.technical.score < -0.5) {
      strategies.push('Technical Breakdown');
    }
    if (components.ml.confidence > 0.7 && components.ml.signal === 'SELL') {
      strategies.push('AI High Confidence');
    }
    if (components.sentiment.score < -0.3) {
      strategies.push('Negative Sentiment');
    }
    if (components.onchain.sentiment.score < -0.3) {
      strategies.push('On-Chain Bearish');
    }

    return strategies.length > 0 ? strategies.join(' + ') : 'Multi-Factor Sell';
  }

  private calculatePriceTargets(
    symbol: string,
    priceData: number[],
    signal: 'BUY' | 'SELL' | 'HOLD',
    technicalIndicators: any
  ): { target: number; stopLoss: number } {
    const currentPrice = priceData[priceData.length - 1];
    const volatility = this.calculateVolatility(priceData);
    
    let target: number;
    let stopLoss: number;

    if (signal === 'BUY') {
      // Target: Upper Bollinger Band or 2x volatility
      target = Math.max(
        technicalIndicators.bollinger.upper,
        currentPrice * (1 + volatility * 2)
      );
      
      // Stop loss: Lower Bollinger Band or 1x volatility
      stopLoss = Math.min(
        technicalIndicators.bollinger.lower,
        currentPrice * (1 - volatility)
      );
    } else if (signal === 'SELL') {
      // Target: Lower Bollinger Band or 2x volatility down
      target = Math.min(
        technicalIndicators.bollinger.lower,
        currentPrice * (1 - volatility * 2)
      );
      
      // Stop loss: Upper Bollinger Band or 1x volatility up
      stopLoss = Math.max(
        technicalIndicators.bollinger.upper,
        currentPrice * (1 + volatility)
      );
    } else {
      target = currentPrice;
      stopLoss = currentPrice * 0.95; // 5% stop loss for holds
    }

    return { target, stopLoss };
  }

  private calculateVolatility(priceData: number[]): number {
    const returns = [];
    for (let i = 1; i < priceData.length; i++) {
      returns.push((priceData[i] - priceData[i - 1]) / priceData[i - 1]);
    }
    
    const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / returns.length;
    
    return Math.sqrt(variance);
  }

  private determineTimeframe(technicalIndicators: any, mlPrediction: any): string {
    // Base timeframe on signal strength and volatility
    if (mlPrediction.confidence > 0.8) {
      return 'Short-term (1-4 hours)';
    } else if (mlPrediction.confidence > 0.6) {
      return 'Medium-term (4-24 hours)';
    } else {
      return 'Long-term (1-7 days)';
    }
  }

  private generateReasoning(
    combinedSignal: any,
    technicalIndicators: any,
    mlPrediction: any,
    sentimentSignal: any,
    onChainSentiment: any,
    riskMetrics: any
  ): string {
    const reasons = [];

    // Technical reasoning
    if (technicalIndicators.rsi < 30) {
      reasons.push('RSI indicates oversold conditions');
    } else if (technicalIndicators.rsi > 70) {
      reasons.push('RSI indicates overbought conditions');
    }

    if (technicalIndicators.macd.macd > technicalIndicators.macd.signal) {
      reasons.push('MACD shows bullish crossover');
    } else {
      reasons.push('MACD shows bearish crossover');
    }

    // ML reasoning
    if (mlPrediction.confidence > 0.7) {
      reasons.push(`AI models show ${mlPrediction.confidence.toFixed(1)}% confidence in ${mlPrediction.signal} signal`);
    }

    // Sentiment reasoning
    if (Math.abs(sentimentSignal.score) > 0.3) {
      reasons.push(`Market sentiment is ${sentimentSignal.overallSentiment.toLowerCase()} (${(sentimentSignal.score * 100).toFixed(1)}%)`);
    }

    // On-chain reasoning
    if (onChainSentiment.confidence > 0.5) {
      reasons.push(`On-chain analysis shows ${onChainSentiment.sentiment.toLowerCase()} signals`);
    }

    // Risk reasoning
    if (riskMetrics.riskScore > 70) {
      reasons.push('High risk environment detected');
    } else if (riskMetrics.riskScore < 30) {
      reasons.push('Low risk environment supports position');
    }

    return reasons.length > 0 ? reasons.join('. ') + '.' : 'Mixed signals from multiple indicators.';
  }

  private async storeSignal(signal: TradingSignal): Promise<void> {
    try {
      await storage.createAlert({
        userId: 'system',
        type: signal.signal === 'BUY' ? 'success' : signal.signal === 'SELL' ? 'danger' : 'info',
        title: `Trading Signal: ${signal.symbol}`,
        message: `${signal.signal} signal with ${(signal.confidence * 100).toFixed(1)}% confidence. Strategy: ${signal.strategy}`
      });
    } catch (error) {
      console.error('Error storing signal:', error);
    }
  }

  private getDefaultSignal(symbol: string): TradingSignal {
    return {
      symbol,
      signal: 'HOLD',
      confidence: 0,
      strategy: 'Error - Analysis Failed',
      reasoning: 'Unable to generate signal due to analysis error',
      technicalScore: 0,
      fundamentalScore: 0,
      sentimentScore: 0,
      riskScore: 100,
      priceTarget: 0,
      stopLoss: 0,
      timeframe: 'Unknown',
      timestamp: new Date(),
      components: {
        technical: null,
        ml: null,
        sentiment: null,
        onchain: null,
        arbitrage: null,
        risk: null
      }
    };
  }

  // Advanced signal generation methods
  async generatePortfolioSignals(symbols: string[]): Promise<TradingSignal[]> {
    const signals = [];
    
    for (const symbol of symbols) {
      try {
        const marketData = await storage.getMarketData(symbol);
        if (marketData.length > 0) {
          const prices = marketData.map(m => parseFloat(m.price));
          const volumes = marketData.map(m => parseFloat(m.volume24h));
          
          const signal = await this.generateSignal(symbol, prices, volumes);
          signals.push(signal);
        }
      } catch (error) {
        console.error(`Error generating signal for ${symbol}:`, error);
      }
    }
    
    return signals;
  }

  async generateCorrelationAdjustedSignals(symbols: string[]): Promise<TradingSignal[]> {
    const signals = await this.generatePortfolioSignals(symbols);
    
    // Calculate correlation matrix
    const correlationMatrix = await this.riskManagement.calculateCorrelationMatrix(symbols);
    
    // Adjust signals based on correlation
    for (let i = 0; i < signals.length; i++) {
      let correlationAdjustment = 1;
      
      for (let j = 0; j < signals.length; j++) {
        if (i !== j && correlationMatrix[i] && correlationMatrix[i][j]) {
          const correlation = correlationMatrix[i][j];
          
          // Reduce confidence for highly correlated assets with same signal
          if (correlation > 0.7 && signals[i].signal === signals[j].signal) {
            correlationAdjustment *= 0.8;
          }
        }
      }
      
      signals[i].confidence *= correlationAdjustment;
      signals[i].reasoning += ` Correlation-adjusted confidence: ${(signals[i].confidence * 100).toFixed(1)}%.`;
    }
    
    return signals;
  }

  async generateRiskAdjustedSignals(userId: string, symbols: string[]): Promise<TradingSignal[]> {
    const signals = await this.generatePortfolioSignals(symbols);
    const portfolio = await storage.getPortfolio(userId);
    const positions = await storage.getPositions(userId);
    
    // Calculate current portfolio exposure
    const currentExposure = positions.reduce((total, position) => {
      return total + (parseFloat(position.size) * parseFloat(position.currentPrice));
    }, 0);
    
    const portfolioValue = parseFloat(portfolio.totalValue);
    const exposureRatio = currentExposure / portfolioValue;
    
    // Adjust signals based on risk
    for (const signal of signals) {
      // Reduce BUY signals if portfolio is overexposed
      if (signal.signal === 'BUY' && exposureRatio > 0.8) {
        signal.confidence *= 0.5;
        signal.reasoning += ' Position size reduced due to portfolio risk.';
      }
      
      // Increase SELL signals if portfolio is overexposed
      if (signal.signal === 'SELL' && exposureRatio > 0.8) {
        signal.confidence *= 1.2;
        signal.reasoning += ' Position reduction recommended due to portfolio risk.';
      }
    }
    
    return signals;
  }
}

export default SignalGenerator;