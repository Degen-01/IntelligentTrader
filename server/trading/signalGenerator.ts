import { sentiment } from 'sentiment';
import { calculateTechnicalIndicators } from './indicators';

export interface TradingSignal {
  symbol: string;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  strategy: string;
  indicators: {
    rsi?: number;
    macd?: { macd: number; signal: number; histogram: number };
    bollinger?: { upper: number; middle: number; lower: number };
    volume?: number;
    sentiment?: number;
  };
  reasoning: string;
}

export class SignalGenerator {
  private sentimentAnalyzer: any;

  constructor() {
    this.sentimentAnalyzer = sentiment;
  }

  async generateSignal(
    symbol: string,
    priceData: number[],
    volumeData: number[],
    newsData?: string[]
  ): Promise<TradingSignal> {
    const technicalIndicators = calculateTechnicalIndicators(priceData, volumeData);
    const sentimentScore = await this.analyzeSentiment(newsData || []);
    
    const signals = {
      technical: this.generateTechnicalSignal(technicalIndicators),
      sentiment: this.generateSentimentSignal(sentimentScore),
      volume: this.generateVolumeSignal(volumeData, technicalIndicators.volume),
    };

    // Combine signals with weighted approach
    const finalSignal = this.combineSignals(signals);
    
    return {
      symbol,
      signal: finalSignal.signal,
      confidence: finalSignal.confidence,
      strategy: 'Multi-factor Analysis',
      indicators: {
        rsi: technicalIndicators.rsi,
        macd: technicalIndicators.macd,
        bollinger: technicalIndicators.bollinger,
        volume: technicalIndicators.volume,
        sentiment: sentimentScore,
      },
      reasoning: this.generateReasoning(signals, technicalIndicators, sentimentScore),
    };
  }

  private generateTechnicalSignal(indicators: any): { signal: 'BUY' | 'SELL' | 'HOLD'; confidence: number } {
    let score = 0;
    let signals = 0;

    // RSI signals
    if (indicators.rsi) {
      if (indicators.rsi < 30) {
        score += 1; // Oversold - buy signal
      } else if (indicators.rsi > 70) {
        score -= 1; // Overbought - sell signal
      }
      signals++;
    }

    // MACD signals
    if (indicators.macd) {
      if (indicators.macd.histogram > 0 && indicators.macd.macd > indicators.macd.signal) {
        score += 1; // Bullish momentum
      } else if (indicators.macd.histogram < 0 && indicators.macd.macd < indicators.macd.signal) {
        score -= 1; // Bearish momentum
      }
      signals++;
    }

    // Bollinger Bands signals
    if (indicators.bollinger && indicators.currentPrice) {
      const currentPrice = indicators.currentPrice;
      if (currentPrice <= indicators.bollinger.lower) {
        score += 1; // Price at lower band - buy signal
      } else if (currentPrice >= indicators.bollinger.upper) {
        score -= 1; // Price at upper band - sell signal
      }
      signals++;
    }

    const normalizedScore = signals > 0 ? score / signals : 0;
    const confidence = Math.abs(normalizedScore);

    if (normalizedScore > 0.3) {
      return { signal: 'BUY', confidence };
    } else if (normalizedScore < -0.3) {
      return { signal: 'SELL', confidence };
    } else {
      return { signal: 'HOLD', confidence };
    }
  }

  private generateSentimentSignal(sentimentScore: number): { signal: 'BUY' | 'SELL' | 'HOLD'; confidence: number } {
    const normalizedScore = Math.max(-1, Math.min(1, sentimentScore));
    const confidence = Math.abs(normalizedScore);

    if (normalizedScore > 0.3) {
      return { signal: 'BUY', confidence };
    } else if (normalizedScore < -0.3) {
      return { signal: 'SELL', confidence };
    } else {
      return { signal: 'HOLD', confidence };
    }
  }

  private generateVolumeSignal(volumeData: number[], avgVolume: number): { signal: 'BUY' | 'SELL' | 'HOLD'; confidence: number } {
    const recentVolume = volumeData.slice(-5).reduce((a, b) => a + b, 0) / 5;
    const volumeRatio = recentVolume / avgVolume;
    
    if (volumeRatio > 1.5) {
      return { signal: 'BUY', confidence: Math.min(0.8, volumeRatio - 1) };
    } else if (volumeRatio < 0.5) {
      return { signal: 'SELL', confidence: Math.min(0.8, 1 - volumeRatio) };
    } else {
      return { signal: 'HOLD', confidence: 0.1 };
    }
  }

  private combineSignals(signals: any): { signal: 'BUY' | 'SELL' | 'HOLD'; confidence: number } {
    const weights = {
      technical: 0.5,
      sentiment: 0.3,
      volume: 0.2,
    };

    let totalScore = 0;
    let totalWeight = 0;

    Object.entries(signals).forEach(([key, signal]: [string, any]) => {
      const weight = weights[key as keyof typeof weights];
      let signalScore = 0;

      if (signal.signal === 'BUY') {
        signalScore = signal.confidence;
      } else if (signal.signal === 'SELL') {
        signalScore = -signal.confidence;
      }

      totalScore += signalScore * weight;
      totalWeight += weight;
    });

    const finalScore = totalScore / totalWeight;
    const confidence = Math.abs(finalScore);

    if (finalScore > 0.3) {
      return { signal: 'BUY', confidence };
    } else if (finalScore < -0.3) {
      return { signal: 'SELL', confidence };
    } else {
      return { signal: 'HOLD', confidence };
    }
  }

  private async analyzeSentiment(newsData: string[]): Promise<number> {
    if (newsData.length === 0) return 0;

    let totalSentiment = 0;
    let totalWords = 0;

    newsData.forEach(text => {
      const result = this.sentimentAnalyzer(text);
      totalSentiment += result.score;
      totalWords += result.words.length;
    });

    // Normalize sentiment score
    return totalWords > 0 ? totalSentiment / totalWords : 0;
  }

  private generateReasoning(signals: any, indicators: any, sentimentScore: number): string {
    const reasons = [];

    // Technical analysis reasoning
    if (indicators.rsi) {
      if (indicators.rsi < 30) {
        reasons.push(`RSI at ${indicators.rsi.toFixed(2)} indicates oversold conditions`);
      } else if (indicators.rsi > 70) {
        reasons.push(`RSI at ${indicators.rsi.toFixed(2)} indicates overbought conditions`);
      }
    }

    if (indicators.macd) {
      if (indicators.macd.histogram > 0) {
        reasons.push('MACD histogram shows bullish momentum');
      } else if (indicators.macd.histogram < 0) {
        reasons.push('MACD histogram shows bearish momentum');
      }
    }

    // Sentiment reasoning
    if (sentimentScore > 0.3) {
      reasons.push('Positive market sentiment detected');
    } else if (sentimentScore < -0.3) {
      reasons.push('Negative market sentiment detected');
    }

    // Volume reasoning
    if (signals.volume.signal === 'BUY') {
      reasons.push('Unusual volume activity suggests buying interest');
    } else if (signals.volume.signal === 'SELL') {
      reasons.push('Low volume suggests lack of interest');
    }

    return reasons.length > 0 ? reasons.join('; ') : 'No clear signals detected';
  }

  // Machine learning-based signal generation (simplified)
  async generateMLSignal(
    symbol: string,
    features: number[]
  ): Promise<TradingSignal> {
    // This is a simplified ML model - in production, you'd use a trained model
    const prediction = this.simpleMLPredict(features);
    
    return {
      symbol,
      signal: prediction.signal,
      confidence: prediction.confidence,
      strategy: 'Machine Learning',
      indicators: {
        rsi: features[0],
        volume: features[1],
        sentiment: features[2],
      },
      reasoning: `ML model prediction based on ${features.length} features`,
    };
  }

  private simpleMLPredict(features: number[]): { signal: 'BUY' | 'SELL' | 'HOLD'; confidence: number } {
    // Simple linear combination (in production, use a real ML model)
    const weights = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1];
    const score = features.reduce((sum, feature, index) => {
      return sum + (feature * (weights[index] || 0));
    }, 0);

    const normalizedScore = Math.tanh(score / 100); // Normalize to [-1, 1]
    const confidence = Math.abs(normalizedScore);

    if (normalizedScore > 0.3) {
      return { signal: 'BUY', confidence };
    } else if (normalizedScore < -0.3) {
      return { signal: 'SELL', confidence };
    } else {
      return { signal: 'HOLD', confidence };
    }
  }
}

export default SignalGenerator;