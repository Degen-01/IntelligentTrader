// Technical indicators for trading signals
export function calculateTechnicalIndicators(prices: number[], volumes: number[] = []) {
  if (prices.length < 14) {
    throw new Error('Not enough price data for technical indicators');
  }

  return {
    rsi: calculateRSI(prices, 14),
    macd: calculateMACD(prices),
    bollinger: calculateBollingerBands(prices, 20),
    sma: calculateSMA(prices, 20),
    ema: calculateEMA(prices, 20),
    volume: volumes.length > 0 ? calculateVolumeAverage(volumes) : 0,
    currentPrice: prices[prices.length - 1],
    priceChange: prices[prices.length - 1] - prices[prices.length - 2],
    priceChangePercent: ((prices[prices.length - 1] - prices[prices.length - 2]) / prices[prices.length - 2]) * 100,
  };
}

// RSI (Relative Strength Index)
export function calculateRSI(prices: number[], period: number = 14): number {
  if (prices.length < period + 1) return 50;

  const gains = [];
  const losses = [];

  for (let i = 1; i < prices.length; i++) {
    const change = prices[i] - prices[i - 1];
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? -change : 0);
  }

  const avgGain = calculateSMA(gains.slice(-period), period);
  const avgLoss = calculateSMA(losses.slice(-period), period);

  if (avgLoss === 0) return 100;
  
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

// MACD (Moving Average Convergence Divergence)
export function calculateMACD(prices: number[]): { macd: number; signal: number; histogram: number } {
  const ema12 = calculateEMA(prices, 12);
  const ema26 = calculateEMA(prices, 26);
  const macd = ema12 - ema26;
  
  // Calculate signal line (9-period EMA of MACD)
  const macdValues = [];
  for (let i = 26; i < prices.length; i++) {
    const ema12Val = calculateEMA(prices.slice(0, i + 1), 12);
    const ema26Val = calculateEMA(prices.slice(0, i + 1), 26);
    macdValues.push(ema12Val - ema26Val);
  }
  
  const signal = macdValues.length >= 9 ? calculateEMA(macdValues, 9) : 0;
  const histogram = macd - signal;

  return { macd, signal, histogram };
}

// Bollinger Bands
export function calculateBollingerBands(prices: number[], period: number = 20): { upper: number; middle: number; lower: number } {
  const sma = calculateSMA(prices, period);
  const standardDeviation = calculateStandardDeviation(prices.slice(-period), sma);
  
  return {
    upper: sma + (2 * standardDeviation),
    middle: sma,
    lower: sma - (2 * standardDeviation),
  };
}

// Simple Moving Average
export function calculateSMA(values: number[], period: number): number {
  if (values.length < period) return values[values.length - 1] || 0;
  
  const slice = values.slice(-period);
  return slice.reduce((sum, value) => sum + value, 0) / slice.length;
}

// Exponential Moving Average
export function calculateEMA(values: number[], period: number): number {
  if (values.length === 0) return 0;
  if (values.length < period) return calculateSMA(values, values.length);
  
  const multiplier = 2 / (period + 1);
  let ema = calculateSMA(values.slice(0, period), period);
  
  for (let i = period; i < values.length; i++) {
    ema = (values[i] * multiplier) + (ema * (1 - multiplier));
  }
  
  return ema;
}

// Standard Deviation
export function calculateStandardDeviation(values: number[], mean: number): number {
  const variance = values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / values.length;
  return Math.sqrt(variance);
}

// Volume Average
export function calculateVolumeAverage(volumes: number[], period: number = 20): number {
  return calculateSMA(volumes, period);
}

// Stochastic Oscillator
export function calculateStochastic(highs: number[], lows: number[], closes: number[], period: number = 14): { k: number; d: number } {
  if (highs.length < period || lows.length < period || closes.length < period) {
    return { k: 50, d: 50 };
  }

  const recentHighs = highs.slice(-period);
  const recentLows = lows.slice(-period);
  const currentClose = closes[closes.length - 1];
  
  const highestHigh = Math.max(...recentHighs);
  const lowestLow = Math.min(...recentLows);
  
  const k = ((currentClose - lowestLow) / (highestHigh - lowestLow)) * 100;
  
  // Calculate %D as 3-period moving average of %K
  const kValues = [];
  for (let i = period - 1; i < closes.length; i++) {
    const periodHighs = highs.slice(i - period + 1, i + 1);
    const periodLows = lows.slice(i - period + 1, i + 1);
    const periodHigh = Math.max(...periodHighs);
    const periodLow = Math.min(...periodLows);
    kValues.push(((closes[i] - periodLow) / (periodHigh - periodLow)) * 100);
  }
  
  const d = calculateSMA(kValues.slice(-3), 3);
  
  return { k, d };
}

// Williams %R
export function calculateWilliamsR(highs: number[], lows: number[], closes: number[], period: number = 14): number {
  if (highs.length < period || lows.length < period || closes.length < period) {
    return -50;
  }

  const recentHighs = highs.slice(-period);
  const recentLows = lows.slice(-period);
  const currentClose = closes[closes.length - 1];
  
  const highestHigh = Math.max(...recentHighs);
  const lowestLow = Math.min(...recentLows);
  
  return ((highestHigh - currentClose) / (highestHigh - lowestLow)) * -100;
}

// Average True Range (ATR)
export function calculateATR(highs: number[], lows: number[], closes: number[], period: number = 14): number {
  if (highs.length < 2 || lows.length < 2 || closes.length < 2) {
    return 0;
  }

  const trueRanges = [];
  
  for (let i = 1; i < highs.length; i++) {
    const high = highs[i];
    const low = lows[i];
    const prevClose = closes[i - 1];
    
    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );
    
    trueRanges.push(tr);
  }
  
  return calculateSMA(trueRanges, Math.min(period, trueRanges.length));
}

// Commodity Channel Index (CCI)
export function calculateCCI(highs: number[], lows: number[], closes: number[], period: number = 20): number {
  if (highs.length < period || lows.length < period || closes.length < period) {
    return 0;
  }

  const typicalPrices = [];
  for (let i = 0; i < highs.length; i++) {
    typicalPrices.push((highs[i] + lows[i] + closes[i]) / 3);
  }
  
  const sma = calculateSMA(typicalPrices, period);
  const currentTP = typicalPrices[typicalPrices.length - 1];
  
  // Calculate mean deviation
  const recentTP = typicalPrices.slice(-period);
  const meanDeviation = recentTP.reduce((sum, tp) => sum + Math.abs(tp - sma), 0) / period;
  
  return meanDeviation !== 0 ? (currentTP - sma) / (0.015 * meanDeviation) : 0;
}

// Money Flow Index (MFI)
export function calculateMFI(highs: number[], lows: number[], closes: number[], volumes: number[], period: number = 14): number {
  if (highs.length < period + 1 || volumes.length < period + 1) {
    return 50;
  }

  const typicalPrices = [];
  const rawMoneyFlows = [];
  
  for (let i = 0; i < highs.length; i++) {
    const tp = (highs[i] + lows[i] + closes[i]) / 3;
    typicalPrices.push(tp);
    rawMoneyFlows.push(tp * volumes[i]);
  }
  
  let positiveMoneyFlow = 0;
  let negativeMoneyFlow = 0;
  
  for (let i = 1; i < typicalPrices.length; i++) {
    if (typicalPrices[i] > typicalPrices[i - 1]) {
      positiveMoneyFlow += rawMoneyFlows[i];
    } else if (typicalPrices[i] < typicalPrices[i - 1]) {
      negativeMoneyFlow += rawMoneyFlows[i];
    }
  }
  
  if (negativeMoneyFlow === 0) return 100;
  
  const moneyFlowRatio = positiveMoneyFlow / negativeMoneyFlow;
  return 100 - (100 / (1 + moneyFlowRatio));
}

// Support and Resistance levels
export function calculateSupportResistance(highs: number[], lows: number[], period: number = 20): { support: number; resistance: number } {
  const recentHighs = highs.slice(-period);
  const recentLows = lows.slice(-period);
  
  const resistance = Math.max(...recentHighs);
  const support = Math.min(...recentLows);
  
  return { support, resistance };
}

// Fibonacci retracement levels
export function calculateFibonacci(high: number, low: number): { [key: string]: number } {
  const diff = high - low;
  
  return {
    '0%': high,
    '23.6%': high - (diff * 0.236),
    '38.2%': high - (diff * 0.382),
    '50%': high - (diff * 0.5),
    '61.8%': high - (diff * 0.618),
    '100%': low,
  };
}

// Pivot Points
export function calculatePivotPoints(high: number, low: number, close: number): { [key: string]: number } {
  const pivot = (high + low + close) / 3;
  
  return {
    pivot,
    r1: (2 * pivot) - low,
    r2: pivot + (high - low),
    r3: high + (2 * (pivot - low)),
    s1: (2 * pivot) - high,
    s2: pivot - (high - low),
    s3: low - (2 * (high - pivot)),
  };
}

export default {
  calculateTechnicalIndicators,
  calculateRSI,
  calculateMACD,
  calculateBollingerBands,
  calculateSMA,
  calculateEMA,
  calculateStochastic,
  calculateWilliamsR,
  calculateATR,
  calculateCCI,
  calculateMFI,
  calculateSupportResistance,
  calculateFibonacci,
  calculatePivotPoints,
};