export function calculateTechnicalIndicators(prices: number[], period: number = 14) {
  // Simple Moving Average
  const sma = (data: number[], period: number) => {
    const result: number[] = [];
    for (let i = period - 1; i < data.length; i++) {
      const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      result.push(sum / period);
    }
    return result;
  };

  // Relative Strength Index
  const rsi = (data: number[], period: number = 14) => {
    const changes = data.slice(1).map((price, i) => price - data[i]);
    const gains = changes.map(change => Math.max(change, 0));
    const losses = changes.map(change => Math.max(-change, 0));
    
    let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
    let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
    
    const result: number[] = [];
    
    for (let i = period; i < changes.length; i++) {
      avgGain = (avgGain * (period - 1) + gains[i]) / period;
      avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
      
      const rs = avgGain / avgLoss;
      const rsiValue = 100 - (100 / (1 + rs));
      result.push(rsiValue);
    }
    
    return result;
  };

  // MACD
  const macd = (data: number[]) => {
    const ema12 = exponentialMovingAverage(data, 12);
    const ema26 = exponentialMovingAverage(data, 26);
    
    const macdLine = ema12.map((value, i) => value - ema26[i]);
    const signalLine = exponentialMovingAverage(macdLine, 9);
    const histogram = macdLine.map((value, i) => value - signalLine[i]);
    
    return { macdLine, signalLine, histogram };
  };

  const exponentialMovingAverage = (data: number[], period: number) => {
    const multiplier = 2 / (period + 1);
    const result: number[] = [];
    
    result[0] = data[0];
    
    for (let i = 1; i < data.length; i++) {
      result[i] = (data[i] * multiplier) + (result[i - 1] * (1 - multiplier));
    }
    
    return result;
  };

  return {
    sma: sma(prices, period),
    rsi: rsi(prices, period),
    macd: macd(prices),
    ema: exponentialMovingAverage(prices, period)
  };
}

export function generateTradingSignal(indicators: any) {
  const { rsi, macd } = indicators;
  
  if (!rsi.length || !macd.macdLine.length) return 'HOLD';
  
  const currentRSI = rsi[rsi.length - 1];
  const currentMACD = macd.macdLine[macd.macdLine.length - 1];
  const currentSignal = macd.signalLine[macd.signalLine.length - 1];
  
  // Simple trading logic
  if (currentRSI < 30 && currentMACD > currentSignal) {
    return 'BUY';
  } else if (currentRSI > 70 && currentMACD < currentSignal) {
    return 'SELL';
  }
  
  return 'HOLD';
}

export function calculateRiskMetrics(portfolio: any, trades: any[]) {
  if (!trades.length) return null;
  
  const returns = trades.map(trade => parseFloat(trade.pnl));
  const totalReturn = returns.reduce((sum, ret) => sum + ret, 0);
  const avgReturn = totalReturn / returns.length;
  
  // Calculate volatility (standard deviation)
  const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length;
  const volatility = Math.sqrt(variance);
  
  // Simple Sharpe ratio (assuming risk-free rate of 0)
  const sharpeRatio = volatility > 0 ? avgReturn / volatility : 0;
  
  // Max drawdown
  let peak = 0;
  let maxDrawdown = 0;
  let runningTotal = 0;
  
  for (const ret of returns) {
    runningTotal += ret;
    if (runningTotal > peak) {
      peak = runningTotal;
    }
    const drawdown = (peak - runningTotal) / peak;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
    }
  }
  
  return {
    sharpeRatio: sharpeRatio.toFixed(2),
    maxDrawdown: (-maxDrawdown * 100).toFixed(2),
    volatility: volatility.toFixed(2),
    totalReturn: totalReturn.toFixed(2)
  };
}
