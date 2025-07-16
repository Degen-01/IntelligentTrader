import { storage } from '../storage-database';

export interface RiskParameters {
  maxPositionSize: number;
  maxDrawdown: number;
  var95: number;
  var99: number;
  sharpeRatio: number;
  portfolioHeat: number;
  concentrationLimit: number;
  correlationThreshold: number;
  leverageRatio: number;
  liquidityRatio: number;
}

export interface VaRResult {
  var95: number;
  var99: number;
  expectedShortfall: number;
  confidenceInterval: [number, number];
}

export class RiskManagement {
  private userId: string;
  private portfolioValue: number;
  private riskParameters: RiskParameters;

  constructor(userId: string, portfolioValue: number, riskParams: RiskParameters) {
    this.userId = userId;
    this.portfolioValue = portfolioValue;
    this.riskParameters = riskParams;
  }

  // Value at Risk calculation using Monte Carlo simulation
  async calculateVaR(returns: number[], confidence: number = 0.95, simulations: number = 10000): Promise<VaRResult> {
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);

    // Monte Carlo simulation
    const simulatedReturns = [];
    for (let i = 0; i < simulations; i++) {
      const randomReturn = this.normalRandom() * stdDev + mean;
      simulatedReturns.push(randomReturn);
    }

    simulatedReturns.sort((a, b) => a - b);
    
    const var95Index = Math.floor(simulations * 0.05);
    const var99Index = Math.floor(simulations * 0.01);
    
    const var95 = -simulatedReturns[var95Index] * this.portfolioValue;
    const var99 = -simulatedReturns[var99Index] * this.portfolioValue;
    
    // Expected Shortfall (Conditional VaR)
    const tailReturns = simulatedReturns.slice(0, var95Index);
    const expectedShortfall = -tailReturns.reduce((a, b) => a + b, 0) / tailReturns.length * this.portfolioValue;

    return {
      var95,
      var99,
      expectedShortfall,
      confidenceInterval: [var95 * 0.9, var95 * 1.1],
    };
  }

  // Box-Muller transformation for normal distribution
  private normalRandom(): number {
    const u = Math.random();
    const v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Portfolio correlation analysis
  async calculateCorrelationMatrix(symbols: string[]): Promise<number[][]> {
    const returns: { [symbol: string]: number[] } = {};
    
    // Get historical returns for each symbol
    for (const symbol of symbols) {
      const marketData = await storage.getMarketData(symbol);
      if (marketData.length > 0) {
        returns[symbol] = this.calculateReturns(marketData.map(m => parseFloat(m.price)));
      }
    }

    // Calculate correlation matrix
    const matrix: number[][] = [];
    for (let i = 0; i < symbols.length; i++) {
      matrix[i] = [];
      for (let j = 0; j < symbols.length; j++) {
        if (i === j) {
          matrix[i][j] = 1;
        } else {
          const correlation = this.calculateCorrelation(
            returns[symbols[i]] || [],
            returns[symbols[j]] || []
          );
          matrix[i][j] = correlation;
        }
      }
    }

    return matrix;
  }

  private calculateReturns(prices: number[]): number[] {
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }
    return returns;
  }

  private calculateCorrelation(returns1: number[], returns2: number[]): number {
    if (returns1.length !== returns2.length || returns1.length === 0) return 0;

    const mean1 = returns1.reduce((a, b) => a + b, 0) / returns1.length;
    const mean2 = returns2.reduce((a, b) => a + b, 0) / returns2.length;

    let numerator = 0;
    let sumSq1 = 0;
    let sumSq2 = 0;

    for (let i = 0; i < returns1.length; i++) {
      const diff1 = returns1[i] - mean1;
      const diff2 = returns2[i] - mean2;
      numerator += diff1 * diff2;
      sumSq1 += diff1 * diff1;
      sumSq2 += diff2 * diff2;
    }

    const denominator = Math.sqrt(sumSq1 * sumSq2);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  // Sharpe ratio calculation
  calculateSharpeRatio(returns: number[], riskFreeRate: number = 0.02): number {
    const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);
    
    return stdDev === 0 ? 0 : (meanReturn - riskFreeRate) / stdDev;
  }

  // Maximum drawdown calculation
  calculateMaxDrawdown(prices: number[]): number {
    let maxDrawdown = 0;
    let peak = prices[0];

    for (let i = 1; i < prices.length; i++) {
      if (prices[i] > peak) {
        peak = prices[i];
      }
      
      const drawdown = (peak - prices[i]) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }

    return maxDrawdown;
  }

  // Portfolio heat calculation
  calculatePortfolioHeat(positions: any[]): number {
    const activePositions = positions.filter(p => p.isActive);
    const totalExposure = activePositions.reduce((sum, p) => {
      return sum + Math.abs(parseFloat(p.size) * parseFloat(p.currentPrice));
    }, 0);

    return (totalExposure / this.portfolioValue) * 100;
  }

  // Concentration risk analysis
  calculateConcentrationRisk(positions: any[]): { [symbol: string]: number } {
    const concentrations: { [symbol: string]: number } = {};
    
    positions.forEach(position => {
      const exposure = Math.abs(parseFloat(position.size) * parseFloat(position.currentPrice));
      const concentration = (exposure / this.portfolioValue) * 100;
      concentrations[position.symbol] = concentration;
    });

    return concentrations;
  }

  // Liquidity risk assessment
  async assessLiquidityRisk(symbol: string): Promise<number> {
    const marketData = await storage.getMarketData(symbol);
    if (marketData.length === 0) return 1; // High risk if no data

    const latestData = marketData[0];
    const volume = parseFloat(latestData.volume24h);
    const price = parseFloat(latestData.price);
    
    // Calculate liquidity score based on volume and price stability
    const volumeScore = Math.min(volume / 1000000, 1); // Normalize to 1M volume
    const liquidityScore = volumeScore * 0.8 + 0.2; // Base score of 0.2

    return Math.max(0, Math.min(1, liquidityScore));
  }

  // Position sizing based on Kelly Criterion
  calculateOptimalPositionSize(
    winRate: number,
    avgWin: number,
    avgLoss: number,
    confidence: number
  ): number {
    // Kelly Criterion: f = (bp - q) / b
    // where b = odds, p = win probability, q = loss probability
    const odds = avgWin / Math.abs(avgLoss);
    const kellyFraction = (odds * winRate - (1 - winRate)) / odds;
    
    // Apply confidence factor and risk limits
    const adjustedFraction = Math.min(kellyFraction * confidence, 0.25); // Max 25% of portfolio
    
    return Math.max(0, adjustedFraction);
  }

  // Stress testing
  async performStressTest(scenarios: { [symbol: string]: number }[]): Promise<number[]> {
    const positions = await storage.getPositions(this.userId);
    const results = [];

    for (const scenario of scenarios) {
      let portfolioChange = 0;
      
      for (const position of positions) {
        if (position.isActive && scenario[position.symbol] !== undefined) {
          const currentValue = parseFloat(position.size) * parseFloat(position.currentPrice);
          const stressedValue = currentValue * (1 + scenario[position.symbol]);
          portfolioChange += stressedValue - currentValue;
        }
      }
      
      results.push(portfolioChange);
    }

    return results;
  }

  // Risk-adjusted performance metrics
  calculateRiskAdjustedMetrics(returns: number[]): {
    sharpeRatio: number;
    sortinoRatio: number;
    calmarRatio: number;
    maxDrawdown: number;
  } {
    const riskFreeRate = 0.02;
    const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);

    // Sharpe Ratio
    const sharpeRatio = stdDev === 0 ? 0 : (meanReturn - riskFreeRate) / stdDev;

    // Sortino Ratio (downside deviation)
    const downsideReturns = returns.filter(r => r < meanReturn);
    const downsideVariance = downsideReturns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / downsideReturns.length;
    const downsideDeviation = Math.sqrt(downsideVariance);
    const sortinoRatio = downsideDeviation === 0 ? 0 : (meanReturn - riskFreeRate) / downsideDeviation;

    // Calculate prices from returns for max drawdown
    const prices = [100]; // Start with 100
    for (let i = 0; i < returns.length; i++) {
      prices.push(prices[i] * (1 + returns[i]));
    }
    const maxDrawdown = this.calculateMaxDrawdown(prices);

    // Calmar Ratio
    const calmarRatio = maxDrawdown === 0 ? 0 : meanReturn / maxDrawdown;

    return {
      sharpeRatio,
      sortinoRatio,
      calmarRatio,
      maxDrawdown,
    };
  }

  // Real-time risk monitoring
  async monitorRealTimeRisk(): Promise<{
    alerts: string[];
    riskScore: number;
    recommendations: string[];
  }> {
    const positions = await storage.getPositions(this.userId);
    const alerts: string[] = [];
    const recommendations: string[] = [];

    // Portfolio heat check
    const portfolioHeat = this.calculatePortfolioHeat(positions);
    if (portfolioHeat > 80) {
      alerts.push('Portfolio heat exceeds 80%');
      recommendations.push('Consider reducing position sizes');
    }

    // Concentration risk check
    const concentrations = this.calculateConcentrationRisk(positions);
    Object.entries(concentrations).forEach(([symbol, concentration]) => {
      if (concentration > 30) {
        alerts.push(`High concentration in ${symbol}: ${concentration.toFixed(1)}%`);
        recommendations.push(`Diversify exposure in ${symbol}`);
      }
    });

    // Correlation check
    const symbols = [...new Set(positions.map(p => p.symbol))];
    if (symbols.length > 1) {
      const correlationMatrix = await this.calculateCorrelationMatrix(symbols);
      const avgCorrelation = correlationMatrix.flat().reduce((a, b) => a + b, 0) / (symbols.length * symbols.length);
      
      if (avgCorrelation > 0.7) {
        alerts.push('High portfolio correlation detected');
        recommendations.push('Consider adding uncorrelated assets');
      }
    }

    // Calculate overall risk score
    const riskScore = Math.min(100, portfolioHeat + (alerts.length * 10));

    return {
      alerts,
      riskScore,
      recommendations,
    };
  }
}

export default RiskManagement;