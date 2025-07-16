import { calculateTechnicalIndicators } from './indicators';

export interface MLFeatures {
  technical: number[];
  sentiment: number[];
  volume: number[];
  macro: number[];
  microstructure: number[];
}

export interface MLPrediction {
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  probability: number;
  features: MLFeatures;
  modelType: string;
}

export class MLTradingModels {
  private models: Map<string, any> = new Map();
  private featureScalers: Map<string, any> = new Map();

  constructor() {
    this.initializeModels();
  }

  private initializeModels() {
    // Random Forest-like ensemble model
    this.models.set('ensemble', {
      trees: this.generateDecisionTrees(50),
      weights: this.generateRandomWeights(50),
    });

    // Neural Network-like model
    this.models.set('neural', {
      layers: [
        { weights: this.generateRandomWeights(20, 10), bias: this.generateRandomWeights(10) },
        { weights: this.generateRandomWeights(10, 5), bias: this.generateRandomWeights(5) },
        { weights: this.generateRandomWeights(5, 3), bias: this.generateRandomWeights(3) },
      ],
    });

    // Support Vector Machine-like model
    this.models.set('svm', {
      supportVectors: this.generateSupportVectors(100),
      alpha: this.generateRandomWeights(100),
      kernel: 'rbf',
    });
  }

  // Feature engineering for ML models
  async extractFeatures(
    symbol: string,
    priceData: number[],
    volumeData: number[],
    sentiment: number = 0
  ): Promise<MLFeatures> {
    const technicalIndicators = calculateTechnicalIndicators(priceData, volumeData);
    
    // Technical features
    const technical = [
      technicalIndicators.rsi / 100,
      technicalIndicators.macd.macd,
      technicalIndicators.macd.signal,
      technicalIndicators.macd.histogram,
      technicalIndicators.bollinger.upper,
      technicalIndicators.bollinger.middle,
      technicalIndicators.bollinger.lower,
      technicalIndicators.sma,
      technicalIndicators.ema,
      technicalIndicators.priceChange,
      technicalIndicators.priceChangePercent / 100,
    ];

    // Sentiment features
    const sentimentFeatures = [
      sentiment,
      sentiment > 0 ? 1 : 0, // Positive sentiment flag
      sentiment < 0 ? 1 : 0, // Negative sentiment flag
      Math.abs(sentiment), // Sentiment magnitude
    ];

    // Volume features
    const volumeFeatures = [
      technicalIndicators.volume,
      this.calculateVolumeMA(volumeData, 5),
      this.calculateVolumeMA(volumeData, 20),
      this.calculateVolumeRatio(volumeData),
      this.calculateVolumeSpike(volumeData),
    ];

    // Macro features (simplified)
    const macroFeatures = [
      this.getBitcoinCorrelation(priceData),
      this.getMarketVolatility(priceData),
      this.getTrendStrength(priceData),
      this.getMomentumOscillator(priceData),
    ];

    // Microstructure features
    const microstructureFeatures = [
      this.calculateSpread(priceData),
      this.calculateImbalance(priceData),
      this.calculateLiquidity(volumeData),
      this.calculateImpact(priceData, volumeData),
    ];

    return {
      technical,
      sentiment: sentimentFeatures,
      volume: volumeFeatures,
      macro: macroFeatures,
      microstructure: microstructureFeatures,
    };
  }

  // Ensemble prediction using multiple models
  async predictSignal(features: MLFeatures): Promise<MLPrediction> {
    const flatFeatures = this.flattenFeatures(features);
    const normalizedFeatures = this.normalizeFeatures(flatFeatures);

    // Get predictions from all models
    const ensemblePred = this.predictEnsemble(normalizedFeatures);
    const neuralPred = this.predictNeural(normalizedFeatures);
    const svmPred = this.predictSVM(normalizedFeatures);

    // Weight the predictions
    const weights = { ensemble: 0.4, neural: 0.35, svm: 0.25 };
    const weightedProbability = 
      ensemblePred * weights.ensemble + 
      neuralPred * weights.neural + 
      svmPred * weights.svm;

    // Convert to signal
    let signal: 'BUY' | 'SELL' | 'HOLD';
    let confidence = Math.abs(weightedProbability - 0.5) * 2;

    if (weightedProbability > 0.6) {
      signal = 'BUY';
    } else if (weightedProbability < 0.4) {
      signal = 'SELL';
    } else {
      signal = 'HOLD';
    }

    return {
      signal,
      confidence,
      probability: weightedProbability,
      features,
      modelType: 'ensemble',
    };
  }

  // Long Short-Term Memory (LSTM) simulation
  async predictLSTM(sequenceData: number[][], lookback: number = 20): Promise<number[]> {
    const predictions = [];
    
    for (let i = lookback; i < sequenceData.length; i++) {
      const sequence = sequenceData.slice(i - lookback, i);
      const prediction = this.simulateLSTMCell(sequence);
      predictions.push(prediction);
    }

    return predictions;
  }

  private simulateLSTMCell(sequence: number[][]): number {
    // Simplified LSTM simulation
    const hiddenState = new Array(10).fill(0);
    const cellState = new Array(10).fill(0);

    for (const input of sequence) {
      // Forget gate
      const forgetGate = this.sigmoid(this.dotProduct(input, this.generateRandomWeights(input.length)));
      
      // Input gate
      const inputGate = this.sigmoid(this.dotProduct(input, this.generateRandomWeights(input.length)));
      const candidateValues = this.tanh(this.dotProduct(input, this.generateRandomWeights(input.length)));
      
      // Update cell state
      for (let i = 0; i < cellState.length; i++) {
        cellState[i] = cellState[i] * forgetGate + candidateValues * inputGate;
      }
      
      // Output gate
      const outputGate = this.sigmoid(this.dotProduct(input, this.generateRandomWeights(input.length)));
      
      // Update hidden state
      for (let i = 0; i < hiddenState.length; i++) {
        hiddenState[i] = outputGate * this.tanh(cellState[i]);
      }
    }

    return this.sigmoid(this.dotProduct(hiddenState, this.generateRandomWeights(hiddenState.length)));
  }

  // Reinforcement Learning Q-Learning simulation
  async trainQLearning(states: number[][], actions: number[], rewards: number[]): Promise<number[][]> {
    const qTable = this.initializeQTable(states.length, 3); // 3 actions: BUY, SELL, HOLD
    const learningRate = 0.1;
    const discountFactor = 0.95;
    const epsilon = 0.1;

    for (let episode = 0; episode < 1000; episode++) {
      for (let i = 0; i < states.length - 1; i++) {
        const state = this.stateToIndex(states[i]);
        const action = actions[i];
        const reward = rewards[i];
        const nextState = this.stateToIndex(states[i + 1]);

        // Q-learning update
        const maxNextQ = Math.max(...qTable[nextState]);
        const currentQ = qTable[state][action];
        const newQ = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);
        
        qTable[state][action] = newQ;
      }
    }

    return qTable;
  }

  // Genetic Algorithm for strategy optimization
  async optimizeStrategy(
    parameters: number[],
    fitnessFunction: (params: number[]) => number,
    generations: number = 100
  ): Promise<number[]> {
    const populationSize = 50;
    let population = this.initializePopulation(populationSize, parameters.length);

    for (let gen = 0; gen < generations; gen++) {
      // Evaluate fitness
      const fitness = population.map(individual => fitnessFunction(individual));
      
      // Selection
      const selected = this.tournamentSelection(population, fitness, populationSize);
      
      // Crossover and mutation
      const newPopulation = [];
      for (let i = 0; i < populationSize; i += 2) {
        const parent1 = selected[i];
        const parent2 = selected[i + 1] || selected[0];
        
        const [child1, child2] = this.crossover(parent1, parent2);
        newPopulation.push(this.mutate(child1), this.mutate(child2));
      }
      
      population = newPopulation;
    }

    // Return best individual
    const finalFitness = population.map(individual => fitnessFunction(individual));
    const bestIndex = finalFitness.indexOf(Math.max(...finalFitness));
    return population[bestIndex];
  }

  // Advanced feature engineering methods
  private calculateVolumeMA(volumes: number[], period: number): number {
    const slice = volumes.slice(-period);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  }

  private calculateVolumeRatio(volumes: number[]): number {
    if (volumes.length < 2) return 1;
    return volumes[volumes.length - 1] / volumes[volumes.length - 2];
  }

  private calculateVolumeSpike(volumes: number[]): number {
    const recent = volumes.slice(-10);
    const average = recent.reduce((a, b) => a + b, 0) / recent.length;
    return volumes[volumes.length - 1] / average;
  }

  private getBitcoinCorrelation(prices: number[]): number {
    // Simplified correlation calculation
    const returns = prices.slice(1).map((price, i) => (price - prices[i]) / prices[i]);
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    return Math.min(1, Math.max(-1, variance)); // Normalized correlation proxy
  }

  private getMarketVolatility(prices: number[]): number {
    const returns = prices.slice(1).map((price, i) => Math.log(price / prices[i]));
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    return Math.sqrt(variance);
  }

  private getTrendStrength(prices: number[]): number {
    const firstHalf = prices.slice(0, Math.floor(prices.length / 2));
    const secondHalf = prices.slice(Math.floor(prices.length / 2));
    
    const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
    
    return (secondAvg - firstAvg) / firstAvg;
  }

  private getMomentumOscillator(prices: number[]): number {
    const period = 14;
    const slice = prices.slice(-period);
    const highest = Math.max(...slice);
    const lowest = Math.min(...slice);
    const current = prices[prices.length - 1];
    
    return (current - lowest) / (highest - lowest);
  }

  private calculateSpread(prices: number[]): number {
    const high = Math.max(...prices.slice(-10));
    const low = Math.min(...prices.slice(-10));
    return (high - low) / ((high + low) / 2);
  }

  private calculateImbalance(prices: number[]): number {
    const upMoves = prices.slice(1).filter((price, i) => price > prices[i]).length;
    const downMoves = prices.slice(1).filter((price, i) => price < prices[i]).length;
    return (upMoves - downMoves) / (upMoves + downMoves);
  }

  private calculateLiquidity(volumes: number[]): number {
    const recent = volumes.slice(-20);
    const average = recent.reduce((a, b) => a + b, 0) / recent.length;
    const variance = recent.reduce((sum, v) => sum + Math.pow(v - average, 2), 0) / recent.length;
    return average / Math.sqrt(variance);
  }

  private calculateImpact(prices: number[], volumes: number[]): number {
    const returns = prices.slice(1).map((price, i) => Math.abs(price - prices[i]) / prices[i]);
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;
    return avgReturn / avgVolume;
  }

  // Utility functions for ML models
  private flattenFeatures(features: MLFeatures): number[] {
    return [
      ...features.technical,
      ...features.sentiment,
      ...features.volume,
      ...features.macro,
      ...features.microstructure,
    ];
  }

  private normalizeFeatures(features: number[]): number[] {
    const mean = features.reduce((a, b) => a + b, 0) / features.length;
    const variance = features.reduce((sum, f) => sum + Math.pow(f - mean, 2), 0) / features.length;
    const stdDev = Math.sqrt(variance);
    
    return features.map(f => stdDev === 0 ? 0 : (f - mean) / stdDev);
  }

  private predictEnsemble(features: number[]): number {
    const ensemble = this.models.get('ensemble');
    let prediction = 0;
    
    for (let i = 0; i < ensemble.trees.length; i++) {
      prediction += this.evaluateTree(ensemble.trees[i], features) * ensemble.weights[i];
    }
    
    return this.sigmoid(prediction);
  }

  private predictNeural(features: number[]): number {
    const neural = this.models.get('neural');
    let activation = features;
    
    for (const layer of neural.layers) {
      activation = this.neuralLayerForward(activation, layer.weights, layer.bias);
    }
    
    return activation[0];
  }

  private predictSVM(features: number[]): number {
    const svm = this.models.get('svm');
    let decision = 0;
    
    for (let i = 0; i < svm.supportVectors.length; i++) {
      const kernel = this.rbfKernel(features, svm.supportVectors[i]);
      decision += svm.alpha[i] * kernel;
    }
    
    return this.sigmoid(decision);
  }

  private evaluateTree(tree: any, features: number[]): number {
    // Simplified decision tree evaluation
    let node = tree;
    while (node.left || node.right) {
      if (features[node.feature] < node.threshold) {
        node = node.left;
      } else {
        node = node.right;
      }
    }
    return node.value;
  }

  private neuralLayerForward(input: number[], weights: number[][], bias: number[]): number[] {
    const output = [];
    for (let i = 0; i < bias.length; i++) {
      let sum = bias[i];
      for (let j = 0; j < input.length; j++) {
        sum += input[j] * weights[j][i];
      }
      output.push(this.sigmoid(sum));
    }
    return output;
  }

  private rbfKernel(x1: number[], x2: number[], gamma: number = 1): number {
    const distance = x1.reduce((sum, val, i) => sum + Math.pow(val - x2[i], 2), 0);
    return Math.exp(-gamma * distance);
  }

  // Helper functions
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  private tanh(x: number): number {
    return Math.tanh(x);
  }

  private dotProduct(a: number[], b: number[]): number {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
  }

  private generateRandomWeights(size: number, size2?: number): number[] | number[][] {
    if (size2) {
      return Array(size).fill(0).map(() => Array(size2).fill(0).map(() => Math.random() - 0.5));
    }
    return Array(size).fill(0).map(() => Math.random() - 0.5);
  }

  private generateDecisionTrees(count: number): any[] {
    return Array(count).fill(0).map(() => this.generateRandomTree(5));
  }

  private generateRandomTree(depth: number): any {
    if (depth === 0) {
      return { value: Math.random() - 0.5 };
    }
    
    return {
      feature: Math.floor(Math.random() * 10),
      threshold: Math.random(),
      left: this.generateRandomTree(depth - 1),
      right: this.generateRandomTree(depth - 1),
    };
  }

  private generateSupportVectors(count: number): number[][] {
    return Array(count).fill(0).map(() => Array(10).fill(0).map(() => Math.random() - 0.5));
  }

  private initializeQTable(states: number, actions: number): number[][] {
    return Array(states).fill(0).map(() => Array(actions).fill(0));
  }

  private stateToIndex(state: number[]): number {
    return Math.floor(state.reduce((sum, val) => sum + val, 0) * 100) % 1000;
  }

  private initializePopulation(size: number, dimensions: number): number[][] {
    return Array(size).fill(0).map(() => Array(dimensions).fill(0).map(() => Math.random()));
  }

  private tournamentSelection(population: number[][], fitness: number[], size: number): number[][] {
    const selected = [];
    for (let i = 0; i < size; i++) {
      const tournament = Array(3).fill(0).map(() => Math.floor(Math.random() * population.length));
      const winner = tournament.reduce((best, current) => fitness[current] > fitness[best] ? current : best);
      selected.push([...population[winner]]);
    }
    return selected;
  }

  private crossover(parent1: number[], parent2: number[]): [number[], number[]] {
    const crossoverPoint = Math.floor(Math.random() * parent1.length);
    const child1 = [...parent1.slice(0, crossoverPoint), ...parent2.slice(crossoverPoint)];
    const child2 = [...parent2.slice(0, crossoverPoint), ...parent1.slice(crossoverPoint)];
    return [child1, child2];
  }

  private mutate(individual: number[], mutationRate: number = 0.1): number[] {
    return individual.map(gene => Math.random() < mutationRate ? Math.random() : gene);
  }
}

export default MLTradingModels;