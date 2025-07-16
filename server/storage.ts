import { 
  users, 
  portfolio, 
  positions, 
  trades, 
  botStatus, 
  marketData, 
  riskMetrics, 
  alerts,
  type User, 
  type InsertUser,
  type Portfolio,
  type Position,
  type InsertPosition,
  type Trade,
  type InsertTrade,
  type BotStatus,
  type InsertBotStatus,
  type MarketData,
  type RiskMetrics,
  type Alert,
  type InsertAlert
} from "@shared/schema";

export interface IStorage {
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  
  getPortfolio(userId: number): Promise<Portfolio | undefined>;
  updatePortfolio(userId: number, data: Partial<Portfolio>): Promise<Portfolio>;
  
  getPositions(userId: number): Promise<Position[]>;
  createPosition(position: InsertPosition): Promise<Position>;
  updatePosition(id: number, data: Partial<Position>): Promise<Position>;
  closePosition(id: number): Promise<Position>;
  
  getTrades(userId: number, limit?: number): Promise<Trade[]>;
  createTrade(trade: InsertTrade): Promise<Trade>;
  
  getBotStatus(userId: number): Promise<BotStatus | undefined>;
  updateBotStatus(userId: number, data: Partial<BotStatus>): Promise<BotStatus>;
  
  getMarketData(symbol?: string): Promise<MarketData[]>;
  updateMarketData(symbol: string, data: Partial<MarketData>): Promise<MarketData>;
  
  getRiskMetrics(userId: number): Promise<RiskMetrics | undefined>;
  updateRiskMetrics(userId: number, data: Partial<RiskMetrics>): Promise<RiskMetrics>;
  
  getAlerts(userId: number): Promise<Alert[]>;
  createAlert(alert: InsertAlert): Promise<Alert>;
  markAlertRead(id: number): Promise<Alert>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private portfolios: Map<number, Portfolio>;
  private positions: Map<number, Position>;
  private trades: Map<number, Trade>;
  private botStatuses: Map<number, BotStatus>;
  private marketDataMap: Map<string, MarketData>;
  private riskMetricsMap: Map<number, RiskMetrics>;
  private alertsMap: Map<number, Alert>;
  private currentId: number;

  constructor() {
    this.users = new Map();
    this.portfolios = new Map();
    this.positions = new Map();
    this.trades = new Map();
    this.botStatuses = new Map();
    this.marketDataMap = new Map();
    this.riskMetricsMap = new Map();
    this.alertsMap = new Map();
    this.currentId = 1;

    // Initialize with demo data
    this.initializeDemo();
  }

  private initializeDemo() {
    // Create demo user
    const demoUser: User = {
      id: 1,
      username: "demo",
      password: "demo123"
    };
    this.users.set(1, demoUser);

    // Initialize portfolio
    const demoPortfolio: Portfolio = {
      id: 1,
      userId: 1,
      totalValue: "54782.45",
      dayChange: "1247.83",
      dayChangePercent: "2.34",
      updatedAt: new Date()
    };
    this.portfolios.set(1, demoPortfolio);

    // Initialize bot status
    const demoBotStatus: BotStatus = {
      id: 1,
      userId: 1,
      isActive: true,
      currentStrategy: "momentum",
      confidence: "94.2",
      riskTolerance: 6,
      positionSizePercent: 15,
      uptime: 1425, // 23h 45m
      tradesCount: 47,
      successRate: "78.2",
      updatedAt: new Date()
    };
    this.botStatuses.set(1, demoBotStatus);

    // Initialize market data
    const btcData: MarketData = {
      id: 1,
      symbol: "BTC/USDT",
      price: "43247.82",
      change24h: "1034.67",
      changePercent24h: "2.45",
      volume24h: "28467523.45",
      high24h: "43589.12",
      low24h: "42156.78",
      updatedAt: new Date()
    };
    this.marketDataMap.set("BTC/USDT", btcData);

    // Initialize risk metrics
    const demoRiskMetrics: RiskMetrics = {
      id: 1,
      userId: 1,
      portfolioHeat: "60.0",
      maxDrawdown: "-2.34",
      sharpeRatio: "2.45",
      var95: "1234.56",
      betaBtc: "0.87",
      riskScore: "7.2",
      updatedAt: new Date()
    };
    this.riskMetricsMap.set(1, demoRiskMetrics);

    this.currentId = 2;
  }

  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(user => user.username === username);
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.currentId++;
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  async getPortfolio(userId: number): Promise<Portfolio | undefined> {
    return Array.from(this.portfolios.values()).find(p => p.userId === userId);
  }

  async updatePortfolio(userId: number, data: Partial<Portfolio>): Promise<Portfolio> {
    const existing = await this.getPortfolio(userId);
    if (!existing) {
      const newPortfolio: Portfolio = {
        id: this.currentId++,
        userId,
        totalValue: "0",
        dayChange: "0",
        dayChangePercent: "0",
        updatedAt: new Date(),
        ...data
      };
      this.portfolios.set(newPortfolio.id, newPortfolio);
      return newPortfolio;
    }
    
    const updated = { ...existing, ...data, updatedAt: new Date() };
    this.portfolios.set(updated.id, updated);
    return updated;
  }

  async getPositions(userId: number): Promise<Position[]> {
    return Array.from(this.positions.values()).filter(p => p.userId === userId);
  }

  async createPosition(position: InsertPosition): Promise<Position> {
    const id = this.currentId++;
    const newPosition: Position = { 
      ...position, 
      id, 
      createdAt: new Date()
    };
    this.positions.set(id, newPosition);
    return newPosition;
  }

  async updatePosition(id: number, data: Partial<Position>): Promise<Position> {
    const existing = this.positions.get(id);
    if (!existing) throw new Error("Position not found");
    
    const updated = { ...existing, ...data };
    this.positions.set(id, updated);
    return updated;
  }

  async closePosition(id: number): Promise<Position> {
    return this.updatePosition(id, { isActive: false });
  }

  async getTrades(userId: number, limit = 50): Promise<Trade[]> {
    const userTrades = Array.from(this.trades.values())
      .filter(t => t.userId === userId)
      .sort((a, b) => b.executedAt.getTime() - a.executedAt.getTime());
    
    return userTrades.slice(0, limit);
  }

  async createTrade(trade: InsertTrade): Promise<Trade> {
    const id = this.currentId++;
    const newTrade: Trade = { 
      ...trade, 
      id, 
      executedAt: new Date()
    };
    this.trades.set(id, newTrade);
    return newTrade;
  }

  async getBotStatus(userId: number): Promise<BotStatus | undefined> {
    return Array.from(this.botStatuses.values()).find(b => b.userId === userId);
  }

  async updateBotStatus(userId: number, data: Partial<BotStatus>): Promise<BotStatus> {
    const existing = await this.getBotStatus(userId);
    if (!existing) {
      const newStatus: BotStatus = {
        id: this.currentId++,
        userId,
        isActive: false,
        currentStrategy: "momentum",
        confidence: "0",
        riskTolerance: 6,
        positionSizePercent: 15,
        uptime: 0,
        tradesCount: 0,
        successRate: "0",
        updatedAt: new Date(),
        ...data
      };
      this.botStatuses.set(newStatus.id, newStatus);
      return newStatus;
    }
    
    const updated = { ...existing, ...data, updatedAt: new Date() };
    this.botStatuses.set(updated.id, updated);
    return updated;
  }

  async getMarketData(symbol?: string): Promise<MarketData[]> {
    if (symbol) {
      const data = this.marketDataMap.get(symbol);
      return data ? [data] : [];
    }
    return Array.from(this.marketDataMap.values());
  }

  async updateMarketData(symbol: string, data: Partial<MarketData>): Promise<MarketData> {
    const existing = this.marketDataMap.get(symbol);
    if (!existing) {
      const newData: MarketData = {
        id: this.currentId++,
        symbol,
        price: "0",
        change24h: "0",
        changePercent24h: "0",
        volume24h: "0",
        high24h: "0",
        low24h: "0",
        updatedAt: new Date(),
        ...data
      };
      this.marketDataMap.set(symbol, newData);
      return newData;
    }
    
    const updated = { ...existing, ...data, updatedAt: new Date() };
    this.marketDataMap.set(symbol, updated);
    return updated;
  }

  async getRiskMetrics(userId: number): Promise<RiskMetrics | undefined> {
    return this.riskMetricsMap.get(userId);
  }

  async updateRiskMetrics(userId: number, data: Partial<RiskMetrics>): Promise<RiskMetrics> {
    const existing = this.riskMetricsMap.get(userId);
    if (!existing) {
      const newMetrics: RiskMetrics = {
        id: this.currentId++,
        userId,
        portfolioHeat: "0",
        maxDrawdown: "0",
        sharpeRatio: "0",
        var95: "0",
        betaBtc: "0",
        riskScore: "0",
        updatedAt: new Date(),
        ...data
      };
      this.riskMetricsMap.set(userId, newMetrics);
      return newMetrics;
    }
    
    const updated = { ...existing, ...data, updatedAt: new Date() };
    this.riskMetricsMap.set(userId, updated);
    return updated;
  }

  async getAlerts(userId: number): Promise<Alert[]> {
    return Array.from(this.alertsMap.values())
      .filter(a => a.userId === userId)
      .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
  }

  async createAlert(alert: InsertAlert): Promise<Alert> {
    const id = this.currentId++;
    const newAlert: Alert = { 
      ...alert, 
      id, 
      createdAt: new Date()
    };
    this.alertsMap.set(id, newAlert);
    return newAlert;
  }

  async markAlertRead(id: number): Promise<Alert> {
    const existing = this.alertsMap.get(id);
    if (!existing) throw new Error("Alert not found");
    
    const updated = { ...existing, isRead: true };
    this.alertsMap.set(id, updated);
    return updated;
  }
}

export const storage = new MemStorage();
