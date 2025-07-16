import { db } from './db';
import { eq } from 'drizzle-orm';
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
  type UpsertUser,
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
  type InsertAlert,
} from '@shared/schema';

export interface IStorage {
  // User operations for Replit Auth
  getUser(id: string): Promise<User | undefined>;
  upsertUser(user: UpsertUser): Promise<User>;
  
  // Trading operations
  getPortfolio(userId: string): Promise<Portfolio | undefined>;
  updatePortfolio(userId: string, data: Partial<Portfolio>): Promise<Portfolio>;
  
  getPositions(userId: string): Promise<Position[]>;
  createPosition(position: InsertPosition): Promise<Position>;
  updatePosition(id: number, data: Partial<Position>): Promise<Position>;
  closePosition(id: number): Promise<Position>;
  
  getTrades(userId: string, limit?: number): Promise<Trade[]>;
  createTrade(trade: InsertTrade): Promise<Trade>;
  
  getBotStatus(userId: string): Promise<BotStatus | undefined>;
  updateBotStatus(userId: string, data: Partial<BotStatus>): Promise<BotStatus>;
  
  getMarketData(symbol?: string): Promise<MarketData[]>;
  updateMarketData(symbol: string, data: Partial<MarketData>): Promise<MarketData>;
  
  getRiskMetrics(userId: string): Promise<RiskMetrics | undefined>;
  updateRiskMetrics(userId: string, data: Partial<RiskMetrics>): Promise<RiskMetrics>;
  
  getAlerts(userId: string): Promise<Alert[]>;
  createAlert(alert: InsertAlert): Promise<Alert>;
  markAlertRead(id: number): Promise<Alert>;
}

export class DatabaseStorage implements IStorage {
  // User operations for Replit Auth
  async getUser(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user;
  }

  async upsertUser(userData: UpsertUser): Promise<User> {
    const [user] = await db
      .insert(users)
      .values(userData)
      .onConflictDoUpdate({
        target: users.id,
        set: {
          ...userData,
          updatedAt: new Date(),
        },
      })
      .returning();
    return user;
  }

  // Trading operations
  async getPortfolio(userId: string): Promise<Portfolio | undefined> {
    const [portfolioData] = await db
      .select()
      .from(portfolio)
      .where(eq(portfolio.userId, userId));
    return portfolioData;
  }

  async updatePortfolio(userId: string, data: Partial<Portfolio>): Promise<Portfolio> {
    const [updated] = await db
      .update(portfolio)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(portfolio.userId, userId))
      .returning();
    
    if (!updated) {
      // Create new portfolio if it doesn't exist
      const [newPortfolio] = await db
        .insert(portfolio)
        .values({
          userId,
          totalValue: "0",
          dayChange: "0",
          dayChangePercent: "0",
          ...data,
        })
        .returning();
      return newPortfolio;
    }
    
    return updated;
  }

  async getPositions(userId: string): Promise<Position[]> {
    return await db
      .select()
      .from(positions)
      .where(eq(positions.userId, userId));
  }

  async createPosition(position: InsertPosition): Promise<Position> {
    const [newPosition] = await db
      .insert(positions)
      .values(position)
      .returning();
    return newPosition;
  }

  async updatePosition(id: number, data: Partial<Position>): Promise<Position> {
    const [updated] = await db
      .update(positions)
      .set(data)
      .where(eq(positions.id, id))
      .returning();
    return updated;
  }

  async closePosition(id: number): Promise<Position> {
    const [closed] = await db
      .update(positions)
      .set({ isActive: false })
      .where(eq(positions.id, id))
      .returning();
    return closed;
  }

  async getTrades(userId: string, limit = 50): Promise<Trade[]> {
    return await db
      .select()
      .from(trades)
      .where(eq(trades.userId, userId))
      .limit(limit)
      .orderBy(trades.executedAt);
  }

  async createTrade(trade: InsertTrade): Promise<Trade> {
    const [newTrade] = await db
      .insert(trades)
      .values(trade)
      .returning();
    return newTrade;
  }

  async getBotStatus(userId: string): Promise<BotStatus | undefined> {
    const [status] = await db
      .select()
      .from(botStatus)
      .where(eq(botStatus.userId, userId));
    return status;
  }

  async updateBotStatus(userId: string, data: Partial<BotStatus>): Promise<BotStatus> {
    const [updated] = await db
      .update(botStatus)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(botStatus.userId, userId))
      .returning();
    
    if (!updated) {
      // Create new bot status if it doesn't exist
      const [newStatus] = await db
        .insert(botStatus)
        .values({
          userId,
          currentStrategy: "Technical Analysis",
          confidence: "75",
          ...data,
        })
        .returning();
      return newStatus;
    }
    
    return updated;
  }

  async getMarketData(symbol?: string): Promise<MarketData[]> {
    if (symbol) {
      return await db
        .select()
        .from(marketData)
        .where(eq(marketData.symbol, symbol));
    }
    return await db.select().from(marketData);
  }

  async updateMarketData(symbol: string, data: Partial<MarketData>): Promise<MarketData> {
    const [updated] = await db
      .update(marketData)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(marketData.symbol, symbol))
      .returning();
    
    if (!updated) {
      // Create new market data if it doesn't exist
      const [newData] = await db
        .insert(marketData)
        .values({
          symbol,
          price: "0",
          change24h: "0",
          changePercent24h: "0",
          volume24h: "0",
          high24h: "0",
          low24h: "0",
          exchange: "binance",
          ...data,
        })
        .returning();
      return newData;
    }
    
    return updated;
  }

  async getRiskMetrics(userId: string): Promise<RiskMetrics | undefined> {
    const [metrics] = await db
      .select()
      .from(riskMetrics)
      .where(eq(riskMetrics.userId, userId));
    return metrics;
  }

  async updateRiskMetrics(userId: string, data: Partial<RiskMetrics>): Promise<RiskMetrics> {
    const [updated] = await db
      .update(riskMetrics)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(riskMetrics.userId, userId))
      .returning();
    
    if (!updated) {
      // Create new risk metrics if it doesn't exist
      const [newMetrics] = await db
        .insert(riskMetrics)
        .values({
          userId,
          portfolioHeat: "0",
          maxDrawdown: "0",
          sharpeRatio: "0",
          var95: "0",
          betaBtc: "0",
          riskScore: "0",
          ...data,
        })
        .returning();
      return newMetrics;
    }
    
    return updated;
  }

  async getAlerts(userId: string): Promise<Alert[]> {
    return await db
      .select()
      .from(alerts)
      .where(eq(alerts.userId, userId))
      .orderBy(alerts.createdAt);
  }

  async createAlert(alert: InsertAlert): Promise<Alert> {
    const [newAlert] = await db
      .insert(alerts)
      .values(alert)
      .returning();
    return newAlert;
  }

  async markAlertRead(id: number): Promise<Alert> {
    const [updated] = await db
      .update(alerts)
      .set({ isRead: true })
      .where(eq(alerts.id, id))
      .returning();
    return updated;
  }
}

export const storage = new DatabaseStorage();