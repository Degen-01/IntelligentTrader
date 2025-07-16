import { pgTable, text, serial, integer, boolean, decimal, timestamp, json } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const portfolio = pgTable("portfolio", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull(),
  totalValue: decimal("total_value", { precision: 20, scale: 8 }).notNull(),
  dayChange: decimal("day_change", { precision: 10, scale: 4 }).notNull(),
  dayChangePercent: decimal("day_change_percent", { precision: 10, scale: 4 }).notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const positions = pgTable("positions", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull(),
  symbol: text("symbol").notNull(),
  side: text("side").notNull(), // 'long' | 'short'
  size: decimal("size", { precision: 20, scale: 8 }).notNull(),
  entryPrice: decimal("entry_price", { precision: 20, scale: 8 }).notNull(),
  currentPrice: decimal("current_price", { precision: 20, scale: 8 }).notNull(),
  pnl: decimal("pnl", { precision: 20, scale: 8 }).notNull(),
  pnlPercent: decimal("pnl_percent", { precision: 10, scale: 4 }).notNull(),
  stopLoss: decimal("stop_loss", { precision: 20, scale: 8 }),
  takeProfit: decimal("take_profit", { precision: 20, scale: 8 }),
  isActive: boolean("is_active").default(true).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const trades = pgTable("trades", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull(),
  symbol: text("symbol").notNull(),
  side: text("side").notNull(), // 'buy' | 'sell'
  amount: decimal("amount", { precision: 20, scale: 8 }).notNull(),
  price: decimal("price", { precision: 20, scale: 8 }).notNull(),
  pnl: decimal("pnl", { precision: 20, scale: 8 }).notNull(),
  strategy: text("strategy").notNull(),
  executedAt: timestamp("executed_at").defaultNow().notNull(),
});

export const botStatus = pgTable("bot_status", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull(),
  isActive: boolean("is_active").default(false).notNull(),
  currentStrategy: text("current_strategy").notNull(),
  confidence: decimal("confidence", { precision: 5, scale: 2 }).notNull(),
  riskTolerance: integer("risk_tolerance").default(6).notNull(),
  positionSizePercent: integer("position_size_percent").default(15).notNull(),
  uptime: integer("uptime").default(0).notNull(), // in minutes
  tradesCount: integer("trades_count").default(0).notNull(),
  successRate: decimal("success_rate", { precision: 5, scale: 2 }).default("0").notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const marketData = pgTable("market_data", {
  id: serial("id").primaryKey(),
  symbol: text("symbol").notNull(),
  price: decimal("price", { precision: 20, scale: 8 }).notNull(),
  change24h: decimal("change_24h", { precision: 10, scale: 4 }).notNull(),
  changePercent24h: decimal("change_percent_24h", { precision: 10, scale: 4 }).notNull(),
  volume24h: decimal("volume_24h", { precision: 20, scale: 8 }).notNull(),
  high24h: decimal("high_24h", { precision: 20, scale: 8 }).notNull(),
  low24h: decimal("low_24h", { precision: 20, scale: 8 }).notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const riskMetrics = pgTable("risk_metrics", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull(),
  portfolioHeat: decimal("portfolio_heat", { precision: 5, scale: 2 }).notNull(),
  maxDrawdown: decimal("max_drawdown", { precision: 10, scale: 4 }).notNull(),
  sharpeRatio: decimal("sharpe_ratio", { precision: 10, scale: 4 }).notNull(),
  var95: decimal("var_95", { precision: 20, scale: 8 }).notNull(),
  betaBtc: decimal("beta_btc", { precision: 10, scale: 4 }).notNull(),
  riskScore: decimal("risk_score", { precision: 3, scale: 1 }).notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const alerts = pgTable("alerts", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull(),
  type: text("type").notNull(), // 'warning' | 'success' | 'info' | 'danger'
  title: text("title").notNull(),
  message: text("message").notNull(),
  isRead: boolean("is_read").default(false).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

// Insert schemas
export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertPositionSchema = createInsertSchema(positions).omit({
  id: true,
  createdAt: true,
});

export const insertTradeSchema = createInsertSchema(trades).omit({
  id: true,
  executedAt: true,
});

export const insertBotStatusSchema = createInsertSchema(botStatus).omit({
  id: true,
  updatedAt: true,
});

export const insertAlertSchema = createInsertSchema(alerts).omit({
  id: true,
  createdAt: true,
});

// Types
export type User = typeof users.$inferSelect;
export type InsertUser = z.infer<typeof insertUserSchema>;

export type Portfolio = typeof portfolio.$inferSelect;
export type Position = typeof positions.$inferSelect;
export type InsertPosition = z.infer<typeof insertPositionSchema>;

export type Trade = typeof trades.$inferSelect;
export type InsertTrade = z.infer<typeof insertTradeSchema>;

export type BotStatus = typeof botStatus.$inferSelect;
export type InsertBotStatus = z.infer<typeof insertBotStatusSchema>;

export type MarketData = typeof marketData.$inferSelect;
export type RiskMetrics = typeof riskMetrics.$inferSelect;

export type Alert = typeof alerts.$inferSelect;
export type InsertAlert = z.infer<typeof insertAlertSchema>;
