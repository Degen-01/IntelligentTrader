import { pgTable, text, serial, integer, boolean, decimal, timestamp, json, varchar, index } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Session storage table for authentication
export const sessions = pgTable(
  "sessions",
  {
    sid: varchar("sid").primaryKey(),
    sess: json("sess").notNull(),
    expire: timestamp("expire").notNull(),
  },
  (table) => [index("IDX_session_expire").on(table.expire)],
);

// Enhanced user table with authentication and wallet integration
export const users = pgTable("users", {
  id: varchar("id").primaryKey().notNull(),
  email: varchar("email").unique(),
  firstName: varchar("first_name"),
  lastName: varchar("last_name"),
  profileImageUrl: varchar("profile_image_url"),
  username: text("username").unique(),
  password: text("password"),
  walletAddress: text("wallet_address"),
  telegramId: text("telegram_id"),
  telegramUsername: text("telegram_username"),
  apiKeys: json("api_keys").$type<{
    binance?: { apiKey: string; secret: string; };
    coinbase?: { apiKey: string; secret: string; passphrase: string; };
  }>(),
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const portfolio = pgTable("portfolio", {
  id: serial("id").primaryKey(),
  userId: varchar("user_id").notNull(),
  totalValue: decimal("total_value", { precision: 20, scale: 8 }).notNull(),
  dayChange: decimal("day_change", { precision: 10, scale: 4 }).notNull(),
  dayChangePercent: decimal("day_change_percent", { precision: 10, scale: 4 }).notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const positions = pgTable("positions", {
  id: serial("id").primaryKey(),
  userId: varchar("user_id").notNull(),
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
  userId: varchar("user_id").notNull(),
  symbol: text("symbol").notNull(),
  side: text("side").notNull(), // 'buy' | 'sell'
  amount: decimal("amount", { precision: 20, scale: 8 }).notNull(),
  price: decimal("price", { precision: 20, scale: 8 }).notNull(),
  pnl: decimal("pnl", { precision: 20, scale: 8 }).notNull(),
  strategy: text("strategy").notNull(),
  orderType: text("order_type").default("market").notNull(), // 'market' | 'limit' | 'twap' | 'vwap'
  exchange: text("exchange").default("binance").notNull(),
  executedAt: timestamp("executed_at").defaultNow().notNull(),
});

export const botStatus = pgTable("bot_status", {
  id: serial("id").primaryKey(),
  userId: varchar("user_id").notNull(),
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
  exchange: text("exchange").default("binance").notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const riskMetrics = pgTable("risk_metrics", {
  id: serial("id").primaryKey(),
  userId: varchar("user_id").notNull(),
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
  userId: varchar("user_id").notNull(),
  type: text("type").notNull(), // 'warning' | 'success' | 'info' | 'danger'
  title: text("title").notNull(),
  message: text("message").notNull(),
  isRead: boolean("is_read").default(false).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

// New tables for advanced features
export const tradingSignals = pgTable("trading_signals", {
  id: serial("id").primaryKey(),
  symbol: text("symbol").notNull(),
  signal: text("signal").notNull(), // 'BUY' | 'SELL' | 'HOLD'
  confidence: decimal("confidence", { precision: 5, scale: 2 }).notNull(),
  indicators: json("indicators").$type<{
    rsi?: number;
    macd?: { macd: number; signal: number; histogram: number; };
    bollinger?: { upper: number; middle: number; lower: number; };
    volume?: number;
    sentiment?: number;
  }>(),
  strategy: text("strategy").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const newsAnalysis = pgTable("news_analysis", {
  id: serial("id").primaryKey(),
  title: text("title").notNull(),
  content: text("content").notNull(),
  sentiment: decimal("sentiment", { precision: 5, scale: 2 }).notNull(), // -1 to 1
  relevantSymbols: json("relevant_symbols").$type<string[]>(),
  source: text("source").notNull(),
  publishedAt: timestamp("published_at").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const backtestResults = pgTable("backtest_results", {
  id: serial("id").primaryKey(),
  userId: varchar("user_id").notNull(),
  strategy: text("strategy").notNull(),
  symbol: text("symbol").notNull(),
  startDate: timestamp("start_date").notNull(),
  endDate: timestamp("end_date").notNull(),
  initialBalance: decimal("initial_balance", { precision: 20, scale: 8 }).notNull(),
  finalBalance: decimal("final_balance", { precision: 20, scale: 8 }).notNull(),
  totalReturn: decimal("total_return", { precision: 10, scale: 4 }).notNull(),
  sharpeRatio: decimal("sharpe_ratio", { precision: 10, scale: 4 }).notNull(),
  maxDrawdown: decimal("max_drawdown", { precision: 10, scale: 4 }).notNull(),
  totalTrades: integer("total_trades").notNull(),
  winRate: decimal("win_rate", { precision: 5, scale: 2 }).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const onChainData = pgTable("on_chain_data", {
  id: serial("id").primaryKey(),
  symbol: text("symbol").notNull(),
  walletAddress: text("wallet_address"),
  transactionCount: integer("transaction_count").default(0),
  volume24h: decimal("volume_24h", { precision: 20, scale: 8 }).notNull(),
  activeAddresses: integer("active_addresses").default(0),
  largeTransactions: integer("large_transactions").default(0),
  exchangeInflow: decimal("exchange_inflow", { precision: 20, scale: 8 }).default("0"),
  exchangeOutflow: decimal("exchange_outflow", { precision: 20, scale: 8 }).default("0"),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const arbitrageOpportunities = pgTable("arbitrage_opportunities", {
  id: serial("id").primaryKey(),
  symbol: text("symbol").notNull(),
  buyExchange: text("buy_exchange").notNull(),
  sellExchange: text("sell_exchange").notNull(),
  buyPrice: decimal("buy_price", { precision: 20, scale: 8 }).notNull(),
  sellPrice: decimal("sell_price", { precision: 20, scale: 8 }).notNull(),
  spread: decimal("spread", { precision: 10, scale: 4 }).notNull(),
  profitPotential: decimal("profit_potential", { precision: 20, scale: 8 }).notNull(),
  isActive: boolean("is_active").default(true).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const telegramNotifications = pgTable("telegram_notifications", {
  id: serial("id").primaryKey(),
  userId: varchar("user_id").notNull(),
  message: text("message").notNull(),
  type: text("type").notNull(), // 'trade' | 'alert' | 'signal' | 'risk'
  sent: boolean("sent").default(false).notNull(),
  sentAt: timestamp("sent_at"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

// Insert schemas
export const insertUserSchema = createInsertSchema(users).pick({
  id: true,
  email: true,
  firstName: true,
  lastName: true,
  profileImageUrl: true,
  username: true,
  password: true,
  walletAddress: true,
  telegramId: true,
  telegramUsername: true,
  apiKeys: true,
});

export const upsertUserSchema = createInsertSchema(users).pick({
  id: true,
  email: true,
  firstName: true,
  lastName: true,
  profileImageUrl: true,
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

export const insertTradingSignalSchema = createInsertSchema(tradingSignals).omit({
  id: true,
  createdAt: true,
});

export const insertNewsAnalysisSchema = createInsertSchema(newsAnalysis).omit({
  id: true,
  createdAt: true,
});

export const insertBacktestResultSchema = createInsertSchema(backtestResults).omit({
  id: true,
  createdAt: true,
});

export const insertOnChainDataSchema = createInsertSchema(onChainData).omit({
  id: true,
  updatedAt: true,
});

export const insertArbitrageOpportunitySchema = createInsertSchema(arbitrageOpportunities).omit({
  id: true,
  createdAt: true,
});

export const insertTelegramNotificationSchema = createInsertSchema(telegramNotifications).omit({
  id: true,
  createdAt: true,
});

// Types
export type User = typeof users.$inferSelect;
export type InsertUser = z.infer<typeof insertUserSchema>;
export type UpsertUser = z.infer<typeof upsertUserSchema>;

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

export type TradingSignal = typeof tradingSignals.$inferSelect;
export type InsertTradingSignal = z.infer<typeof insertTradingSignalSchema>;

export type NewsAnalysis = typeof newsAnalysis.$inferSelect;
export type InsertNewsAnalysis = z.infer<typeof insertNewsAnalysisSchema>;

export type BacktestResult = typeof backtestResults.$inferSelect;
export type InsertBacktestResult = z.infer<typeof insertBacktestResultSchema>;

export type OnChainData = typeof onChainData.$inferSelect;
export type InsertOnChainData = z.infer<typeof insertOnChainDataSchema>;

export type ArbitrageOpportunity = typeof arbitrageOpportunities.$inferSelect;
export type InsertArbitrageOpportunity = z.infer<typeof insertArbitrageOpportunitySchema>;

export type TelegramNotification = typeof telegramNotifications.$inferSelect;
export type InsertTelegramNotification = z.infer<typeof insertTelegramNotificationSchema>;
