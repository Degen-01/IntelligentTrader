import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertBotStatusSchema, insertTradeSchema, insertAlertSchema } from "@shared/schema";

export async function registerRoutes(app: Express): Promise<Server> {
  // Portfolio endpoints
  app.get("/api/portfolio/:userId", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const portfolio = await storage.getPortfolio(userId);
      if (!portfolio) {
        return res.status(404).json({ message: "Portfolio not found" });
      }
      res.json(portfolio);
    } catch (error) {
      res.status(500).json({ message: "Failed to get portfolio" });
    }
  });

  // Positions endpoints
  app.get("/api/positions/:userId", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const positions = await storage.getPositions(userId);
      res.json(positions);
    } catch (error) {
      res.status(500).json({ message: "Failed to get positions" });
    }
  });

  // Trades endpoints
  app.get("/api/trades/:userId", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const limit = req.query.limit ? parseInt(req.query.limit as string) : undefined;
      const trades = await storage.getTrades(userId, limit);
      res.json(trades);
    } catch (error) {
      res.status(500).json({ message: "Failed to get trades" });
    }
  });

  app.post("/api/trades", async (req, res) => {
    try {
      const trade = insertTradeSchema.parse(req.body);
      const newTrade = await storage.createTrade(trade);
      res.json(newTrade);
    } catch (error) {
      res.status(400).json({ message: "Invalid trade data" });
    }
  });

  // Bot status endpoints
  app.get("/api/bot-status/:userId", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const botStatus = await storage.getBotStatus(userId);
      if (!botStatus) {
        return res.status(404).json({ message: "Bot status not found" });
      }
      res.json(botStatus);
    } catch (error) {
      res.status(500).json({ message: "Failed to get bot status" });
    }
  });

  app.patch("/api/bot-status/:userId", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const updates = req.body;
      const botStatus = await storage.updateBotStatus(userId, updates);
      res.json(botStatus);
    } catch (error) {
      res.status(500).json({ message: "Failed to update bot status" });
    }
  });

  // Market data endpoints
  app.get("/api/market-data", async (req, res) => {
    try {
      const symbol = req.query.symbol as string;
      const marketData = await storage.getMarketData(symbol);
      res.json(marketData);
    } catch (error) {
      res.status(500).json({ message: "Failed to get market data" });
    }
  });

  // Risk metrics endpoints
  app.get("/api/risk-metrics/:userId", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const riskMetrics = await storage.getRiskMetrics(userId);
      if (!riskMetrics) {
        return res.status(404).json({ message: "Risk metrics not found" });
      }
      res.json(riskMetrics);
    } catch (error) {
      res.status(500).json({ message: "Failed to get risk metrics" });
    }
  });

  // Alerts endpoints
  app.get("/api/alerts/:userId", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const alerts = await storage.getAlerts(userId);
      res.json(alerts);
    } catch (error) {
      res.status(500).json({ message: "Failed to get alerts" });
    }
  });

  app.post("/api/alerts", async (req, res) => {
    try {
      const alert = insertAlertSchema.parse(req.body);
      const newAlert = await storage.createAlert(alert);
      res.json(newAlert);
    } catch (error) {
      res.status(400).json({ message: "Invalid alert data" });
    }
  });

  app.patch("/api/alerts/:id/read", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const alert = await storage.markAlertRead(id);
      res.json(alert);
    } catch (error) {
      res.status(500).json({ message: "Failed to mark alert as read" });
    }
  });

  // Trading actions
  app.post("/api/bot/:userId/start", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const botStatus = await storage.updateBotStatus(userId, { isActive: true });
      res.json(botStatus);
    } catch (error) {
      res.status(500).json({ message: "Failed to start bot" });
    }
  });

  app.post("/api/bot/:userId/stop", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const botStatus = await storage.updateBotStatus(userId, { isActive: false });
      res.json(botStatus);
    } catch (error) {
      res.status(500).json({ message: "Failed to stop bot" });
    }
  });

  app.post("/api/bot/:userId/pause", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const botStatus = await storage.updateBotStatus(userId, { isActive: false });
      res.json(botStatus);
    } catch (error) {
      res.status(500).json({ message: "Failed to pause bot" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
