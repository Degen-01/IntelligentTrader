import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { Portfolio, BotStatus, Trade, MarketData, RiskMetrics, Alert } from "@shared/schema";

export function useTradingData(userId: number) {
  const queryClient = useQueryClient();

  const { data: portfolio, isLoading: portfolioLoading } = useQuery<Portfolio>({
    queryKey: ["/api/portfolio", userId.toString()],
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  const { data: botStatus, isLoading: botStatusLoading } = useQuery<BotStatus>({
    queryKey: ["/api/bot-status", userId.toString()],
    refetchInterval: 5000, // Refetch every 5 seconds
  });

  const { data: trades, isLoading: tradesLoading } = useQuery<Trade[]>({
    queryKey: ["/api/trades", userId.toString()],
    refetchInterval: 10000, // Refetch every 10 seconds
  });

  const { data: marketData, isLoading: marketDataLoading } = useQuery<MarketData[]>({
    queryKey: ["/api/market-data"],
    refetchInterval: 3000, // Refetch every 3 seconds
  });

  const { data: riskMetrics, isLoading: riskMetricsLoading } = useQuery<RiskMetrics>({
    queryKey: ["/api/risk-metrics", userId.toString()],
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  const { data: alerts, isLoading: alertsLoading } = useQuery<Alert[]>({
    queryKey: ["/api/alerts", userId.toString()],
    refetchInterval: 15000, // Refetch every 15 seconds
  });

  return {
    portfolio,
    botStatus,
    trades,
    marketData,
    riskMetrics,
    alerts,
    isLoading: portfolioLoading || botStatusLoading || tradesLoading || marketDataLoading || riskMetricsLoading || alertsLoading,
    queryClient
  };
}
