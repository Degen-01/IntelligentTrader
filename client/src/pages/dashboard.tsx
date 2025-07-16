import Sidebar from "@/components/layout/sidebar";
import TopBar from "@/components/layout/topbar";
import StatsOverview from "@/components/dashboard/stats-overview";
import PriceChart from "@/components/dashboard/price-chart";
import OrderBook from "@/components/dashboard/order-book";
import AIStrategyPanel from "@/components/dashboard/ai-strategy-panel";
import RecentTrades from "@/components/dashboard/recent-trades";
import PerformanceChart from "@/components/dashboard/performance-chart";
import RiskMetrics from "@/components/dashboard/risk-metrics";
import AlertsPanel from "@/components/dashboard/alerts-panel";

export default function Dashboard() {
  return (
    <div className="min-h-screen flex bg-background text-foreground">
      <Sidebar />
      
      <main className="flex-1 ml-64">
        <TopBar />
        
        <div className="p-6 space-y-6">
          <StatsOverview />
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <PriceChart />
            </div>
            <OrderBook />
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <AIStrategyPanel />
            <RecentTrades />
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <PerformanceChart />
            </div>
            <RiskMetrics />
          </div>
          
          <AlertsPanel />
        </div>
      </main>
    </div>
  );
}
