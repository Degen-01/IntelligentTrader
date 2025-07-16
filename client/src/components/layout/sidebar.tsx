import { Bot, ChartLine, Wallet, Shield, Brain, History, Bell, Settings, ArrowRightLeft } from "lucide-react";
import { useTradingData } from "@/hooks/use-trading-data";

export default function Sidebar() {
  const { botStatus } = useTradingData(1); // Demo user ID

  const formatUptime = (minutes: number) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
  };

  return (
    <aside className="w-64 bg-card border-r border-border fixed h-full z-10">
      <div className="p-6">
        <div className="flex items-center space-x-3 mb-8">
          <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center">
            <Bot className="text-primary-foreground text-lg" size={20} />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground">AI Trading Bot</h1>
            <p className="text-sm text-muted-foreground">v2.1.0</p>
          </div>
        </div>

        <nav className="space-y-2">
          <a href="#" className="flex items-center space-x-3 px-4 py-3 bg-primary/20 text-primary rounded-lg border border-primary/30">
            <ChartLine size={20} />
            <span className="font-medium">Dashboard</span>
          </a>
          <a href="#" className="flex items-center space-x-3 px-4 py-3 text-muted-foreground hover:bg-muted rounded-lg transition-colors">
            <ArrowRightLeft size={20} />
            <span>Trading</span>
          </a>
          <a href="#" className="flex items-center space-x-3 px-4 py-3 text-muted-foreground hover:bg-muted rounded-lg transition-colors">
            <Wallet size={20} />
            <span>Portfolio</span>
          </a>
          <a href="#" className="flex items-center space-x-3 px-4 py-3 text-muted-foreground hover:bg-muted rounded-lg transition-colors">
            <Shield size={20} />
            <span>Risk Management</span>
          </a>
          <a href="#" className="flex items-center space-x-3 px-4 py-3 text-muted-foreground hover:bg-muted rounded-lg transition-colors">
            <Brain size={20} />
            <span>AI Strategies</span>
          </a>
          <a href="#" className="flex items-center space-x-3 px-4 py-3 text-muted-foreground hover:bg-muted rounded-lg transition-colors">
            <History size={20} />
            <span>Backtesting</span>
          </a>
          <a href="#" className="flex items-center space-x-3 px-4 py-3 text-muted-foreground hover:bg-muted rounded-lg transition-colors">
            <Bell size={20} />
            <span>Alerts</span>
          </a>
          <a href="#" className="flex items-center space-x-3 px-4 py-3 text-muted-foreground hover:bg-muted rounded-lg transition-colors">
            <Settings size={20} />
            <span>Settings</span>
          </a>
        </nav>

        <div className="mt-8 p-4 bg-muted rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-muted-foreground">Bot Status</span>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full animate-pulse ${botStatus?.isActive ? 'bg-success' : 'bg-danger'}`}></div>
              <span className={`text-xs font-medium ${botStatus?.isActive ? 'text-success' : 'text-danger'}`}>
                {botStatus?.isActive ? 'ACTIVE' : 'INACTIVE'}
              </span>
            </div>
          </div>
          <div className="space-y-2 text-xs text-muted-foreground">
            <div className="flex justify-between">
              <span>Uptime:</span>
              <span>{botStatus ? formatUptime(botStatus.uptime) : '0h 0m'}</span>
            </div>
            <div className="flex justify-between">
              <span>Trades Today:</span>
              <span className="text-primary">{botStatus?.tradesCount || 0}</span>
            </div>
            <div className="flex justify-between">
              <span>Success Rate:</span>
              <span className="text-success">{botStatus?.successRate || '0'}%</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}
