import { Card, CardContent } from "@/components/ui/card";
import { TrendingUp, Wallet, Activity, Brain, ArrowUp } from "lucide-react";
import { useTradingData } from "@/hooks/use-trading-data";

export default function StatsOverview() {
  const { portfolio, botStatus } = useTradingData(1);

  const formatCurrency = (value: string) => {
    const num = parseFloat(value);
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(num);
  };

  const formatPercent = (value: string) => {
    return `${parseFloat(value).toFixed(2)}%`;
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <Card className="bg-card border-border">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Total Portfolio Value</p>
              <p className="text-3xl font-bold text-foreground">
                {portfolio ? formatCurrency(portfolio.totalValue) : '$0.00'}
              </p>
              <div className="flex items-center mt-2">
                <ArrowUp className="text-success text-sm mr-1" size={16} />
                <span className="text-success text-sm font-medium">
                  {portfolio ? formatPercent(portfolio.dayChangePercent) : '0%'}
                </span>
                <span className="text-muted-foreground text-sm ml-1">24h</span>
              </div>
            </div>
            <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center">
              <Wallet className="text-primary text-xl" size={24} />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-card border-border">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Today's P&L</p>
              <p className="text-3xl font-bold text-success">
                {portfolio ? `+${formatCurrency(portfolio.dayChange)}` : '+$0.00'}
              </p>
              <div className="flex items-center mt-2">
                <span className="text-muted-foreground text-sm">Win Rate: </span>
                <span className="text-foreground text-sm font-medium ml-1">
                  {botStatus ? formatPercent(botStatus.successRate) : '0%'}
                </span>
              </div>
            </div>
            <div className="w-12 h-12 bg-success/20 rounded-lg flex items-center justify-center">
              <TrendingUp className="text-success text-xl" size={24} />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-card border-border">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Active Positions</p>
              <p className="text-3xl font-bold text-foreground">12</p>
              <div className="flex items-center mt-2">
                <span className="text-muted-foreground text-sm">Risk: </span>
                <span className="text-warning text-sm font-medium ml-1">Medium</span>
              </div>
            </div>
            <div className="w-12 h-12 bg-warning/20 rounded-lg flex items-center justify-center">
              <Activity className="text-warning text-xl" size={24} />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-card border-border">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">AI Confidence</p>
              <p className="text-3xl font-bold text-foreground">
                {botStatus ? formatPercent(botStatus.confidence) : '0%'}
              </p>
              <div className="flex items-center mt-2">
                <span className="text-muted-foreground text-sm">Strategy: </span>
                <span className="text-primary text-sm font-medium ml-1 capitalize">
                  {botStatus?.currentStrategy || 'None'}
                </span>
              </div>
            </div>
            <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center">
              <Brain className="text-primary text-xl" size={24} />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
