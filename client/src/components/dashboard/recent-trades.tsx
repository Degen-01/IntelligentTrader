import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowUp, ArrowDown, ArrowRight } from "lucide-react";
import { useTradingData } from "@/hooks/use-trading-data";

export default function RecentTrades() {
  const { trades } = useTradingData(1);

  // Sample data for demo
  const sampleTrades = [
    {
      id: 1,
      symbol: "BTC",
      side: "buy",
      amount: "0.0234",
      price: "43156.78",
      pnl: "127.34",
      executedAt: new Date(Date.now() - 2 * 60 * 1000)
    },
    {
      id: 2,
      symbol: "ETH",
      side: "sell",
      amount: "1.45",
      price: "2567.89",
      pnl: "89.45",
      executedAt: new Date(Date.now() - 5 * 60 * 1000)
    },
    {
      id: 3,
      symbol: "ADA",
      side: "buy",
      amount: "2450",
      price: "0.4234",
      pnl: "45.67",
      executedAt: new Date(Date.now() - 8 * 60 * 1000)
    }
  ];

  const displayTrades = trades && trades.length > 0 ? trades : sampleTrades;

  const formatTimeAgo = (date: Date) => {
    const now = new Date();
    const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60));
    
    if (diffInMinutes < 1) return "Just now";
    if (diffInMinutes === 1) return "1m ago";
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    
    const diffInHours = Math.floor(diffInMinutes / 60);
    if (diffInHours === 1) return "1h ago";
    return `${diffInHours}h ago`;
  };

  const formatCurrency = (value: string | number) => {
    const num = typeof value === 'string' ? parseFloat(value) : value;
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(num);
  };

  return (
    <Card className="bg-card border-border">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-bold text-foreground">Recent Trades</h3>
          <Button variant="ghost" size="sm" className="text-primary hover:text-primary/80">
            View All <ArrowRight size={16} className="ml-1" />
          </Button>
        </div>

        <div className="space-y-3">
          {displayTrades.slice(0, 3).map((trade) => (
            <div key={trade.id} className="flex items-center justify-between p-3 bg-muted rounded-lg">
              <div className="flex items-center space-x-3">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                  trade.side === 'buy' ? 'bg-success/20' : 'bg-danger/20'
                }`}>
                  {trade.side === 'buy' ? (
                    <ArrowUp className="text-success" size={16} />
                  ) : (
                    <ArrowDown className="text-danger" size={16} />
                  )}
                </div>
                <div>
                  <div className="font-medium text-foreground">
                    {trade.side.toUpperCase()} {trade.symbol}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {trade.amount} {trade.symbol} @ {formatCurrency(trade.price)}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-success font-medium">+{formatCurrency(trade.pnl)}</div>
                <div className="text-xs text-muted-foreground">
                  {formatTimeAgo(trade.executedAt)}
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
