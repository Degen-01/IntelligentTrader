import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useTradingData } from "@/hooks/use-trading-data";

export default function RiskMetrics() {
  const { riskMetrics } = useTradingData(1);

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

  const portfolioHeat = riskMetrics ? parseFloat(riskMetrics.portfolioHeat) : 60;
  const riskScore = riskMetrics ? parseFloat(riskMetrics.riskScore) : 7.2;

  return (
    <Card className="bg-card border-border">
      <CardContent className="p-6">
        <h3 className="text-lg font-bold text-foreground mb-6">Risk Metrics</h3>
        
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <span className="text-sm text-muted-foreground">Portfolio Heat</span>
            <div className="flex items-center space-x-2">
              <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                <div 
                  className="h-full bg-warning rounded-full transition-all duration-300"
                  style={{ width: `${portfolioHeat}%` }}
                ></div>
              </div>
              <span className="text-sm font-medium text-warning w-12">
                {formatPercent(String(portfolioHeat))}
              </span>
            </div>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-muted-foreground">Max Drawdown</span>
            <span className="text-sm font-medium text-foreground">
              {riskMetrics ? formatPercent(riskMetrics.maxDrawdown) : '-2.34%'}
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-muted-foreground">Sharpe Ratio</span>
            <span className="text-sm font-medium text-success">
              {riskMetrics ? riskMetrics.sharpeRatio : '2.45'}
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-muted-foreground">VaR (95%)</span>
            <span className="text-sm font-medium text-foreground">
              {riskMetrics ? formatCurrency(riskMetrics.var95) : '$1,234.56'}
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-muted-foreground">Beta vs BTC</span>
            <span className="text-sm font-medium text-foreground">
              {riskMetrics ? riskMetrics.betaBtc : '0.87'}
            </span>
          </div>

          <div className="pt-4 border-t border-border">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Risk Score</span>
              <span className="text-sm font-medium text-warning">
                {riskScore.toFixed(1)}/10
              </span>
            </div>
            <Progress 
              value={(riskScore / 10) * 100} 
              className="w-full h-2"
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
