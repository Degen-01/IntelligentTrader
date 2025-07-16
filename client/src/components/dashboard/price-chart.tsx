import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowUp } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTradingData } from "@/hooks/use-trading-data";

export default function PriceChart() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { marketData } = useTradingData(1);
  const [timeframe, setTimeframe] = useState("1H");

  const btcData = marketData?.find(data => data.symbol === "BTC/USDT");

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Generate sample data points
    const dataPoints = Array.from({ length: 50 }, (_, i) => {
      const basePrice = 43000;
      const variation = Math.sin(i * 0.3) * 500 + Math.random() * 200 - 100;
      return basePrice + variation;
    });

    // Clear canvas
    ctx.clearRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);

    // Draw grid
    ctx.strokeStyle = 'rgba(156, 163, 175, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const y = (canvas.offsetHeight / 10) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.offsetWidth, y);
      ctx.stroke();
    }

    // Draw price line
    ctx.strokeStyle = '#3B82F6';
    ctx.lineWidth = 2;
    ctx.beginPath();

    const maxPrice = Math.max(...dataPoints);
    const minPrice = Math.min(...dataPoints);
    const priceRange = maxPrice - minPrice;

    dataPoints.forEach((price, index) => {
      const x = (canvas.offsetWidth / (dataPoints.length - 1)) * index;
      const y = canvas.offsetHeight - ((price - minPrice) / priceRange) * canvas.offsetHeight;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Fill area under curve
    ctx.fillStyle = 'rgba(59, 130, 246, 0.1)';
    ctx.lineTo(canvas.offsetWidth, canvas.offsetHeight);
    ctx.lineTo(0, canvas.offsetHeight);
    ctx.closePath();
    ctx.fill();

  }, [timeframe]);

  const formatPrice = (price: string) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(parseFloat(price));
  };

  const formatPercent = (percent: string) => {
    return `${parseFloat(percent).toFixed(2)}%`;
  };

  return (
    <Card className="bg-card border-border">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4">
            <h3 className="text-xl font-bold text-foreground">BTC/USDT</h3>
            <div className="flex items-center space-x-2">
              <span className="text-2xl font-bold text-foreground">
                {btcData ? formatPrice(btcData.price) : '$43,247.82'}
              </span>
              <div className="flex items-center">
                <ArrowUp className="text-success text-sm mr-1" size={16} />
                <span className="text-success font-medium">
                  {btcData ? formatPercent(btcData.changePercent24h) : '+2.45%'}
                </span>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {['1H', '4H', '1D', '1W'].map((tf) => (
              <Button
                key={tf}
                onClick={() => setTimeframe(tf)}
                size="sm"
                variant={timeframe === tf ? "default" : "ghost"}
                className={`text-xs ${
                  timeframe === tf 
                    ? "bg-primary text-primary-foreground" 
                    : "bg-muted text-muted-foreground hover:bg-muted/80"
                }`}
              >
                {tf}
              </Button>
            ))}
          </div>
        </div>
        <div className="h-80">
          <canvas 
            ref={canvasRef}
            className="w-full h-full"
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </CardContent>
    </Card>
  );
}
