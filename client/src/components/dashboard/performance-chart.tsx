import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useEffect, useRef, useState } from "react";

export default function PerformanceChart() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [period, setPeriod] = useState("7D");

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Generate sample data
    const portfolioData = [50000, 51200, 50800, 52100, 53400, 54200, 54782];
    const benchmarkData = [50000, 50500, 50200, 51000, 51800, 52500, 53200];
    const labels = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'];

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

    const maxValue = Math.max(...portfolioData, ...benchmarkData);
    const minValue = Math.min(...portfolioData, ...benchmarkData);
    const valueRange = maxValue - minValue;

    // Draw portfolio line
    ctx.strokeStyle = '#10B981';
    ctx.lineWidth = 2;
    ctx.beginPath();

    portfolioData.forEach((value, index) => {
      const x = (canvas.offsetWidth / (portfolioData.length - 1)) * index;
      const y = canvas.offsetHeight - ((value - minValue) / valueRange) * canvas.offsetHeight;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Fill area under portfolio curve
    ctx.fillStyle = 'rgba(16, 185, 129, 0.1)';
    ctx.lineTo(canvas.offsetWidth, canvas.offsetHeight);
    ctx.lineTo(0, canvas.offsetHeight);
    ctx.closePath();
    ctx.fill();

    // Draw benchmark line
    ctx.strokeStyle = '#F59E0B';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();

    benchmarkData.forEach((value, index) => {
      const x = (canvas.offsetWidth / (benchmarkData.length - 1)) * index;
      const y = canvas.offsetHeight - ((value - minValue) / valueRange) * canvas.offsetHeight;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();
    ctx.setLineDash([]);

  }, [period]);

  return (
    <Card className="bg-card border-border">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-bold text-foreground">Performance Analytics</h3>
          <div className="flex items-center space-x-2">
            {['7D', '30D', '90D'].map((p) => (
              <Button
                key={p}
                onClick={() => setPeriod(p)}
                size="sm"
                variant={period === p ? "default" : "ghost"}
                className={`text-xs ${
                  period === p 
                    ? "bg-primary text-primary-foreground" 
                    : "bg-muted text-muted-foreground hover:bg-muted/80"
                }`}
              >
                {p}
              </Button>
            ))}
          </div>
        </div>
        
        <div className="h-64 mb-4">
          <canvas 
            ref={canvasRef}
            className="w-full h-full"
            style={{ width: '100%', height: '100%' }}
          />
        </div>

        <div className="flex items-center justify-center space-x-6 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-success rounded-full"></div>
            <span className="text-muted-foreground">Portfolio Value</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-warning rounded-full"></div>
            <span className="text-muted-foreground">BTC Benchmark</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
