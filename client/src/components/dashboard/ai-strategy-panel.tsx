import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Play, Zap } from "lucide-react";
import { useTradingData } from "@/hooks/use-trading-data";
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

export default function AIStrategyPanel() {
  const { botStatus, queryClient } = useTradingData(1);
  const [strategy, setStrategy] = useState(botStatus?.currentStrategy || "momentum");
  const [riskTolerance, setRiskTolerance] = useState([botStatus?.riskTolerance || 6]);
  const [positionSize, setPositionSize] = useState(botStatus?.positionSizePercent || 15);
  const { toast } = useToast();

  const startBotMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/bot/1/start"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/bot-status", "1"] });
      toast({
        title: "Bot Started",
        description: "AI trading bot has been activated",
        variant: "default"
      });
    }
  });

  const updateBotMutation = useMutation({
    mutationFn: (data: any) => apiRequest("PATCH", "/api/bot-status/1", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/bot-status", "1"] });
      toast({
        title: "Strategy Optimized",
        description: "Bot configuration has been updated",
        variant: "default"
      });
    }
  });

  const handleStartBot = () => {
    startBotMutation.mutate();
  };

  const handleOptimize = () => {
    const updates = {
      currentStrategy: strategy,
      riskTolerance: riskTolerance[0],
      positionSizePercent: positionSize
    };
    updateBotMutation.mutate(updates);
  };

  return (
    <Card className="bg-card border-border">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-bold text-foreground">AI Strategy Control</h3>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-success rounded-full animate-pulse"></div>
            <span className="text-xs text-success font-medium">LEARNING</span>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <Label className="text-sm font-medium text-muted-foreground mb-2 block">
              Active Strategy
            </Label>
            <Select value={strategy} onValueChange={setStrategy}>
              <SelectTrigger className="w-full bg-muted border-border text-foreground">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="momentum">Momentum Trading</SelectItem>
                <SelectItem value="meanReversion">Mean Reversion</SelectItem>
                <SelectItem value="arbitrage">Cross-Exchange Arbitrage</SelectItem>
                <SelectItem value="scalping">Scalping</SelectItem>
                <SelectItem value="ensemble">Ensemble Strategy</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label className="text-sm font-medium text-muted-foreground mb-2 block">
              Risk Tolerance
            </Label>
            <div className="flex items-center space-x-4">
              <Slider
                value={riskTolerance}
                onValueChange={setRiskTolerance}
                max={10}
                min={1}
                step={1}
                className="flex-1"
              />
              <span className="text-foreground font-medium w-8">{riskTolerance[0]}</span>
            </div>
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>Conservative</span>
              <span>Aggressive</span>
            </div>
          </div>

          <div>
            <Label className="text-sm font-medium text-muted-foreground mb-2 block">
              Position Size (%)
            </Label>
            <Input
              type="number"
              value={positionSize}
              onChange={(e) => setPositionSize(parseInt(e.target.value) || 0)}
              min={1}
              max={100}
              className="w-full bg-muted border-border text-foreground"
            />
          </div>

          <div className="flex space-x-3">
            <Button 
              onClick={handleStartBot}
              disabled={startBotMutation.isPending}
              className="flex-1 bg-success hover:bg-success/80 text-white"
            >
              <Play size={16} className="mr-2" />
              Start Bot
            </Button>
            <Button 
              onClick={handleOptimize}
              disabled={updateBotMutation.isPending}
              className="flex-1 bg-primary hover:bg-primary/80 text-primary-foreground"
            >
              <Zap size={16} className="mr-2" />
              Optimize
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
