import { Clock, StopCircle, Pause, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTradingData } from "@/hooks/use-trading-data";
import { useState, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

export default function TopBar() {
  const { botStatus, queryClient } = useTradingData(1);
  const [currentTime, setCurrentTime] = useState(new Date());
  const { toast } = useToast();

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const stopBotMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/bot/1/stop"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/bot-status", "1"] });
      toast({
        title: "Emergency Stop",
        description: "Bot has been stopped successfully",
        variant: "destructive"
      });
    }
  });

  const pauseBotMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/bot/1/pause"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/bot-status", "1"] });
      toast({
        title: "Bot Paused",
        description: "Bot has been paused successfully",
        variant: "default"
      });
    }
  });

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour12: false, 
      timeZone: 'UTC' 
    }) + ' UTC';
  };

  return (
    <header className="bg-card border-b border-border px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <h2 className="text-2xl font-bold text-foreground">Trading Dashboard</h2>
          <div className="flex items-center space-x-2 text-sm text-muted-foreground">
            <Clock size={16} />
            <span>{formatTime(currentTime)}</span>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <Button 
            onClick={() => stopBotMutation.mutate()}
            disabled={stopBotMutation.isPending}
            variant="destructive"
            size="sm"
          >
            <StopCircle size={16} className="mr-2" />
            Emergency Stop
          </Button>
          <Button 
            onClick={() => pauseBotMutation.mutate()}
            disabled={pauseBotMutation.isPending}
            className="bg-warning hover:bg-warning/80 text-black"
            size="sm"
          >
            <Pause size={16} className="mr-2" />
            Pause Bot
          </Button>
          
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center">
              <User className="text-primary-foreground" size={20} />
            </div>
            <div className="text-sm">
              <div className="font-medium text-foreground">John Trader</div>
              <div className="text-muted-foreground">Pro Plan</div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
