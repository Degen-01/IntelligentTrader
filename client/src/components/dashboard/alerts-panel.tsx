import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertTriangle, CheckCircle, Info, Shield, Settings } from "lucide-react";
import { useTradingData } from "@/hooks/use-trading-data";

export default function AlertsPanel() {
  const { alerts } = useTradingData(1);

  // Sample alerts for demo
  const sampleAlerts = [
    {
      id: 1,
      type: "warning" as const,
      title: "High Volatility",
      message: "BTC volatility increased by 45% in the last hour",
      createdAt: new Date(Date.now() - 2 * 60 * 1000),
      isRead: false
    },
    {
      id: 2,
      type: "success" as const,
      title: "Profit Target Hit",
      message: "ETH position reached +15% profit target",
      createdAt: new Date(Date.now() - 5 * 60 * 1000),
      isRead: false
    },
    {
      id: 3,
      type: "info" as const,
      title: "Strategy Update",
      message: "AI model confidence increased to 94.2%",
      createdAt: new Date(Date.now() - 12 * 60 * 1000),
      isRead: false
    },
    {
      id: 4,
      type: "danger" as const,
      title: "Risk Limit",
      message: "Approaching daily loss limit (85% used)",
      createdAt: new Date(Date.now() - 18 * 60 * 1000),
      isRead: false
    }
  ];

  const displayAlerts = alerts && alerts.length > 0 ? alerts : sampleAlerts;

  const getAlertIcon = (type: string) => {
    switch (type) {
      case "warning":
        return <AlertTriangle className="text-warning" size={16} />;
      case "success":
        return <CheckCircle className="text-success" size={16} />;
      case "info":
        return <Info className="text-primary" size={16} />;
      case "danger":
        return <Shield className="text-danger" size={16} />;
      default:
        return <Info className="text-primary" size={16} />;
    }
  };

  const getAlertStyles = (type: string) => {
    switch (type) {
      case "warning":
        return "bg-warning/10 border-warning/30";
      case "success":
        return "bg-success/10 border-success/30";
      case "info":
        return "bg-primary/10 border-primary/30";
      case "danger":
        return "bg-danger/10 border-danger/30";
      default:
        return "bg-primary/10 border-primary/30";
    }
  };

  const getAlertTextColor = (type: string) => {
    switch (type) {
      case "warning":
        return "text-warning";
      case "success":
        return "text-success";
      case "info":
        return "text-primary";
      case "danger":
        return "text-danger";
      default:
        return "text-primary";
    }
  };

  const formatTimeAgo = (date: Date) => {
    const now = new Date();
    const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60));
    
    if (diffInMinutes < 1) return "Just now";
    if (diffInMinutes === 1) return "1 minute ago";
    if (diffInMinutes < 60) return `${diffInMinutes} minutes ago`;
    
    const diffInHours = Math.floor(diffInMinutes / 60);
    if (diffInHours === 1) return "1 hour ago";
    return `${diffInHours} hours ago`;
  };

  return (
    <Card className="bg-card border-border">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-bold text-foreground">Active Alerts & Notifications</h3>
          <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground">
            <Settings size={16} className="mr-1" />
            Configure
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {displayAlerts.slice(0, 4).map((alert) => (
            <div 
              key={alert.id} 
              className={`p-4 border rounded-lg ${getAlertStyles(alert.type)}`}
            >
              <div className="flex items-center space-x-2 mb-2">
                {getAlertIcon(alert.type)}
                <span className={`text-sm font-medium ${getAlertTextColor(alert.type)}`}>
                  {alert.title}
                </span>
              </div>
              <p className="text-xs text-muted-foreground mb-2">{alert.message}</p>
              <span className="text-xs text-muted-foreground">
                {formatTimeAgo(alert.createdAt)}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
