import { Card, CardContent } from "@/components/ui/card";

export default function OrderBook() {
  // Generate sample order book data
  const sellOrders = [
    { price: 43289.45, amount: 0.0234 },
    { price: 43278.21, amount: 0.1456 },
    { price: 43267.89, amount: 0.0789 },
  ];

  const buyOrders = [
    { price: 43235.67, amount: 0.0567 },
    { price: 43223.45, amount: 0.2134 },
    { price: 43210.12, amount: 0.0923 },
  ];

  const currentPrice = 43247.82;

  return (
    <Card className="bg-card border-border">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-bold text-foreground">Order Book</h3>
          <span className="text-sm text-muted-foreground">BTC/USDT</span>
        </div>
        
        <div className="space-y-4">
          {/* Sell Orders */}
          <div>
            <div className="flex justify-between text-xs text-muted-foreground mb-2">
              <span>Price (USDT)</span>
              <span>Amount (BTC)</span>
            </div>
            <div className="space-y-1">
              {sellOrders.map((order, index) => (
                <div key={index} className="flex justify-between text-xs bg-danger/10 p-1 rounded">
                  <span className="text-danger">{order.price.toFixed(2)}</span>
                  <span className="text-muted-foreground">{order.amount.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Current Price */}
          <div className="text-center py-2 border-t border-b border-border">
            <span className="text-lg font-bold text-foreground">{currentPrice.toFixed(2)}</span>
          </div>

          {/* Buy Orders */}
          <div>
            <div className="space-y-1">
              {buyOrders.map((order, index) => (
                <div key={index} className="flex justify-between text-xs bg-success/10 p-1 rounded">
                  <span className="text-success">{order.price.toFixed(2)}</span>
                  <span className="text-muted-foreground">{order.amount.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
