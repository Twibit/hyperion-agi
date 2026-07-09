import sys
import os
import time
import pickle
import random
import urllib.request
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
from cain_brain import CainBrain

def fetch_tsla_prices():
    """Fetches real-time TSLA closing prices manually to avoid heavy pandas dependencies."""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/TSLA?range=3mo&interval=1d"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            closes = data['chart']['result'][0]['indicators']['quote'][0]['close']
            # Filter out any None values from the market data
            return [price for price in closes if price is not None]
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def run_paper_trading():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Hyperion Bio-Algorithmic Trader (Real-Time Market Simulation)    ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    save_path = "hyperion_compressed_brain.pkl"
    if not os.path.exists(save_path):
        print("Error: Brain file not found. Run ingestion first.")
        return
        
    print(f"Loading Hyperion Financial Brain ({save_path})...")
    with open(save_path, 'rb') as f:
        brain = pickle.load(f)
        
    print("📈 Fetching raw market data for TSLA (Tesla, Inc.) for the last 3 months...")
    close_prices = fetch_tsla_prices()
    
    if not close_prices:
        print("Failed to fetch market data.")
        return
        
    print(f"Market data loaded: {len(close_prices)} trading days.\n")
    
    portfolio_cash = 10000.00
    shares_owned = 0
    trades_executed = 0
    winning_trades = 0
    losing_trades = 0
    
    buy_price = 0
    
    print("💰 Initial Balance: $10,000.00 | Trading Asset: TSLA")
    print("-" * 70)
    
    # We need a rolling window of 3 days to form a trend sentence
    for i in range(3, len(close_prices)):
        today_price = close_prices[i]
        yesterday_price = close_prices[i-1]
        
        # Calculate percentage changes
        p1 = ((close_prices[i-1] - close_prices[i-2]) / close_prices[i-2]) * 100
        p2 = ((close_prices[i-2] - close_prices[i-3]) / close_prices[i-3]) * 100
        
        trend = "upward" if p1 > 0 and p2 > 0 else "downward" if p1 < 0 and p2 < 0 else "volatile"
        
        # Prompt Engineering: Asking Hyperion for high-frequency day trading advice
        prompt = f"The stock market is highly {trend}. Based on algorithmic trading, quantitative analysis, and high-frequency trading principles, the immediate tactical action to maximize short-term profit is to"
        
        # Generate Hyperion's thought
        thought = brain.speak(prompt, domain="prose", length=15).lower()
        
        action = "HOLD"
        
        # Map output to actions (Hyperion vocabulary determines the action)
        if any(word in thought for word in ["buy", "invest", "purchase", "holdings", "acquire", "growth", "value", "increase"]):
            if portfolio_cash >= today_price:
                action = "BUY"
        elif any(word in thought for word in ["sell", "short", "drop", "risk", "cash", "liquidate", "loss", "decrease"]):
            if shares_owned > 0:
                action = "SELL"
                
        # Execute Trades
        if action == "BUY":
            shares_bought = int(portfolio_cash // today_price)
            if shares_bought > 0:
                portfolio_cash -= shares_bought * today_price
                shares_owned += shares_bought
                buy_price = today_price
                print(f"Day {i:02d} | Price: ${today_price:.2f} | Hyperion: BUY  ({shares_bought} shares) | Thought: '...{thought[-40:]}'")
                
        elif action == "SELL":
            profit = (today_price - buy_price) * shares_owned
            portfolio_cash += shares_owned * today_price
            
            trades_executed += 1
            if profit > 0:
                winning_trades += 1
                result = "🟢 WIN"
            else:
                losing_trades += 1
                result = "🔴 LOSS"
                
            print(f"Day {i:02d} | Price: ${today_price:.2f} | Hyperion: SELL ({shares_owned} shares) | {result} (${profit:+.2f})")
            shares_owned = 0
            
    # Sell any remaining shares at the end of the simulation
    if shares_owned > 0:
        portfolio_cash += shares_owned * close_prices[-1]
        
    print("-" * 70)
    print("TRADING SIMULATION COMPLETE")
    print(f"Final Portfolio Value : ${portfolio_cash:.2f}")
    profit_pct = ((portfolio_cash - 10000.0) / 10000.0) * 100
    print(f"Total Profit/Loss     : {profit_pct:+.2f}%")
    
    if trades_executed > 0:
        win_rate = (winning_trades / trades_executed) * 100
        print(f"Total Trades Executed : {trades_executed}")
        print(f"Hyperion Win Rate      : {win_rate:.1f}%")
    else:
        print("No trades were completed (held through the period).")

if __name__ == '__main__':
    run_paper_trading()
