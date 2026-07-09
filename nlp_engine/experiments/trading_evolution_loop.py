#!/usr/bin/env python3
"""
Hyperion Autonomous Trading Evolution Loop
==========================================
1. Scrape curated trading/prediction books & concepts via Firecrawl
2. Fix the brain's internal dict structure so new learning works after pickle load
3. Run the paper trading simulation on real TSLA data
4. Measure win rate
5. Repeat with more knowledge until best win rate is achieved
"""
import sys
import os
import time
import pickle
import random
import urllib.request
import json
import re
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
from cain_brain import CainBrain

FIRECRAWL_KEY = "fc-eb008efd07a24fd2b0dc659162b603b4"
BRAIN_PATH = "hyperion_compressed_brain.pkl"

# ═══════════════════════════════════════════════════════════════
# CURATED KNOWLEDGE ROUNDS (Each round = a thematic curriculum)
# ═══════════════════════════════════════════════════════════════
KNOWLEDGE_ROUNDS = [
    {
        "name": "Round 1: Trading Psychology & Market Mechanics",
        "urls": [
            "https://en.wikipedia.org/wiki/Flash_Boys",
            "https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow",
            "https://en.wikipedia.org/wiki/Fooled_by_Randomness",
            "https://en.wikipedia.org/wiki/The_Black_Swan:_The_Impact_of_the_Highly_Improbable",
            "https://en.wikipedia.org/wiki/Day_trading",
            "https://en.wikipedia.org/wiki/Market_microstructure",
            "https://en.wikipedia.org/wiki/High-frequency_trading",
            "https://en.wikipedia.org/wiki/Technical_analysis",
        ]
    },
    {
        "name": "Round 2: Quantitative Strategies & Risk",
        "urls": [
            "https://en.wikipedia.org/wiki/Algorithmic_trading",
            "https://en.wikipedia.org/wiki/Statistical_arbitrage",
            "https://en.wikipedia.org/wiki/Arbitrage",
            "https://en.wikipedia.org/wiki/Dark_pool",
            "https://en.wikipedia.org/wiki/Market_maker",
            "https://en.wikipedia.org/wiki/Order_book",
            "https://en.wikipedia.org/wiki/Automated_trading_system",
            "https://en.wikipedia.org/wiki/Quantitative_analysis_(finance)",
        ]
    },
    {
        "name": "Round 3: Prediction Science & Decision Theory",
        "urls": [
            "https://en.wikipedia.org/wiki/Prediction_market",
            "https://en.wikipedia.org/wiki/Bayesian_inference",
            "https://en.wikipedia.org/wiki/Decision_theory",
            "https://en.wikipedia.org/wiki/Game_theory",
            "https://en.wikipedia.org/wiki/Monte_Carlo_method",
            "https://en.wikipedia.org/wiki/Regression_analysis",
            "https://en.wikipedia.org/wiki/Time_series",
            "https://en.wikipedia.org/wiki/Behavioral_economics",
        ]
    },
    {
        "name": "Round 4: Advanced Financial Instruments",
        "urls": [
            "https://en.wikipedia.org/wiki/Moving_average",
            "https://en.wikipedia.org/wiki/Bollinger_Bands",
            "https://en.wikipedia.org/wiki/Relative_strength_index",
            "https://en.wikipedia.org/wiki/Candlestick_chart",
            "https://en.wikipedia.org/wiki/Support_and_resistance",
            "https://en.wikipedia.org/wiki/Momentum_(finance)",
            "https://en.wikipedia.org/wiki/Swing_trading",
            "https://en.wikipedia.org/wiki/Trend_following",
        ]
    },
]


def repair_brain_dicts(brain):
    """
    CRITICAL FIX: After pickle loading, all defaultdicts become regular dicts.
    This function restores them back to defaultdicts so new learning works.
    """
    for domain, bank in brain.banks.items():
        new_counts = []
        for order_dict in bank.counts:
            new_order = defaultdict(lambda: defaultdict(float))
            if isinstance(order_dict, dict):
                for ctx, word_counts in order_dict.items():
                    inner = defaultdict(float)
                    if isinstance(word_counts, dict):
                        inner.update(word_counts)
                    new_order[ctx] = inner
            else:
                new_order.update(order_dict)
            new_counts.append(new_order)
        bank.counts = new_counts
    return brain


def fetch_firecrawl(target_url):
    """Fetches a specific page via Firecrawl (Thread-Safe, returns raw text)."""
    url = "https://api.firecrawl.dev/v2/scrape"
    payload = json.dumps({"url": target_url}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers={
        'Authorization': f'Bearer {FIRECRAWL_KEY}',
        'Content-Type': 'application/json'
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data.get('success') and 'data' in data:
                text = data['data'].get('markdown', '')
                title = data['data'].get('metadata', {}).get('title', 'Unknown')
                text = re.sub(r'\[.*?\]\(.*?\)', '', text)
                text = re.sub(r'\[.*?\]', '', text)
                if len(text) > 100:
                    return title, text
    except Exception as e:
        print(f"    [Firecrawl Error]: {e}")
    return None, None


def fetch_tsla_prices():
    """Fetches real-time TSLA closing prices from Yahoo Finance raw API."""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/TSLA?range=3mo&interval=1d"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            closes = data['chart']['result'][0]['indicators']['quote'][0]['close']
            return [price for price in closes if price is not None]
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return []


def run_simulation(brain, close_prices, silent=False):
    """
    Runs the paper trading simulation. Returns (final_value, trades, wins, losses).
    """
    portfolio_cash = 10000.00
    shares_owned = 0
    trades_executed = 0
    winning_trades = 0
    losing_trades = 0
    buy_price = 0
    trade_log = []

    for i in range(3, len(close_prices)):
        today_price = close_prices[i]

        p1 = ((close_prices[i-1] - close_prices[i-2]) / close_prices[i-2]) * 100
        p2 = ((close_prices[i-2] - close_prices[i-3]) / close_prices[i-3]) * 100

        trend = "upward" if p1 > 0 and p2 > 0 else "downward" if p1 < 0 and p2 < 0 else "volatile"

        prompt = f"The stock market is highly {trend}. Based on algorithmic trading, quantitative analysis, and high-frequency trading principles, the immediate tactical action to maximize short-term profit is to"
        thought = brain.speak(prompt, domain="prose", length=15).lower()

        action = "HOLD"

        buy_words = ["buy", "invest", "purchase", "holdings", "acquire", "growth", "value", "increase", "long", "accumulate", "opportunity"]
        sell_words = ["sell", "short", "drop", "risk", "cash", "liquidate", "loss", "decrease", "exit", "reduce", "hedge", "stop"]

        if any(word in thought for word in buy_words):
            if portfolio_cash >= today_price:
                action = "BUY"
        elif any(word in thought for word in sell_words):
            if shares_owned > 0:
                action = "SELL"

        if action == "BUY":
            shares_bought = int(portfolio_cash // today_price)
            if shares_bought > 0:
                portfolio_cash -= shares_bought * today_price
                shares_owned += shares_bought
                buy_price = today_price
                if not silent:
                    print(f"    Day {i:02d} | ${today_price:.2f} | BUY  {shares_bought} shares")

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
            if not silent:
                print(f"    Day {i:02d} | ${today_price:.2f} | SELL {shares_owned} shares | {result} (${profit:+.2f})")
            trade_log.append(profit)
            shares_owned = 0

    # Liquidate remaining position
    if shares_owned > 0:
        final_profit = (close_prices[-1] - buy_price) * shares_owned
        portfolio_cash += shares_owned * close_prices[-1]
        trades_executed += 1
        if final_profit > 0:
            winning_trades += 1
        else:
            losing_trades += 1

    return portfolio_cash, trades_executed, winning_trades, losing_trades


def save_brain(brain):
    """Saves brain with dict conversion for pickling."""
    for domain, bank in brain.banks.items():
        for i in range(len(bank.counts)):
            bank.counts[i] = {k: dict(v) for k, v in bank.counts[i].items()}
    with open(BRAIN_PATH, 'wb') as f:
        pickle.dump(brain, f)


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Hyperion Autonomous Trading Evolution Loop                      ║")
    print("║  Scrape → Learn → Trade → Measure → Repeat Until Best Win Rate  ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    # Step 1: Load or create brain
    if os.path.exists(BRAIN_PATH):
        print(f"Loading existing brain from {BRAIN_PATH}...")
        with open(BRAIN_PATH, 'rb') as f:
            brain = pickle.load(f)
        print("🔧 Repairing internal dictionary structures (defaultdict restoration)...")
        brain = repair_brain_dicts(brain)
    else:
        print("Creating fresh CainBrain...")
        brain = CainBrain()

    # Step 2: Fetch market data once
    print("\n📈 Fetching real-time TSLA market data (3 months)...")
    close_prices = fetch_tsla_prices()
    if not close_prices:
        print("Failed to fetch market data. Aborting.")
        return
    print(f"Loaded {len(close_prices)} trading days.\n")

    # Step 3: Baseline simulation
    print("─" * 66)
    print("BASELINE SIMULATION (Before new knowledge)")
    print("─" * 66)
    val, trades, wins, losses = run_simulation(brain, close_prices, silent=True)
    baseline_pct = ((val - 10000.0) / 10000.0) * 100
    baseline_wr = (wins / trades * 100) if trades > 0 else 0
    print(f"    Portfolio: ${val:.2f} | P/L: {baseline_pct:+.2f}% | Trades: {trades} | Win Rate: {baseline_wr:.1f}%")

    best_win_rate = baseline_wr
    best_round_name = "Baseline"
    best_portfolio = val

    # Step 4: Evolution Loop — scrape a round, learn, simulate, compare
    for round_idx, knowledge_round in enumerate(KNOWLEDGE_ROUNDS):
        print(f"\n{'═' * 66}")
        print(f"📚 {knowledge_round['name']}")
        print(f"{'═' * 66}")

        learned_count = 0
        for url in knowledge_round['urls']:
            title, text = fetch_firecrawl(url)
            if title and text:
                try:
                    brain.learn(text, domain='prose', source=title)
                    learned_count += 1
                    print(f"    Learned: '{title}' | Vocab: {len(brain.banks['prose'].vocab)}")
                except Exception as e:
                    print(f"    Learn Error on '{title}': {e}")
            time.sleep(0.5)  # Gentle rate limiting

        print(f"\n    📖 Ingested {learned_count}/{len(knowledge_round['urls'])} articles.")

        # Run simulation after this round
        print(f"\n    Simulating trades with new knowledge...")
        val, trades, wins, losses = run_simulation(brain, close_prices, silent=False)
        pct = ((val - 10000.0) / 10000.0) * 100
        wr = (wins / trades * 100) if trades > 0 else 0

        print(f"\n    ┌────────────────────────────────────────────────┐")
        print(f"    │ Portfolio: ${val:.2f}  |  P/L: {pct:+.2f}%           │")
        print(f"    │ Trades: {trades}  |  Wins: {wins}  |  Losses: {losses}         │")
        print(f"    │ Win Rate: {wr:.1f}%                                │")
        if wr > best_win_rate:
            print(f"    │ 🏆 NEW BEST WIN RATE! (was {best_win_rate:.1f}%)            │")
            best_win_rate = wr
            best_round_name = knowledge_round['name']
            best_portfolio = val
        elif wr == best_win_rate and val > best_portfolio:
            print(f"    │ 📈 Same win rate but higher profit!            │")
            best_portfolio = val
            best_round_name = knowledge_round['name']
        else:
            print(f"    │ ⏳ Best remains: {best_win_rate:.1f}% from {best_round_name[:20]}  │")
        print(f"    └────────────────────────────────────────────────┘")

        # Save brain after each round
        save_brain(brain)
        # Repair dicts again for the next round's learning
        brain = repair_brain_dicts(brain)

    # Final Summary
    print(f"\n{'═' * 66}")
    print(f"🏁 EVOLUTION COMPLETE — FINAL RESULTS")
    print(f"{'═' * 66}")
    print(f"  🏆 Best Win Rate    : {best_win_rate:.1f}%")
    print(f"  📚 Best Round       : {best_round_name}")
    print(f"  💰 Best Portfolio   : ${best_portfolio:.2f}")
    print(f"  📈 Best P/L         : {((best_portfolio - 10000.0) / 10000.0) * 100:+.2f}%")
    print(f"  Final Brain Size : {os.path.getsize(BRAIN_PATH) / (1024*1024):.2f} MB")
    print(f"  Final Vocab      : {len(brain.banks['prose'].vocab)} words")
    print(f"{'═' * 66}")


if __name__ == '__main__':
    main()
