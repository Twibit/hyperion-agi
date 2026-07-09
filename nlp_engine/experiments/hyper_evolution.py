#!/usr/bin/env python3
"""
Hyper-Evolution Script for Hyperion Neural Trader
Uses Genetic Algorithm concepts to find the absolute best trading hyperparameters.
"""
import sys
import os
import random
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.experiments.neural_trader_v2 import run_neural_trading

def mutate(params):
    """Slightly mutate the parameters."""
    new_params = params.copy()
    mutation_choice = random.choice(list(params.keys()))
    
    if mutation_choice == 'rsi_buy':
        new_params['rsi_buy'] = max(10, min(50, new_params['rsi_buy'] + random.randint(-5, 5)))
    elif mutation_choice == 'rsi_sell':
        new_params['rsi_sell'] = max(60, min(95, new_params['rsi_sell'] + random.randint(-5, 5)))
    elif mutation_choice == 'sma_short':
        new_params['sma_short'] = max(2, min(10, new_params['sma_short'] + random.choice([-1, 1])))
    elif mutation_choice == 'sma_long':
        new_params['sma_long'] = max(10, min(50, new_params['sma_long'] + random.choice([-2, 2, 5, -5])))
    elif mutation_choice == 'signal_threshold':
        new_params['signal_threshold'] = max(0.01, min(0.5, new_params['signal_threshold'] + random.uniform(-0.05, 0.05)))
    elif mutation_choice == 'take_profit_rsi':
        new_params['take_profit_rsi'] = max(60, min(95, new_params['take_profit_rsi'] + random.randint(-5, 5)))
        
    return new_params

def run_evolution(generations=15, population_size=5):
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Hyperion Darwinian Evolution Loop (Trading Strategy)            ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    ticker = "NVDA"
    time_range = "6mo"
    
    # Initial population
    population = []
    for _ in range(population_size):
        population.append({
            'rsi_buy': random.randint(20, 40),
            'rsi_sell': random.randint(65, 85),
            'sma_short': random.randint(3, 8),
            'sma_long': random.randint(15, 30),
            'signal_threshold': random.uniform(0.05, 0.2),
            'take_profit_rsi': random.randint(70, 90)
        })
        
    best_overall_score = -999
    best_overall_params = None
    
    for gen in range(generations):
        print(f"\nGeneration {gen + 1}/{generations}")
        print("─" * 70)
        
        results = []
        for i, params in enumerate(population):
            profit_pct, win_rate, trades = run_neural_trading(
                ticker=ticker, time_range=time_range,
                rsi_buy=params['rsi_buy'], rsi_sell=params['rsi_sell'],
                sma_short=params['sma_short'], sma_long=params['sma_long'],
                signal_threshold=params['signal_threshold'],
                take_profit_rsi=params['take_profit_rsi'],
                silent=True
            )
            
            # Fitness function: Reward profit heavily, but penalize a low win rate or no trades
            if trades == 0:
                score = -100
            else:
                score = profit_pct + (win_rate * 0.1) # Balance profit and consistency
                
            results.append({'params': params, 'score': score, 'profit': profit_pct, 'win_rate': win_rate, 'trades': trades})
            
            print(f"   Variant {i+1} | Score: {score:>7.2f} | P/L: {profit_pct:>6.2f}% | Win Rate: {win_rate:>5.1f}% | Trades: {trades:>2}")
            
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Track absolute best
        if results[0]['score'] > best_overall_score:
            best_overall_score = results[0]['score']
            best_overall_params = results[0].copy()
            print(f"   🏆 NEW BEST! P/L: {best_overall_params['profit']:>6.2f}%, Win Rate: {best_overall_params['win_rate']:>5.1f}%")
            
        # Select top 2 to survive and reproduce
        survivors = [results[0]['params'], results[1]['params']]
        
        # Create next generation
        population = survivors.copy() # Elitism (keep the best 2)
        while len(population) < population_size:
            parent = random.choice(survivors)
            child = mutate(parent)
            population.append(child)
            
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  EVOLUTION COMPLETE — OPTIMAL PARAMETERS FOUND                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"   Win Rate : {best_overall_params['win_rate']:.1f}%")
    print(f"   Profit   : {best_overall_params['profit']:+.2f}%")
    print(f"   Trades   : {best_overall_params['trades']}")
    print("   Optimal Hyperparameters:")
    for k, v in best_overall_params['params'].items():
        if isinstance(v, float):
            print(f"     - {k}: {v:.3f}")
        else:
            print(f"     - {k}: {v}")

if __name__ == '__main__':
    run_evolution()
