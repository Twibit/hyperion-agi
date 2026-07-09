#!/usr/bin/env python3
"""
Meta-Cortex Prediction Evolution
Demonstrates Hyperion encountering chaotic/non-linear sequences and 
autonomously writing curve-fitting algorithms to predict the future.
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
from cain_brain import CainBrain

def run_prediction_demo():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Hyperion Meta-Cortex — Evolving Predictive Functions            ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    brain = CainBrain()
    
    # 1. Simple Sequence (Built-in knowledge)
    seq1 = [5, 10, 15, 20]
    print(f"Test 1: Simple Sequence {seq1}")
    result = brain.predict_sequence(seq1)
    print(f"   Using Base Instincts (NumericalCortex): Detected '{result['type']}'")
    print(f"   Predicted Next: {result['next_predicted']}")
    
    time.sleep(1)
    
    # 2. Complex Non-Linear Sequence (Squares: n^2)
    # 1, 4, 9, 16, 25... Next should be 36
    seq2 = [1, 4, 9, 16, 25]
    print(f"\nTest 2: Quadratic Sequence {seq2}")
    result = brain.predict_sequence(seq2)
    print(f"   Base Instincts Output: {result['type']} (confidence: {result['confidence']})")
    
    if result['confidence'] < 0.5 or result['type'] == 'random':
        print(f"   ⚠️ Base instincts failed. Triggering MetaCortex to write custom predictive algorithm...")
        
        # Ask MetaCortex to solve it and self-improve
        problem = f"predict next in sequence: {seq2}"
        meta_result = brain.self_improve(problem, 36)
        
        if meta_result.get("improved"):
            print(f"   MetaCortex successfully synthesized a curve-fitting script!")
            print(f"   Predicted Next: {meta_result['result']} (Expected: 36)")
            print(f"\n[Generated Python Script]:\n{meta_result['new_code'].strip()}")
        else:
            print(f"   MetaCortex synthesis failed. Error: {meta_result.get('error', 'Unknown')}")
            
    time.sleep(1)
    
    # 3. Chaotic Cubic Sequence (Cubes: n^3)
    # 1, 8, 27, 64, 125... Next should be 216
    seq3 = [1, 8, 27, 64, 125]
    print(f"\nTest 3: Cubic Sequence {seq3}")
    problem = f"predict next in sequence: {seq3}"
    meta_result = brain.self_improve(problem, 216)
    
    if meta_result.get("improved") or meta_result.get("already_correct"):
        print(f"   MetaCortex successfully applied polynomial regression!")
        print(f"   Predicted Next: {meta_result['result']} (Expected: 216)")
    else:
        print(f"   MetaCortex synthesis failed. Error: {meta_result.get('error', 'Unknown')}")

if __name__ == '__main__':
    run_prediction_demo()
