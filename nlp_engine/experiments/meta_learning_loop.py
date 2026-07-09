#!/usr/bin/env python3
"""
Meta-Learning Loop: Hyperion autonomously learning to solve new mathematical problems
by generating code and writing it to its own brain.
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
from cain_brain import CainBrain

def run_meta_learning():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Hyperion Meta-Learning Loop (Self-Modifying Architecture)       ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    brain = CainBrain()
    
    print(f"Initializing MetaCortex...")
    print(f"   Current capabilities: {', '.join(brain.meta.capabilities)}")
    
    # Curriculum of problems. 
    # Some are known (arithmetic), some require writing new code.
    curriculum = [
        {"problem": "calculate 25 * 4", "expected": 100},
        {"problem": "solve x^2 - 9 = 0", "expected": (3.0, -3.0)}, # Note: solver returns +3, -3
        {"problem": "gcd of 1071 and 462", "expected": 21}, # Needs code generation
        {"problem": "factorial of 7", "expected": 5040}, # Needs code generation
        {"problem": "sum of squares from 1 to 10", "expected": 385}, # Needs code generation
        {"problem": "combination 10, 3", "expected": 120} # Needs code generation
    ]
    
    for item in curriculum:
        problem = item["problem"]
        expected = item["expected"]
        
        print(f"\n📝 Problem: {problem}")
        print(f"   Expected: {expected}")
        
        # Self-Improve loop
        print(f"    Attempting to solve...")
        result = brain.self_improve(problem, expected)
        
        if result.get("already_correct"):
            print(f"   Solved using existing capability!")
            print(f"   Answer: {result['result']}")
        elif result.get("improved"):
            print(f"   Capability lacked! Hyperion generated NEW CODE to solve this.")
            print(f"   Answer: {result['result']} (Matches Expected! )")
            print(f"   [Generated Code]:\n{result.get('new_code', '').strip()}")
            
            # Extract the capability name
            cap_name = problem.split()[0].lower()
            
            # Teach it permanently
            print(f"   💾 Integrating '{cap_name}' into MetaCortex templates...")
            brain.teach_capability(cap_name, result.get('new_code', ''), problem)
            
        else:
            print(f"   Failed to solve. Error: {result.get('error', 'Unknown')}")
            if result.get('result') is not None:
                print(f"   Got: {result['result']}")
        
        time.sleep(1)

    print("\n" + "═" * 66)
    print("🏁 META-LEARNING COMPLETE")
    print(f"   Updated Capabilities ({len(brain.meta.capabilities)}): {', '.join(brain.meta.capabilities)}")
    print("═" * 66)

if __name__ == '__main__':
    run_meta_learning()
