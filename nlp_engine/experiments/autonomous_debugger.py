#!/usr/bin/env python3
"""
Autonomous Debugger Test
Demonstrates Hyperion fixing "real world" bugs like Off-By-One errors 
and wrong operators completely on its own using AST Mutation.
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.debugger_lobe import AutonomousDebugger

def run_bug_hunting():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Hyperion Autonomous Debugger (Genetic Program Repair)           ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    debugger = AutonomousDebugger()
    
    # ----------------------------------------------------------------
    # BUG 1: The Off-By-One Error
    # ----------------------------------------------------------------
    print("🐞 BUG 1: Off-By-One Error in 'find_last_index'")
    
    # Intentionally buggy code: returns -1 because len(lst) is out of bounds
    buggy_code_1 = '''
def find_last_index(lst, target):
    for i in range(len(lst), -1, -1):
        if lst[i] == target:
            return i
    return -1
'''
    
    test_code_1 = '''
try:
    idx = find_last_index([10, 20, 30, 20], 20)
    result = (idx == 3)
except IndexError:
    result = False
'''
    
    print("   [Original Buggy Code]:")
    print("   " + buggy_code_1.strip().replace('\n', '\n   '))
    print("\n   Analyzing and mutating Abstract Syntax Tree...")
    
    start_t = time.time()
    repair_result = debugger.debug(buggy_code_1, test_code_1, max_generations=500)
    elapsed = time.time() - start_t
    
    if repair_result['success']:
        print(f"   BUG FIXED in {repair_result['attempts']} generations! ({elapsed:.2f}s)")
        print(f"   [Fixed Code]:")
        print("   " + repair_result['fixed_code'].strip().replace('\n', '\n   '))
    else:
        print(f"   Failed to fix bug: {repair_result.get('error')}")
        
    time.sleep(1)

    # ----------------------------------------------------------------
    # BUG 2: Wrong Operator (Classic >= vs >)
    # ----------------------------------------------------------------
    print("\n" + "─" * 70)
    print("\n🐞 BUG 2: Logical Operator Error in 'is_eligible_for_discount'")
    
    # Intentionally buggy code: discount is only for >= 10, but code says > 10
    buggy_code_2 = '''
def is_eligible_for_discount(items_in_cart):
    # Rule: Customers with 10 or more items get a discount
    if items_in_cart > 10:
        return True
    else:
        return False
'''
    
    test_code_2 = '''
# 10 items SHOULD trigger the discount
try:
    ans = is_eligible_for_discount(10)
    result = (ans is True)
except Exception:
    result = False
'''
    
    print("   [Original Buggy Code]:")
    print("   " + buggy_code_2.strip().replace('\n', '\n   '))
    print("\n   Mutating comparators...")
    
    start_t = time.time()
    repair_result2 = debugger.debug(buggy_code_2, test_code_2, max_generations=500)
    elapsed = time.time() - start_t
    
    if repair_result2['success']:
        print(f"   BUG FIXED in {repair_result2['attempts']} generations! ({elapsed:.2f}s)")
        print(f"   [Fixed Code]:")
        print("   " + repair_result2['fixed_code'].strip().replace('\n', '\n   '))
    else:
        print(f"   Failed to fix bug: {repair_result2.get('error')}")


if __name__ == '__main__':
    run_bug_hunting()
