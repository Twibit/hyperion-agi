#!/usr/bin/env python3
import sys
import os
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.cain_brain import CainBrain

def setup_isolated_environment():
    sandbox_dir = "/tmp/cain_isolated_env"
    
    # Clean and recreate sandbox
    if os.path.exists(sandbox_dir):
        shutil.rmtree(sandbox_dir)
    os.makedirs(sandbox_dir)
    
    # Complex Bug Scenario: A python data processor with missing imports, 
    # key errors, and type casting failures reading from a JSON file.
    
    broken_code = """
def process_data(data):
    total = 0
    # Bug 1: Tries to add a string to an integer
    for item in data['items']:
        total += item['price']
    return total

if __name__ == '__main__':
    # Bug 2: Missing 'import json' causes NameError
    with open('config.json') as f:
        data = json.loads(f.read())
    print("Total:", process_data(data))
"""
    
    json_data = """{
    "items": [
        {"name": "server_node", "price": "500"},
        {"name": "database_node", "price": "1200"}
    ]
}"""

    with open(os.path.join(sandbox_dir, "processor.py"), "w") as f:
        f.write(broken_code.strip())
        
    with open(os.path.join(sandbox_dir, "config.json"), "w") as f:
        f.write(json_data.strip())
        
    return sandbox_dir

def run_isolated_bench():
    print("==================================================================")
    print("  HYPERION V2 - COMPLEX REAL-WORLD TERMINAL-BENCH                  ")
    print("==================================================================\n")
    
    sandbox_dir = setup_isolated_environment()
    print(f"[System]: Created Isolated Sandbox at {sandbox_dir}")
    print("[System]: Seeded sandbox with a complex multi-file bug (processor.py & config.json).")
    
    brain = CainBrain.awaken()
    
    task_prompt = f"cd {sandbox_dir} && python3 processor.py. Fix the bugs in processor.py so it correctly calculates the total price."
    print(f"\n[Terminal-Bench Task]: {task_prompt}")
    
    brain.speak(task_prompt)
    
if __name__ == "__main__":
    run_isolated_bench()
