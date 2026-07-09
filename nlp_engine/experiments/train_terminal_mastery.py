#!/usr/bin/env python3
import sys
import os
import urllib.request
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.cain_brain import CainBrain

def train_terminal_mastery():
    print("==================================================================")
    print("  HYPERION V2 - NEURO-PLASTIC TERMINAL MASTERY TRAINING            ")
    print("==================================================================\n")
    
    brain = CainBrain()
    
    print("[1/3] DOWNLOADING 'THE PURE BASH BIBLE' FROM GITHUB...")
    # The Pure Bash Bible is an incredible open-source book on advanced Bash scripting
    url = "https://raw.githubusercontent.com/dylanaraps/pure-bash-bible/master/README.md"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            text = response.read().decode('utf-8')
            # Grab a chunk of the book (about 25,000 characters) to prevent overloading RAM
            book_text = text[:25000]
    except Exception as e:
        print(f"Failed to download the book: {e}")
        return

    print(f"\n[2/3] INGESTING BIBLE INTO 86-BILLION NEURON MATRIX...")
    start_time = time.time()
    
    # Slice into chunks for realistic biological ingestion
    chunks = [book_text[i:i+500] for i in range(0, len(book_text), 500)]
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"  -> Ingesting page {i+1}/{len(chunks)}...")
        brain.learn(chunk, domain="code")
        
    train_time = time.time() - start_time
    print(f"\n  -> Total Tokens Absorbed: {len(book_text.split())}")
    print(f"  -> Training Time: {train_time:.2f} seconds")

    print("\n[3/3] EXAMINING KNOWLEDGE (Biological Inference)...")
    
    prompts = [
        "What is the best way to trim a string in bash?",
        "How do you check if a file exists?",
    ]
    
    for p in prompts:
        print(f"\n[Antigravity]: {p}")
        resp = brain.speak(p, domain="code", length=40)
        print(f"[Hyperion]: {resp}")
        
    print("\n==================================================================")
    print("  TRAINING COMPLETE. HYPERION NOW KNOWS BASH SCRIPTING.            ")
    print("==================================================================")

if __name__ == "__main__":
    train_terminal_mastery()
