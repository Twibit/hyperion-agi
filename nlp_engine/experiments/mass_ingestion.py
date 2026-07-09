#!/usr/bin/env python3
import sys
import os
import urllib.request
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.cain_brain import CainBrain

def mass_ingest():
    print("==================================================================")
    print("  HYPERION V2 - MASS KNOWLEDGE INGESTION & PERSISTENCE             ")
    print("==================================================================\n")
    
    # 1. Awaken the Brain (Loads previous synaptic weights from disk)
    brain = CainBrain.awaken()
    
    print("\n[1/3] LOCATING ADVANCED KNOWLEDGE DATASETS...")
    
    # Let's download a slice of the Pro Git book (for git version control knowledge)
    url_git = "https://raw.githubusercontent.com/progit/progit2/main/book/01-introduction/sections/what-is-git.asc"
    
    # Let's also download a slice of Python documentation
    url_python = "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/controlflow.rst"

    urls = [url_git, url_python]
    corpus = ""
    
    for url in urls:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req) as response:
                corpus += response.read().decode('utf-8')[:15000] + "\n"
        except Exception as e:
            print(f"Failed to download {url}: {e}")

    print(f"\n[2/3] INGESTING {len(corpus.split())} TOKENS INTO BIOLOGICAL MATRIX...")
    start_time = time.time()
    
    chunks = [corpus[i:i+500] for i in range(0, len(corpus), 500)]
    for i, chunk in enumerate(chunks):
        if i % 5 == 0:
            print(f"  -> Ingesting page {i+1}/{len(chunks)}...")
        brain.learn(chunk, domain="code")
        
    train_time = time.time() - start_time
    print(f"\n  -> Training Complete in {train_time:.2f} seconds.")

    print("\n[3/3] INITIATING DEEP SLEEP (Memory Crystallization)...")
    # This will save the 86B neuron weights to disk so it remembers everything tomorrow!
    brain.sleep_and_save()
    
    print("\n==================================================================")
    print("  MASS INGESTION COMPLETE. HYPERION IS ASLEEP AND SMARTER.         ")
    print("==================================================================")

if __name__ == "__main__":
    mass_ingest()
