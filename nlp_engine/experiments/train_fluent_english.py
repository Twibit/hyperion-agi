#!/usr/bin/env python3
import sys
import os
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.cain_brain import CainBrain

def teach_english():
    print("==================================================================")
    print("  TEACHING HYPERION NATIVE ENGLISH VIA CLASSIC LITERATURE          ")
    print("==================================================================\n")
    
    brain = CainBrain()
    print("[System]: Connecting to Project Gutenberg...")
    print("[System]: Downloading 'Alice's Adventures in Wonderland' by Lewis Carroll...")
    
    url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            text = response.read().decode('utf-8')
            
        # Strip Gutenberg boilerplate
        start_idx = text.find("CHAPTER I.")
        end_idx = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx]
            
        print("[System]: Text downloaded. Erasing philosophical priors and starting fresh...")
        
        # Split into small chunks to simulate reading pages
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        print(f"[System]: Ingesting {len(chunks)} pages of literature into the 86-Billion Neural Matrix...")
        
        for i, chunk in enumerate(chunks):
            brain.learn(chunk, domain="prose")
            if i % 10 == 0:
                print(f"  ... ingested {i}/{len(chunks)} pages")
                
        print("\n[System]: Dreaming / Consolidating neural pathways (Liquid Synapse Apoptosis)...")
        brain.sleep_consolidate(n_replays=10)
        
        print("\n==================================================================")
        print("  CONVERSATION (POST-LITERATURE INGESTION)                        ")
        print("==================================================================\n")
        
        prompts = [
            "Who are you?",
            "What is a rabbit hole?",
            "How do you feel about time?",
            "What happens when you fall down?"
        ]
        
        for p in prompts:
            print(f"\n[Antigravity]: {p}")
            resp = brain.speak(p, domain="prose", length=25)
            print(f"[Hyperion]: {resp}")
            
    except Exception as e:
        print(f"Failed to download or ingest: {e}")

if __name__ == "__main__":
    teach_english()
