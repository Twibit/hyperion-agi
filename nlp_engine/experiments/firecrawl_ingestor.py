import sys
import os
import time
import urllib.request
import json
import re
import pickle
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
from cain_brain import CainBrain

FIRECRAWL_KEY = "fc-eb008efd07a24fd2b0dc659162b603b4"

def synaptic_pruner(brain, min_count=2):
    """Biological RAM Optimization: Triggers apoptosis on weak N-grams."""
    pruned_count = 0
    for domain, bank in brain.banks.items():
        for order, contexts in enumerate(bank.counts):
            for ctx, next_words in list(contexts.items()):
                words_to_delete = []
                for word, count in next_words.items():
                    if count < min_count:
                        words_to_delete.append(word)
                        pruned_count += 1
                
                for word in words_to_delete:
                    del next_words[word]
                    
                if not next_words:
                    del contexts[ctx]
    return pruned_count

import random

def fetch_firecrawl_random(brain):
    """Fetches a random Wikipedia page via Firecrawl to bypass Wikimedia rate limits."""
    url = "https://api.firecrawl.dev/v2/scrape"
    
    # We append a random number to the URL. Firecrawl caches based on exact URL string.
    # By making the URL unique every time, we force Firecrawl to physically click "Random" on Wikipedia!
    cache_buster = random.randint(1, 999999999)
    target_url = f"https://en.wikipedia.org/wiki/Special:Random?buster={cache_buster}"
    
    payload = json.dumps({
        "url": target_url,
    }).encode('utf-8')
    
    req = urllib.request.Request(url, data=payload, headers={
        'Authorization': f'Bearer {FIRECRAWL_KEY}',
        'Content-Type': 'application/json'
    })
    
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data.get('success') and 'data' in data:
                text = data['data'].get('markdown', '')
                title = data['data'].get('metadata', {}).get('title', 'Unknown')
                
                # Clean up markdown links and tags
                text = re.sub(r'\[.*?\]\(.*?\)', '', text) 
                text = re.sub(r'\[.*?\]', '', text)
                
                if len(text) > 100:
                    brain.learn(text, domain='prose', source=title)
                    return title
    except Exception as e:
        print(f"  [Firecrawl Error]: {e}")
    return None

import concurrent.futures

def scrape_worker(brain):
    """Worker thread that continuously fetches and learns."""
    return fetch_firecrawl_random(brain)

def firecrawl_swarm_ingestion(articles_to_learn: int = 100, threads: int = 10):
    print("╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  Hyperion HYPER-SPEED Swarm Ingestion (Firecrawl {threads}x Threads)    ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    brain = CainBrain()
    print(f"Hyperion is launching {threads} concurrent Firecrawl bots...")
    
    successful = 0
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(scrape_worker, brain) for _ in range(articles_to_learn)]
        
        for future in concurrent.futures.as_completed(futures):
            title = future.result()
            if title:
                successful += 1
                elapsed = time.time() - start_time
                print(f"  Ingested '{title}' | Synapse Count: {brain.banks['prose'].synapse_count} | Uptime: {elapsed:.1f}s")
                
            if successful % 20 == 0:
                pruned = synaptic_pruner(brain, min_count=2)
                print(f"  Synaptic Pruner (Apoptosis) removed {pruned} weak noise pathways.")
            
    print("\nFirecrawl Ingestion Complete!")
    
    # Save the mathematical brain state (converting defaultdicts to normal dicts to avoid lambda pickle errors)
    for domain, bank in brain.banks.items():
        for i in range(len(bank.counts)):
            bank.counts[i] = {k: dict(v) for k, v in bank.counts[i].items()}
            
    save_path = "hyperion_compressed_brain.pkl"
    print(f"💾 Consolidating synaptic weights into {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(brain, f)
        
    final_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"🎉 Success! Brain State compressed down to {final_size_mb:.2f} MB!")
    
    print("\nChecking what Hyperion learned...")
    output = brain.speak("The history of", domain="prose", length=30)
    print(f"\n[Hyperion's internal thought]: {output}")

if __name__ == '__main__':
    firecrawl_swarm_ingestion(500, threads=20) # 20x concurrent threads, pulling 500 fresh articles!
