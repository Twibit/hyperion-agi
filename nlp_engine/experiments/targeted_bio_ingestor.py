import sys
import os
import time
import urllib.request
import json
import re
import pickle
import concurrent.futures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
from cain_brain import CainBrain

FIRECRAWL_KEY = "fc-eb008efd07a24fd2b0dc659162b603b4"

TARGET_URLS = [
    "https://en.wikipedia.org/wiki/Flash_Boys",
    "https://en.wikipedia.org/wiki/Liar%27s_Poker",
    "https://en.wikipedia.org/wiki/The_Black_Swan:_The_Impact_of_the_Highly_Improbable",
    "https://en.wikipedia.org/wiki/Fooled_by_Randomness",
    "https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow",
    "https://en.wikipedia.org/wiki/Market_microstructure",
    "https://en.wikipedia.org/wiki/Technical_analysis",
    "https://en.wikipedia.org/wiki/Algorithmic_trading",
    "https://en.wikipedia.org/wiki/Day_trading",
    "https://en.wikipedia.org/wiki/Behavioral_economics",
    "https://en.wikipedia.org/wiki/Arbitrage",
    "https://en.wikipedia.org/wiki/Statistical_arbitrage",
    "https://en.wikipedia.org/wiki/Order_book",
    "https://en.wikipedia.org/wiki/Order_(exchange)",
    "https://en.wikipedia.org/wiki/Dark_pool",
    "https://en.wikipedia.org/wiki/Market_maker",
    "https://en.wikipedia.org/wiki/High-frequency_trading",
    "https://en.wikipedia.org/wiki/Quantitative_analysis_(finance)",
    "https://en.wikipedia.org/wiki/Automated_trading_system"
]

def fetch_specific_firecrawl(target_url):
    """Fetches a specific Wikipedia page via Firecrawl (Thread-Safe)."""
    url = "https://api.firecrawl.dev/v2/scrape"
    payload = json.dumps({"url": target_url}).encode('utf-8')
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
                    return title, text
    except Exception as e:
        print(f"  [Firecrawl Error]: {e}")
    return None, None

def targeted_bio_ingestion():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Hyperion TARGETED Bio-Swarm Ingestion (Finance & Trading)        ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    save_path = "hyperion_compressed_brain.pkl"
    if os.path.exists(save_path):
        print(f"Loading existing AI/CS Brain State from {save_path} to compound knowledge...")
        with open(save_path, 'rb') as f:
            brain = pickle.load(f)
    else:
        brain = CainBrain()
        
    print(f"Hyperion is launching 5 concurrent bots to learn {len(TARGET_URLS)} core subjects...")
    
    successful = 0
    start_time = time.time()
    
    # Use 5 threads for HTTP fetching (IO Bound), but ingest sequentially (CPU Bound) for thread safety!
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_specific_firecrawl, url): url for url in TARGET_URLS}
        
        for future in concurrent.futures.as_completed(futures):
            title, text = future.result()
            if title and text:
                try:
                    brain.learn(text, domain='prose', source=title)
                    successful += 1
                    elapsed = time.time() - start_time
                    print(f"  Learned Core Subject: '{title}' | Synapse Count: {brain.banks['prose'].synapse_count} | Uptime: {elapsed:.1f}s")
                except Exception as e:
                    print(f"  [Learn Error]: {e}")
            
    print("\nTargeted Curriculum Complete!")
    
    # Save the mathematical brain state
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
    output = brain.speak("Artificial intelligence is", domain="prose", length=30)
    print(f"\n[Hyperion's internal thought]: {output}")

if __name__ == '__main__':
    targeted_bio_ingestion()
