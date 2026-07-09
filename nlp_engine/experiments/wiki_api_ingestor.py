import sys
import os
import time
import urllib.request
import json
import re
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
from cain_brain import CainBrain

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

def fetch_random_wikipedia_batch(limit=20):
    """Fetches a batch of random Wikipedia articles instantly using a single API call."""
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&generator=random&grnnamespace=0&grnlimit={limit}&prop=extracts&exchars=3000&explaintext"
    req = urllib.request.Request(url, headers={'User-Agent': 'Hyperion-Mobile/1.0'})
    
    articles = []
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            for page_id in pages:
                title = pages[page_id].get('title', 'Unknown')
                extract = pages[page_id].get('extract', '')
                if extract:
                    articles.append((title, extract))
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("  [Rate Limited] Pausing for 5 seconds...")
            time.sleep(5)
        else:
            print(f"  [HTTP Error]: {e.code}")
    except Exception as e:
        print(f"  [Network Error]: {e}")
    return articles

def live_stream_learning(batches_to_learn: int = 5):
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Hyperion HYPER-SPEED Internet Ingestion Stream (Wikipedia API)   ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    brain = CainBrain()
    print("Hyperion is online and connecting to the internet...")
    
    total_articles = 0
    for i in range(batches_to_learn):
        print(f"\n[Batch {i+1}/{batches_to_learn}] Fetching 20 articles at once...")
        start = time.time()
        articles = fetch_random_wikipedia_batch(limit=20)
        
        for title, text in articles:
            # Strip wiki markup brackets if any
            text = re.sub(r'\[.*?\]', '', text)
            # Learn the text
            brain.learn(text, domain='prose', source=title)
            total_articles += 1
            
        elapsed = time.time() - start
        print(f"  Ingested 20 articles in {elapsed:.2f} seconds! Synapse Count: {brain.banks['prose'].synapse_count}")
        
        # Prune every 10 batches (200 articles) to save memory
        if (i + 1) % 10 == 0:
            pruned = synaptic_pruner(brain, min_count=2)
            print(f"  Synaptic Pruner (Apoptosis) removed {pruned} weak noise pathways.")
            
        time.sleep(1.5) # Be polite to Wikipedia's API
            
    print("\nIngestion Complete!")
    print(f"Total Articles Learned: {total_articles}")
    
    # Save the mathematical brain state
    save_path = "hyperion_compressed_brain.pkl"
    print(f"💾 Consolidating synaptic weights into {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(brain, f)
        
    final_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"🎉 Success! Brain State compressed down to {final_size_mb:.2f} MB!")

if __name__ == '__main__':
    live_stream_learning(500) # 500 batches = 10,000 articles
