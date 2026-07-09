import sys
import os
import time
import pickle
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
from cain_brain import CainBrain

try:
    from libzim.reader import Archive
except ImportError:
    print("Missing libzim! Please install it by running: pip install libzim")
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Missing BeautifulSoup! Please install it by running: pip install beautifulsoup4")
    sys.exit(1)


def synaptic_pruner(brain, min_count=2):
    """
    Biological RAM Optimization: Scans the brain's neural pathways and 
    triggers 'apoptosis' (cell death) on N-grams that have only been seen 
    rarely, assuming they are typos or noise.
    """
    pruned_count = 0
    for domain, bank in brain.banks.items():
        for order, contexts in bank.counts.items():
            for ctx, next_words in list(contexts.items()):
                words_to_delete = []
                for word, count in next_words.items():
                    if count < min_count:
                        words_to_delete.append(word)
                        pruned_count += 1
                
                for word in words_to_delete:
                    del next_words[word]
                    
                # Clean up empty contexts
                if not next_words:
                    del contexts[ctx]
                    
    return pruned_count


def scrape_and_learn(zim_path: str, max_articles: int = 100000):
    print(f"Initializing Hyperion Cortex for ZIM Ingestion...")
    brain = CainBrain()
    
    if not os.path.exists(zim_path):
        print(f"Could not find ZIM file at {zim_path}")
        print("Please download 'wikipedia_en_simple_all_nopic.zim' or similar from download.kiwix.org")
        return
        
    print(f"📖 Mounting ZIM Archive: {zim_path}")
    zim = Archive(zim_path)
    total_entries = zim.entry_count
    print(f"Found {total_entries} total entries in archive.")
    
    articles_processed = 0
    start_time = time.time()
    
    for i in range(total_entries):
        if articles_processed >= max_articles:
            break
            
        try:
            entry = zim._get_entry_by_id(i)
            # Namespace 'A' represents actual readable articles
            if entry.namespace == b'A':
                item = entry.get_item()
                raw_html = item.content().tobytes().decode('utf-8', errors='ignore')
                
                # Strip HTML to get pure grammatical text
                soup = BeautifulSoup(raw_html, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                
                # Basic cleaning (remove bracketed citations like [1], [citation needed])
                text = re.sub(r'\[.*?\]', '', text)
                
                if len(text) > 500:  # Only ingest substantial articles
                    brain.learn(text, domain='prose', source=entry.title)
                    articles_processed += 1
                    
                    if articles_processed % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"[{articles_processed}] Ingested '{entry.title}'... ({elapsed:.2f}s)")
                    
                    # Run biological RAM optimization every 5,000 articles
                    if articles_processed % 5000 == 0:
                        print("Running Synaptic Pruner (Apoptosis) to free RAM...")
                        pruned = synaptic_pruner(brain, min_count=2)
                        print(f"🧹 Pruned {pruned} weak neural pathways (noise/typos).")
                        
        except Exception as e:
            # Skip corrupted or unreadable entries silently
            continue
            
    print("\nIngestion Complete!")
    print(f"Total Articles Learned: {articles_processed}")
    
    # Save the mathematical brain state to a tiny compressed file
    save_path = "hyperion_compressed_brain.pkl"
    print(f"💾 Consolidating synaptic weights into {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(brain, f)
        
    final_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"🎉 Success! 50GB Wikipedia compressed down to a {final_size_mb:.2f} MB Brain State!")

if __name__ == '__main__':
    print("Welcome to the Hyperion ZIM Ingestor")
    zim_file = input("Enter the path to your .zim file (e.g., wikipedia.zim): ")
    scrape_and_learn(zim_file)
