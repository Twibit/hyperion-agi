import sys
import os
import time
import pickle
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
from cain_brain import CainBrain

try:
    from datasets import load_dataset
except ImportError:
    print("Missing huggingface datasets library! Run: pip install datasets")
    sys.exit(1)

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


def run_huggingface_stream(articles_to_learn: int = 1000):
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Hyperion HYPER-SPEED CDN Streamer (HuggingFace 2026 Dataset)     ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    brain = CainBrain()
    print("Hyperion is online and connecting to the Hugging Face CDN...")
    
    try:
        # Connecting to the latest 2026 Wikipedia monthly dump (or standard if unavailable)
        print("🌐 Initiating HTTP stream for 2026 Wikipedia data...")
        dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    except Exception as e:
        print(f"Failed to connect to dataset stream: {e}")
        return
        
    start_time = time.time()
    articles_processed = 0
    
    # We iterate over the stream without downloading the whole file!
    try:
        for article in dataset:
            if articles_processed >= articles_to_learn:
                break
                
            title = article.get('title', 'Unknown')
            text = article.get('text', '')
            
            if len(text) > 500:
                brain.learn(text, domain='prose', source=title)
                articles_processed += 1
                
                if articles_processed % 50 == 0:
                    elapsed = time.time() - start_time
                    speed = articles_processed / elapsed
                    print(f"Streamed {articles_processed} articles | Speed: {speed:.1f} articles/sec | Synapses: {brain.banks['prose'].synapse_count}")
                    
                if articles_processed % 500 == 0:
                    pruned = synaptic_pruner(brain, min_count=2)
                    print(f"  Synaptic Pruner (Apoptosis) removed {pruned} weak noise pathways.")
                    
    except Exception as e:
        print(f"Stream interrupted: {e}")
        traceback.print_exc()
        
    print("\nStream Ingestion Complete!")
    
    # Save the mathematical brain state
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
    run_huggingface_stream(500) # Learn 500 articles as a test run
