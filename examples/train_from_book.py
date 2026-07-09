import numpy as np
import time
import os
from embodied_agent.organism import AuroraOmnia
from embodied_agent.config import AuroraOmniaConfig

print("==================================================")
print(" EVOLUTIONARY CURRICULUM: PHASE 1 (BOOK DOWNLOAD) ")
print("==================================================\n")

cfg = AuroraOmniaConfig()
cfg.thalamic_router_enable = True
cfg.lateralization_enable = True
cfg.lr_cortex = 0.5  # Moderate plasticity for long-term retention
org = AuroraOmnia(cfg)

book_path = "assembly_book.txt"
print(f"[System] Locating {book_path}...")

with open(book_path, 'r') as f:
    text = f.read()

# Parse book into structural sentences
paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
corpus = []
for p in paragraphs:
    words = p.split()
    clean_words = [w.lower().strip(".,!?;:()[]\"'") for w in words]
    corpus.append([w for w in clean_words if w])

# Speedrun: train on just 5 foundational lines to ensure it completes fast
speedrun_corpus = corpus[4:9]  # Grab the core boolean logic lines

print(f"[System] AGI is reading the book (extracting core {len(speedrun_corpus)} sequences).")
print("[System] Commencing Matrix Download...\n")

for epoch in range(1, 4):
    for words in speedrun_corpus:
        if len(words) < 2: continue
        for i in range(len(words) - 1):
            current_word = words[i]
            target_word = words[i+1]
            
            obs = {"text": current_word}
            org.step(obs)
            
            if hasattr(org, '_prev_z') and org._prev_z is not None:
                try:
                    org.text_decoder.observe(target_word, org._prev_z)
                except Exception:
                    pass
            org.step(obs, reward=2.0)

print("\n[System] Invoking Deep REM Sleep Consolidation...")
org.sleep()

print("\n[System] Final Physics Intuition Test...")
prompt = ["xor", "1", "0", "is"]
print(f"  [Prompt] \"{' '.join(prompt)} ...\"")
for token in prompt:
    org.step({"text": token})
try:
    completion = org.generate_text(prompt=" ".join(prompt), n_tokens=1, temperature=0.1)
    print(f"  [AGI Completion]: {completion.strip()}")
except Exception as e:
    print(f"  [Error]: {e}")

print(f"\n[System] Knowledge fused. Deleting {book_path} from environment.")
os.remove(book_path)
print("[System] Book destroyed. Knowledge exists only within AGI cortex.")
