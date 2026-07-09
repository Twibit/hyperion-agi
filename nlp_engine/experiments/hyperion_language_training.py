#!/usr/bin/env python3
"""
Project HYPERION: Language Fluency Training Loop
=================================================
Continuous training loop that:
  1. Feeds Wikipedia articles into the Language Cortex (Broca's Area)
  2. Every N articles, generates text and grades it (Wernicke's Area)
  3. Applies Dopamine/Cortisol neuromodulation based on grade
  4. Periodically runs sleep consolidation (Hippocampal replay)
  5. Tracks progress toward A+ academic fluency

The organism learns by reading real human language and building an internal
statistical model of word transitions — exactly how a biological brain
acquires language through exposure.
"""

import sys
import os
import time
import urllib.request
import json
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.language_cortex import HyperionLanguageCortex


def fetch_random_wikipedia():
    """Fetch a random Wikipedia article as sensory input."""
    try:
        url = ("https://en.wikipedia.org/w/api.php?format=json&action=query"
               "&generator=random&grnnamespace=0&prop=extracts&exchars=3000&explaintext")
        req = urllib.request.Request(url, headers={'User-Agent': 'Hyperion-LangCortex/4.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data['query']['pages']
            for page_id in pages:
                return pages[page_id].get('title', '?'), pages[page_id].get('extract', '')
    except Exception:
        return None, None


def grade_to_letter(score):
    """Convert numeric score to letter grade."""
    if score >= 80:
        return "A+"
    elif score >= 60:
        return "A"
    elif score >= 40:
        return "B+"
    elif score >= 20:
        return "B"
    elif score >= 10:
        return "C"
    elif score >= 0:
        return "D"
    else:
        return "F"


def print_status_bar(cortex):
    """Print a compact status line."""
    s = cortex.status()
    print(f"  Vocab: {s['vocabulary_size']:,} | "
          f"Synapses: {s['synapse_count']:,} | "
          f"Articles: {s['articles_read']} | "
          f"Tokens: {s['tokens_seen']:,} | "
          f"Temp: {s['temperature']:.2f} | "
          f"DA: {s['dopamine']:.1f} | "
          f"CORT: {s['cortisol']:.1f}")


def run_language_training():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Language Recognition Engine v4.0              ║")
    print("║  Architecture: Broca + Wernicke + Hippocampal Consolidation     ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    cortex = HyperionLanguageCortex(max_order=4)
    
    # === PHASE 1: MASSIVE KNOWLEDGE INGESTION ===
    print("PHASE 1: SENSORY EXPOSURE (Building synaptic foundation)")
    print("   Loading Wikipedia articles into Broca's production area...\n")
    
    initial_articles = 20
    for i in range(initial_articles):
        title, text = fetch_random_wikipedia()
        if not text:
            time.sleep(0.5)
            continue
        n_tokens = cortex.learn(text, source=title or "wiki")
        if n_tokens > 0:
            print(f"  [{i+1}/{initial_articles}] Ingested {n_tokens:,} tokens from '{title}'")
    
    print(f"\n  Initial knowledge base complete.")
    print_status_bar(cortex)
    
    # === PHASE 1.5: SLEEP CONSOLIDATION ===
    print("\n💤 SLEEP CYCLE: Hippocampal replay (strengthening key memories)...")
    replayed = cortex.sleep_consolidate(n_replays=15)
    print(f"  Replayed {replayed} memories through Broca's area.\n")
    
    # === PHASE 2: GENERATE + EVALUATE + LEARN LOOP ===
    print("═" * 68)
    print("🎓 PHASE 2: ACADEMIC TRAINING (Generate → Evaluate → Modulate)")
    print("   Goal: Reach A+ fluency through Dopamine/Cortisol feedback.")
    print("═" * 68 + "\n")
    
    prompts = [
        "The", "In", "However", "During", "Although",
        "Furthermore", "Scientists", "The government",
        "According to", "Throughout history",
        "The development", "Research has", "It was",
        "Many scholars", "The concept"
    ]
    
    consecutive_good = 0
    
    while True:
        # Ingest fresh knowledge every epoch
        title, text = fetch_random_wikipedia()
        if text:
            n = cortex.learn(text, source=title or "wiki")
            if n > 0:
                print(f"  [📖] Read '{title}' ({n:,} tokens)")
        
        # Pick a prompt
        prompt = prompts[cortex.epoch % len(prompts)]
        
        # Generate
        generated = cortex.speak(prompt=prompt, length=80)
        
        # Evaluate
        score, feedback = cortex.evaluate(generated)
        grade = grade_to_letter(score)
        
        # Neuromodulate
        modulation = cortex.neuromodulate(score, generated)
        
        # Display
        epoch = cortex.epoch
        print(f"\n{'─' * 68}")
        print(f"  EPOCH {epoch} | Grade: {grade} (Score: {score}) | Modulation: {modulation}")
        print(f"  Feedback: {feedback}")
        print(f"  ┌─ Generated Text ─────────────────────────────────────────────")
        
        # Word-wrap the text for display
        words = generated.split()
        lines = []
        current_line = "  │ "
        for word in words:
            if len(current_line) + len(word) + 1 > 68:
                lines.append(current_line)
                current_line = "  │ " + word
            else:
                current_line += " " + word if current_line != "  │ " else current_line + word
        lines.append(current_line)
        for line in lines:
            print(line)
        print(f"  └──────────────────────────────────────────────────────────────")
        print_status_bar(cortex)
        
        # Track consecutive good scores
        if score >= 20:
            consecutive_good += 1
        else:
            consecutive_good = 0
        
        # A+ Achievement Check
        if grade == "A+" and consecutive_good >= 3:
            print("\n" + "🌟" * 34)
            print("  🏆 HYPERION HAS ACHIEVED A+ FLUENCY!")
            print("  Three consecutive high-quality generations confirmed.")
            print("  The biological language cortex is fully operational.")
            print("🌟" * 34)
            
            # Generate a showcase
            print("\n  === SHOWCASE GENERATION ===")
            for showcase_prompt in ["The history of", "Scientists have discovered", "In conclusion"]:
                showcase = cortex.speak(prompt=showcase_prompt, length=100, temperature=0.7)
                print(f"\n  Prompt: '{showcase_prompt}'")
                print(f"  Output: {showcase}")
            
            break
        
        # Periodic sleep consolidation
        if epoch % 10 == 0:
            print(f"\n  💤 Sleep consolidation cycle...")
            replayed = cortex.sleep_consolidate(n_replays=10)
            print(f"     Replayed {replayed} memories.")
        
        # Brief pause to not hammer Wikipedia
        time.sleep(1)


if __name__ == "__main__":
    run_language_training()
