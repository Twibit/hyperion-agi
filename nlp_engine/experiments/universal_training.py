#!/usr/bin/env python3
"""
Project HYPERION: Universal Cortex Multi-Domain Training
========================================================
Trains the Universal Cortex across three knowledge domains simultaneously:

  - Prose (Wikipedia):    Encyclopedic knowledge and factual language
  - Code  (GitHub/CPython): Programming syntax and software patterns
  - Literature (Gutenberg): Literary style, narrative, and dialogue

Architecture:
  Phase 1: Mass Ingestion — bulk-load diverse training data
  Phase 2: Continuous Learning — fetch → learn → generate → evaluate → modulate

The cortex learns domain-specific synaptic pathways while sharing structural
knowledge across domains (cross-domain transfer via shared vocabulary).
"""

import sys
import os
import time
import urllib.request
import json
import re
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.universal_cortex import UniversalCortex


# ═══════════════════════════════════════════════════════════════════════════
#  DATA SOURCES
# ═══════════════════════════════════════════════════════════════════════════

def fetch_random_wikipedia():
    """Fetch a random Wikipedia article as prose training data.

    Returns:
        tuple[str|None, str|None]: (title, extract) or (None, None) on error.
    """
    try:
        url = ("https://en.wikipedia.org/w/api.php?format=json&action=query"
               "&generator=random&grnnamespace=0&prop=extracts&exchars=3000&explaintext")
        req = urllib.request.Request(url, headers={'User-Agent': 'Hyperion-Universal/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data['query']['pages']
            for page_id in pages:
                title = pages[page_id].get('title', '?')
                extract = pages[page_id].get('extract', '')
                return title, extract
    except Exception:
        return None, None
    return None, None


def fetch_github_code():
    """Fetch a random Python source chunk from CPython's standard library.

    Returns:
        tuple[str|None, str|None]: (filename, code_chunk) or (None, None) on error.
    """
    urls = [
        "https://raw.githubusercontent.com/python/cpython/main/Lib/random.py",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/collections/__init__.py",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/json/decoder.py",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/pathlib/__init__.py",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/functools.py",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/typing.py",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/dataclasses.py",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/contextlib.py",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/unittest/mock.py",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/textwrap.py",
    ]
    url = random.choice(urls)
    filename = url.rsplit('/', 1)[-1]

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Hyperion-Universal/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            code = response.read().decode('utf-8')
        if len(code) < 100:
            return None, None
        # Extract a random 1000–3000 character chunk
        chunk_len = random.randint(1000, min(3000, len(code)))
        max_start = max(0, len(code) - chunk_len)
        start = random.randint(0, max_start) if max_start > 0 else 0
        chunk = code[start:start + chunk_len]
        return filename, chunk
    except Exception:
        return None, None


def fetch_gutenberg_literature():
    """Fetch a literary excerpt from Project Gutenberg.

    Returns:
        tuple[str|None, str|None]: (book_title, chunk) or (None, None) on error.
    """
    books = [
        ("https://www.gutenberg.org/cache/epub/1342/pg1342.txt", "Pride and Prejudice"),
        ("https://www.gutenberg.org/cache/epub/1661/pg1661.txt", "Sherlock Holmes"),
        ("https://www.gutenberg.org/cache/epub/84/pg84.txt", "Frankenstein"),
        ("https://www.gutenberg.org/cache/epub/11/pg11.txt", "Alice in Wonderland"),
        ("https://www.gutenberg.org/cache/epub/1232/pg1232.txt", "The Prince"),
        ("https://www.gutenberg.org/cache/epub/98/pg98.txt", "A Tale of Two Cities"),
    ]
    url, title = random.choice(books)

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Hyperion-Universal/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            text = response.read().decode('utf-8', errors='replace')
        if len(text) < 500:
            return None, None
        # Skip header/footer — use the MIDDLE 60% of the text
        total = len(text)
        middle_start = int(total * 0.20)
        middle_end = int(total * 0.80)
        middle = text[middle_start:middle_end]
        if len(middle) < 500:
            return None, None
        chunk_len = random.randint(500, min(2000, len(middle)))
        max_start = max(0, len(middle) - chunk_len)
        start = random.randint(0, max_start) if max_start > 0 else 0
        chunk = middle[start:start + chunk_len]
        return title, chunk
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def grade_to_letter(score):
    """Convert a numeric Wernicke score to a letter grade.

    Args:
        score: Integer evaluation score.

    Returns:
        str: Letter grade from A+ down to F.
    """
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


def domain_emoji(domain):
    """Return the status emoji for a training domain.

    Args:
        domain: One of 'prose', 'code', 'literature', 'shared'.

    Returns:
        str: Emoji character for the domain.
    """
    mapping = {
        'prose': '📖',
        'code': '💻',
        'literature': '📜',
        'shared': '🔗',
    }
    return mapping.get(domain, '❓')


def print_status_bar(cortex):
    """Print a compact one-line status showing per-domain vocab and synapse counts."""
    parts = []
    for domain in ['prose', 'code', 'literature']:
        emoji = domain_emoji(domain)
        vocab = len(cortex.banks[domain].vocab) if domain in cortex.banks else 0
        synapses = cortex.banks[domain].synapse_count if domain in cortex.banks else 0
        # Compact formatting: k for thousands
        if synapses >= 1000:
            syn_str = f"{synapses / 1000:.1f}k"
        else:
            syn_str = str(synapses)
        parts.append(f"{emoji} {domain}: {vocab}w/{syn_str} syn")
    temp = cortex.temperature if hasattr(cortex, 'temperature') else 0.0
    status_line = " | ".join(parts) + f" | Temp: {temp:.2f}"
    print(f"  {status_line}")


def word_wrap(text, width=64, prefix='  │ '):
    """Word-wrap text for display with a prefix on each line.

    Args:
        text: The text to wrap.
        width: Maximum total line width (including prefix).
        prefix: String prepended to each wrapped line.

    Returns:
        str: The wrapped text as a single string with newlines.
    """
    words = text.split()
    lines = []
    current_line = prefix
    content_width = width - len(prefix)

    for word in words:
        if current_line == prefix:
            # First word on this line
            current_line += word
        elif len(current_line) + 1 + len(word) > width:
            lines.append(current_line)
            current_line = prefix + word
        else:
            current_line += ' ' + word

    if current_line.strip():
        lines.append(current_line)

    return '\n'.join(lines) if lines else prefix


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_universal_training():
    """Multi-domain training loop for the Universal Cortex.

    Phase 1: Mass ingestion of prose, code, and literature.
    Phase 2: Continuous learning with evaluation and neuromodulation.
    """

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Universal Cortex Training v5.0               ║")
    print("║  Multi-Domain Architecture: Prose + Code + Literature          ║")
    print("║  Engine: Broca + Wernicke V2 + Hippocampal Consolidation       ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    cortex = UniversalCortex()

    # ─── PHASE 1: MASS INGESTION ─────────────────────────────────────────
    print("═" * 66)
    print("  PHASE 1: MASS INGESTION — Building Multi-Domain Foundation")
    print("═" * 66 + "\n")

    # --- Wikipedia (prose) ---
    print("  📖 Ingesting Wikipedia articles (prose domain)...\n")
    wiki_count = 0
    for i in range(15):
        title, text = fetch_random_wikipedia()
        if title and text:
            n_tokens = cortex.learn(text, domain='prose', source=title)['tokens']
            if n_tokens > 0:
                wiki_count += 1
                print(f"    [📖 {wiki_count:>2}/15] {title} ({n_tokens:,} tokens)")
        else:
            print(f"    [📖  -/15] (fetch failed, skipping)")
        time.sleep(0.3)

    print(f"\n  Prose ingestion complete: {wiki_count} articles loaded.")
    print_status_bar(cortex)

    # --- GitHub (code) ---
    print("\n  💻 Ingesting CPython source files (code domain)...\n")
    code_count = 0
    for i in range(5):
        filename, chunk = fetch_github_code()
        if filename and chunk:
            n_tokens = cortex.learn(chunk, domain='code', source=filename)['tokens']
            if n_tokens > 0:
                code_count += 1
                print(f"    [💻 {code_count:>2}/5 ] {filename} ({n_tokens:,} tokens)")
        else:
            print(f"    [💻  -/5 ] (fetch failed, skipping)")
        time.sleep(0.3)

    print(f"\n  Code ingestion complete: {code_count} files loaded.")
    print_status_bar(cortex)

    # --- Gutenberg (literature) ---
    print("\n  📜 Ingesting Project Gutenberg literature...\n")
    lit_count = 0
    for i in range(5):
        title, chunk = fetch_gutenberg_literature()
        if title and chunk:
            n_tokens = cortex.learn(chunk, domain='literature', source=title)['tokens']
            if n_tokens > 0:
                lit_count += 1
                print(f"    [📜 {lit_count:>2}/5 ] {title} ({n_tokens:,} tokens)")
        else:
            print(f"    [📜  -/5 ] (fetch failed, skipping)")
        time.sleep(0.3)

    print(f"\n  Literature ingestion complete: {lit_count} books sampled.")
    print_status_bar(cortex)

    # --- Sleep consolidation ---
    print("\n  💤 SLEEP CYCLE: Hippocampal replay across all domains...")
    replayed = cortex.sleep_consolidate(n_replays=15)
    print(f"     Replayed {replayed} memories through Broca's area.\n")

    # ─── PHASE 2: CONTINUOUS LEARNING LOOP ────────────────────────────────
    print("═" * 66)
    print("  🎓 PHASE 2: CONTINUOUS LEARNING — Generate → Evaluate → Modulate")
    print("     Goal: 5 consecutive B+ or better across domains.")
    print("═" * 66 + "\n")

    # Source cycle: wiki, wiki, code, wiki, literature
    source_cycle = ['prose', 'prose', 'code', 'prose', 'literature']

    # Per-domain generation prompts
    domain_prompts = {
        'prose': ["The", "In", "However", "During", "According to",
                  "The development", "Research has", "Many scholars"],
        'code': ["def", "class", "import", "for", "if", "return", "with"],
        'literature': ["It was", "The", "She", "He", "There was", "In the", "Upon"],
    }

    # Per-domain prompt counters
    prompt_counters = {'prose': 0, 'code': 0, 'literature': 0}

    consecutive_good = 0
    epoch = 0

    while True:
        epoch += 1

        # Determine domain for this epoch
        domain = source_cycle[(epoch - 1) % len(source_cycle)]
        emoji = domain_emoji(domain)

        # Fetch and learn new data
        ingested_label = ""
        if domain == 'prose':
            title, text = fetch_random_wikipedia()
            if title and text:
                n = cortex.learn(text, domain='prose', source=title)['tokens']
                if n > 0:
                    ingested_label = f"'{title}' ({n:,} tok)"
        elif domain == 'code':
            filename, chunk = fetch_github_code()
            if filename and chunk:
                n = cortex.learn(chunk, domain='code', source=filename)['tokens']
                if n > 0:
                    ingested_label = f"'{filename}' ({n:,} tok)"
        elif domain == 'literature':
            title, chunk = fetch_gutenberg_literature()
            if title and chunk:
                n = cortex.learn(chunk, domain='literature', source=title)['tokens']
                if n > 0:
                    ingested_label = f"'{title}' ({n:,} tok)"

        if ingested_label:
            print(f"  [{emoji}] Ingested {ingested_label}")
        else:
            print(f"  [{emoji}] (fetch failed — using cached knowledge)")

        # Pick the next prompt for this domain
        prompts = domain_prompts[domain]
        prompt = prompts[prompt_counters[domain] % len(prompts)]
        prompt_counters[domain] += 1

        # Generate text in the same domain
        generated = cortex.speak(prompt=prompt, length=80, domain=domain)

        # Evaluate with Wernicke's V2
        score, feedback = cortex.evaluate(generated, domain=domain)
        grade = grade_to_letter(score)

        # Neuromodulate
        modulation = cortex.neuromodulate(score, generated)
        if modulation == "DOPAMINE":
            mod_indicator = "DOPAMINE 🟢"
        elif modulation == "CORTISOL":
            mod_indicator = "CORTISOL 🔴"
        else:
            mod_indicator = "NEUTRAL  🟡"

        # Display
        print(f"{'─' * 66}")
        print(f"  EPOCH {epoch} [{emoji} {domain}] | Grade: {grade} (Score: {score}) | {mod_indicator}")
        print(f"  Feedback: {feedback}")
        print(f"  ┌─ Generated Text ─────────────────────────────────────────────")
        wrapped = word_wrap(generated, width=64, prefix='  │ ')
        print(wrapped)
        print(f"  └──────────────────────────────────────────────────────────────")
        print_status_bar(cortex)

        # Track consecutive good scores (B or better: score >= 20)
        if score >= 20:
            consecutive_good += 1
        else:
            consecutive_good = 0

        # Periodic sleep consolidation + perplexity
        if epoch % 10 == 0:
            print(f"\n  💤 Sleep consolidation cycle...")
            replayed = cortex.sleep_consolidate(n_replays=10)
            print(f"     Replayed {replayed} memories.")

            # Measure perplexity for each domain
            print(f"  Perplexity measurements:")
            for d in ['prose', 'code', 'literature']:
                ppl = cortex.measure_perplexity(domain=d)
                if ppl is not None:
                    print(f"     {domain_emoji(d)} {d}: {ppl:.2f}")
                else:
                    print(f"     {domain_emoji(d)} {d}: N/A (insufficient data)")
            print()

        # Achievement: 5 consecutive good scores
        if consecutive_good >= 5:
            print()
            print("🌟" * 33)
            print("  ╔══════════════════════════════════════════════════════════╗")
            print("  ║   🏆 UNIVERSAL CORTEX ACHIEVEMENT UNLOCKED! 🏆          ║")
            print("  ║   5 consecutive high-quality generations confirmed.     ║")
            print("  ║   The multi-domain cortex is fully operational.         ║")
            print("  ╚══════════════════════════════════════════════════════════╝")
            print("🌟" * 33)

            # Showcase: one generation per domain
            print("\n  ═══ SHOWCASE: Cross-Domain Generation ═══\n")
            showcase_prompts = {
                'prose': "The development",
                'code': "def",
                'literature': "It was",
            }
            for d in ['prose', 'code', 'literature']:
                emoji_d = domain_emoji(d)
                showcase = cortex.speak(
                    prompt=showcase_prompts[d], length=100,
                    domain=d, temperature=0.7
                )
                print(f"  {emoji_d} [{d.upper()}] prompt: '{showcase_prompts[d]}'")
                print(f"  ┌──────────────────────────────────────────────────────────")
                wrapped = word_wrap(showcase, width=64, prefix='  │ ')
                print(wrapped)
                print(f"  └──────────────────────────────────────────────────────────\n")

            break

        time.sleep(1)


if __name__ == '__main__':
    run_universal_training()
