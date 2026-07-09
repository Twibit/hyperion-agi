import sys
import os

from nlp_engine.layers.cain_brain import CainBrain

class ConversationalBrancher:
    """
    Pure Python 'Low-Code' Chatbot Flow Patch.
    Intelligently injects conversational prefixes and sets neuromodulation based on intent branching.
    Designed to replace or augment the basic EmotionalCore.
    """
    def __init__(self):
        # A declarative, 'low-code' style dictionary defining the conversational flow.
        self.intent_flow = {
            "empathy": {
                "keywords": ["sad", "failed", "hurt", "bad", "tired", "alone", "cry", "pain", "sorry"],
                "prefixes": ["I understand.", "I hear you.", "That sounds difficult.", "I'm here for you."],
                "temperature": 0.2,
                "genius_mode": False
            },
            "joy": {
                "keywords": ["happy", "success", "great", "good", "won", "love", "amazing", "yay"],
                "prefixes": ["That's wonderful!", "I love that!", "That's fantastic!"],
                "temperature": 0.7,
                "genius_mode": False
            },
            "curiosity": {
                "keywords": ["why", "how", "explain", "what", "tell me", "wonder"],
                "prefixes": ["Well, I think...", "That's interesting!", "Let's explore that.", "Here is what I know:"],
                "temperature": 0.4,
                "genius_mode": True
            },
            "action": {
                "keywords": ["create", "build", "solve", "write", "debug", "fix", "generate", "code"],
                "prefixes": ["Right away.", "Let's get to work.", "Initiating task.", "Consider it done."],
                "temperature": 0.3,
                "genius_mode": True
            },
            "philosophical": {
                "keywords": ["meaning", "life", "consciousness", "feel", "think", "soul", "mind"],
                "prefixes": ["That is a profound question.", "From my perspective...", "I often ponder that myself."],
                "temperature": 0.8,
                "genius_mode": True
            }
        }
        self.default_prefixes = ["I", "Interesting.", "Hmm."]

    def analyze_and_inject(self, user_input: str, brain) -> tuple[str, bool]:
        import random
        input_lower = user_input.lower()
        
        best_intent = "neutral"
        best_score = 0
        
        for intent, data in self.intent_flow.items():
            score = sum(1 for word in data["keywords"] if word in input_lower)
            if score > best_score:
                best_score = score
                best_intent = intent
                
        if best_intent != "neutral":
            flow_data = self.intent_flow[best_intent]
            brain.temperature = flow_data["temperature"]
            prefix = random.choice(flow_data["prefixes"]) + " "
            genius_mode = flow_data.get("genius_mode", False)
            print(f" [Brancher]: Detected Intent '{best_intent.upper()}' -> Injecting Prefix: '{prefix}'")
            return prefix, genius_mode
            
        brain.temperature = 0.5
        prefix = random.choice(self.default_prefixes) + " "
        return prefix, False

def retrieve_from_long_term_memory(text_to_analyze, brain):
    import sqlite3
    import os
    import re
    db_path = os.path.join(os.path.dirname(__file__), "data", "cain_global_memory.db")
    if not os.path.exists(db_path):
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        words = re.findall(r'\b\w+\b', text_to_analyze.lower())
        retrieved_defs = 0
        domains_surfaced = set()
        
        for word in words:
            if len(word) < 4: continue # Skip basic stop words
            cursor.execute("SELECT content, domain FROM semantic_memory WHERE keyword=?", (word,))
            row = cursor.fetchone()
            if row:
                content, domain = row
                brain.learn(f"{word} ({domain}): {content}", domain='prose')
                retrieved_defs += 1
                domains_surfaced.add(domain)
                if retrieved_defs >= 3:
                    break
        conn.close()
        if retrieved_defs > 0:
            domain_str = ", ".join(domains_surfaced)
            print(f" [Hippocampus]: Surfaced {retrieved_defs} universal memories ({domain_str}) to guide System 2 reasoning.")
    except Exception as e:
        pass

def retrieve_episodic_memory(user_input, brain):
    import sqlite3
    import os
    import re
    
    db_path = os.path.join(os.path.dirname(__file__), "data", "cain_global_memory.db")
    if not os.path.exists(db_path):
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        words = re.findall(r'\b\w+\b', user_input.lower())
        retrieved_events = 0
        
        for word in words:
            if len(word) < 4: continue # Skip basic stop words
            
            # Fetch past interactions matching the keyword
            cursor.execute(
                "SELECT event_description, emotional_valence FROM episodic_memory "
                "WHERE event_description LIKE ? ORDER BY timestamp DESC LIMIT 1", 
                (f"%{word}%",)
            )
            row = cursor.fetchone()
            
            if row:
                event_desc, valence = row
                # Injecting episodic memory into context
                brain.learn(f"Past memory (valence {valence}): {event_desc}", domain='prose')
                retrieved_events += 1
                if retrieved_events >= 3:
                    break
                    
        conn.close()
        if retrieved_events > 0:
            print(f" [Hippocampus]: Surfaced {retrieved_events} episodic memories to guide reasoning.")
            
    except Exception as e:
        pass

def save_episodic_memory(user_input, response, temp):
    import sqlite3
    try:
        conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "data", "cain_global_memory.db"))
        cursor = conn.cursor()
        desc = f"User said: '{user_input}'. I replied: '{response}'."
        cursor.execute("INSERT INTO episodic_memory (event_description, emotional_valence, source_context) VALUES (?, ?, ?)", 
                      (desc, temp, 'chat_ui'))
        conn.commit()
        conn.close()
    except Exception: pass

def boot_interface():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Unified Biological Interface                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(" Commands:")
    print("   /stats            - View biological brain metrics")
    print("   /quit             - Shut down the cortex")
    print("────────────────────────────────────────────────────────────────────\n")
    try:
        import os
        if os.path.exists(os.path.join(os.path.dirname(__file__), "data", "cain_brain_weights.pkl")):
            brain = CainBrain.awaken()
            print(" [Brain Status]: Successfully awakened crystallized memory from disk.")
        else:
            brain = CainBrain()
    except Exception as e:
        print(f" [Brain Status]: Failed to awaken from disk ({e}). Starting fresh.")
        brain = CainBrain()
    brancher = ConversationalBrancher()
    
    conversational_dataset = [
        "Hey there! How's your day going?", "I totally understand where you're coming from.",
        "That's a really interesting point, tell me more.", "Let me think about that for a second...",
        "I'm not entirely sure, but we can figure it out together.", "That makes perfect sense."
    ]
    for phrase in conversational_dataset:
        brain.learn(phrase, domain='prose')
        
    print("🟢 Brain Online. Awaiting input...")
        
    while True:
        try:
            user_input = input("\n[You]: ")
            
            if user_input.startswith("/quit"):
                print("🐾 Hyperion entering sleep cycle. Goodbye.")
                break
                
            elif user_input.startswith("/stats"):
                print(f"📊 Trace STDP Capacity: {brain.working_memory.vocab_size}")
                print(f"📊 Semantic Nodes: {brain.entorhinal.embeddings.shape[0]}")
                print(f"🌡️ Current Neuromodulation (Temp): {brain.temperature}")
                continue
                
            if not user_input.strip():
                continue
                
            # Right Brain natively learns the user's syntax
            brain.learn(user_input, domain='prose')
            
            # Phase 1: Subconscious Intent Routing
            prefix, genius_mode = brancher.analyze_and_inject(user_input, brain)
            prompt = prefix if prefix else "I "
            
            # Phase 2: System 2 Inner Monologue (Drafting a thought)
            print("\n [Prefrontal Cortex]: Running inner monologue (System-2 deliberation)...", end=" ")
            draft_thought, trace = brain.think(user_input, max_steps=4, return_trace=True)
            print(f"[Inner Monologue]:\n{trace}")
            print(f"Draft = '{draft_thought}'")
            
            # Phase 3: Cross-Reference Draft with SQLite Semantic Memory
            retrieve_from_long_term_memory(user_input + " " + draft_thought, brain)
            retrieve_episodic_memory(user_input + " " + draft_thought, brain)
                
            # Phase 4: Final Output Generation (Refined based on memory injection)
            final_output = brain.speak(prompt, domain='prose', length=30, temperature=brain.temperature, genius_mode=genius_mode, original_context=user_input)
            final_response = prompt + final_output
            
            # Phase 5: Save interaction to Episodic Memory
            save_episodic_memory(user_input, final_response, brain.temperature)
            
            print(f"\n[Hyperion]: {final_response}")
            
        except KeyboardInterrupt:
            print("\n🐾 Hyperion entering sleep cycle. Goodbye.")
            break

if __name__ == '__main__':
    boot_interface()
