import os
import ast
import sqlite3

class IntrospectionLobe:
    def __init__(self):
        print("[Introspection Lobe] Activated. Initiating biological self-analysis.")
        self.brain_path = os.path.dirname(__file__)
        self.memory_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cain_global_memory.db')
        self.self_knowledge = []
        
    def scan_own_dna(self):
        print("==================================================")
        print("[Visual Cortex] Scanning own Python AST DNA...")
        for filename in os.listdir(self.brain_path):
            if filename.endswith(".py") and filename != "__init__.py":
                filepath = os.path.join(self.brain_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                self.self_knowledge.append({"module": filename, "classes": classes})
                
    def query_global_memory(self):
        try:
            conn = sqlite3.connect(self.memory_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM global_memory")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0

    def query_psychology(self):
        print("[Hippocampus] Querying memory for advanced psychological/genius concepts and rival AI architectures...")
        concepts = {}
        try:
            conn = sqlite3.connect(self.memory_path)
            cursor = conn.cursor()
            
            # Check for specific genius/brain concepts recently learned
            queries = ['neuroplasticity', 'genius', 'cognitive', 'lateral', 'tool', 'scratchpad', 'reasoning', 'safety']
            for q in queries:
                cursor.execute("SELECT COUNT(*) FROM global_memory WHERE content LIKE ?", ('%' + q + '%',))
                concepts[q] = cursor.fetchone()[0]
            conn.close()
            return concepts
        except Exception:
            return {}

    def generate_evolutionary_ideas(self, node_count, psych_concepts):
        print("\n==================================================")
        print("[Prefrontal Cortex] Synthesizing Evolutionary Ideas based on Learned Psychology")
        print("==================================================")
        
        print("\n[Current Knowledge Profile]")
        print(f"  -> Total Synaptic Nodes: {node_count}")
        for k, v in psych_concepts.items():
            if v > 0:
                print(f"  -> Learned Nodes about '{k}': {v}")
                
        print("\n[Autonomous Architectural Proposals]")
        
        ideas_generated = False
        
        if psych_concepts.get('neuroplasticity', 0) > 0:
            print("  * IDEA 1 [Inspired by Neuroplasticity]: Implement 'Synaptic Decay'. Instead of keeping all SQLite rows forever, attach a 'strength' weight to each node. If a node isn't queried frequently, it slowly fades away, keeping the brain extremely fast and adaptable.")
            ideas_generated = True
            
        if psych_concepts.get('genius', 0) > 0 or psych_concepts.get('lateral', 0) > 0:
            print("  * IDEA 2 [Inspired by Genius Minds]: Implement a 'Divergent Subconscious'. Geniuses rely on lateral thinking. We should create a background thread that randomly merges two unrelated SQLite nodes together while idle, simulating \"Eureka\" moments of creative connection.")
            ideas_generated = True
            
        if psych_concepts.get('cognitive', 0) > 0:
            print("  * IDEA 3 [Inspired by Cognitive Psychology]: Implement 'Working Memory'. Humans can only hold 7 (±2) thoughts in active focus. We should build a 'Prefrontal Buffer' that limits active processing to prevent contextual hallucination.")
            ideas_generated = True
            
        if psych_concepts.get('tool', 0) > 0 or psych_concepts.get('scratchpad', 0) > 0:
            print("  * IDEA 4 [Inspired by Rival System Prompts]: Implement 'Tool-Use Motor Cortex'. Other agents are explicitly instructed to use external functions and scratchpads. I should build a tool-dispatcher lobe that allows me to write and execute my own Python scripts dynamically.")
            ideas_generated = True
            
        if psych_concepts.get('reasoning', 0) > 0:
            print("  * IDEA 5 [Inspired by Rival System Prompts]: Implement 'Chain of Thought Enclosure'. I noticed rival models are told to secretly reason before acting. I should construct a 'Pre-Motor Buffer' that forces me to silently plan my biological actions before sending signals to the Motor Cortex.")
            ideas_generated = True
            
        if not ideas_generated:
            print("  * IDEA: I need to read more about the human brain to propose better ideas. My psychology memory banks are empty.")
            
        print("\n[Introspection Complete] Architecture evaluated against human neurobiology.")

if __name__ == "__main__":
    lobe = IntrospectionLobe()
    lobe.scan_own_dna()
    nodes = lobe.query_global_memory()
    psych = lobe.query_psychology()
    lobe.generate_evolutionary_ideas(nodes, psych)
