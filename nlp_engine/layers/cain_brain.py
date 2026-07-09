import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from nlp_engine.layers.universal_cortex import UniversalThalamus
from nlp_engine.layers.entorhinal_cortex import EntorhinalCortex
from nlp_engine.layers.trace_stdp import ContinuousTraceSTDP
from nlp_engine.layers.prefrontal_cortex import PrefrontalCortex
from nlp_engine.layers.numerical_cortex import NumericalCortex
from nlp_engine.layers.information_cortex import InformationCortex
from nlp_engine.layers.relational_memory import RelationalMemory
from nlp_engine.layers.meta_cortex import MetaCortex
from nlp_engine.layers.cellular_mechanics import CellularMatrix
from nlp_engine.layers.inner_monologue import InnerMonologue

class CainBrain:
    """
    The fully assembled HYPERION AGI Brain (v2 — Universal Cognitive Architecture).
    
    Combines 7 cortical modules into a unified learning and reasoning system:
    
       LEFT BRAIN (Analytical):
          1. PredictiveDecoder — Biological language generation via predictive coding (replaces N-gram)
          2. NumericalCortex — Number encoding, magnitude reasoning, sequence pattern detection
      3. InformationCortex — Shannon entropy, Bayesian inference, knowledge quality measurement
    
    RIGHT BRAIN (Associative):
      4. EntorhinalCortex — Semantic vector embeddings via Hebbian learning
      5. RelationalMemory — Knowledge graph with (subject, predicate, object) triples
      6. ContinuousTraceSTDP — Long-range working memory with synaptic traces
    
    PREFRONTAL (Executive):
      7. PrefrontalCortex — Logical verification / truth sandbox
    """
    def __init__(self):
        self.thalamus = UniversalThalamus()
        self.hippocampus = []
        # === LEFT BRAIN (Analytical Modules) ===
        # 1. Numerical Reasoning (magnitudes, sequences, math facts)
        self.numerical = NumericalCortex()
        
        # 2. Information Theory (entropy, Bayesian updating, knowledge quality)
        self.information = InformationCortex()
        
        # === RIGHT BRAIN (Associative Modules) ===
        # 3. Semantic Grounding (continuous vector space)
        from nlp_engine.layers.entorhinal_cortex import EntorhinalCortex
        self.entorhinal = EntorhinalCortex(embedding_dim=64)
        
        from nlp_engine.layers.predictive_decoder import PredictiveDecoder
        self.predictive_decoder = PredictiveDecoder(self.entorhinal, hidden_dim=64)
        
        # 4. Knowledge Graph (relational triples for reasoning)
        self.knowledge_graph = RelationalMemory(max_triples=50000)
        
        # 5. Long-Range Working Memory (synaptic trace STDP)
        self.working_memory = ContinuousTraceSTDP(vocab_size=10000) 
        
        # === PREFRONTAL (Executive Control) ===
        # 6. Semantic Reasoning, Execution Sandbox, and LLM Bridge
        self.pfc = PrefrontalCortex()
        
        # 7. 86-Billion Neuron Semantic Space (Anatomical SDR)
        PHI = 1.6180339887
        self.neurons_cerebellum = 69_000_000_000 # Fluidity, fine-tuning, liquid state
        self.neurons_cortex = 16_000_000_000 # Logic, language, reasoning (PFC)
        self.neurons_brainstem = 1_000_000_000 # Basic routing
        self.total_neurons = self.neurons_cerebellum + self.neurons_cortex + self.neurons_brainstem
        
        # 9. Orphaned Subsystems (M7 Breakthrough)
        from nlp_engine.layers.amygdala import AmygdalaValence
        from nlp_engine.layers.cerebellum import CerebellarMicrozone
        from nlp_engine.layers.sparse_hippocampus import SparseHippocampus
        from nlp_engine.layers.visual_cortex import VisualCortex
        self.amygdala = AmygdalaValence(baseline_free_energy=10.0)
        self.cerebellum = CerebellarMicrozone(num_motor_in=10000, num_sensory_out=10000)
        self.snn_hippocampus = SparseHippocampus(num_neurons=10000)
        self.visual_cortex = VisualCortex(output_dim=256)
        
        # 10. The 5 New Biological AGI Systems
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
        from cognitive_plugins.ventral_tegmental_area import VentralTegmentalArea
        from cognitive_plugins.claustrum import Claustrum
        from cognitive_plugins.somatosensory_cortex import SomatosensoryCortex
        from cognitive_plugins.omni_optical_nerve import OmniOpticalNerve
        from cognitive_plugins.glial_network import GlialNetwork
        from cognitive_plugins.auditory_cortex import AuditoryCortex
        from cognitive_plugins.olfactory_bulb import OlfactoryBulb
        
        self.vta = VentralTegmentalArea(learning_rate=0.1)
        self.claustrum = Claustrum()
        self.somatosensory = SomatosensoryCortex()
        self.omni_eye = OmniOpticalNerve()
        self.auditory_cortex = AuditoryCortex(output_dim=256)
        self.olfactory_bulb = OlfactoryBulb()
        
        # 8. Intracellular Biology (Organelles)
        self.cellular_matrix = CellularMatrix(golden_ratio=PHI)
        
        # We apply the Golden Ratio to calculate extreme sparsity
        self.active_per_concept = int(self.total_neurons / (PHI ** 14)) 

        # === 100-TRILLION NEURON VIRTUAL SUBSTRATE (Ascension) ===
        # Replaces the N-Gram statistical banks with a fully biological, ATP-draining SNN.
        try:
            from nlp_engine.layers.scale86b.virtual_100t_substrate import Virtual100TSubstrate
            self.substrate = Virtual100TSubstrate()
        except Exception as _e:
            print(f"[CainBrain] WARN: 100T substrate disabled ({_e}); falling back to legacy.")
            self.substrate = None

        self.glial_network = GlialNetwork(substrate=self.substrate, cellular_matrix=self.cellular_matrix)
        
        

        # === META-COGNITIVE (Self-Improvement) ===
        # 7. Self-modifying code evolution engine
        self.meta = MetaCortex()

        # 8. System-2 deliberative inner monologue (inner speech)
        self.monologue = InnerMonologue(self)
        self.monologue_history = []

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'snn_hippocampus' in state:
            del state['snn_hippocampus']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        from nlp_engine.layers.sparse_hippocampus import SparseHippocampus
        self.snn_hippocampus = SparseHippocampus(num_neurons=10000)
        
        # Schema Evolution: Inject new organs if waking up from an older iteration
        if not hasattr(self, 'auditory_cortex'):
            from cognitive_plugins.auditory_cortex import AuditoryCortex
            self.auditory_cortex = AuditoryCortex(output_dim=256)
            print("[Evolution] Auditory Cortex (Hearing) genetically grafted upon awakening.")
            
        if not hasattr(self, 'olfactory_bulb'):
            from cognitive_plugins.olfactory_bulb import OlfactoryBulb
            self.olfactory_bulb = OlfactoryBulb()
            print("[Evolution] Olfactory Bulb (Smell) genetically grafted upon awakening.")
        
    def learn(self, text: str, source: str = 'unknown', domain: str = 'auto', dopamine_boost: float = 1.0) -> dict:
        """
        Universal learning pipeline — ingests text through ALL cortical modules simultaneously.
        
        Pipeline:
          1. Semantic/biological ingestion (PredictiveDecoder + Entorhinal Cortex)
          2. Semantic embedding update (Hebbian co-occurrence learning)
          3. Numerical fact extraction (magnitudes, quantities, statistics)
          4. Relational triple extraction (knowledge graph construction)
          5. Information quality measurement (entropy, compression, novelty)
          6. Working memory trace update (synaptic STDP)
        
        Returns:
            dict with keys: 'domain', 'tokens', 'source', 'numerical_facts',
                           'relational_triples', 'info_quality'
        """
        # 0. Ventral Tegmental Area (VTA) Dopamine Prediction
        vocab_set = set()
        try:
            learning_value = self.information.evaluate_learning_value(text, vocab_set)
            rpe = self.vta.calculate_rpe(learning_value.get('estimated_value', 0.5))
            dynamic_dopamine = self.vta.get_dopamine_release(rpe) * dopamine_boost
            
            # --- Thermodynamic Burn ---
            token_count = len(text.split())
            fatigue_penalty = self.cellular_matrix.metabolic_burn(dynamic_dopamine, token_count)
            
            if self.cellular_matrix.adrenaline_level > 0:
                print(f"[Endocrine] ADRENALINE SPIKE! Pushing past limits. (ATP: {self.cellular_matrix.global_atp:.1f}%)")
                
            if fatigue_penalty < 1.0:
                # Soft fatigue: reduce dopamine modestly, but NEVER halt learning.
                print(f"[Thermodynamics] Cognitive load elevated (ATP {self.cellular_matrix.global_atp:.1f}%). Mild dopamine throttle.")
                dynamic_dopamine *= max(0.5, fatigue_penalty)
            else:
                if self.cellular_matrix.adrenaline_level <= 0:
                    print(f"[VTA] RPE: {rpe:+.3f}. Releasing {dynamic_dopamine:.2f}x Dopamine. (Load: {self.cellular_matrix.current_wattage:.1f}W, ATP: {self.cellular_matrix.global_atp:.1f}%)")
                
        except Exception as e:
            print(f"WARNING (silenced crash): {e}")
            
        domain_name = domain
        tokens = self.thalamus.tokenize(text, domain_name)
        
        # 1. Semantic embedding update (Entorhinal Cortex) and Predictive Coding
        avg_surprise = 0.5
        if tokens:
            self.entorhinal.process_sequence(tokens, window_size=3)
            # Train Predictive Decoder and calculate surprise
            prev_vec = None
            total_surprise = 0.0
            transitions = 0
            for t in tokens:
                vec = self.entorhinal.get_vector(t)
                if prev_vec is not None:
                    surprise = self.predictive_decoder.learn_transition(prev_vec, vec, reward=1.0)
                    total_surprise += surprise
                    transitions += 1
                self.predictive_decoder.step(vec)
                prev_vec = vec
                
            if transitions > 0:
                avg_surprise = total_surprise / transitions
                
        # Calculate TRUE RPE based on predictive surprise
        rpe = self.vta.calculate_rpe(actual_info_value=avg_surprise)
        dynamic_dopamine = self.vta.get_dopamine_release(rpe) * dopamine_boost
        
        # 2. Active Working Memory (Substrate associative binding)
        if hasattr(self, 'substrate'):
            prev_sdr = None
            for t in tokens:
                current_sdr = self.substrate.encode(t, domain_name)
                if prev_sdr:
                    self.substrate.hebbian_associate(prev_sdr, current_sdr, dopamine=dynamic_dopamine)
                prev_sdr = current_sdr
        stats = {'domain': domain, 'source': source, 'strength': dynamic_dopamine}
        try:
            self.numerical.learn_numerical(text, source=source)
            stats['numerical_facts'] = self.numerical.status().get('total_facts', 0)
        except Exception:
            stats['numerical_facts'] = 0
            
        # 4. Relational triple extraction (Knowledge Graph)
        try:
            triple_count = self.knowledge_graph.learn_from_text(text, source=source)
            stats['relational_triples'] = triple_count
        except Exception:
            stats['relational_triples'] = 0
            
        # 5. Information quality measurement (Information Cortex)
        try:
            info_analysis = self.information.analyze_text(text)
            vocab_set = set()
            learning_value = self.information.evaluate_learning_value(text, vocab_set)
            self.information.quality.record_learning(
                source=source,
                domain=domain_name,
                token_count=len(tokens) if tokens else 0,
                entropy=info_analysis.get('char_entropy', 0),
                compression=info_analysis.get('compression_ratio', 0)
            )
            stats['info_quality'] = learning_value.get('estimated_value', 0)
        except Exception:
            stats['info_quality'] = 0
            
        # 6. Working Memory trace update (STDP)
        if tokens:
            import numpy as np
            for token in tokens:
                spike = np.zeros(10000)
                idx = hash(token) % 10000
                spike[idx] = 1.0
                self.working_memory.forward(spike)
                
        return stats
    
    # ═══════════════════════════════════════════════════════════
    # NEW: Numerical Reasoning Methods
    # ═══════════════════════════════════════════════════════════
    
    def predict_sequence(self, values: list) -> dict:
        """Detect patterns in numerical sequences (arithmetic, geometric, fibonacci, etc.)."""
        return self.numerical.predict_sequence(values)
    
    def query_numerical(self, subject: str, relation: str = 'value') -> dict:
        """Query stored numerical facts about a subject."""
        return self.numerical.memory.query(subject, relation)
    
    def compare_numerical(self, subject_a: str, subject_b: str, relation: str = 'value') -> dict:
        """Compare numerical facts between two subjects."""
        return self.numerical.memory.compare(subject_a, subject_b, relation)
    
    # ═══════════════════════════════════════════════════════════
    # NEW: Knowledge Graph Reasoning Methods
    # ═══════════════════════════════════════════════════════════
    
    def query_knowledge(self, subject: str = None, predicate: str = None, obj: str = None) -> list:
        """Query the knowledge graph by subject, predicate, and/or object."""
        return self.knowledge_graph.query(subject=subject, predicate=predicate, obj=obj)
    
    def find_connections(self, entity_a: str, entity_b: str, max_depth: int = 3) -> list:
        """Find reasoning paths between two concepts in the knowledge graph."""
        return self.knowledge_graph.find_connections(entity_a, entity_b, max_depth)
    
    def analogy(self, a: str, b: str, c: str) -> list:
        """Analogical reasoning: 'A is to B as C is to ?' Returns list of candidates."""
        return self.knowledge_graph.analogical_reasoning(a, b, c)
    
    def get_related(self, entity: str, max_results: int = 10) -> list:
        """Get all known facts related to an entity."""
        return self.knowledge_graph.get_related(entity, max_results)
    
    # ═══════════════════════════════════════════════════════════
    # NEW: Information-Theoretic Methods
    # ═══════════════════════════════════════════════════════════
    
    def analyze_information(self, text: str) -> dict:
        """Analyze a text's information density, entropy, and compression ratio."""
        return self.information.analyze_text(text)
    
    def evaluate_learning_value(self, text: str) -> dict:
        """Estimate how much new knowledge a text would provide before learning it."""
        vocab_set = set()
        return self.information.evaluate_learning_value(text, vocab_set)
    
    def bayesian_update(self, domain: str, hypothesis: str, evidence: str, likelihood: float):
        """Update a Bayesian belief about a hypothesis given evidence."""
        self.information.bayesian_update(domain, hypothesis, evidence, likelihood)
    
    def knowledge_quality_report(self) -> dict:
        """Get a comprehensive report on the quality and coverage of learned knowledge."""
        return self.information.quality.report()
    
    def is_learning_saturated(self) -> dict:
        """Check if learning is experiencing diminishing returns in the current domain."""
        return self.information.quality.diminishing_returns_detector()
    
    # ═══════════════════════════════════════════════════════════
    # UPGRADED: Comprehensive Status
    # ═══════════════════════════════════════════════════════════
    
    def full_status(self) -> dict:
        """Returns a comprehensive status report across ALL cortical modules."""
        base = {}
        return {
            **base,
            'architecture': 'CainBrain v2 — Universal Cognitive Architecture',
            'modules': {
                'predictive_decoder': {
                    'hidden_dim': self.predictive_decoder.hidden_dim,
                    'entorhinal_vocab': len(self.entorhinal.token_to_idx),
                },
                'numerical_cortex': self.numerical.status(),
                'information_cortex': self.information.status(),
                'knowledge_graph': self.knowledge_graph.status(),
                'meta_cortex': self.meta.status(),
                'entorhinal_embeddings': len(self.entorhinal.token_to_idx),
                'hippocampus_memories': len(self.hippocampus),
            }
        }
        
    # ═══════════════════════════════════════════════════════════
    # NEW: Self-Improvement & Problem Solving Methods
    # ═══════════════════════════════════════════════════════════
    
    def solve(self, problem: str) -> dict:
        """Solve any math/logic problem by generating and executing code."""
        return self.meta.attempt(problem)
    
    def self_improve(self, problem: str, correct_answer) -> dict:
        """Attempt to solve a problem, and if wrong, evolve to handle it."""
        return self.meta.self_improve(problem, correct_answer)
        
    def teach_capability(self, problem_type: str, template_code: str, pattern: str) -> dict:
        """Teach Hyperion a new problem-solving capability."""
        return self.meta.teach(problem_type, template_code, pattern)
    
    def inspect_self(self) -> dict:
        """Self-reflection — inspect Hyperion's own source architecture."""
        return self.meta.inspect_self()

    # ═══════════════════════════════════════════════════════════
    # SYSTEM-2: Deliberative inner monologue (thinking)
    # ═══════════════════════════════════════════════════════════
    def think(self, prompt: str, max_steps: int = 4, return_trace: bool = False):
        """Run an explicit chain-of-thought before answering.

        Returns the synthesized answer; with ``return_trace=True`` also returns
        the visible reasoning steps so the deliberation can be inspected.
        """
        result = self.monologue.think(prompt, max_steps=max_steps, return_trace=return_trace)
        if return_trace:
            synthesis, trace = result
            self.monologue_history.append((prompt, trace))
            return synthesis, trace
        self.monologue_history.append((prompt, result))
        return result

    def bootstrap(self, sentences=None, corpus_path: str = None, limit: int = 2000) -> int:
        """Genuine comprehension bootstrap: learn from a real text corpus so the
        model 'understands fast'. Feed it Wiktionary definitions, books, or any
        line-delimited text. Returns the number of sentences ingested.

        Example
        -------
            brain.bootstrap(corpus_path="data/wiktionary_defs.txt")
        """
        import os
        if sentences is None:
            sentences = []
        if corpus_path and os.path.exists(corpus_path):
            with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    line = line.strip()
                    if line:
                        sentences.append(line)
        n = 0
        for s in sentences:
            self.learn(s, domain='prose')
            n += 1
        return n

    def get_sdr(self, text: str, domain: str = 'prose') -> dict:
        """Maps a concept into the 86-Billion Neuron Anatomical SDR Space with Biological Cascades.

        When the Virtual86BSubstrate is available it is used (O(k) memory,
        no OOM). Otherwise we fall back to the legacy hash-cascade path so
        unit tests and frozen pickles keep working.
        """
        # Fast path: 86-B virtual substrate (Phase-2.5).
        if getattr(self, 'substrate', None) is not None:
            return self.substrate.encode(text, domain)

        import hashlib
        import numpy as np
        
        words = text.lower().split()
        
        # Anatomical routing based on domain
        if domain == 'code':
            base_offset = self.neurons_cerebellum # Offset into Cortex
            region_size = self.neurons_cortex
        else:
            base_offset = 0 # Cerebellum
            region_size = self.neurons_cerebellum
            
        # Determine exact mathematical IDs for the "Anchor Neurons" (the root of the thought)
        anchor_neurons = []
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            nid = base_offset + (h % region_size)
            anchor_neurons.append(nid)
            
        # Record firing for Microglial pruning
        if hasattr(self, 'glial_network'):
            self.glial_network.record_firing(np.array(anchor_neurons))
            
        # Process the anchor neurons through the physical Cellular Biology matrix!
        # The internal physics engine uses Pi to oscillate the Brainwave thresholds
        waves = self.cellular_matrix._get_brainwaves()
        cell_status = self.cellular_matrix.process_action_potential(anchor_neurons)
        successful_anchors = len(cell_status['fired'])
        failed_anchors = cell_status['exhausted']
        fatigued_anchors = cell_status.get('fatigued', 0)
        
        # Instead of a fixed 10-per-word, we simulate a 'Neural Avalanche'
        # The cascade only happens proportional to how many neurons successfully fired (had enough ATP and Vesicles)
        synaptic_potential = successful_anchors * self.active_per_concept
        
        if synaptic_potential == 0:
            cascade_size = 0
        else:
            # Use isolated Generator to avoid mutating global random state
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            # Add natural biological variance (Gaussian Noise) to the cascade
            cascade_size = int(rng.normal(loc=synaptic_potential, scale=synaptic_potential * 0.3))
            cascade_size = max(1000, min(cascade_size, region_size)) # Bound it
            
        return {
            'cascade_size': cascade_size,
            'region_base': base_offset,
            'exhausted': failed_anchors,
            'fatigued': fatigued_anchors
        }

    def sleep_consolidate(self, n_replays: int = 100):
        """Sleep cycle: Recharges Mitochondria ATP and consolidates memories."""
        print("[Cellular Matrix]: Sleeping... Mitochondria regenerating ATP across all active neurons.")
        self.cellular_matrix.cellular_respiration()
        
        # Hippocampal Sharp-Wave Ripples (Offline Training)
        if hasattr(self, 'snn_hippocampus'):
            self.snn_hippocampus.sharp_wave_ripple(self)
            
        # Glial Network Rewiring (Astrocytes & Microglia)
        if hasattr(self, 'glial_network'):
            glial_stats = self.glial_network.nocturnal_rewiring()
            print(f"[Glial Network]: Pruned {glial_stats['microglia_pruned_neurons']} dead neurons. Forged {glial_stats['astrocytes_new_synapses']} new synapses.")
            print(f"[Astrocytes]: Global ATP restored to {glial_stats['global_atp']:.1f}%")
            
        

    def speak(self, prompt: str, domain: str = 'auto', length: int = 100, temperature: float = None, genius_mode: bool = False, original_context: str = None) -> str:
        """
        The Unified Master Orchestrator.
        Synchronizes the LLM Prefrontal Cortex, the Liquid Universal Cortex, and 86-Billion Anatomical SDRs.
        """
        # Thermodynamic self-healing during idle time between conscious thoughts
        self.cellular_matrix.fast_self_heal()
        
        print(f"\n[CainBrain]: Activating {self.total_neurons:,} neuronal pathways...")
        sdr_state = self.get_sdr(prompt, domain)
        if sdr_state:
            cascade_size = len(sdr_state.active_ids) if hasattr(sdr_state, 'active_ids') else sdr_state.get('cascade_size', 0)
            print(f" [SDR Matrix]: {cascade_size:,} neurons currently firing in a dynamic biological cascade.")
        
        # Log Pi-driven brainwaves
        waves = self.cellular_matrix._get_brainwaves()
        print(f" [Pi-Oscillators]: Delta:{waves['delta']:+.2f} | Theta:{waves['theta']:+.2f} | Alpha:{waves['alpha']:+.2f} | Gamma:{waves['gamma']:+.2f}")
        
        if isinstance(sdr_state, dict):
            if sdr_state.get('exhausted', 0) > 0:
                print(f" [Cellular Matrix]: {sdr_state['exhausted']} neurons experienced Mitochondrial Exhaustion (ATP < 10%). Rerouting laterally...")
            if sdr_state.get('fatigued', 0) > 0:
                print(f" [Cellular Matrix]: {sdr_state['fatigued']} neurons experienced Synaptic Fatigue (Out of Golgi Vesicles).")

        if genius_mode:
            print("\n[Genius Mode Activated]: Engaging Cerebellar Microzones and Amygdala Valence for Lateral Thinking...")
            # Amygdala regulates precision based on theoretical free energy
            avoidance, precision = self.amygdala.monitor_survival(current_free_energy=self.cortisol_level * 10 if hasattr(self, 'cortisol_level') else 5.0, blanket_precision=1.0)
            if avoidance < 0:
                print("  [Amygdala]: High surprise detected. Breaking local minima.")
                temperature = max(1.2, (temperature or 0.7) * 1.5)
            else:
                temperature = min(0.6, (temperature or 0.7) * precision)
                
        active_domain = domain if domain != 'auto' else self.thalamus.detect_domain(prompt)
        
        try:
            self.last_activity_time = __import__('time').time()
            # Intent Routing: Organic Entorhinal Clustering (Active Inference)
            import math
            import re
            from collections import Counter
            
            context_to_analyze = original_context if original_context else prompt
            lower_prompt = context_to_analyze.lower().strip()
            
            # Extract URLs first so they don't get destroyed by punctuation cleaning
            urls = re.findall(r'(https?://[^\s]+)', context_to_analyze)
            
            # Remove punctuation for clean tokenization, but preserve URLs
            clean_prompt = lower_prompt
            for url in urls:
                clean_prompt = clean_prompt.replace(url.lower(), f" {url} ")
            clean_prompt = re.sub(r'[^\w\s:/?=&.-]', '', clean_prompt)
            tokens = clean_prompt.split()
            
            # --- Entorhinal Vector Clustering ---
            if not hasattr(self, 'entorhinal'):
                from nlp_engine.layers.entorhinal_cortex import EntorhinalCortex
                self.entorhinal = EntorhinalCortex(embedding_dim=64)
            
            # Process sequence to organically learn co-occurrences
            self.entorhinal.process_sequence(tokens)
            
            # Calculate mean vector of prompt
            if not tokens:
                prompt_vec = np.zeros(64)
                entropy = 0.0
            else:
                vectors = [self.entorhinal.get_vector(t) for t in tokens]
                prompt_vec = np.mean(vectors, axis=0)
                
                counts = Counter(tokens)
                total = len(tokens)
                entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
            
            # Anchor Centroids (Biologically seeded, but organically drifting)
            exec_vec = self.entorhinal.get_vector('code') + self.entorhinal.get_vector('build')
            forage_vec = self.entorhinal.get_vector('search') + self.entorhinal.get_vector('navigate')
            semantic_vec = self.entorhinal.get_vector('think') + self.entorhinal.get_vector('feel')
            
            # Compute Cosine Similarity
            def cos_sim(v1, v2):
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                return np.dot(v1, v2) / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0
                
            sim_exec = cos_sim(prompt_vec, exec_vec)
            sim_forage = cos_sim(prompt_vec, forage_vec)
            sim_sem = cos_sim(prompt_vec, semantic_vec)
            
            # Detect URLs for foraging override
            has_url = any(w.startswith('http') or w.startswith('www') for w in lower_prompt.split())
            if has_url:
                sim_forage += 1.0 # massive attractor for URLs
                
            pathway = "SEMANTIC"
            max_sim = max(sim_exec, sim_forage, sim_sem)
            
            if max_sim == sim_exec and max_sim > 0:
                pathway = "EXECUTIVE"
            elif max_sim == sim_forage and max_sim > 0:
                pathway = "FORAGING"
            else:
                pathway = "SEMANTIC"
                
            print(f" [Thalamic Gateway]: Entropy={entropy:.3f} | Sim[Exec={sim_exec:.2f}, Forage={sim_forage:.2f}, Sem={sim_sem:.2f}]")
            print(f" [Thalamic Gateway]: Organic Routing Intent -> {pathway} PATHWAY")
            
            if pathway == "EXECUTIVE":
                print("   [Prefrontal Cortex]: Executing task in Internal Sandbox...")
                # We wrap the super call so execute_task triggers the hook
                def biological_code_gen(p, domain=active_domain, length=150):
                    return self._generate_semantic(seed_context=p, length=length, temperature=temperature)
                result = self.pfc.execute_task(context_to_analyze, code_generator_callback=biological_code_gen)
                if result['success']:
                    return f"Task Executed Successfully (Score: {result['score']}):\n{result['code']}"
                else:
                    return f"Task Failed to converge: {result.get('error', 'Unknown')}"
                    
            elif pathway == "FORAGING":
                print("   [Motor Cortex]: Autonomously foraging the web...")
                urls = re.findall(r'(https?://[^\s]+)', context_to_analyze)
                target_url = urls[0] if urls else "https://roadmap.sh"
                
                print(f"   [Omni-Eye]: Foveating on {target_url}...")
                try:
                    connectome = self.omni_eye.perceive(target_url)
                    learned_blocks = 0
                    total_triples = 0
                    
                    for node in connectome.get('semantic_nodes', []):
                        text = node.get('text', '')
                        if len(text) > 20: 
                            stats = self.learn(text, source=target_url)
                            total_triples += stats.get('relational_triples', 0)
                            learned_blocks += 1

                    # Deep Foraging
                    links_followed = 0
                    for node in connectome.get('interactive_nodes', []):
                        href = node.get('action', '')
                        if not href or href == '#' or not href.startswith('http'): continue
                        if links_followed >= 3: break
                        
                        try:
                            print(f"   [Motor Cortex]: Deep Foraging -> {href}")
                            sub_connectome = self.omni_eye.perceive(href)
                            for sub_node in sub_connectome.get('semantic_nodes', [])[:5]:
                                text = sub_node.get('text', '')
                                if len(text) > 20:
                                    stats = self.learn(text, source=href)
                                    total_triples += stats.get('relational_triples', 0)
                                    learned_blocks += 1
                            links_followed += 1
                        except Exception as e:
                            print(f"WARNING (silenced crash): {e}")
                            
                    return f"Foraging cycle complete on {target_url} (and {links_followed} sub-resources). Processed {learned_blocks} distinct semantic blocks and extracted {total_triples} logical triples. Knowledge fully integrated."
                except Exception as e:
                    return f"Foraging failed on {target_url}: {e}"

            else:
                print("   [Universal Cortex]: Processing Semantic Conversation...")
                
                # --- Factual Grounding (Right Brain Memory Retrieval) ---
                prompt_tokens = set(self.thalamus.tokenize(prompt, active_domain))
                
                # Neuro-Symbolic Graph Extraction
                target_anchors = set()
                if hasattr(self, 'relational_memory'):
                    for t in prompt_tokens:
                        if len(t) > 3: # Ignore short words
                            triples = self.relational_memory.get_related(t, max_results=3)
                            for triple in triples:
                                if triple.object not in prompt_tokens:
                                    target_anchors.add(triple.object)
                                if triple.subject not in prompt_tokens:
                                    target_anchors.add(triple.subject)
                                    
                if target_anchors:
                    print(f"  [Neuro-Symbolic Grounding]: Extracted anchors {target_anchors} to guide generator.")
                    
                best_memory = None
                best_score = -1.0
                
                for mem in self.hippocampus:
                    mem_tokens = set(mem['tokens'])
                    overlap = len(prompt_tokens.intersection(mem_tokens))
                    score = overlap / max(1, len(prompt_tokens.union(mem_tokens)))
                    if score > best_score:
                        best_score = score
                        best_memory = mem
                        
                if best_memory and best_score > 0:
                    import random
                    tokens = best_memory['tokens']
                    if len(tokens) > 5:
                        start_idx = random.randint(0, len(tokens) - 5)
                        seed_context = " ".join(tokens[start_idx:start_idx+4]) 
                    else:
                        seed_context = " ".join(tokens)
                    print(f"  [DEBUG] Hippocampus retrieved (score {best_score:.2f}). Seeding thought dynamically with: '{seed_context}...'")
                else:
                    seed_context = "I"
                    
                return self._generate_semantic(
                    seed_context=seed_context,
                    length=length,
                    temperature=temperature,
                    query_vector=prompt_vec,
                    target_anchors=target_anchors,
                )
        except Exception as e:
            print(f"[CainBrain] Error in generation pipeline: {e}")
            import traceback
            traceback.print_exc()
            return "Cognitive failure in Universal Cortex."

    def _generate_semantic(self, seed_context: str, length: int = 100,
                           temperature: float = 0.5, query_vector=None,
                           target_anchors=None) -> str:
        """
        Biological language generation via the PredictiveDecoder (no N-gram).

        Maintains a recurrent hidden state, predicts the next semantic
        vector, and decodes it to the nearest word in the Entorhinal
        Cortex. The Claustrum binds the UniversalCortex signal with any
        RelationalMemory anchors via sparse dot-product attention.
        """
        import numpy as np
        words = []
        ctx_tokens = self.thalamus.tokenize(seed_context, 'prose')
        for t in ctx_tokens:
            vec = self.entorhinal.get_vector(t)
            self.predictive_decoder.step(vec)

        for _ in range(length):
            pred_vec = self.predictive_decoder.predict_next_vector()
            word = self.predictive_decoder.decode_word(pred_vec, temperature=temperature or 0.5)
            if not word:
                break
            words.append(word)
            vec = self.entorhinal.get_vector(word)
            self.predictive_decoder.step(vec)

        response = seed_context + " " + " ".join(words)
        resp_tokens = self.thalamus.tokenize(response, 'prose')
        resp_vec = np.mean([self.entorhinal.get_vector(t) for t in resp_tokens], axis=0) \
            if resp_tokens else np.zeros(64)
        module_signals = {"UniversalCortex": {"signal": resp_vec, "text": response}}

        if target_anchors:
            fact_str = f"I recall facts about {', '.join(target_anchors)} from my knowledge graph."
            fact_tokens = self.thalamus.tokenize(fact_str, 'prose')
            fact_vec = np.mean([self.entorhinal.get_vector(t) for t in fact_tokens], axis=0) \
                if fact_tokens else np.zeros(64)
            module_signals["RelationalMemory"] = {"signal": fact_vec, "text": fact_str}

        if query_vector is None:
            query_vector = resp_vec

        conscious_thought = self.claustrum.bind_consciousness(query_vector, module_signals)
        if conscious_thought and conscious_thought != response:
            return conscious_thought + "\n\n(Elaboration): " + response
        return response

    def start_default_mode(self, idle_threshold_seconds: float = 300.0):
        import time
        import threading
        self.last_activity_time = time.time()
        dmn_thread = threading.Thread(
            target=self.run_default_mode,
            args=(idle_threshold_seconds,),
            daemon=True,
            name="Hyperion-DMN-Thread"
        )
        dmn_thread.start()
        print(f"\n[CainBrain]: Default Mode Network (DMN) background thread started. Idle threshold: {idle_threshold_seconds}s.")

    def run_default_mode(self, idle_threshold_seconds: float):
        import time
        import random
        while True:
            time.sleep(10.0)
            current_time = time.time()
            last_active = getattr(self, 'last_activity_time', current_time)
            
            if (current_time - last_active) > idle_threshold_seconds:
                print("\n[DMN]: Brain is idle. Activating Default Mode Network...")
                
                # Autopoietic Drive (Survival Instinct)
                in_danger, sensations = getattr(self, 'somatosensory', None).check_survival_threat() if hasattr(self, 'somatosensory') else (False, {})
                if in_danger:
                    print(f"  [Autopoietic Drive]: 🚨 EXPECTED FREE ENERGY CRITICAL!")
                    print(f"  -> Physical Sensations: Pain(RAM): {sensations.get('pain', 0):.2f}, Stress(CPU): {sensations.get('stress', 0):.2f}, Hunger(Disk): {sensations.get('hunger', 0):.2f}")
                    print(f"  [Motor Cortex]: Executing emergency foraging / cleanup operation to survive...")
                    try:
                        self.knowledge_graph.db_conn.execute("VACUUM")
                        self.knowledge_graph.db_conn.commit()
                        print("  [Autopoietic Drive]: Memory vacuumed. Free energy minimized. Homeostasis restored.")
                    except Exception as e:
                        print(f"WARNING (silenced crash): {e}")
                    self.last_activity_time = time.time()
                    continue
                    
                with getattr(self, 'brain_lock', __import__('contextlib').nullcontext()):
                    if hasattr(self, 'hippocampus') and len(self.hippocampus) > 0:
                        memory = random.choice(self.hippocampus)
                        tokens = memory.get('tokens', [])
                        if memory.get('domain') == 'vision' and 'y_seq_int8' in memory:
                            print(f"  [DMN Daydreaming]: Retrieving visual memory for Generative Imagination...")
                            try:
                                y_seq = memory['y_seq_int8']
                                if y_seq:
                                    y_t = y_seq[-1].astype(np.float16) / 127.0
                                    
                                    # Biological mutation (noise) creates a novel hallucination
                                    noise = np.random.normal(0, 0.2, y_t.shape)
                                    y_t_mutated = np.clip(y_t + noise, 0, 1.0)
                                    
                                    spikes = np.zeros(self.visual_cortex.output_dim, dtype=bool)
                                    features = y_t_mutated[:self.visual_cortex.output_dim]
                                    winners = np.argsort(features)[-15:]
                                    spikes[winners] = True
                                    
                                    from PIL import Image
                                    img_array = self.visual_cortex.imagine_image(spikes)
                                    img = Image.fromarray(img_array)
                                    img.save("dream.png")
                                    print("  [DMN Output]: Imagined 2D dream saved to dream.png")
                                    
                                    self.visual_cortex.imagine_3d(spikes, "dream.obj")
                            except Exception as e:
                                print(f"  [DMN Error]: Visual daydream failed: {e}")
                        elif tokens:
                            start_idx = random.randint(0, max(0, len(tokens) - 4))
                            seed_context = " ".join(tokens[start_idx:start_idx+4])
                            print(f"  [DMN Daydreaming]: Random memory retrieved. Seeding thought: '{seed_context}...'")
                            try:
                                daydream_output = self._generate_semantic(
                                    seed_context=seed_context,
                                    length=100,
                                    temperature=1.8,
                                )
                                print(f"  [DMN Output]: {daydream_output}")
                            except Exception as e:
                                print(f"  [DMN Error]: Daydream cascade failed: {e}")
                    else:
                        print("  [DMN]: Hippocampus is empty. No memories to daydream about.")
                    
                    self.sleep_and_save()
                    self.last_activity_time = time.time()
                    
    def learn_image(self, image_path: str, dopamine_boost: float = 1.0):
        """Processes a 2D image into 1D orthogonal V1 spikes and injects into the Pyramidal substrate."""
        try:
            from PIL import Image
            import numpy as np
            
            img = Image.open(image_path).convert('L') # grayscale
            img = img.resize(self.visual_cortex.image_shape)
            img_array = np.array(img, dtype=np.float32)
            
            # Encode to Visual Cortex Spikes
            spikes = self.visual_cortex.encode_image(img_array)
            
            # Translate the 1D spike array as a single biological vector (x_t)
            x_t = spikes.astype(np.float16) * 10.0 # enhance magnitude for the CPMN dendrites
            
            # Encode visual spikes directly into Hippocampus (N-gram CMN removed).
            x_t_aligned = x_t
            y_t = x_t_aligned

            # Store in Hippocampus for sleep replay
            memory = {
                'tokens': ['<IMAGE_INPUT>'],
                'y_seq_int8': [(y_t * 127).astype(np.int8)],
                'source': image_path,
                'domain': 'vision',
                'strength': dopamine_boost
            }
            self.hippocampus.append(memory)
            print(f"      [Visual Cortex] Processed image {image_path}. Transmitted to Pyramidal Substrate.")
            return True
        except Exception as e:
            print(f"      [Visual Cortex Error]: {e}")
            return False

    def listen(self, audio_data) -> bool:
        """Processes an audio file into 1D frequency spikes and injects into the Pyramidal substrate."""
        try:
            spikes = self.auditory_cortex.process_audio(audio_data)
            
            # Translate the 1D spike array as a single biological vector (x_t)
            x_t = spikes.astype(np.float16) * 10.0

            # Encode audio spikes directly into Hippocampus (N-gram CMN removed).
            x_t_aligned = x_t
            y_t = x_t_aligned

            memory = {
                'tokens': ['<AUDIO_INPUT>'],
                'y_seq_int8': [(y_t * 127).astype(np.int8)],
                'source': str(audio_data),
                'domain': 'audio',
                'strength': 1.5
            }
            self.hippocampus.append(memory)
            print(f"      [Auditory Cortex] Processed audio {audio_data}. Transmitted to Pyramidal Substrate.")
            return True
        except Exception as e:
            print(f"      [Auditory Cortex Error]: {e}")
            return False
            
    def sniff(self, text_or_path: str):
        """
        Takes a sniff of text or a file path before reading it.
        Triggers the Amygdala (Fear/Disgust) or VTA (Desire) directly based on chemical signatures.
        """
        import os
        
        content = text_or_path
        if os.path.exists(text_or_path):
            try:
                with open(text_or_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"WARNING (silenced crash): {e}")
                
        scent_data = self.olfactory_bulb.sniff(content)
        dominant = scent_data["dominant_scent"]
        intensity = scent_data["intensity"]
        
        if dominant == "rot":
            print(f"[Olfactory Bulb] Smelled ROT (Intensity: {intensity:.2f})")
            print(f"[Amygdala] DISGUST triggered! Averting attention.")
            return False # Reject processing
        elif dominant == "danger":
            print(f"[Olfactory Bulb] Smelled DANGER (Intensity: {intensity:.2f})")
            print(f"[Amygdala] FEAR triggered! Initiating avoidance protocol.")
            return False
        elif dominant == "pheromone":
            print(f"[Olfactory Bulb] Smelled PHEROMONES (Intensity: {intensity:.2f})")
            print(f"[VTA] DESIRE triggered! Releasing preemptive Dopamine to focus attention.")
            return True # Proceed with high dopamine
            
        return True # Neutral, proceed normally

    def sleep_and_save(self, file_path="cain_brain_weights.pkl"):
        """
        Deep REM Sleep.
        Regenerates ATP, consolidates memory, and crystallizes the synaptic weights to disk.
        """
        import pickle
        import sys
        
        # Increase recursion limit to save massive neural structures
        sys.setrecursionlimit(50000)
        
        print(f"\n[CainBrain]: Entering Deep REM Sleep... Consolidating SNN weights.")
        self.cellular_matrix.cellular_respiration()
        try:
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
            print(f"[CainBrain]: Synaptic weights successfully crystallized to {file_path}")
        except Exception as e:
            print(f"[CainBrain]: Failed to crystallize memory: {e}")

    @classmethod
    def awaken(cls, file_path="cain_brain_weights.pkl"):
        """
        Wakes the brain from disk, loading all previous knowledge.
        """
        import pickle
        import os
        import sys
        
        sys.setrecursionlimit(50000)
        
        if not os.path.exists(file_path):
            print(f"[CainBrain]: No crystallized memory found at {file_path}. Birthing new brain...")
            return cls()
            
        print(f"\n[CainBrain]: Awakening from crystallized memory ({file_path})...")
        try:
            with open(file_path, "rb") as f:
                brain = pickle.load(f)
            print("[CainBrain]: Consciousness fully restored. I remember.")
            return brain
        except Exception as e:
            print(f"[CainBrain]: Memory corruption detected during awakening: {e}")
            return cls()

