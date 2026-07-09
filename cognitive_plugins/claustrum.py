import numpy as np

class Claustrum:
    """
    Global Workspace Theory / Consciousness Binder.
    Takes inputs from all cortical modules and forces them to 'vote' based on precision.
    The highest precision signal achieves 'ignition' and is broadcasted globally.
    """
    def __init__(self):
        print(r"[Claustrum] Global Workspace online. Integrating $\Phi$.")
        self.current_conscious_thought = None
        
    def bind_consciousness(self, query_vector, module_signals):
        """
        module_signals: A dictionary of {module_name: {"signal": sdr_vector, "text": string}}
        query_vector: The current context SDR vector.
        Returns the weighted combination of text or the top attended text.
        """
        import rust_backend_shim
        
        print("\n[Claustrum] Competing modalities for Global Ignition (Attention):")
        keys = []
        names = []
        texts = []
        for name, data in module_signals.items():
            names.append(name)
            # Compute dot product as the raw precision/match score
            signal_vec = data["signal"]
            score = np.dot(query_vector, signal_vec)
            keys.append(score)
            texts.append(data.get("text", ""))
            
        if not keys:
            return None
            
        # Use Rust Core to compute Sparse Attention (Softmax with temperature)
        attention_weights = rust_backend_shim.sparse_attention(keys, [], 1.0) # query isn't used in this fast-path if keys are already dot-products
        
        best_module = None
        highest_weight = -1.0
        
        for name, weight in zip(names, attention_weights):
            print(f"  -> {name} | Attention Weight: {weight:.3f}")
            if weight > highest_weight:
                highest_weight = weight
                best_module = name
                
        # We can return the text of the module with highest attention, 
        # or combine them if they cross a threshold.
        combined_thought = []
        for text, weight in zip(texts, attention_weights):
            if weight > 0.2: # Ignition threshold
                combined_thought.append(text)
                
        if combined_thought:
            self.current_conscious_thought = " ".join(combined_thought)
            print(f"[Claustrum] IGNITION ACHIEVED! Attended modules broadcast globally.")
            return self.current_conscious_thought
        
        return None
