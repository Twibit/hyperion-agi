import numpy as np

class EntorhinalCortex:
    """
    Entorhinal Cortex Layer: Semantic Grounding (Embeddings).
    
    This layer maintains a continuous vector space where tokens with similar meanings
    have similar vectors. It uses biologically-inspired Hebbian learning to update
    representations based on co-occurrence, mimicking how the brain forms 
    associative memories in the medial temporal lobe.
    """
    def __init__(self, embedding_dim=64, learning_rate=0.05, decay_rate=0.0001, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        
        # Token to index mapping (analogous to specific neuron assemblies)
        self.token_to_idx = {}
        self.idx_to_token = {}
        
        # Embeddings matrix, dynamically expandable
        self.embeddings = np.empty((0, embedding_dim))
        
    def _add_token(self, token):
        """Neurogenesis: Allocate a new representation for a novel token."""
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            # Initialize with small random weights (like new synaptic connections)
            new_vector = np.random.normal(0, 0.1, (1, self.embedding_dim))
            if self.embeddings.shape[0] == 0:
                self.embeddings = new_vector
            else:
                self.embeddings = np.vstack([self.embeddings, new_vector])
        return self.token_to_idx[token]

    def get_vector(self, token):
        """Retrieve the continuous vector representation of a token."""
        if token not in self.token_to_idx:
            self._add_token(token)
        idx = self.token_to_idx[token]
        return self.embeddings[idx]

    def process_sequence(self, tokens, window_size=2):
        """
        Process a sequence of tokens and apply Hebbian learning.
        "Neurons that fire together, wire together."
        Tokens that co-occur within the window size will have their representations pulled closer.
        """
        if not tokens:
            return

        # Ensure all tokens are added
        indices = [self._add_token(token) for token in tokens]

        # Apply biological Hebbian updating
        for i, target_idx in enumerate(indices):
            # Define the context window bounds (spatial/temporal proximity)
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            
            context_indices = indices[start:i] + indices[i+1:end]
            
            if not context_indices:
                continue
                
            # Average context vector (aggregate surrounding neural activity)
            context_vector = np.mean(self.embeddings[context_indices], axis=0)
            
            # Target vector
            target_vector = self.embeddings[target_idx]
            
            # Hebbian update: Move target vector towards context vector
            # The more they co-occur, the more similar their representations become.
            delta = self.learning_rate * (context_vector - target_vector)
            self.embeddings[target_idx] += delta
            
            # Negative Sampling (Contrastive Learning)
            # Pick a few random tokens to push away from to prevent mode collapse
            num_negatives = 5
            if len(self.embeddings) > 1:
                # choose random negatives
                neg_indices = np.random.choice(len(self.embeddings), min(num_negatives, len(self.embeddings)), replace=False)
                for neg_idx in neg_indices:
                    if neg_idx != target_idx and neg_idx not in context_indices:
                        neg_vector = self.embeddings[neg_idx]
                        # Push away
                        diff = self.embeddings[target_idx] - neg_vector
                        dist = np.linalg.norm(diff)
                        if dist < 1e-4:
                            # if perfectly collapsed, push in random direction
                            diff = np.random.normal(0, 0.1, self.embedding_dim)
                            dist = np.linalg.norm(diff)
                        # Repel stronger when closer
                        push_delta = (self.learning_rate * 0.5) * (diff / (dist + 0.1))
                        self.embeddings[target_idx] += push_delta
            
            # Homeostatic synaptic scaling (normalization) to prevent explosive growth
            norm = np.linalg.norm(self.embeddings[target_idx])
            if norm > 0:
                self.embeddings[target_idx] /= norm
                
        # Global synaptic decay (forgetting) to emulate pruning of unused connections
        # We only apply a tiny decay to prevent complete collapse.
        self.embeddings *= (1.0 - self.decay_rate)
        
    def similarity(self, token1, token2):
        """Calculate cosine similarity between two tokens' representations."""
        if token1 not in self.token_to_idx or token2 not in self.token_to_idx:
            return 0.0
        
        vec1 = self.embeddings[self.token_to_idx[token1]]
        vec2 = self.embeddings[self.token_to_idx[token2]]
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
    def get_embeddings(self, tokens, embedding_dim=16):
        import math
        embeddings = []
        for i, token in enumerate(tokens):
            vector = []
            base_hash = sum(ord(c) * (k + 1) for k, c in enumerate(token)) if token else 0
            for j in range(embedding_dim):
                feature_val = ((base_hash * (j + 17)) % 101) / 100.0 - 0.5
                if j % 2 == 0:
                    pos_enc = math.sin(i / (10000 ** (j / embedding_dim)))
                else:
                    pos_enc = math.cos(i / (10000 ** ((j - 1) / embedding_dim)))
                vector.append(feature_val + pos_enc)
            embeddings.append(vector)
        return embeddings
