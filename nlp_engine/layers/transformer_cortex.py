import numpy as np

class AttentionHead:
    """
    The Mathematical Core of an LLM: Scaled Dot-Product Self-Attention.
    This allows Hyperion to understand context (e.g., distinguishing 'apple' the fruit from 'Apple' the company).
    """
    def __init__(self, d_model: int, d_k: int):
        self.d_k = d_k
        # Initialize weight matrices for Queries, Keys, and Values
        self.W_q = np.random.randn(d_model, d_k) * 0.1
        self.W_k = np.random.randn(d_model, d_k) * 0.1
        self.W_v = np.random.randn(d_model, d_k) * 0.1

    def forward(self, x: np.ndarray) -> tuple:
        """
        x shape: (sequence_length, d_model)
        Returns: Contextualized output and attention weights
        """
        # 1. Linear projections
        Q = np.dot(x, self.W_q) # (seq_len, d_k)
        K = np.dot(x, self.W_k) # (seq_len, d_k)
        V = np.dot(x, self.W_v) # (seq_len, d_k)

        # 2. Scaled Dot-Product (How much each word cares about other words)
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k) # (seq_len, seq_len)

        # 3. Softmax to get probabilities (attention weights)
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # 4. Multiply by Values to get contextualized embeddings
        output = np.dot(attention_weights, V) # (seq_len, d_k)

        return output, attention_weights

class TransformerBlock:
    """A complete Transformer layer rivaling modern LLM architecture."""
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Multi-Head Attention
        self.heads = [AttentionHead(d_model, self.d_k) for _ in range(num_heads)]
        self.W_o = np.random.randn(num_heads * self.d_k, d_model) * 0.1
        
        # Feed Forward Network
        self.W1 = np.random.randn(d_model, d_model * 4) * 0.1
        self.b1 = np.zeros(d_model * 4)
        self.W2 = np.random.randn(d_model * 4, d_model) * 0.1
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Multi-Head Attention
        head_outputs = []
        for head in self.heads:
            out, _ = head.forward(x)
            head_outputs.append(out)
            
        # Concatenate heads and project
        multi_head_out = np.concatenate(head_outputs, axis=-1)
        attention_out = np.dot(multi_head_out, self.W_o)
        
        # Residual Connection & LayerNorm (Simplified)
        x = x + attention_out
        x = (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-8)
        
        # Feed Forward (ReLU activation)
        ff_hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)
        ff_out = np.dot(ff_hidden, self.W2) + self.b2
        
        # Residual Connection & LayerNorm
        x = x + ff_out
        x = (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-8)
        
        return x

class TransformerCortex:
    """The Ultimate Semantic Understanding Engine"""
    def __init__(self, vocab_size: int, d_model: int = 64, num_layers: int = 2, num_heads: int = 4):
        self.d_model = d_model
        self.embeddings = np.random.randn(vocab_size, d_model) * 0.1
        self.blocks = [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        self.vocab_size = vocab_size

    def get_positional_encoding(self, seq_len: int) -> np.ndarray:
        """Adds temporal awareness (word order) using sine/cosine waves."""
        pe = np.zeros((seq_len, self.d_model))
        position = np.arange(0, seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def process_sequence(self, token_indices: list[int]) -> np.ndarray:
        """Passes a sequence through the Transformer to achieve deep semantic understanding."""
        seq_len = len(token_indices)
        
        # 1. Lookup embeddings
        x = np.array([self.embeddings[idx] for idx in token_indices])
        
        # 2. Add Positional Encoding (Contextual time)
        x = x + self.get_positional_encoding(seq_len)
        
        # 3. Pass through Transformer Blocks
        for block in self.blocks:
            x = block.forward(x)
            
        return x
