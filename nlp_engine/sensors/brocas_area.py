from typing import List, Dict, Set

class BrocaMotorCortex:
    """
    Broca's Area Motor Architect (Right Brain Swarm)
    Generative output module for decoding Sparse Distributed Representations (SDRs).
    """
    
    def __init__(self, sdr_size: int = 2048, activation_threshold: float = 0.5):
        """
        Initializes the BrocaMotorCortex.
        
        Args:
            sdr_size: The total bit-capacity of the SDR (default is 2048).
            activation_threshold: Minimum overlap ratio required to activate a token.
        """
        self.sdr_size = sdr_size
        self.activation_threshold = activation_threshold

    def decode_sdr_to_sequence(
        self, 
        active_sdr: List[int], 
        reverse_mapping: Dict[str, List[int]]
    ) -> List[str]:
        """
        Decodes a 2048-bit SDR back into a sequence of tokens using overlap analysis.
        
        Args:
            active_sdr: The 2048-bit array (can be a dense list of 0s/1s or a sparse list of active indices).
            reverse_mapping: Dictionary mapping tokens (or text) to their reference SDR representations.
            
        Returns:
            A sequenced list of decoded tokens.
        """
        sparse_active_sdr = self._to_sparse(active_sdr)
        
        token_scores = []
        for token, token_sdr in reverse_mapping.items():
            sparse_token_sdr = self._to_sparse(token_sdr)
            
            # Compute intersection (overlap) between active bits
            overlap = len(sparse_active_sdr.intersection(sparse_token_sdr))
            
            if len(sparse_token_sdr) > 0:
                # Activation ratio measures how much of the target token's meaning is present
                activation_ratio = overlap / len(sparse_token_sdr)
                
                if activation_ratio >= self.activation_threshold:
                    token_scores.append((token, activation_ratio, overlap))
        
        # Order tokens by activation strength and raw overlap to emulate motor sequencing
        token_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        decoded_sequence = [token for token, _, _ in token_scores]
        
        return self._apply_motor_sequencing(decoded_sequence)

    def generate_text(self, active_sdr: List[int], reverse_mapping: Dict[str, List[int]]) -> str:
        """
        Decodes the SDR and generates a cohesive text string.
        """
        sequence = self.decode_sdr_to_sequence(active_sdr, reverse_mapping)
        return " ".join(sequence)

    def _to_sparse(self, sdr: List[int]) -> Set[int]:
        """
        Helper method to normalize SDR arrays into a sparse Set of active bit indices.
        """
        if not sdr:
            return set()
            
        # Detect if it's a dense 2048-bit array (composed of 0s and 1s)
        if len(sdr) == self.sdr_size and all(bit in (0, 1) for bit in sdr):
            return {i for i, bit in enumerate(sdr) if bit == 1}
            
        # Otherwise, assume it's already a sparse array of indices
        return set(sdr)

    def _apply_motor_sequencing(self, tokens: List[str]) -> List[str]:
        """
        Syntactic alignment and filtering.
        In a broader Swarm architecture, this integrates with grammar constraint modules.
        Here we provide a foundational sequencing that maintains activation order.
        """
        # A simple uniqueness filter simulating temporal refractory periods
        seen = set()
        sequenced = []
        for token in tokens:
            if token not in seen:
                sequenced.append(token)
                seen.add(token)
        return sequenced
