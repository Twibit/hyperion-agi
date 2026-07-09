import numpy as np
from nlp_engine.layers.blanket import CAINBlanket
from nlp_engine.layers.predictive import PredictiveMinicolumn
from nlp_engine.optim.tree_search import CAINTreeSearch

class CorpusCallosum:
    """
    Category Theory Functor mapping between the Continuous Right Brain 
    and the Discrete/Physics Left Brain (CAIN).
    Ensures mathematical synchronization between creative heuristics and strict bounds.
    """
    def __init__(self, dim):
        self.dim = dim
        # Identity matrix for direct mapping, can be trained for complex dimensional shifts
        self.functor_matrix = np.eye(dim) 
        
    def map_right_to_left(self, right_brain_vector):
        """Maps creative, continuous heuristic ideas into mathematically bounded physics nodes."""
        return self.functor_matrix @ right_brain_vector
        
    def map_left_to_right(self, left_brain_state):
        """Maps strict, validated physics bounds back out to the creative language swarm."""
        return np.linalg.pinv(self.functor_matrix) @ left_brain_state


class HyperionRightBrain:
    """
    The Creative Swarm.
    Generates heuristics, linguistic predictions, and unconstrained ideas.
    In a live environment, this interfaces with external LLMs and API Swarms.
    """
    def __init__(self, dim):
        self.dim = dim
        
    def generate_proposals(self, n_proposals=1000):
        """Generates continuous 'messy' hallucinated ideas/predictions."""
        # Simulated as gaussian noise representing diverse creative thought vectors
        return [np.random.randn(self.dim) * 0.5 for _ in range(n_proposals)]


class HyperionLeftBrain:
    """
    The CAIN Physics Engine.
    Acts as the absolute source of mathematical truth and error-free bounds.
    """
    def __init__(self, dim):
        self.blanket = CAINBlanket(dim)
        self.minicolumn = PredictiveMinicolumn(in_dim=dim, out_dim=dim, sparsity=0.1)
        self.search_engine = CAINTreeSearch(dim, self.minicolumn.generative_weights)
        
    def evaluate_and_collapse(self, mapped_proposals, target_prior):
        """
        Runs the Stockfish Expected Free Energy (EFE) Alpha-Beta protocol 
        to find the single mathematically perfect proposal.
        """
        best_policy, best_efe, pruned_count = self.search_engine.aggressive_lookahead(
            current_state=self.blanket.internal_state, 
            target_prior=target_prior, 
            possible_policies=mapped_proposals
        )
        return best_policy, best_efe, pruned_count


class HyperionOrganism:
    """
    The fully unified, synchronous organism.
    Combines the generative power of the Right Brain with the absolute truth of the CAIN Left Brain.
    """
    def __init__(self, dim=16):
        self.dim = dim
        self.left_brain = HyperionLeftBrain(dim)
        self.right_brain = HyperionRightBrain(dim)
        self.corpus_callosum = CorpusCallosum(dim)
        
    def process_thought(self, target_concept, n_proposals=5000):
        """
        The Master Execution Loop:
        1. Right Brain generates wildly creative proposals.
        2. Corpus Callosum Maps them to CAIN physics vectors.
        3. Left Brain brutally evaluates them using Expected Free Energy bounds.
        4. The winning, error-free thought is selected and synced back to both hemispheres.
        """
        print("\n[Hyperion] 🧠 Thought initiated. Engaging dual-hemisphere sync...")
        
        # 1. Right Brain Generation
        creative_proposals = self.right_brain.generate_proposals(n_proposals=n_proposals)
        
        # 2. Sync / Mapping (Functor application)
        mapped_proposals = [self.corpus_callosum.map_right_to_left(p) for p in creative_proposals]
        
        # 3. Left Brain Physics Evaluation (CAIN Alpha-Beta Search)
        best_physics_state, best_efe, pruned = self.left_brain.evaluate_and_collapse(
            mapped_proposals, target_concept
        )
        
        print(f"[Hyperion] ⚔️ Left Brain evaluated {n_proposals} Right Brain ideas.")
        print(f"[Hyperion] ✂️ Left Brain pruned {pruned} hallucinated ideas instantly via FEP bounds.")
        
        # 4. Assimilation & Sync Back
        # Update Left Brain's internal state
        self.left_brain.blanket.internal_state += (self.left_brain.minicolumn.generative_weights @ best_physics_state)
        
        # Reverse Functor: Push the verified truth back to the Right Brain
        synced_creative_state = self.corpus_callosum.map_left_to_right(self.left_brain.blanket.internal_state)
        
        # Verify Extrinsic Error (Surprise)
        final_prediction = self.left_brain.minicolumn.view(self.left_brain.blanket)
        extrinsic_error = np.mean((target_concept - final_prediction)**2)
        
        print(f"[Hyperion] ⚡ Thought resolved. Best Expected Free Energy: {best_efe:.4f}")
        print(f"[Hyperion] 🔗 Hemispheres synced. Final Error vs Truth: {extrinsic_error:.4f}\n")
        
        return synced_creative_state

if __name__ == "__main__":
    print("Initializing Unified Hyperion Organism...")
    dual_planner = HyperionOrganism(dim=16)
    
    # Target concept is a 'truth' vector Hyperion needs to figure out
    np.random.seed(42)
    truth_concept = np.random.randn(16)
    
    dual_planner.process_thought(target_concept=truth_concept, n_proposals=10000)
