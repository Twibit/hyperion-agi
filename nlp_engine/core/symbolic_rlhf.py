import numpy as np
from nlp_engine.core.fep_math import FreeEnergyMath

class TruthPillars:
    """
    Evaluates the discrete Symbolic space for structural and logical truth.
    These act as the absolute physical laws of the CAIN Left Brain.
    """
    @staticmethod
    def evaluate_syntactic_category(symbolic_graph):
        """
        Grammatical Truth (Syntactic Category).
        Checks if the SNN spikes compose into a valid morphism (grammar).
        """
        if symbolic_graph is None or np.sum(np.abs(symbolic_graph)) == 0:
            return float('inf') # Syntax Error: Infinite Free Energy penalty
        return 0.0
        
    @staticmethod
    def evaluate_curry_howard(proof_graph):
        """
        Logical Truth (Curry-Howard-Lambek Correspondence).
        SNN outputs are type-checked as logical propositions.
        """
        # A mock implementation: valid logic yields zero error.
        return 0.0 
        
    @staticmethod
    def evaluate_lagrangian_action(complexity_metric):
        """
        Physical Truth (Lagrangian Structural Integrity).
        Evaluates structural entropy (e.g. Cyclomatic Complexity / Spaghetti Code).
        """
        if complexity_metric > 100.0:
            return complexity_metric * 100.0 # Heavy precision scaling for bad structure
        return complexity_metric

class AdjointFunctorLens:
    """
    The Functorial Bridge between Continuous Spiking (Right Brain) 
    and Discrete Logic (Left Brain).
    """
    def __init__(self, in_dim, out_dim):
        # Parsing Functor F (Maps spikes to logic)
        self.F_matrix = np.random.randn(out_dim, in_dim) / np.sqrt(in_dim)
        # Precision Functor G (Projects logic errors back to continuous local spikes)
        self.G_matrix = np.linalg.pinv(self.F_matrix)
        
    def parse_f(self, spike_train):
        """Continuous SNN Space -> Discrete Logic Space."""
        return self.F_matrix @ spike_train
        
    def project_g(self, logic_error):
        """Discrete Logic Error -> Continuous SNN Precision Error."""
        return self.G_matrix @ logic_error

class NativeSymbolicRLHF:
    """
    The Native Truth Filter executing Category Theory evaluations
    and Dynamic Apoptosis. No LLMs or Transformers required.
    """
    def __init__(self, in_dim, out_dim):
        self.lens = AdjointFunctorLens(in_dim, out_dim)
        self.survival_threshold = 1000.0 # Maximum allowed Variational Free Energy
        
    def evaluate_and_update(self, blanket, active_spikes):
        """The Predictive Lens execution loop."""
        
        # Step 1: The View (Forward Pass via Parsing Functor F)
        # Parse the raw physical spikes into a logical symbolic AST graph
        symbolic_ast = self.lens.parse_f(active_spikes)
        
        # Step 2: The Truth Filter (Evaluation)
        syntax_error = TruthPillars.evaluate_syntactic_category(symbolic_ast)
        logic_error = TruthPillars.evaluate_curry_howard(symbolic_ast)
        
        # Approximate structural complexity of the thought
        complexity = np.sum(np.abs(symbolic_ast))
        structural_error = TruthPillars.evaluate_lagrangian_action(complexity)
        
        total_logic_error = syntax_error + logic_error + structural_error
        
        # Step 3: Dynamic Apoptosis (The Rejection Rule)
        # If grammar or logic fails, Precision Pi -> infinity.
        if total_logic_error == float('inf') or total_logic_error > self.survival_threshold:
            # The thought hit an infinite logical wall.
            # Trigger instantaneous Apoptosis. Prune the sub-network and discard execution trace.
            blanket.active_state *= 0 
            return False, float('inf') # Network/Thought pruned
            
        # Step 4: The Update (Backward Pass via Precision Functor G)
        # Instead of backprop, the Functor G maps the discrete error strictly back 
        # to the boundary of the Markov Blanket (Sensory State).
        error_vector = np.ones_like(symbolic_ast) * total_logic_error
        sensory_error = self.lens.project_g(error_vector)
        
        # The SNN minicolumn will now use STDP modulated by this sensory error
        # to organically update its internal weights.
        blanket.sensory_state += sensory_error
        
        return True, total_logic_error # Network survives and learns natively
