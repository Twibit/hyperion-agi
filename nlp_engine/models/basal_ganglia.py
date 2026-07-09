import numpy as np
from nlp_engine.optim.tree_search import CAINTreeSearch

class BasalGangliaPolicyGate:
    """
    The Basal Ganglia: The Master Action Gatekeeper.
    It natively maps to our Expected Free Energy (EFE) Alpha-Beta Pruning protocol.
    Instead of executing every thought the Cortex proposes, the BG strictly evaluates
    them and inhibits everything except the absolute optimal path.
    """
    def __init__(self, state_dim, transition_matrix):
        # The Striatum is biologically equivalent to our FESENN evaluator
        self.striatum = CAINTreeSearch(state_dim, transition_matrix)
        
    def select_action(self, current_state, target_prior, proposed_policies):
        """
        The Cortex proposes multiple diverse actions (policies).
        The Basal Ganglia evaluates their Expected Free Energy and gates them.
        """
        # The Globus Pallidus physically executes the Active Alpha-Beta pruning
        best_policy, best_efe, pruned_count = self.striatum.aggressive_lookahead(
            current_state, target_prior, proposed_policies
        )
        
        if best_policy is None:
            # Complete behavioral inhibition (Freezing Response)
            # If all proposed policies hit infinite error walls, execute nothing.
            return np.zeros_like(current_state), float('inf'), pruned_count
            
        # Dopaminergic Go-Signal: The winning policy is disinhibited and executed
        return best_policy, best_efe, pruned_count
