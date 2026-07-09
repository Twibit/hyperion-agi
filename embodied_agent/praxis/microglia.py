"""
Microglial Immune Pruning — V11 (Aurora-Praxis).
=================================================

Microglia are the resident immune cells of the brain. They actively patrol
the neural tissue, phagocytizing (eating) dead cells and pruning weak or
noisy synapses.

In Aurora V11, the Microglia module operates during the sleep cycle.
It traverses the StigmergicGraph (the sparse concept network) and hunts
down nodes/edges that are contributing to noise or are simply unused,
preventing catastrophic forgetting through biological garbage collection.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any
from ..cortex.stigmergic_graph import StigmergicGraph

class Microglia:
    def __init__(self, 
                 phagocytosis_threshold: float = 0.05,
                 node_death_threshold: float = 1.0,
                 seed: int = 1729):
        """
        phagocytosis_threshold: Edges with weight below this are pruned.
        node_death_threshold: Nodes with count <= this are marked for death.
        """
        self.edge_thresh = float(phagocytosis_threshold)
        self.node_thresh = float(node_death_threshold)
        self.rng = np.random.default_rng(seed)
        
        # Diagnostic tracking
        self._total_edges_pruned = 0
        self._total_nodes_pruned = 0
        
    def patrol_and_prune(self, graph: StigmergicGraph) -> Dict[str, Any]:
        """
        Sweep the graph during sleep, pruning weak synapses and dead nodes.
        Returns pruning statistics.
        """
        pruned_edges = 0
        pruned_nodes = 0
        
        # 1. Prune weak edges (synapses)
        surviving_edges = {}
        for edge, weight in graph._edges.items():
            if weight >= self.edge_thresh:
                surviving_edges[edge] = weight
            else:
                pruned_edges += 1
                
        graph._edges = surviving_edges
        
        # 2. Prune dead/noisy nodes
        if len(graph._counts) > 0:
            # Find nodes that have barely been used
            dead_indices = np.where(graph._counts <= self.node_thresh)[0]
            if len(dead_indices) > 0:
                dead_set = set(dead_indices)
                pruned_nodes = len(dead_set)
                
                # We zero out the counts and centroids, making them targets 
                # for the graph's `_add_node` eviction logic on the next cycle, 
                # effectively freeing the memory slots (phagocytosis).
                for idx in dead_indices:
                    graph._counts[idx] = -1.0 # Set to -1 so argmin definitely evicts them
                    graph._centroids[idx] = 0.0
                    
                # Ensure no edges connect to dead nodes
                surviving_edges = {
                    e: w for e, w in graph._edges.items()
                    if e[0] not in dead_set and e[1] not in dead_set
                }
                graph._edges = surviving_edges
                
        self._total_edges_pruned += pruned_edges
        self._total_nodes_pruned += pruned_nodes
        
        return {
            "edges_pruned_this_cycle": pruned_edges,
            "nodes_pruned_this_cycle": pruned_nodes,
            "total_edges_pruned": self._total_edges_pruned,
            "total_nodes_pruned": self._total_nodes_pruned
        }
