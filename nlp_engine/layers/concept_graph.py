"""
Concept Graph module for Macro-Planning in zero-FLOP architectures.

This module implements a state machine graph where:
- V (Vertices): Set of abstract concepts.
- E (Edges): Directed transitions indicating logical progression between concepts.
"""

import re
from typing import List, Dict, Set

class ConceptGraph:
    def __init__(self):
        # A topological adjacency list for our state machine transitions.
        # This defines the edges E in our concept graph.
        self.adjacency: Dict[str, List[str]] = {
            'bug': ['cause', 'fix'],
            'error': ['cause', 'fix'],
            'cause': ['fix', 'test'],
            'fix': ['test', 'deploy'],
            'feature': ['design', 'implement'],
            'design': ['implement', 'test'],
            'implement': ['test', 'deploy'],
            'concept': ['design', 'prototype'],
            'prototype': ['test']
        }
        self.default_outline = ['introduction', 'body', 'conclusion']

    def _parse_prompt(self, prompt: str) -> List[str]:
        """
        Lexical mapping function f: text -> V.
        Extracts foundational concepts from raw text.
        """
        tokens = re.findall(r'\b\w+\b', prompt.lower())
        concepts = [token for token in tokens if token in self.adjacency]
        return concepts

    def generate_outline(self, prompt: str) -> List[str]:
        """
        Given an initial stimulus (prompt), identifies the starting vertex
        and performs a deterministic graph walk to generate a sequence (outline).
        """
        extracted_concepts = self._parse_prompt(prompt)
        
        if not extracted_concepts:
            return self.default_outline
            
        # Initialize state machine at the first matched concept
        start_node = extracted_concepts[0]
        outline = [start_node]
        current_node = start_node
        
        # State machine traversal (linear walk)
        visited: Set[str] = {start_node}
        
        while current_node in self.adjacency:
            next_nodes = self.adjacency[current_node]
            
            # Select next state (deterministic, acyclic progression)
            next_node = None
            for candidate in next_nodes:
                if candidate not in visited:
                    next_node = candidate
                    break
                    
            if next_node:
                outline.append(next_node)
                visited.add(next_node)
                current_node = next_node
            else:
                # Terminal state reached
                break
                
        return outline
