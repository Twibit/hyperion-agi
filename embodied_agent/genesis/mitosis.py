"""
Somatic Evolution (Mitosis & Apoptosis) — V12 (Aurora-Genesis).
===============================================================

In contrast to Evolutionary Algorithms that breed entire populations of brains,
Aurora V12 implements Darwinian evolution *inside* a single brain's tissue.
This mirrors biological neurogenesis and synaptic pruning.

During the `sleep()` consolidation cycle, the SomaticEvolution core evaluates
the "fitness" of the different Mixture-of-Experts (MoE) clusters and anatomical
regions based on two metrics:
  1. Prediction Error (Accuracy)
  2. Metabolic Usage (Efficiency)

* Mitosis (Cellular Division): 
  If a neural cluster is highly used and highly accurate, it has reached its 
  capacity. It undergoes mitosis: splitting into two clusters to specialize 
  further, introducing slight *mutations* to its hyper-parameters (e.g., 
  learning rate, sparsity).
  
* Apoptosis (Programmed Cell Death): 
  If a mutated cluster is rarely used or consistently generates high errors, 
  the Microglia tag it, and it undergoes apoptosis. Its synaptic resources 
  are reallocated to healthy tissue.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List


class SomaticEvolution:
    """
    Manages the internal Darwinian evolution of the brain's hyperparameters.
    """
    def __init__(self, 
                 mutation_rate: float = 0.05, 
                 mitosis_threshold: float = 0.9,
                 apoptosis_threshold: float = 0.1,
                 seed: int = 1729):
        """
        mutation_rate: How much the hyperparameters drift during mitosis.
        mitosis_threshold: Fitness score > this triggers division.
        apoptosis_threshold: Fitness score < this triggers death.
        """
        self.mutation_rate = float(mutation_rate)
        self.mitosis_thresh = float(mitosis_threshold)
        self.apoptosis_thresh = float(apoptosis_threshold)
        self.rng = np.random.default_rng(seed)
        
        # Track statistics
        self._n_mitosis = 0
        self._n_apoptosis = 0
        
        # Per-region hyperparameter DNA.
        # Bootstrapped from real brain regions on the first sleep() call.
        # Map of: region_name -> {'learning_rate': float, 'sparsity': float, 'fitness': float}
        self.cellular_dna: Dict[str, Dict[str, float]] = {}

    def evaluate_and_evolve(self, 
                            region_usages: Dict[str, float], 
                            region_errors: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluates the fitness of regions and triggers mitosis/apoptosis.
        Called during the sleep cycle.
        
        region_usages: Dict of region_name -> active fraction [0.0 - 1.0]
        region_errors: Dict of region_name -> recent EMA prediction error
        """
        # Bootstrap: on first call, populate cellular_dna from the actual
        # brain regions so we track real tissue, not phantom entries.
        all_regions = set(region_usages.keys()) | set(region_errors.keys())
        for r in all_regions:
            if r not in self.cellular_dna:
                self.cellular_dna[r] = {
                    "learning_rate": 3e-4,
                    "sparsity": 0.02,
                    "fitness": 0.5,
                }

        apoptosis_events = []
        mitosis_events = []
        
        # Evaluate fitness and evolve
        keys = list(self.cellular_dna.keys())
        for region in keys:
            dna = self.cellular_dna[region]
            usage = region_usages.get(region, 0.0)
            error = region_errors.get(region, 1.0)
            
            # Fitness formula: High usage is good, high error is bad.
            # Normalised loosely to [0.0, 1.0]
            fitness = usage * (1.0 - min(error, 1.0))
            
            # EMA update of fitness
            dna["fitness"] = 0.8 * dna["fitness"] + 0.2 * fitness
            
            # --- APOPTOSIS ---
            # If the region is failing and it's not the last core region
            if dna["fitness"] < self.apoptosis_thresh and len(self.cellular_dna) > 1:
                del self.cellular_dna[region]
                apoptosis_events.append(region)
                self._n_apoptosis += 1
                continue
                
            # --- MITOSIS ---
            if dna["fitness"] > self.mitosis_thresh:
                # Split! Create a new specialized region
                child_name = f"{region}_mut_{self._n_mitosis}"
                child_dna = {
                    # Mutate learning rate (e.g. +/- 5%)
                    "learning_rate": dna["learning_rate"] * self.rng.uniform(1.0 - self.mutation_rate, 1.0 + self.mutation_rate),
                    # Mutate sparsity
                    "sparsity": dna["sparsity"] * self.rng.uniform(1.0 - self.mutation_rate, 1.0 + self.mutation_rate),
                    "fitness": dna["fitness"] * 0.5  # Start with half fitness
                }
                
                # Parent loses some fitness due to division cost
                dna["fitness"] *= 0.5
                
                self.cellular_dna[child_name] = child_dna
                mitosis_events.append(child_name)
                self._n_mitosis += 1
                
        return {
            "apoptosis_events": apoptosis_events,
            "mitosis_events": mitosis_events,
            "total_mitosis": self._n_mitosis,
            "total_apoptosis": self._n_apoptosis,
            "active_cellular_strains": len(self.cellular_dna)
        }
