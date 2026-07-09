import numpy as np

class EpigeneticDNA:
    """
    Stores the SNN matrix weights and their Epigenetic Trauma tags.
    """
    def __init__(self, weights):
        self.weights = weights
        # Methylation tags (0.0 to 1.0). High methylation = blocked/feared pathway (Trauma)
        self.methylation_tags = np.zeros_like(weights, dtype=np.float32)

class EvolutionaryGauntlet:
    """
    The Generative Evolution Protocol.
    Simulates Reward, Pain, Trauma, Death, and Reproduction across generations of CAIN organisms.
    """
    def __init__(self, population_size=10):
        self.population_size = population_size
        self.population_dna = []
        
    def initialize_population(self, shape):
        for _ in range(self.population_size):
            initial_weights = np.random.randint(-10, 10, size=shape, dtype=np.int16)
            self.population_dna.append(EpigeneticDNA(initial_weights))
            
    def natural_selection(self, fitness_scores, amygdala_trauma_signals):
        """
        Evaluates a generation.
        fitness_scores: The inverse of Free Energy (Reward/Success).
        amygdala_trauma_signals: Spikes from the Amygdala indicating extreme pain/failure.
        """
        print("\n[Evolution] Generation complete. Commencing Natural Selection...")
        
        # 1. Death (Dynamic Apoptosis)
        # Cull the bottom 50% of the population who failed the logic/physics walls
        sorted_indices = np.argsort(fitness_scores)[::-1] # Highest fitness first
        survivors_indices = sorted_indices[:self.population_size // 2]
        deaths = self.population_size - len(survivors_indices)
        print(f"[Evolution] ☠️ {deaths} organisms died. DNA discarded.")
        
        new_generation = []
        
        # 2. Epigenetic Trauma Absorption & Reproduction
        for idx in survivors_indices:
            parent_dna = self.population_dna[idx]
            pain_level = amygdala_trauma_signals[idx]
            
            # Absorb Trauma: If the organism survived but experienced massive pain/error,
            # it physically "methylates" (locks) the synapses that caused the pain.
            if pain_level > 0:
                print(f"[Evolution] 🩸 Survivor {idx} absorbed Trauma. Encoding epigenetic fear.")
                parent_dna.methylation_tags += (pain_level * 0.1)
                parent_dna.methylation_tags = np.clip(parent_dna.methylation_tags, 0.0, 1.0)
                
            # 3. Reproduction (Mitosis / Mutation)
            # Child inherits the base logic but mutates slightly to explore new ideas
            mutation_noise = np.random.randint(-2, 3, size=parent_dna.weights.shape).astype(np.int16)
            child_weights = parent_dna.weights + mutation_noise
            
            # The Critical Rule: Epigenetic Inheritance
            # The child instinctively fears what traumatized the parent. The mutated weights 
            # are mathematically blocked (zeroed out) by the parent's trauma tags.
            epigenetic_blocker = (1.0 - parent_dna.methylation_tags)
            child_weights = np.round(child_weights * epigenetic_blocker).astype(np.int16)
            
            child_dna = EpigeneticDNA(child_weights)
            # Pass the trauma tags down to the next generation
            child_dna.methylation_tags = parent_dna.methylation_tags.copy() 
            
            # Both parent and child pass to the next generation
            new_generation.extend([parent_dna, child_dna])
            
        self.population_dna = new_generation
        print("[Evolution] 🌱 New generation seeded. Trauma and Reward inherited.")
