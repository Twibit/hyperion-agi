import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nlp_engine.layers.sparse_hippocampus import SparseHippocampus
from nlp_engine.experiments.teacher_curriculum import OrthogonalLanguageEncoder
from nlp_engine.layers.visual_cortex import VisualCortex
from nlp_engine.layers.prefrontal_cortex import PrefrontalCortex

def generate_mock_image(is_cat=True):
    # A simple 28x28 grayscale image
    img = np.zeros((28, 28), dtype=np.uint8)
    if is_cat:
        # Draw a simple triangle for ear
        img[5:15, 5:15] = 255
    else:
        # Draw a circle for dog
        img[10:20, 10:20] = 255
    return img

def run_multimodal_exam():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Multimodal Logic & Sensory Exam              ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    brain_dim = 100000
    encoder = OrthogonalLanguageEncoder(spike_dim=brain_dim)
    sparse_brain = SparseHippocampus(num_neurons=brain_dim)
    visual_cortex = VisualCortex(output_dim=brain_dim)
    pfc = PrefrontalCortex(spike_dim=brain_dim)
    
    # 1. VISUAL INGESTION
    print("=== SENSORY INGESTION ===")
    print("👩‍🏫 TEACHER: 'Look at this object. It is a cat.'")
    
    cat_img = generate_mock_image(is_cat=True)
    visual_spike = visual_cortex.encode_image(cat_img)
    word_spike = encoder.get_spike("cat")
    
    # STDP Bind: Vision -> Word -> "is" -> "feline"
    sparse_brain.stdp_sequence_memorize([visual_spike, word_spike, encoder.get_spike("is"), encoder.get_spike("feline")])
    
    print("👩‍🏫 TEACHER: 'Look at this object. It is a dog.'")
    dog_img = generate_mock_image(is_cat=False)
    dog_visual_spike = visual_cortex.encode_image(dog_img)
    dog_word_spike = encoder.get_spike("dog")
    
    sparse_brain.stdp_sequence_memorize([dog_visual_spike, dog_word_spike, encoder.get_spike("is"), encoder.get_spike("canine")])
    
    # Intentionally add a hallucinated connection to test the PFC
    sparse_brain.stdp_sequence_memorize([word_spike, encoder.get_spike("is"), encoder.get_spike("canine")])
    
    # 2. LOGICAL RULES
    print("\n=== LOGICAL ONTOLOGY ===")
    print("👩‍🏫 TEACHER: 'A cat is a feline. It CANNOT be a canine.'")
    
    # Teach the PFC rules
    pfc.add_rule(encoder.get_spike("cat"), encoder.get_spike("canine"), is_valid=False) # Contradiction!
    
    # 3. MULTIMODAL TEST
    print("\n=== SENSORY LOGIC EXAM ===")
    print("👩‍🏫 TEACHER: 'I am showing you an image. Run a stream of consciousness on what it is.'")
    
    # Present the Cat Image
    test_visual_spike = visual_cortex.encode_image(cat_img)
    
    generated_spikes = sparse_brain.sequence_generate(test_visual_spike, sequence_length=5, threshold=1, max_active=1)
    
    # Decode raw thoughts (Without PFC)
    raw_sentence = []
    for step_spike in generated_spikes:
        active_neurons = np.nonzero(step_spike)[0]
        recalled_words = set()
        for n in active_neurons:
            word_idx = n // 3
            if word_idx in encoder.idx_to_word:
                recalled_words.add(encoder.idx_to_word[word_idx])
        if recalled_words:
            raw_sentence.append("/".join(recalled_words))
            
    print(f"\n  👦 HYPERION (Raw Animal Instinct): {' -> '.join(raw_sentence)}")
    
    # PFC Evaluation
    print("\n[Prefrontal Cortex Evaluating Stream of Consciousness...]")
    filtered_spikes = pfc.filter_stream(generated_spikes)
    
    spoken_sentence = []
    for step_spike in filtered_spikes:
        active_neurons = np.nonzero(step_spike)[0]
        recalled_words = set()
        for n in active_neurons:
            word_idx = n // 3
            if word_idx in encoder.idx_to_word:
                recalled_words.add(encoder.idx_to_word[word_idx])
        
        if recalled_words:
            spoken_sentence.append("/".join(recalled_words))
            
    print(f"  👦 HYPERION (Logical Executive Speech): {' -> '.join(spoken_sentence)}")

if __name__ == "__main__":
    run_multimodal_exam()
