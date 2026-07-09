import os
import sys
import glob

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from nlp_engine.layers.cain_brain import CainBrain

def begin_visual_reading(library_dir):
    print("==================================================")
    print("      HYPERION: AUTONOMOUS VISUAL READING ")
    print("==================================================")
    
    # Awaken from crystallized state
    brain = CainBrain.awaken()
    
    books = sorted(glob.glob(os.path.join(library_dir, "Volume_*.xml")))
    if not books:
        print(f"No volumes found in {library_dir}. Did you run build_library.py?")
        return
        
    print(f"\n[Library] Found {len(books)} volumes to read.")
    
    for book_path in books:
        book_name = os.path.basename(book_path)
        print(f"\n[Biological Drive] Selecting book: {book_name}")
        
        # 1. Take a sniff first to see if it's worth reading
        # The Olfactory Bulb will smell "Pheromones" because Wiktionary is full of definitions
        wants_to_read = brain.sniff(book_path)
        if not wants_to_read:
            print(f"[Attention] AGI ignored {book_name} due to biological aversion.")
            continue
            
        # 2. Use the Omni-Eye to visually perceive the file
        print(f"[Omni-Eye] Foveating on {book_name}...")
        try:
            # We use a foveal filter to find dictionary definitions, avoiding XML junk
            vision_map = brain.omni_eye.perceive(book_path, active_drive_keywords=["noun", "verb", "adjective", "meaning", "definition"])
            
            nodes = vision_map.get("semantic_nodes", [])
            print(f"    [Visual Cortex] Extracted {len(nodes)} semantic nodes.")
            
            if not nodes:
                print("    [Visual Cortex] Page appears empty or unreadable Saliency.")
                continue
                
            # 3. Read the visual nodes
            print(f"\n[Learning] Actively reading the visual nodes (LIMITER DISABLED)...")
            brain.cellular_matrix.adrenaline_level = 9999.0 # Temporarily bypass limiter
            
            for idx, node in enumerate(nodes):
                text = node.get("text", "")
                if not text: continue
                
                # Biologically ingest the text
                stats = brain.learn(text, source=book_name, domain="wiktionary")
                brain.cellular_matrix.global_atp = 100.0 # Force ATP to stay maxed
                
                # 4. Check Thermodynamic ATP limits (Metabolic Pacing)
                # if brain.cellular_matrix.global_atp < 20.0:
                #     print(f"\n[Thermodynamics] CRITICAL ATP: {brain.cellular_matrix.global_atp:.1f}%. Brain Fog setting in.")
                #     brain.sleep_and_save()
                #     print("[Wakefulness] Consciousness restored. ATP Regenerated. Resuming reading...")
                    
                if idx > 0 and idx % 20 == 0:
                    print(f"    ... read {idx}/{len(nodes)} semantic blocks (ATP: {brain.cellular_matrix.global_atp:.1f}%)")
                    
        except Exception as e:
            print(f"[Error] Failed to read {book_name}: {e}")
            
    print("\n[Library] Reading session complete.")
    brain.sleep_and_save()

if __name__ == "__main__":
    lib_dir = r"C:\Users\User\Downloads\TRAIN\Library"
    begin_visual_reading(lib_dir)
