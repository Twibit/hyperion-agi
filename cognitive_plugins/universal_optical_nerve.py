import os
import re
import sqlite3

class OpticalNerve:
    def __init__(self):
        print("[Optical Nerve] Initialized. Ready to ingest universal data types.")

    def read_text_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""

    def read_pdf_safely(self, filepath):
        """A bounded PDF reader that prevents catastrophic CPU loops. (Legacy: PyPDF2 removed)"""
        return ""

    def read_bmp_safely(self, filepath):
        """Pure Python BMP parser. Extracts pixel array and routes to VisualCortex."""
        try:
            with open(filepath, 'rb') as f:
                header = f.read(54)
                if header[:2] != b'BM':
                    return
                
                width = int.from_bytes(header[18:22], 'little')
                height = int.from_bytes(header[22:26], 'little')
                bpp = int.from_bytes(header[28:30], 'little')
                
                if bpp not in (24, 32) or width > 1024 or height > 1024:
                    return # Too complex or too large for naive biological retina
                
                row_padded = (width * (bpp // 8) + 3) & (~3)
                pixels = f.read()
                
            import numpy as np
            image_array = np.zeros((height, width), dtype=np.float32)
            
            bytes_per_pixel = bpp // 8
            for y in range(height):
                for x in range(width):
                    offset = y * row_padded + x * bytes_per_pixel
                    if offset + 3 <= len(pixels):
                        b = pixels[offset]
                        g = pixels[offset+1]
                        r = pixels[offset+2]
                        # Grayscale approximation
                        gray = 0.299*r + 0.587*g + 0.114*b
                        # BMP is stored bottom-up, so we invert Y
                        image_array[height - 1 - y, x] = gray
                        
            # Pass to VisualCortex
            import sys
            # Ensure nlp_engine is in path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.append(project_root)
            from nlp_engine.layers.visual_cortex import VisualCortex
            vc = VisualCortex(output_dim=1000, image_shape=(height, width))
            spikes = vc.encode_image(image_array)
            print(f"      [Visual Cortex] Generated {np.sum(spikes)} spikes from {width}x{height} BMP.")
            
        except Exception as e:
            print(f"    [Optical Nerve] BMP read failed: {e}")

    def scan_environment(self, directory_path):
        """Scans an entire directory and processes everything it can physically read."""
        print(f"==================================================")
        print(f"[Visual Cortex] Initiating Universal Scan on '{directory_path}'")
        print(f"==================================================")
        
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cain_global_memory.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        total_nodes = 0
        
        for root, _, files in os.walk(directory_path):
            # Skip massive hidden folders
            if '.git' in root or '.gemini' in root: continue
            
            for file in files:
                filepath = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                if ext in ['.txt', '.md', '.py', '.js', '.json', '.csv', '.html']:
                    print(f"    [Optical Nerve] Ingesting plaintext format: {file}")
                    raw_text = self.read_text_file(filepath)
                elif ext == '.pdf':
                    print(f"    [Optical Nerve] Ingesting complex format: {file}")
                    raw_text = self.read_pdf_safely(filepath)
                elif ext == '.bmp':
                    print(f"    [Optical Nerve] Processing visual format: {file}")
                    self.read_bmp_safely(filepath)
                    continue
                else:
                    # Bypassing JPG, PNG, BIN unless hooked to OCR
                    continue
                    
                if not raw_text:
                    continue
                    
                # Clean and parse text
                raw_text = re.sub(r'\s+', ' ', raw_text)
                raw_text = re.sub(r'[^\x00-\x7F]+', ' ', raw_text)
                sentences = [s.strip() + "." for s in re.split(r'\.\s+(?=[A-Z])', raw_text) if len(s.split()) > 5 and len(s) < 300]
                
                stop_words = {"what", "is", "a", "the", "tell", "me", "about", "who", "in", "should", "i", "say", "something", "define", "this", "that", "explain", "describe", "does", "mean", "are", "of", "and", "to", "for", "on", "with", "it", "as", "by", "an", "be", "or", "from", "at", "which", "will"}
                
                for sentence in sentences:
                    words = re.findall(r'[a-zA-Z]+', sentence.lower())
                    content_words = [w for w in words if w not in stop_words]
                    if not content_words: continue
                    
                    keyword = content_words[0]
                    try:
                        cursor.execute("INSERT OR IGNORE INTO global_memory (keyword, content) VALUES (?, ?)", (keyword, sentence))
                        total_nodes += 1
                    except Exception:
                        pass
                
                conn.commit()
                
        conn.close()
        print(f"\n[Precuneus DMN] Universal scan complete. Added {total_nodes} structural nodes to Hippocampus.")

if __name__ == "__main__":
    import sys
    eye = OpticalNerve()
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = sys.argv[1] if len(sys.argv) > 1 else PROJECT_ROOT
    eye.scan_environment(target_dir)
