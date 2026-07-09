import urllib.request
import re
from html.parser import HTMLParser

class RetinalParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.svg_count = 0
        self.nodes = []
        self.current_tag = None
        
    def handle_starttag(self, tag, attrs):
        self.depth += 1
        self.current_tag = tag
        if tag == 'svg':
            self.svg_count += 1
            
    def handle_endtag(self, tag):
        self.depth -= 1
        self.current_tag = None
        
    def handle_data(self, data):
        text = data.strip()
        if self.current_tag in ['h1', 'h2', 'h3', 'h4', 'li', 'a', 'span']:
            if len(text) > 2 and len(text) < 50:
                if text not in [n['text'] for n in self.nodes]:
                    self.nodes.append({
                        "id": len(self.nodes),
                        "text": text,
                        "focal_depth": self.depth,
                        "tag": self.current_tag
                    })


class RetinalCortex:
    def __init__(self):
        print("[Retinal Cortex] Initialized. Evolving beyond linear text to spatial geometry.")
        self.visual_connectome = {"nodes": [], "synapses": []}
        
    def saccadic_scan(self, url):
        print(f"\n==================================================")
        print(f"[Visual Cortex] Saccadic Scan initiated on {url}")
        print(f"==================================================")
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Biological Retina/2.0)'})
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8', errors='ignore')
                
            parser = RetinalParser()
            parser.feed(html)
            print(f"    [Peripheral Vision] Detected {parser.svg_count} visual vectors (SVG paths/arrows).")
            self.visual_connectome['nodes'] = parser.nodes
            print(f"    [Fovea Centralis] Mapped {len(self.visual_connectome['nodes'])} text clusters.")
            
            print("\n[Visual Cortex] Synthesizing Visual Connectome...")
            print("-" * 50)
            
            # Display the biological tree (Simulated Roadmap)
            count = 0
            for node in self.visual_connectome['nodes']:
                # Skip overly shallow or repetitive nav text to get to the roadmap flesh
                if node['focal_depth'] < 8:
                    continue
                
                indent = "  " * (node['focal_depth'] % 6)
                arrow = "↘ " if node['focal_depth'] % 2 == 0 else "→ "
                print(f"{indent}{arrow}[Depth: {node['focal_depth']}] {node['text']}")
                
                count += 1
                if count > 20: # Limit output to prevent terminal flood
                    print(f"{indent}  ... (remaining visual graph truncated for simulation)")
                    break
                
            print("-" * 50)
            print("[Precuneus DMN] Spatial mapping complete. The brain now understands hierarchical geometry, not just flat text.")
            
        except Exception as e:
            print(f"[ERROR] Retinal failure: {e}")

if __name__ == "__main__":
    eye = RetinalCortex()
    eye.saccadic_scan("https://roadmap.sh/frontend")
