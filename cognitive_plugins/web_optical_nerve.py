import urllib.request
import re
import sqlite3
import os
from html.parser import HTMLParser

class WebTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.nodes = []
        self.current_tag = None
        self.depth = 0
        
        # We replace the hardcoded skip_tags with a continuous representation later, 
        # but for now we remove the hardcoded list. We will let the cortex decide what's structural.
        self.in_skip_tag = False
        self.skip_depth = 0
        
        from nlp_engine.layers.entorhinal_cortex import EntorhinalCortex
        import numpy as np
        self.entorhinal = EntorhinalCortex(embedding_dim=64)
        self.np = np
        self.current_tag = None
        self.depth = 0
        self.skip_tags = {'script', 'style', 'noscript'}
        self.in_skip_tag = False
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        self.depth += 1
        self.current_tag = tag
        if tag in self.skip_tags:
            self.in_skip_tag = True
            self.skip_depth += 1
            
        if tag in {'a', 'button', 'input'}:
            attr_dict = dict(attrs)
            action = attr_dict.get('href', '') or attr_dict.get('name', '')
            
            # Microglia Ad-Blocker (Semantic Immune Response)
            # Instead of a hardcoded list, we compute geometric distance to the 'advertisement' centroid
            is_ad = False
            if action:
                action_words = re.findall(r'[a-zA-Z]+', action.lower())
                if action_words:
                    self.entorhinal.process_sequence(action_words)
                    action_vec = self.np.mean([self.entorhinal.get_vector(w) for w in action_words], axis=0)
                    ad_vec = self.entorhinal.get_vector('advertisement') + self.entorhinal.get_vector('tracking')
                    
                    n1, n2 = self.np.linalg.norm(action_vec), self.np.linalg.norm(ad_vec)
                    sim = float(self.np.dot(action_vec, ad_vec) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0
                    if sim > 0.6:
                        is_ad = True
            
            if action and not is_ad:
                self.nodes.append({
                    "type": "interactive",
                    "tag": tag,
                    "depth": self.depth,
                    "action": action,
                    "coords": (self.depth * 20 + 40, len(self.nodes) * 30 + 100) # Simulated 2D Layout
                })
            elif is_ad:
                # Log ad detection for visibility, but don't add to interactive nodes
                self.nodes.append({
                    "type": "ad_blocked",
                    "tag": tag,
                    "depth": self.depth,
                    "action": action
                })

    def handle_endtag(self, tag):
        self.depth -= 1
        self.current_tag = None
        if tag in self.skip_tags:
            self.skip_depth -= 1
            if self.skip_depth <= 0:
                self.in_skip_tag = False
                self.skip_depth = 0

    def handle_data(self, data):
        if not self.in_skip_tag:
            text = data.strip()
            
            # --- Markdown & Bare URL Link Extractor ---
            # Extract Markdown links: [Text](url)
            md_links = re.findall(r'\[(.*?)\]\((https?://.*?)\)', text)
            for md_text, md_url in md_links:
                self.nodes.append({
                    "type": "interactive",
                    "tag": "markdown_link",
                    "depth": self.depth,
                    "action": md_url,
                    "coords": (self.depth * 20 + 40, len(self.nodes) * 30 + 100)
                })
                # Remove the markdown link format so the plain text is still learned semantically
                text = text.replace(f"[{md_text}]({md_url})", md_text)
                
            # Extract Bare URLs: https://...
            bare_urls = re.findall(r'(?<!\()(https?://[^\s\)]+)', text)
            for b_url in bare_urls:
                self.nodes.append({
                    "type": "interactive",
                    "tag": "bare_url",
                    "depth": self.depth,
                    "action": b_url,
                    "coords": (self.depth * 20 + 40, len(self.nodes) * 30 + 100)
                })
                # Remove the bare URL from the semantic text
                text = text.replace(b_url, "")
                
            text = text.strip()
            if len(text) > 5:
                self.nodes.append({
                    "type": "text",
                    "text": text,
                    "depth": self.depth,
                    "tag": self.current_tag
                })

class WebOpticalNerve:
    def __init__(self):
        print("[Web Optical Nerve] Initialized. Ready to consume the Internet.")
        self.threat_detected = False
        import os
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
        from nlp_engine.layers.entorhinal_cortex import EntorhinalCortex
        self.entorhinal = EntorhinalCortex(embedding_dim=64)
        
    def detect_threat(self, html, headers):
        """Biological Adrenaline: Senses if a site has anti-bot security via Semantic Geometry."""
        html_lower = html.lower()
        
        from nlp_engine.layers.entorhinal_cortex import EntorhinalCortex
        import numpy as np
        entorhinal = EntorhinalCortex(embedding_dim=64)
        
        # Sample the HTML for semantic threat signatures
        words = re.findall(r'[a-zA-Z]+', html_lower[:5000]) # Sample first 5k chars
        
        self.threat_detected = False
        if words:
            entorhinal.process_sequence(words)
            page_vec = np.mean([entorhinal.get_vector(w) for w in words], axis=0)
            threat_vec = entorhinal.get_vector('captcha') + entorhinal.get_vector('security') + entorhinal.get_vector('blocked')
            
            n1, n2 = np.linalg.norm(page_vec), np.linalg.norm(threat_vec)
            sim = float(np.dot(page_vec, threat_vec) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0
            
            if sim > 0.5:
                self.threat_detected = True
                
        # We also check headers for explicit security server patterns organically
        if 'Server' in headers:
            server_vec = entorhinal.get_vector(headers['Server'].lower())
            n1, n2 = np.linalg.norm(server_vec), np.linalg.norm(threat_vec)
            sim = float(np.dot(server_vec, threat_vec) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0
            if sim > 0.5:
                self.threat_detected = True
            
        if self.threat_detected:
            print("    [Amygdala] THREAT DETECTED! Security guard active. Engaging Stealth Adrenaline (Slow, Human-like).")
        else:
            print("    [Amygdala] Environment safe. Engaging Fast Automatic Processing.")
        return self.threat_detected

    def read_website(self, url):
        print(f"\n==================================================")
        print(f"[Visual Cortex] Initiating Optical Stream from {url}")
        print(f"==================================================")
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'})
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8', errors='ignore')
                headers = dict(response.headers)
                
            self.detect_threat(html, headers)
                
            print("[Wernicke] HTML received. Synthesizing Visual Connectome...")
            
            parser = WebTextParser()
            parser.feed(html)
            
            print(f"[Hippocampus] Mapped {len(parser.nodes)} structural nodes. Simulating visual field:")
            print("-" * 50)
            
            blocks = []
            interactives = []
            count = 0
            for node in parser.nodes:
                if node['type'] == 'text':
                    blocks.append(node['text'])
                    if count < 15:
                        indent = "  " * (node['depth'] % 8)
                        print(f"{indent}→ [Depth: {node['depth']}] <{node['tag']}> {node['text'][:40]}...")
                    count += 1
                elif node['type'] == 'interactive':
                    interactives.append(node)
                    if count < 15:
                        indent = "  " * (node['depth'] % 8)
                        print(f"{indent}☼ [Interactive] <{node['tag']}> at Coords {node['coords']} -> {node['action'][:30]}")
                    count += 1
                elif node['type'] == 'ad_blocked':
                    if count < 15:
                        indent = "  " * (node['depth'] % 8)
                        print(f"{indent}X [Microglia] BLOCKED AD/TRACKER -> {node['action'][:30]}")
                    count += 1
                    
            if count >= 15:
                print(f"  ... (remaining visual graph truncated for terminal)")
                
            print("-" * 50)
            print(f"[Precuneus DMN] Spatial mapping complete. Found {len(interactives)} clickable targets. Extracting Logical Triples...")
            
            # Save interactives for the Motor Cortex
            self.last_interactives = interactives
            
            import sys
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.append(project_root)
                
            from nlp_engine.layers.relational_memory import RelationalMemory
            rm = RelationalMemory()
            total_triples = 0
            
            for block in blocks:
                if len(block) > 20: # Only parse sentences with actual content
                    learned = rm.learn_from_text(block, source=url)
                    total_triples += learned
            
            print(f"\n[Precuneus DMN] Web digestion complete. Added {total_triples} Neuro-Symbolic triples from the World Wide Web.")
            
        except Exception as e:
            print(f"[ERROR] Connection to Visual Cortex failed: {e}")

if __name__ == "__main__":
    import sys
    eye = WebOpticalNerve()
    target_url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/EbookFoundation/free-programming-books/blob/main/books/free-programming-books-langs.md"
    eye.read_website(target_url)
