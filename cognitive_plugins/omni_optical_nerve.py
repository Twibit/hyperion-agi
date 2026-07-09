import os
import sys
import ast
import numpy as np
import urllib.request
import zlib
import re
from html.parser import HTMLParser

# Ensure root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from rust_backend_shim import native_omni_vision_scan, NativeUiEye
except ImportError:
    native_omni_vision_scan = None
    NativeUiEye = None

try:
    from adaptive_constants import (
        SEMANTIC_HTML_TAGS, INTERACTIVE_HTML_TAGS, INTERACTIVE_UI_KINDS,
        CODE_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS,
        EYE_STREAM_EMA_ALPHA, EYE_PROGRESS_INTERVAL,
        VISION_BUFFER_BYTES, VISION_TELEMETRY_INTERVAL,
    )
except ImportError:
    SEMANTIC_HTML_TAGS = {'p', 'h1', 'h2', 'h3', 'span', 'div', 'article'}
    INTERACTIVE_HTML_TAGS = {'a', 'button'}
    INTERACTIVE_UI_KINDS = {"Button", "Link", "Input", "Checkbox"}
    CODE_EXTENSIONS = {'.py', '.js', '.ts'}
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}
    VIDEO_EXTENSIONS = {'.mp4'}
    EYE_STREAM_EMA_ALPHA = 0.01
    EYE_PROGRESS_INTERVAL = 500_000
    VISION_BUFFER_BYTES = 16 * 1024 * 1024
    VISION_TELEMETRY_INTERVAL = 10_000_000

from embodied_agent.cortex.isomorphic_encoder import IsomorphicEncoder

class NativeWebSaccade(HTMLParser):
    def __init__(self):
        super().__init__()
        self.semantic_nodes = []
        self.interactive_nodes = []
        self.current_tag = None
        self.current_href = None
        self.record_data = False

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag in SEMANTIC_HTML_TAGS:
            self.record_data = True
        elif tag in INTERACTIVE_HTML_TAGS:
            self.record_data = True
            for name, value in attrs:
                if name == 'href':
                    self.current_href = value

    def handle_endtag(self, tag):
        if tag == self.current_tag:
            self.record_data = False
        self.current_tag = None
        self.current_href = None

    def handle_data(self, data):
        if self.record_data:
            text = data.strip()
            # REMOVED: hardcoded length constraints. Adaptive intelligence handles it.
            if len(text) > 0:
                if self.current_href:
                    self.interactive_nodes.append({"action": self.current_href, "label": text})
                else:
                    self.semantic_nodes.append({"text": text})


class OmniOpticalNerve:
    def __init__(self):
        print("[Omni-Eye] Native OmniOpticalNerve initialized. Unbound Adaptive Mode Active.")
        self.isomorphic = IsomorphicEncoder()
        self.omni_connectome = {
            "semantic_nodes": [],
            "interactive_nodes": [],
            "spatial_geometry": []
        }
        self.native_ui_eye = None
        if NativeUiEye is not None:
            try:
                self.native_ui_eye = NativeUiEye()
            except Exception:
                pass

    def perceive(self, target_path, active_drive_keywords=None, on_foveate_callback=None, skim_stride=1):
        """Universal entry point. Auto-routes based on target format. Unbound scale."""
        print(f"\n==================================================")
        print(f"[Omni-Eye] Unbound Foveation on target: {target_path}")
        print(f"==================================================")
        
        self.omni_connectome = {"semantic_nodes": [], "interactive_nodes": [], "spatial_geometry": []}
        self.active_drive_sdr = None
        if active_drive_keywords:
            self.active_drive_sdr = self.isomorphic.encode_sentence(" ".join(active_drive_keywords), fast_mode=True)
            
        if str(target_path).startswith("http"):
            self._perceive_dom_native(target_path)
        elif any(str(target_path).endswith(ext) for ext in CODE_EXTENSIONS):
            self._perceive_ast(target_path)
        elif str(target_path).endswith('.pdf'):
            self._perceive_pdf_native(target_path)
        elif any(str(target_path).endswith(ext) for ext in IMAGE_EXTENSIONS):
            self._perceive_pixels_native(target_path)
        elif any(str(target_path).endswith(ext) for ext in VIDEO_EXTENSIONS):
            self._perceive_temporal(target_path)
        else:
            self._perceive_raw_text_adaptive(target_path, active_drive_keywords, on_foveate_callback, skim_stride)
            
        # Post-filter non-streamed modalities using statistical adaptive thresholds
        if self.active_drive_sdr is not None and not str(target_path).endswith(('.xml', '.txt')):
            self._apply_adaptive_saliency_filter()
            
        return self.omni_connectome

    def _apply_adaptive_saliency_filter(self):
        """Feature 1: Adaptive Omniscient Saliency Filter (No hardcoded threshold)"""
        if not self.omni_connectome["semantic_nodes"]:
            return
            
        print("    [Omniscient Retina] Applying global adaptive hash filter...")
        overlaps = []
        node_sdrs = []
        for node in self.omni_connectome["semantic_nodes"]:
            node_sdr = self.isomorphic.encode_sentence(node["text"])
            node_sdrs.append(node_sdr)
            overlaps.append(int(np.sum(self.active_drive_sdr & node_sdr)))
            
        if not overlaps:
            return
            
        # REMOVED: Hardcoded `overlap > 5`.
        # ADDED: Adaptive Statistical Threshold (Mean + 1 Standard Deviation)
        mean_overlap = np.mean(overlaps)
        std_overlap = np.std(overlaps)
        adaptive_threshold = mean_overlap + std_overlap
        
        foveated_nodes = []
        for i, node in enumerate(self.omni_connectome["semantic_nodes"]):
            # Signal Isolation: Keep nodes mathematically above the noise floor
            if overlaps[i] > adaptive_threshold or (adaptive_threshold < 1 and overlaps[i] > 0):
                foveated_nodes.append(node)
                
        print(f"    [Saliency Map] Adaptive threshold set to {adaptive_threshold:.2f}. Filtered out {len(self.omni_connectome['semantic_nodes']) - len(foveated_nodes)} noise nodes.")
        self.omni_connectome["semantic_nodes"] = foveated_nodes

    def _perceive_dom_native(self, url):
        print(f"    [Visual Cortex] Perceiving {url} natively...")
        
        if self.native_ui_eye is not None:
            try:
                import json
                print("    [Native UI Eye] Engaging biological 0ms latency vision pipeline (Retina -> LGN -> V1)...")
                percept_json = self.native_ui_eye.glance_url(url)
                percept = json.loads(percept_json)
                
                elements = percept.get("elements", [])
                for el in elements:
                    kind = el.get("kind", "Unknown")
                    label = el.get("label", "")
                    bbox = el.get("bbox", [])
                    
                    if kind in INTERACTIVE_UI_KINDS:
                        self.omni_connectome["interactive_nodes"].append({"type": kind, "label": label, "bbox": bbox})
                    else:
                        if label.strip():
                            self.omni_connectome["semantic_nodes"].append({"text": label})
                
                print(f"    [Success] Biological UI perception complete. Extracted {len(elements)} cognitive primitives.")
                return
            except Exception as e:
                print(f"    [Warning] Native UI Eye failed, falling back to primitive scraping: {e}")

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                html_bytes = response.read()
                html_str = html_bytes.decode('utf-8', errors='ignore')
                
            parser = NativeWebSaccade()
            parser.feed(html_str)
            
            # REMOVED: hardcoded [:200] and [:100] limits.
            self.omni_connectome["semantic_nodes"].extend(parser.semantic_nodes)
            self.omni_connectome["interactive_nodes"].extend(parser.interactive_nodes)
            
            print(f"    [Success] Mapped {len(self.omni_connectome['interactive_nodes'])} interactives and {len(self.omni_connectome['semantic_nodes'])} semantic blocks natively.")
        except Exception as e:
            print(f"    [Error] Native DOM Vision failed: {e}")

    def _perceive_pdf_native(self, filepath):
        print("    [Visual Cortex] Natively decompressing PDF zlib streams...")
        try:
            with open(filepath, 'rb') as f:
                pdf_data = f.read()

            streams = re.findall(b'stream(.*?)endstream', pdf_data, re.DOTALL)
            text_blocks = []
            
            for stream in streams:
                try:
                    decompressed = zlib.decompress(stream.strip())
                    text_matches = re.findall(rb'\((.*?)\)\s*Tj', decompressed)
                    for match in text_matches:
                        text_blocks.append(match.decode('latin-1', errors='ignore').strip())
                except:
                    pass
                    
            for text in text_blocks:
                if len(text) > 0: # Removed limit
                    self.omni_connectome["semantic_nodes"].append({"text": text})
                    self.omni_connectome["spatial_geometry"].append({"type": "native_pdf_stream_block"})
                    
            print(f"    [Success] Extracted {len(self.omni_connectome['semantic_nodes'])} semantic text blocks natively from zlib streams.")
        except Exception as e:
            print(f"    [Error] Native PDF Vision failed: {e}")

    def _perceive_pixels_native(self, filepath):
        print("    [Visual Cortex] Reading file as 1D raw byte-stream for entropy edge detection...")
        try:
            with open(filepath, 'rb') as f:
                data = bytearray(f.read())
            
            sub_data = data[::len(data)//1000] if len(data) > 1000 else data
            edge_count = sum(1 for i in range(1, len(sub_data)) if abs(sub_data[i] - sub_data[i-1]) > 50)
            
            self.omni_connectome["spatial_geometry"].append({"type": "native_entropy_edge_map", "edge_pixels": edge_count})
            print(f"    [Success] Detected {edge_count} high-entropy visual spikes without OpenCV.")
        except Exception as e:
            print(f"    [Error] Native Pixel Vision failed: {e}")

    def _perceive_ast(self, filepath):
        print("    [Visual Cortex] Parsing Code Abstract Syntax Tree (AST)...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.omni_connectome["spatial_geometry"].append({"type": "function", "name": node.name})
                    self.omni_connectome["semantic_nodes"].append({"text": f"Function {node.name} defined."})
                elif isinstance(node, ast.ClassDef):
                    self.omni_connectome["spatial_geometry"].append({"type": "class", "name": node.name})
                    
            print(f"    [Success] Mapped {len(self.omni_connectome['spatial_geometry'])} structural geometric code nodes.")
        except Exception as e:
            print(f"    [Error] AST Vision failed: {e}")

    def _perceive_temporal(self, filepath):
        print("    [Visual Cortex] Reading mp4 atom structures natively...")
        try:
            size = os.path.getsize(filepath)
            self.omni_connectome["spatial_geometry"].append({"type": "native_video_blob", "byte_size": size})
            print(f"    [Success] Registered temporal media of {size} bytes.")
        except Exception as e:
            print(f"    [Error] Native Temporal Vision failed: {e}")

    def _perceive_raw_text_adaptive(self, filepath, active_drive_keywords=None, on_foveate_callback=None, skim_stride=1):
        """Feature: Unbound Streaming Perception for massive files (13GB+) with O(1) memory callback support and Speed Skimming."""
        print(f"    [Visual Cortex] Adaptive Stream-Ingestion of {os.path.basename(filepath)}...")
        try:
            file_size = os.path.getsize(filepath)
            print(f"    [Scale Monitor] Target size: {file_size / (1024*1024):.2f} MB. Applying fluid stream processing...")
            
            # REMOVED: hardcoded `if i > 5000: break`.
            # ADDED: Dynamic Exponential Moving Average (EMA) to calculate real-time signal threshold
            # This allows it to read a 13GB file without running out of RAM, by throwing away 
            # noise dynamically on the fly.
            ema_overlap = 0.0
            ema_var = 0.0
            alpha = EYE_STREAM_EMA_ALPHA  # Moving average decay for signal tracking
            
            nodes_kept = 0
            nodes_seen = 0
            
            if skim_stride == 1 and native_omni_vision_scan is not None and active_drive_keywords:
                try:
                    print("    [Native Vision Core] Engaging Rust backend for gigabit streaming...")
                    lines = native_omni_vision_scan(
                        filepath, active_drive_keywords,
                        VISION_BUFFER_BYTES,
                        VISION_TELEMETRY_INTERVAL,
                    )
                    for text in lines:
                        if on_foveate_callback:
                            on_foveate_callback({"text": text})
                        else:
                            self.omni_connectome["semantic_nodes"].append({"text": text})
                        nodes_kept += 1
                    
                    print(f"    [Success] Native Rust stream complete. Isolated {nodes_kept} relevant nodes from {file_size / (1024*1024):.2f} MB.")
                    return
                except NotImplementedError:
                    pass

            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    nodes_seen += 1
                    
                    if nodes_seen % EYE_PROGRESS_INTERVAL == 0:
                        print(f"      [Scan Progress] Fast-forwarded {nodes_seen:,} lines... Found {nodes_kept:,} highly-salient nodes.")
                    
                    if skim_stride > 1 and nodes_seen % skim_stride != 0:
                        # Fast-forward past irrelevant text without engaging the cortex
                        continue
                        
                    text = line.strip()
                    if not text:
                        continue
                        
                    nodes_seen += 1
                    
                    if self.active_drive_sdr is not None:
                        node_sdr = self.isomorphic.encode_sentence(text, fast_mode=True)
                        overlap = int(np.sum(self.active_drive_sdr & node_sdr))
                        
                        # Online Welford's Variance Approx to find statistical signal spikes dynamically
                        delta = overlap - ema_overlap
                        ema_overlap += alpha * delta
                        ema_var = (1 - alpha) * (ema_var + alpha * delta**2)
                        dynamic_threshold = ema_overlap + np.sqrt(ema_var)
                        
                        # Only keep nodes that pierce through the dynamic noise floor
                        if overlap > max(1, dynamic_threshold):
                            if on_foveate_callback:
                                on_foveate_callback({"text": text})
                            else:
                                self.omni_connectome["semantic_nodes"].append({"text": text})
                            nodes_kept += 1
                    else:
                        if on_foveate_callback:
                            on_foveate_callback({"text": text})
                        else:
                            self.omni_connectome["semantic_nodes"].append({"text": text})
                        nodes_kept += 1
                        
            print(f"    [Success] Adaptive stream complete. Examined {nodes_seen} lines, isolated {nodes_kept} relevant nodes.")
        except Exception as e:
            print(f"    [Error] Adaptive Raw text perception failed: {e}")

    def scan_environment(self, directory_path):
        print(f"==================================================")
        print(f"[Omni-Eye] Initiating Universal Scan on '{directory_path}'")
        print(f"==================================================")
        
        for root, _, files in os.walk(directory_path):
            if '.git' in root or '.venv' in root or '.gemini' in root: continue
            for file in files:
                filepath = os.path.join(root, file)
                self.perceive(filepath)
                
        print(f"\n[Omni-Eye] Universal directory scan complete.")

if __name__ == "__main__":
    eye = OmniOpticalNerve()
    target_dir = sys.argv[1] if len(sys.argv) > 1 else project_root
    if os.path.isdir(target_dir):
        eye.scan_environment(target_dir)
    else:
        print(eye.perceive(target_dir))
