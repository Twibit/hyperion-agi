import math
import xml.etree.ElementTree as ET

class SVGSpatialCortex:
    def __init__(self):
        print("[Spatial Cortex] Initialized. Operating entirely on Geometric Euclidean Mathematics (No LLMs).")
        self.nodes = []
        self.edges = []
        
    def calculate_distance(self, x1, y1, x2, y2):
        # Pythagorean theorem to find distance between two points on the screen
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def parse_visual_canvas(self, svg_string):
        print("==================================================")
        print("[Visual Cortex] Parsing pure SVG Geometry...")
        
        # Simulated SVG from a roadmap
        root = ET.fromstring(svg_string)
        
        # 1. Map all Text Boxes (Nodes)
        for text_elem in root.findall('.//text'):
            x = float(text_elem.attrib['x'])
            y = float(text_elem.attrib['y'])
            label = text_elem.text
            self.nodes.append({"label": label, "x": x, "y": y})
            
        print(f"    [Fovea] Detected {len(self.nodes)} Text Nodes based on absolute X/Y coordinates.")
        
        # 2. Map all Lines/Arrows (Edges)
        for line in root.findall('.//line'):
            x1, y1 = float(line.attrib['x1']), float(line.attrib['y1'])
            x2, y2 = float(line.attrib['x2']), float(line.attrib['y2'])
            self.edges.append({"start": (x1, y1), "end": (x2, y2)})
            
        print(f"    [Peripheral] Detected {len(self.edges)} Connecting Arrows.")
        
        # 3. Geometric Synthesis: Connect the dots!
        print("\n[Parietal Lobe] Synthesizing Spatial Graph using Euclidean Proximity...")
        print("--------------------------------------------------")
        
        for edge in self.edges:
            start_x, start_y = edge['start']
            end_x, end_y = edge['end']
            
            # Find the closest text node to the START of the arrow
            source_node = None
            min_start_dist = float('inf')
            for node in self.nodes:
                dist = self.calculate_distance(start_x, start_y, node['x'], node['y'])
                if dist < min_start_dist:
                    min_start_dist = dist
                    source_node = node['label']
                    
            # Find the closest text node to the END of the arrow
            target_node = None
            min_end_dist = float('inf')
            for node in self.nodes:
                dist = self.calculate_distance(end_x, end_y, node['x'], node['y'])
                if dist < min_end_dist:
                    min_end_dist = dist
                    target_node = node['label']
                    
            print(f"    [Spatial Logic] {source_node} ---> points to ---> {target_node}")
            
        print("--------------------------------------------------")
        print("[Success] Reconstructed the visual roadmap geometrically!")

if __name__ == "__main__":
    cortex = SVGSpatialCortex()
    
    # A simulated SVG canvas containing a mini-roadmap
    simulated_svg = """
    <svg>
        <!-- The Text Nodes -->
        <text x="100" y="50">Internet</text>
        <text x="100" y="150">HTML</text>
        <text x="100" y="250">CSS</text>
        <text x="250" y="250">Tailwind</text>
        
        <!-- The Arrows -->
        <line x1="100" y1="55" x2="100" y2="145"/> <!-- Internet to HTML -->
        <line x1="100" y1="155" x2="100" y2="245"/> <!-- HTML to CSS -->
        <line x1="110" y1="250" x2="240" y2="250"/> <!-- CSS to Tailwind -->
    </svg>
    """
    
    cortex.parse_visual_canvas(simulated_svg)
