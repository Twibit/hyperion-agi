import time
import random
import math
import urllib.request
import re
class HumanMotorCortex:
    def __init__(self):
        print("[Motor Cortex] Initializing Biometric Camouflage Protocol.")
        import os
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
        from cognitive_plugins.pineal_gland import PinealGland
        self.pineal = PinealGland()
        
        self.current_pos = (0, 0) # Simulated mouse pointer starting position
        self.base_url = "https://roadmap.sh"
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    def generate_bezier_mouse_path(self, start, end):
        dist = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        control_x = (start[0] + end[0]) / 2 + random.randint(-50, 50)
        control_y = (start[1] + end[1]) / 2 + random.randint(-50, 50)
        
        points = []
        steps = max(5, int(dist / 40)) 
        
        for t in range(steps + 1):
            t_normalized = t / steps
            x = (1 - t_normalized)**2 * start[0] + 2 * (1 - t_normalized) * t_normalized * control_x + t_normalized**2 * end[0]
            y = (1 - t_normalized)**2 * start[1] + 2 * (1 - t_normalized) * t_normalized * control_y + t_normalized**2 * end[1]
            points.append((int(x), int(y)))
            
        return points

    def simulate_human_click(self, target_name):
        target_coords = (random.randint(200, 800), random.randint(100, 900))
        print(f"\n[Motor Cortex] Planning movement to '{target_name}' at {target_coords}")
        
        path = self.generate_bezier_mouse_path(self.current_pos, target_coords)
        
        for i, point in enumerate(path):
            self.pineal.sleep(int(random.uniform(1, 10)))
            
        self.current_pos = target_coords
        
        hesitation = random.uniform(0.1, 0.5)
        print(f"    [Prefrontal Cortex] Pausing for {hesitation:.2f}s (Human hesitation)...")
        self.pineal.sleep(int(hesitation * 1000))
        print(f"    [Motor Cortex] *CLICK* on '{target_name}'")

    def global_exploration(self, target_url):
        print("==================================================")
        print(f"[Biological Engine] Initiating Adaptive Biological Navigation on {target_url}")
        print("==================================================")
        
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
            
        from cognitive_plugins.omni_optical_nerve import OmniOpticalNerve
        eye = OmniOpticalNerve()
        
        # Initial scan to build the Visual Connectome and detect threats
        connectome = eye.perceive(target_url, active_drive_keywords=['navigate', 'link', 'click', 'learn'])
        
        threat_detected = False
        interactives = connectome.get('interactive_nodes', [])
        
        if not interactives:
            print("[Motor Cortex] No interactive elements found to navigate.")
            return
            
        print(f"\n[Motor Cortex] Found {len(interactives)} navigational paths.")
        
        for i, target in enumerate(interactives[:5]): # Explore up to 5 links
            target_name = target.get('action', 'Unknown Link')
            target_coords = target.get('coords', (0,0))
            
            if threat_detected:
                print(f"\n[Motor Cortex] [STEALTH MODE] Planning biological movement to '{target_name}' at {target_coords}")
                path = self.generate_bezier_mouse_path(self.current_pos, target_coords)
                for point in path:
                    self.pineal.sleep(int(random.uniform(1, 15)))
                self.current_pos = target_coords
                
                hesitation = random.uniform(0.5, 2.0)
                print(f"    [Prefrontal Cortex] Pausing for {hesitation:.2f}s (Human hesitation / reading time)...")
                self.pineal.sleep(int(hesitation * 1000))
                print(f"    [Motor Cortex] *CLICK* on '{target_name}'")
            else:
                print(f"\n[Motor Cortex] [FAST MODE] Instant jump to '{target_name}'. No delay.")
                self.current_pos = target_coords
            
            # Form full URL and fetch
            full_url = target_name
            if full_url.startswith('/'):
                full_url = target_url.rstrip('/') + full_url
                
            if not full_url.startswith('http'):
                continue
                
            self.headers['Referer'] = target_url
            
            print(f"    [Visual Cortex] Fetching: {full_url}")
            try:
                page_req = urllib.request.Request(full_url, headers=self.headers)
                with urllib.request.urlopen(page_req, timeout=10) as page_response:
                    page_html = page_response.read().decode('utf-8', errors='ignore')
                    
                print(f"    [Success] Loaded {len(page_html)} bytes.")
                
                if threat_detected:
                    print(f"    [Motor Cortex] Simulating mouse wheel scroll down...")
                    self.pineal.sleep(int(random.uniform(500, 1200)))
                    
            except Exception as e:
                print(f"    [ERROR] Blocked by security on {target_name}: {e}")
                
            if threat_detected:
                rest_time = random.uniform(1.0, 3.0)
                print(f"    [Rest] Absorbing page layout for {rest_time:.2f}s before next action...")
                self.pineal.sleep(int(rest_time * 1000))
            
        print("\n[Biological Engine] Adaptive Exploration Complete.")

if __name__ == "__main__":
    bot = HumanMotorCortex()
    bot.global_exploration("https://roadmap.sh")
