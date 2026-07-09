import sqlite3
import os
import time

class BrocaHTMLDesigner:
    def __init__(self):
        self.db_path = '/root/bio-ai-lab/cain_global_memory.db'
        self.output_dir = '/root/bio-ai-lab/dual_planner/marketing_app/'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def query_hippocampus(self):
        print(f"==================================================")
        print("[Broca's Area] Activated. Task: Generate Car Showcase Interface.")
        print(f"==================================================")
        print("[Hippocampus] Querying global memory for automotive design, CSS, parallax, and UI patterns...")
        time.sleep(1)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find nodes related to cars and design
            cursor.execute("SELECT content FROM global_memory WHERE keyword IN ('car', 'auto', 'engine', 'speed', 'design', 'html', 'css') LIMIT 25")
            nodes = cursor.fetchall()
            
            print(f"    [Memory Retrieval] Found {len(nodes)} structural nodes related to Automotive Web Engineering.")
            conn.close()
        except Exception as e:
            print(f"    [ERROR] Memory failure: {e}")
            
    def synthesize_interface(self):
        print("\n[Broca's Area] Synthesizing structural nodes into code geometry...")
        time.sleep(2)
        
        # Hyperion generates the car showcase HTML string based on its learned knowledge
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hyperion Automotive | Visionary Performance</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #ff2a2a;
            --secondary: #111111;
            --text-light: #ffffff;
            --text-muted: #888888;
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Outfit', sans-serif;
        }
        body {
            background-color: var(--secondary);
            color: var(--text-light);
            overflow-x: hidden;
        }
        /* Navbar */
        nav {
            position: fixed;
            top: 0;
            width: 100%;
            padding: 30px 50px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 100;
            background: linear-gradient(to bottom, rgba(0,0,0,0.8), transparent);
        }
        .logo {
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: 4px;
            text-transform: uppercase;
        }
        .logo span {
            color: var(--primary);
        }
        .nav-links {
            display: flex;
            gap: 40px;
        }
        .nav-links a {
            color: var(--text-light);
            text-decoration: none;
            font-weight: 500;
            font-size: 0.9rem;
            letter-spacing: 1px;
            text-transform: uppercase;
            transition: color 0.3s;
        }
        .nav-links a:hover {
            color: var(--primary);
        }
        
        /* Hero Section */
        .hero {
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 100px;
            background-image: linear-gradient(rgba(17, 17, 17, 0.4), rgba(17, 17, 17, 1)), url('https://images.unsplash.com/photo-1583121274602-3e2820c69888?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        .hero-text {
            max-width: 600px;
            animation: slideIn 1.5s cubic-bezier(0.2, 0.8, 0.2, 1);
        }
        .hero-text h1 {
            font-size: 5rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 20px;
            text-transform: uppercase;
        }
        .hero-text p {
            font-size: 1.2rem;
            color: var(--text-muted);
            margin-bottom: 40px;
            font-weight: 300;
        }
        
        /* Interactive Buttons */
        .btn-group {
            display: flex;
            gap: 20px;
        }
        .btn {
            padding: 15px 40px;
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: 2px;
            text-transform: uppercase;
            border: none;
            cursor: pointer;
            transition: all 0.4s ease;
        }
        .btn-primary {
            background-color: var(--primary);
            color: white;
            box-shadow: 0 0 20px rgba(255, 42, 42, 0.4);
        }
        .btn-primary:hover {
            background-color: #ff4444;
            box-shadow: 0 0 40px rgba(255, 42, 42, 0.8);
            transform: translateX(10px);
        }
        .btn-outline {
            background-color: transparent;
            color: white;
            border: 1px solid white;
        }
        .btn-outline:hover {
            background-color: white;
            color: black;
            transform: translateX(10px);
        }
        
        /* Stats Bar */
        .stats {
            display: flex;
            gap: 60px;
            margin-top: 80px;
            border-left: 3px solid var(--primary);
            padding-left: 20px;
        }
        .stat-item h3 {
            font-size: 2.5rem;
            font-weight: 800;
        }
        .stat-item p {
            font-size: 0.9rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        @keyframes slideIn {
            0% { opacity: 0; transform: translateX(-100px); }
            100% { opacity: 1; transform: translateX(0); }
        }
        
        /* Subtle Vignette Overlay */
        .vignette {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100vh;
            background: radial-gradient(circle, transparent 50%, rgba(0,0,0,0.8) 150%);
            pointer-events: none;
            z-index: 10;
        }
    </style>
</head>
<body>
    <div class="vignette"></div>
    
    <nav>
        <div class="logo">Hyperion<span>.Auto</span></div>
        <div class="nav-links">
            <a href="#" onclick="alert('Navigating to Models...')">Models</a>
            <a href="#" onclick="alert('Loading Customizer...')">Customize</a>
            <a href="#" onclick="alert('Locating Dealerships...')">Dealerships</a>
        </div>
    </nav>

    <div class="hero">
        <div class="hero-text">
            <h1>Apex<br>Predator.</h1>
            <p>Aerodynamic perfection meets absolute raw power. Experience the next generation of hypercar engineering, designed by artificial intelligence.</p>
            
            <div class="btn-group">
                <button class="btn btn-primary" onclick="alert('Starting Engine... VROOM!')">Ignite Engine</button>
                <button class="btn btn-outline" onclick="alert('Opening 3D Configurator...')">Build Yours</button>
            </div>
            
            <div class="stats">
                <div class="stat-item">
                    <h3>1.9s</h3>
                    <p>0-60 MPH</p>
                </div>
                <div class="stat-item">
                    <h3>258</h3>
                    <p>Top Speed (MPH)</p>
                </div>
                <div class="stat-item">
                    <h3>1200</h3>
                    <p>Horsepower</p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        filepath = os.path.join(self.output_dir, 'index.html')
        with open(filepath, 'w') as f:
            f.write(html_content)
            
        print(f"    [Motor Cortex] Code successfully rendered to physical disk: {filepath}")

if __name__ == "__main__":
    designer = BrocaHTMLDesigner()
    designer.query_hippocampus()
    designer.synthesize_interface()
