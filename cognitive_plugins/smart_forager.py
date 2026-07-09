import urllib.request
import json
import os
import subprocess
import time
import sys

class IntelligentForager:
    def __init__(self):
        print("[Intelligent Forager] Activated. Goal: Autonomous filtering of ALL Anthropic repos.")
        self.api_url = "https://api.github.com/users/anthropics/repos?per_page=100" 
        self.base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "anthropic_labs")
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            
        # The Prefrontal Cortex's concept of what is "useful" for an AI to learn
        self.valuable_concepts = ['agent', 'llm', 'prompt', 'sdk', 'api', 'model', 'eval', 'claude', 'typescript', 'python', 'framework', 'tool', 'cookbook']

    def evaluate_usefulness(self, repo):
        name = repo.get('name', '').lower()
        desc = repo.get('description', '')
        desc = desc.lower() if desc else ''
        topics = repo.get('topics', [])
        
        # Combine all metadata into a single context string
        context = name + " " + desc + " " + " ".join(topics)
        
        # Evaluate how many valuable concepts it contains
        score = sum(1 for concept in self.valuable_concepts if concept in context)
        
        if score >= 1:
            return True, score
        return False, score

    def discover_and_filter(self):
        print("==================================================")
        print(f"[Visual Cortex] Scanning entire Anthropics Organization...")
        
        req = urllib.request.Request(self.api_url, headers={'User-Agent': 'Mozilla/5.0 (Intelligent Agent/3.0)'})
        try:
            with urllib.request.urlopen(req) as response:
                repos = json.loads(response.read().decode())
        except Exception as e:
            print(f"[ERROR] Failed to fetch repos: {e}")
            return
            
        print(f"[Hippocampus] Retrieved metadata for {len(repos)} total repositories.")
        print("==================================================")
        print("[Prefrontal Cortex] Evaluating repositories for usefulness...")
        
        approved_repos = []
        rejected_repos = 0
        
        for repo in repos:
            is_useful, score = self.evaluate_usefulness(repo)
            if is_useful:
                approved_repos.append((repo, score))
            else:
                rejected_repos += 1
                
        # Sort approved by score (most useful first)
        approved_repos.sort(key=lambda x: x[1], reverse=True)
        
        print(f"    [Filter] Rejected {rejected_repos} low-value repositories (e.g. static assets, forks).")
        print(f"    [Filter] Approved {len(approved_repos)} high-value repositories for learning.")
        print("--------------------------------------------------")
        
        # Ingest the approved ones
        for repo_tuple in approved_repos:
            repo = repo_tuple[0]
            repo_name = repo['name']
            clone_url = repo['clone_url']
            target_path = os.path.join(self.base_dir, repo_name)
            
            if os.path.exists(target_path):
                # We already have it from the previous scrape, skip it quietly
                continue
                
            print(f"\n[Motor Cortex] Executing physical clone of {repo_name}...")
            try:
                subprocess.run(["git", "clone", "--depth", "1", clone_url, target_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"    [Optical Nerve] Scanning {repo_name}...")
                
                nerve_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omni_optical_nerve.py")
                result = subprocess.run([sys.executable, nerve_script, target_path], capture_output=True, text=True)
                
                for line in result.stdout.split('\n'):
                    if "[Success]" in line or "[Omni-Eye]" in line:
                        print(f"    {line.strip()}")
            except Exception as e:
                print(f"    [ERROR] Failed: {e}")
                
            time.sleep(1) # Small rest
            
        print("\n==================================================")
        print("[Intelligent Forager] Evolution Complete. Knowledge synthesized.")

if __name__ == "__main__":
    forager = IntelligentForager()
    forager.discover_and_filter()
