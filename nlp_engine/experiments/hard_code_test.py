import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from nlp_engine.layers.cain_brain import CainBrain

def run_hard_code_test():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Project HYPERION: Hard Algorithm Bench Test                    ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    brain = CainBrain()
    
    # We provide two famous LeetCode Hard problems:
    # 1. Trapping Rain Water (Two Pointers)
    # 2. Edit Distance (Dynamic Programming)
    hard_algorithms = """
def trap(height):
    if not height: return 0
    l, r = 0, len(height) - 1
    left_max, right_max = height[l], height[r]
    res = 0
    while l < r:
        if left_max < right_max:
            l += 1
            left_max = max(left_max, height[l])
            res += left_max - height[l]
        else:
            r -= 1
            right_max = max(right_max, height[r])
            res += right_max - height[r]
    return res

def minDistance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]
    """
    
    print("Ingesting Hard Algorithms (Trapping Rain Water, Edit Distance)...")
    for _ in range(100): # Ingest thoroughly to map abstract paths
        brain.learn(hard_algorithms, domain='code')
        
    print("\n" + "="*60)
    print(" 💻 BRAIN GENERATING ALGORITHMIC CODE")
    print("="*60)
    
    # We will prompt the brain to write a function and mix the states
    prompt = "def trap(height):"
    print(f"Prompting with: '{prompt}'")
    
    # High temperature so it gets creative and forces the PFC to do heavy filtering
    output = brain.speak(prompt, domain='code', length=80, temperature=0.8)
    
    print("────────────────────────────────────────────────────────────")
    print(output)
    print("────────────────────────────────────────────────────────────")

if __name__ == '__main__':
    run_hard_code_test()
