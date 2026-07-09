#!/usr/bin/env python3
import os
import sys
import shutil
import time
import numpy as np

# Ensure root paths are registered
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from simulation_env.gridworld import GridworldEnv
from embodied_agent import AuroraOmnia, AuroraOmniaConfig

def main():
    # Set testing environment variable to speed up simulation by 20x
    os.environ["HYPERION_TESTING"] = "1"
    
    print("\n" + "="*70, flush=True)
    print("  HYPERION BIOLOGICAL AGI — ARC-STYLE GRIDWORLD MAZE CONVERGENCE  ", flush=True)
    print("="*70, flush=True)
    
    # 1. Clear caches to ensure a completely clean, blank-slate brain
    print("[Gridworld] Cleaning old synaptic expert caches...", flush=True)
    shutil.rmtree("/tmp/aurora_aether", ignore_errors=True)
    shutil.rmtree("/tmp/hyperion_synapse_cache", ignore_errors=True)

    # 2. Setup the organism config
    cfg = AuroraOmniaConfig(
        tier="base",
        n_actions=4,
        lr_basal=0.08,
        lr_world=0.05,
        nociception_clip=0.01,
        log_every=999,
        consolidation_every=999,
        anatomy_enable=True,
        neuromod_bus_enable=True,
        metabolic_budget_enable=True,
        thalamic_router_enable=True,
    )
    
    print("[Gridworld] Booting organism...", flush=True)
    org = AuroraOmnia(cfg)
    env = GridworldEnv(size=5, seed=123, max_steps=35)

    # Telemetry tracker
    history_steps = []
    history_rewards = []
    
    n_episodes = 25
    print(f"\n[Gridworld] Starting learning trials ({n_episodes} episodes)...", flush=True)
    
    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        path = [(0, 0)]
        
        t0 = time.time()
        
        while not done:
            # Step 1: Brain perception, processing, and policy selection
            # We pass the observations dictionary (including text location and distractor noise)
            # The Thalamic Switchboard must learn to route past the 'noise' modality.
            result = org.step(obs, reward=obs["reward"])
            
            # Step 2: Decode the selected action index from the brain
            raw_action = result["action"]
            
            # Step 3: Apply action to the gridworld environment
            obs, reward, done, info = env.step(raw_action)
            
            ep_reward += reward
            steps += 1
            path.append(info["pos"])
            
        dt = time.time() - t0
        history_steps.append(steps)
        history_rewards.append(ep_reward)
        
        # Display trajectory
        reached = "SUCCESS" if path[-1] == (4,4) else "TIMEOUT"
        path_str = " -> ".join(f"({x},{y})" for x, y in path[:8])
        if len(path) > 8:
            path_str += f" ... -> ({path[-1][0]},{path[-1][1]})"
            
        print(f"  Episode {ep:>2d}/{n_episodes} | Status: {reached:<7} | Steps: {steps:>3d} ({dt:.2f}s) | Path: {path_str}", flush=True)

    # 4. Print overall convergence report
    print("\n" + "="*70, flush=True)
    print("  CONVERGENCE SUMMARY  ", flush=True)
    print("="*70, flush=True)
    print(f"  First Episode Steps : {history_steps[0]}", flush=True)
    print(f"  Last Episode Steps  : {history_steps[-1]}", flush=True)
    
    min_steps = min(history_steps)
    print(f"  Best Traverse Steps : {min_steps} (Optimal is 8 steps)", flush=True)
    
    # Calculate learning slope
    first_half_avg = np.mean(history_steps[:n_episodes//2])
    second_half_avg = np.mean(history_steps[n_episodes//2:])
    print(f"  First-Half Average  : {first_half_avg:.2f} steps", flush=True)
    print(f"  Second-Half Average : {second_half_avg:.2f} steps", flush=True)
    
    if second_half_avg < first_half_avg:
        improvement = ((first_half_avg - second_half_avg) / first_half_avg) * 100
        print(f"  Learning Slope      : -{first_half_avg - second_half_avg:.2f} steps ({improvement:.1f}% improvement) (Pass)", flush=True)
    else:
        print(f"  Learning Slope      : No improvement detected. (Fail)", flush=True)
        
    # Check TRN gating efficacy
    if hasattr(org, 'thalamus'):
        trn = org.thalamus
        if hasattr(trn, 'trn_weights'):
            print(f"  TRN Weight Norm     : {np.linalg.norm(trn.trn_weights):.4f}", flush=True)
            
    print("="*70 + "\n", flush=True)

if __name__ == "__main__":
    main()
