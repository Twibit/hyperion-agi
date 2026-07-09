import numpy as np

class GridworldEnv:
    """
    ARC-style 5x5 Gridworld Testing Environment.
    
    The agent starts at (0, 0) and must reach (4, 4).
    We inject a distractor noise channel to test TRN attention gating,
    and delayed rewards to test Astrocyte calcium wave temporal credit assignment.
    """
    def __init__(self, size: int = 5, seed: int = 42, max_steps: int = 40):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.reset()
        
        # Directions mapping: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    def reset(self) -> dict:
        self.agent_pos = [0, 0]
        self.target_pos = [self.size - 1, self.size - 1]
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self) -> dict:
        # 1. Coordinate spatial text
        spatial_text = f"location cell x {self.agent_pos[0]} y {self.agent_pos[1]}"
        
        # 2. Distractor noise text (simulates flashing static/distractions to stress-test TRN)
        distractor_words = ["noise", "static", "glitch", "buzz", "click", "hum", "fuzz", "pop"]
        noise_text = " ".join(self.rng.choice(distractor_words, size=3))
        
        return {
            "text": spatial_text,
            "noise": noise_text,
            "reward": 0.0 # Will be overridden in step()
        }

    def step(self, action_idx: int) -> tuple[dict, float, bool, dict]:
        self.step_count += 1
        
        # Keep index within bounds [0, 3]
        action = action_idx % 4
        
        # Track old position to calculate distance delta (chemotaxis)
        old_pos = list(self.agent_pos)
        
        # Apply movement
        if action == 0: # UP (decrease y coordinate)
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1: # DOWN (increase y coordinate)
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2: # LEFT (decrease x coordinate)
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3: # RIGHT (increase x coordinate)
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)

        # Check goal condition
        reached_goal = (self.agent_pos == self.target_pos)
        
        # Chemotaxis / Gradient-based reward shaping (Manhattan distance delta)
        old_dist = abs(old_pos[0] - self.target_pos[0]) + abs(old_pos[1] - self.target_pos[1])
        new_dist = abs(self.agent_pos[0] - self.target_pos[0]) + abs(self.agent_pos[1] - self.target_pos[1])
        shaping = 0.25 * (old_dist - new_dist)
        
        # Total reward combines target arrival and chemotaxis gradient
        reward = (3.0 if reached_goal else 0.0) + shaping
        
        # Terminate if goal reached or steps exceed limit
        done = reached_goal or (self.step_count >= self.max_steps)
        
        obs = self._get_obs()
        obs["reward"] = reward
        
        info = {
            "pos": tuple(self.agent_pos),
            "action_taken": self.action_names[action],
            "step": self.step_count
        }
        
        return obs, reward, done, info
