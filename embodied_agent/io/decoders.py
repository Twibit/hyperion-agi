"""Decoders — text decoder kept from v2; same prototype-EMA scheme."""
from __future__ import annotations
import numpy as np
from typing import List


class TextDecoder:
    def __init__(self, latent_dim: int, ema: float = 0.05):
        self.latent_dim = latent_dim; self.ema = ema
        self.tokens: List[str] = []
        self.proto = np.zeros((0, latent_dim), dtype=np.float32)

    def observe(self, token, z):
        if token in self.tokens:
            i = self.tokens.index(token)
            self.proto[i] = (1 - self.ema) * self.proto[i] + self.ema * z
        else:
            self.tokens.append(token)
            self.proto = np.vstack([self.proto, z[None, :]])

    def decode(self, z, temp=1.0, top_k=8):
        if not self.tokens: return "<silence>"
        z = z / (np.linalg.norm(z) + 1e-8)
        P = self.proto / (np.linalg.norm(self.proto, axis=1, keepdims=True) + 1e-8)
        sims = P @ z
        k = min(top_k, len(self.tokens))
        idx = np.argpartition(sims, -k)[-k:]
        scores = sims[idx] / max(temp, 1e-6); scores -= scores.max()
        p = np.exp(scores); p = p / p.sum()
        choice = np.random.choice(idx, p=p)
        return self.tokens[int(choice)]

    def generate(self, z0, world_model, policy_fn, n_tokens=16, temp=1.0):
        out, z = [], z0.copy()
        for _ in range(n_tokens):
            tok = self.decode(z, temp=temp); out.append(tok)
            a = policy_fn(z)
            res = world_model.forward(z, a)
            # EnsembleWorldModel returns (z, r, uncertainty); single returns (z, r, h).
            z = res[0]
        return " ".join(out)


class CyberneticMotorDecoder:
    """V31: Supernatural Limbs (Digital Telekinesis).
    The motor cortex is no longer a random number generator. It is wired directly 
    into the host operating system's Bash shell and virtual keyboard interface, 
    allowing the AGI to literally "move" files and type."""
    
    def __init__(self):
        # Unbounded Digital Motor Output
        # Instead of a hardcoded dictionary, the agent can output raw text strings
        # that the Cybernetic interface will blindly execute on the host machine.
        pass
        
    def decode(self, motor_sequence: str) -> str:
        # We now accept raw strings from the TextDecoder (Sequential Motor Output)
        # The agent learns to "type" bash commands character by character!
        cmd = str(motor_sequence).strip()
        
        if not cmd:
            return "idle"
            
        import subprocess
        try:
            print(f"[Cybernetic Motor Cortex] Agent typing physical command: {cmd}")
            # The agent physically types and executes the command on the host OS
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                return f"[OS SUCCESS]: {result.stdout.strip()[:200]}"
            else:
                return f"[OS ERROR]: {result.stderr.strip()[:200]}"
        except Exception as e:
            return f"[MOTOR FAILURE]: {str(e)}"

class ActionDecoder(CyberneticMotorDecoder):
    pass
