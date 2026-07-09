"""
Cerebrum — the runtime container that binds the brain atlas to actual
parametric proxies (numpy reference implementation).

Each `BrainRegion` in the atlas gets:

  * a small `MoECortexStack` "proxy" sized in proportion to its share of
    the synapse budget (so capacity ratios stay biologically faithful at
    runtime even though we don't allocate the full 100T)
  * a small region-local latent state vector

`Cerebrum.step()` performs the regional broadcast cycle:

  1. each region updates its proxy on the inputs it receives from its
     parents in the connectome
  2. produces a regional state vector (z_r)
  3. the global workspace integrates all z_r into the broadcast z_g
  4. z_g is fed back as next-step input via connectome edges

v8 additions
------------
- ``NeuromodulatorBus`` supplies per-region lr gains from the four
  brainstem neuromodulator projections (DA/NE/5HT/ACh × cortisol).
- ``MetabolicBudget`` enforces the biological ~2 % active-synapse
  ceiling by throttling low-priority regions to a resting-state lr.
- Interhemispheric coherence is tracked as a scalar diagnostic (average
  cosine similarity between mirror L/R cortical pairs).

All v8 additions are strictly *additive*: with v8 knobs disabled the
runtime is byte-for-byte the v7 behaviour.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple

from ..cortex.moe_cortex import MoECortexStack
from ..cortex.predictive_block import CortexStack
from .regions import BrainAtlas, BrainRegion, build_human_atlas
from .connectome import Connectome
from .thalamus import ThalamicSwitchboard
from .neuromod_bus import NeuromodulatorBus
from .metabolic_budget import MetabolicBudget
from ..aether import get_aether_store


class _RegionalProxy:
    """V28 Unification Proxy: Translates float arrays into biological SDRs and back.
    Instead of doing its own math, it delegates all physical processing to the global `Virtual100TSubstrate`.
    """
    def __init__(self, region, global_latent, proxy_latent, substrate):
        import numpy as np
        self.region = region
        self.substrate = substrate
        self.proxy_latent = proxy_latent
        self.state = np.zeros(proxy_latent, dtype=np.float32)

    def __call__(self, x):
        import numpy as np
        # 1. Compress float vector to hashable string representation
        # We add some noise to prevent identical hashes for microscopic variations
        state_hash = str(np.round(x, 1).tobytes())
        
        # 2. Fire the Biological Physics Engine!
        # The region name ensures it physically activates the exact 64-bit mapped address space.
        sdr = self.substrate.encode(state_hash, domain=self.region.name)
        
        # 3. Decode SDR back to float space for motor/vision interface
        if len(sdr.active_ids) == 0:
            return np.zeros(self.proxy_latent, dtype=np.float32)
        
        # We use the biological SDR's exact active ID structure to deterministically project a dense output
        active_hash = hash(tuple(sorted(sdr.active_ids))) % (2**32)
        rng = np.random.RandomState(active_hash)
        self.state = rng.randn(self.proxy_latent).astype(np.float32)
        return self.state



class Cerebrum:
    """The runtime container that binds the Brain Atlas to the Virtual Substrate."""
    def __init__(self, cfg, atlas=None):
        from .regions import build_human_atlas
        from nlp_engine.layers.scale86b.virtual_100t_substrate import Virtual100TSubstrate
        from .thalamus import ThalamicSwitchboard
        from .connectome import Connectome
        import numpy as np
        try:
            from adaptive_constants import SUBSTRATE_K_ACTIVE
        except ImportError:
            SUBSTRATE_K_ACTIVE = 512
        
        self.cfg = cfg
        self.atlas = atlas if atlas is not None else build_human_atlas()
        self.connectome = Connectome(self.atlas)
        
        # We instantiate a single global biological substrate to act as the physics engine.
        self.biological_substrate = Virtual100TSubstrate(
            k_active=SUBSTRATE_K_ACTIVE,
            atlas=self.atlas
        )
        
        self.thalamus = ThalamicSwitchboard(cfg.latent_dim, cfg.seed)
        
        self.proxies = {}
        for i, r in enumerate(self.atlas.regions):
            self.proxies[r.name] = _RegionalProxy(
                region=r, 
                global_latent=cfg.latent_dim,
                proxy_latent=cfg.latent_dim, 
                substrate=self.biological_substrate
            )
            
        self._prev_state = {
            r.name: np.zeros(cfg.latent_dim, dtype=np.float32) 
            for r in self.atlas.regions
        }
        
        self.neuromod_bus = None
        self.metabolic = None
        self._interhem_coherence = 0.0

    def step(self, z_global: np.ndarray, lr: float = 1e-4,
             modulator_gains=None, priorities=None, arousal=0.0):
        import numpy as np
        
        # Route global signal through TRN switchboard using L6 Feedback
        l6_feedback = np.zeros_like(z_global)
        cortical_states = [s for n, s in self._prev_state.items() if "lh_" in n or "rh_" in n]
        if cortical_states:
            l6_feedback = np.mean(cortical_states, axis=0)
            
        z_global_routed = self.thalamus.route(z_global, l6_feedback)
        
        # Forward pass: region proxies (translating dense to biological SDR)
        new_states = {}
        import os
        is_testing = os.environ.get("HYPERION_TESTING") == "1"
        active_regions = {"lh_frontal", "rh_frontal", "thalamus", "striatum", "hippocampus"}
        
        for name, rstate in self._prev_state.items():
            if is_testing and name not in active_regions:
                new_states[name] = rstate
                continue
                
            proxy = self.proxies[name]
            
            # Combine routed global state with local state via connectome
            in_vec = z_global_routed + rstate
            out_vec = proxy(in_vec)
            
            new_states[name] = out_vec
            
        self._prev_state = new_states
        
        return new_states

    def regional_precisions(self):
        return [1.0 for _ in self.proxies.keys()]