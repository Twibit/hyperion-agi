"""
Aurora-Cerebrum (v7) / Aurora-Sapiens (v8) — anatomical brain-region layer.

Distributes the 100-trillion-synapse capacity across biologically-grounded
regions of the human brain.  See `regions.py` for the regional spec
(left/right cerebral hemispheres × 4 lobes, cerebellum, brainstem, etc.)
and `connectome.py` for the inter-region wiring (corpus callosum,
thalamo-cortical loops, cortico-cerebellar loops, mesolimbic projections).

v8 additions (Aurora-Sapiens):

  * `NeuromodulatorBus` (neuromod_bus.py) — the four brainstem
    neuromodulator projections now actually modulate per-region plasticity.
  * `MetabolicBudget` (metabolic_budget.py) — the ~2 % biological
    activation ceiling is enforced at runtime, not just reported.
"""
from .regions import (
    BrainRegion, BrainAtlas, build_human_atlas,
    CEREBRUM_LEFT, CEREBRUM_RIGHT, CEREBELLUM,
    BRAINSTEM, LIMBIC, DIENCEPHALON,
)
from .cerebrum import Cerebrum
from .connectome import Connectome, ConnectomeEdge
from .neuromod_bus import NeuromodulatorBus
from .metabolic_budget import MetabolicBudget
