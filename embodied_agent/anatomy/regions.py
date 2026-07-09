"""
Biological brain-region atlas for Aurora-Cerebrum (v7).
========================================================

Defines every macroscopic region we model and how the
**100 trillion synapse** budget is distributed across them.

Numbers are grounded in the neuroscience literature:

  * Total synapses in adult human brain   ≈ 1.5 × 10¹⁴   (we round to 1.0 × 10¹⁴)
  * Cerebellum holds ~50% of all neurons but ~12% of synapses (granule cells
    have tiny synaptic counts each)
  * Each cerebral hemisphere ≈ 40% of synaptic capacity
  * Brainstem + diencephalon (thalamus + hypothalamus) ≈ 4-5%
  * Limbic / basal ganglia ≈ 3%

Each region carries:

    name        : canonical anatomical name
    parent_id   : top-level division it belongs to
    side        : 'left' / 'right' / 'midline'
    synapses    : nominal synaptic count (the "capacity")
    neurons     : nominal neuron count
    active_frac : fraction of synapses considered "firing" each step
                  (used to compute active params per region)
    role        : short functional description
    modalities  : tuple of input modalities this region preferentially
                  consumes

The reference Python implementation does NOT allocate 100T floats — it
allocates one small `MoECortexStack` (or proxy module) per region and
*reports* the spec-level synapse budget through `BrainAtlas.census()`.

This is the same pattern as v3's tier system ("we ship the spec; the
runtime is the numpy reference"); v7 simply makes it anatomical.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# -------- top-level divisions ---------------------------------------
CEREBRUM_LEFT   = "cerebrum_left"
CEREBRUM_RIGHT  = "cerebrum_right"
CEREBELLUM      = "cerebellum"
BRAINSTEM       = "brainstem"
LIMBIC          = "limbic"
DIENCEPHALON    = "diencephalon"


@dataclass
class BrainRegion:
    name: str
    parent_id: str
    side: str  # 'left' | 'right' | 'midline'
    synapses: int
    neurons: int
    active_frac: float
    role: str
    modalities: Tuple[str, ...] = ()
    # Runtime book-keeping (set by Cerebrum at construction time).
    latent_dim: int = 0
    proxy_id: int = -1

    @property
    def active_synapses(self) -> int:
        return int(self.synapses * self.active_frac)


def build_human_atlas(target_total: int = 100_000_000_000_000) -> "BrainAtlas":
    """Construct the canonical 100T-synapse human-brain atlas.

    The numbers below were chosen so that the SUM equals `target_total`
    (default 1.0e14).  All proportions stay within the neuroscience
    ranges referenced at the top of this file.
    """
    # ---- proportional shares (must sum to 1.0) ----
    shares: Dict[str, float] = {
        # Cerebral cortex — left & right, each gets 4 lobes plus deep nuclei.
        "lh_frontal":       0.090,
        "lh_parietal":      0.060,
        "lh_temporal":      0.070,
        "lh_occipital":     0.055,
        "lh_insula":        0.015,
        "lh_cingulate":     0.020,

        "rh_frontal":       0.090,
        "rh_parietal":      0.060,
        "rh_temporal":      0.070,
        "rh_occipital":     0.055,
        "rh_insula":        0.015,
        "rh_cingulate":     0.020,

        # Cerebellum — ~12% of synapses despite ~50% of neurons.
        "cerebellum_left":  0.060,
        "cerebellum_right": 0.060,
        "vermis":           0.010,

        # Diencephalon
        "thalamus":         0.025,
        "hypothalamus":     0.010,

        # Limbic / basal ganglia
        "hippocampus":      0.015,
        "amygdala":         0.008,
        "striatum":         0.020,   # caudate + putamen
        "globus_pallidus":  0.007,
        "nucleus_accumbens":0.005,

        # Brainstem
        "midbrain":         0.020,
        "pons":             0.020,
        "medulla":          0.010,

        # Subcortical white-matter / basal forebrain
        "basal_forebrain":  0.005,
        "corpus_callosum":  0.005,
    }
    s = sum(shares.values())
    # Normalise (account for rounding).
    for k in shares:
        shares[k] /= s

    # ---- regional spec ----
    spec: List[BrainRegion] = []

    def add(name, parent, side, share, neurons, active_frac, role, mods=()):
        spec.append(BrainRegion(
            name=name, parent_id=parent, side=side,
            synapses=int(share * target_total),
            neurons=int(neurons), active_frac=float(active_frac),
            role=role, modalities=mods,
        ))

    # --- LEFT hemisphere — language, logic, sequential analysis ---
    add("lh_frontal",   CEREBRUM_LEFT,  "left", shares["lh_frontal"],
        1.6e9, 0.012, "Executive function, Broca's area, motor planning (L)",
        ("text", "scalar", "proprio"))
    add("lh_parietal",  CEREBRUM_LEFT,  "left", shares["lh_parietal"],
        1.2e9, 0.012, "Somatosensory map (R-body), spatial calc, math",
        ("proprio", "scalar"))
    add("lh_temporal",  CEREBRUM_LEFT,  "left", shares["lh_temporal"],
        1.4e9, 0.012, "Wernicke's area, semantic memory, hearing (R-ear)",
        ("text", "audio"))
    add("lh_occipital", CEREBRUM_LEFT,  "left", shares["lh_occipital"],
        1.1e9, 0.013, "V1-V5 (right visual field), object recognition",
        ("vision",))
    add("lh_insula",    CEREBRUM_LEFT,  "left", shares["lh_insula"],
        0.3e9, 0.010, "Interoceptive awareness, self-other distinction (L)",
        ("interoception",))
    add("lh_cingulate", CEREBRUM_LEFT,  "left", shares["lh_cingulate"],
        0.4e9, 0.010, "Conflict monitoring, error detection, attention",
        ("text", "scalar"))

    # --- RIGHT hemisphere — gestalt, spatial, prosodic, holistic ---
    add("rh_frontal",   CEREBRUM_RIGHT, "right", shares["rh_frontal"],
        1.6e9, 0.012, "Prosody, holistic planning, social inference (R)",
        ("audio", "vision"))
    add("rh_parietal",  CEREBRUM_RIGHT, "right", shares["rh_parietal"],
        1.2e9, 0.012, "Visuospatial reasoning, body schema (L-body)",
        ("vision", "proprio"))
    add("rh_temporal",  CEREBRUM_RIGHT, "right", shares["rh_temporal"],
        1.4e9, 0.012, "Prosody, melody, face recognition, music",
        ("audio", "vision"))
    add("rh_occipital", CEREBRUM_RIGHT, "right", shares["rh_occipital"],
        1.1e9, 0.013, "V1-V5 (left visual field), gestalt vision",
        ("vision",))
    add("rh_insula",    CEREBRUM_RIGHT, "right", shares["rh_insula"],
        0.3e9, 0.010, "Emotional salience, empathy, gut feelings",
        ("interoception",))
    add("rh_cingulate", CEREBRUM_RIGHT, "right", shares["rh_cingulate"],
        0.4e9, 0.010, "Emotional regulation, default-mode anchor (R)",
        ("text", "interoception"))

    # --- Cerebellum — fast supervised forward/inverse models ---
    add("cerebellum_left",  CEREBELLUM, "left",  shares["cerebellum_left"],
        3.5e10, 0.008, "Motor coordination & cognitive timing (R-body)",
        ("proprio", "scalar"))
    add("cerebellum_right", CEREBELLUM, "right", shares["cerebellum_right"],
        3.5e10, 0.008, "Motor coordination & cognitive timing (L-body)",
        ("proprio", "scalar"))
    add("vermis",           CEREBELLUM, "midline", shares["vermis"],
        0.5e10, 0.008, "Axial posture, balance, affect modulation", ())

    # --- Diencephalon — thalamic relay + hypothalamic homeostat ---
    add("thalamus", DIENCEPHALON, "midline", shares["thalamus"],
        1.5e8, 0.020, "Sensory relay + precision-weighted gating",
        ("text", "vision", "audio", "proprio", "scalar", "interoception"))
    add("hypothalamus", DIENCEPHALON, "midline", shares["hypothalamus"],
        1.0e8, 0.020, "Homeostatic set-points, cortisol/HPA axis, drives",
        ("interoception",))

    # --- Limbic + basal ganglia ---
    add("hippocampus", LIMBIC, "midline", shares["hippocampus"],
        3.0e8, 0.030, "Episodic memory, pattern separation/completion",
        ("text", "vision", "audio", "proprio"))
    add("amygdala", LIMBIC, "midline", shares["amygdala"],
        1.2e7, 0.030, "Threat detection, affective tagging, fear extinction",
        ("vision", "audio", "interoception"))
    add("striatum", LIMBIC, "midline", shares["striatum"],
        1.7e8, 0.015, "Action selection (go/no-go), habit formation",
        ("scalar",))
    add("globus_pallidus", LIMBIC, "midline", shares["globus_pallidus"],
        1.0e7, 0.015, "Tonic inhibition of thalamus, gating",
        ())
    add("nucleus_accumbens", LIMBIC, "midline", shares["nucleus_accumbens"],
        1.0e7, 0.020, "Reward prediction, motivation, hedonic tone",
        ("scalar", "interoception"))

    # --- Brainstem — arousal, autonomic, neuromodulator source nuclei ---
    add("midbrain", BRAINSTEM, "midline", shares["midbrain"],
        3.0e7, 0.025,
        "VTA/SN dopamine source, superior/inferior colliculi, RAS",
        ("vision", "audio"))
    add("pons", BRAINSTEM, "midline", shares["pons"],
        3.0e7, 0.025,
        "Locus coeruleus (NE), pontine nuclei (cerebellar relay), REM gate",
        ())
    add("medulla", BRAINSTEM, "midline", shares["medulla"],
        2.0e7, 0.025,
        "Autonomic life-support (HR, BP, breathing), raphe (5HT)",
        ("interoception",))

    # --- Basal forebrain (ACh) + corpus callosum (inter-hemisphere) ---
    add("basal_forebrain", BRAINSTEM, "midline", shares["basal_forebrain"],
        1.0e7, 0.025, "Acetylcholine source -> cortex, attention/arousal", ())
    add("corpus_callosum", CEREBRUM_LEFT, "midline", shares["corpus_callosum"],
        2.0e8, 0.010, "200M axons bridging L/R hemispheres", ())

    return BrainAtlas(spec, target_total=target_total)


@dataclass
class BrainAtlas:
    """A constructed, queryable atlas of brain regions."""
    regions: List[BrainRegion]
    target_total: int = 100_000_000_000_000

    def by_name(self, name: str) -> BrainRegion:
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(f"unknown region {name!r}")

    def by_side(self, side: str) -> List[BrainRegion]:
        return [r for r in self.regions if r.side == side]

    def by_parent(self, parent: str) -> List[BrainRegion]:
        return [r for r in self.regions if r.parent_id == parent]

    def total_synapses(self) -> int:
        return int(sum(r.synapses for r in self.regions))

    def total_active(self) -> int:
        return int(sum(r.active_synapses for r in self.regions))

    def total_neurons(self) -> int:
        return int(sum(r.neurons for r in self.regions))

    def census(self) -> dict:
        """Pretty-printable distribution of the 100T synapse budget."""
        by_parent: Dict[str, dict] = {}
        for r in self.regions:
            d = by_parent.setdefault(r.parent_id, dict(synapses=0, active=0,
                                                       neurons=0, regions=[]))
            d["synapses"] += r.synapses
            d["active"]   += r.active_synapses
            d["neurons"]  += r.neurons
            d["regions"].append(r.name)
        tot_s = self.total_synapses()
        tot_a = self.total_active()
        for k, d in by_parent.items():
            d["share_synapses"] = d["synapses"] / max(1, tot_s)
            d["share_active"]   = d["active"]   / max(1, tot_a)
        return dict(
            total_synapses=tot_s,
            total_active_synapses=tot_a,
            total_neurons=self.total_neurons(),
            n_regions=len(self.regions),
            by_parent=by_parent,
        )
