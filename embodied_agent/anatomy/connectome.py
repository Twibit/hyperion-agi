"""
Anatomical connectome — v7.
============================

Wires the brain regions defined in `regions.py` into a coarse-grained
connectivity graph.  Edge weights are normalised cross-region influences
(0..1) used by the `Cerebrum` to route information.

Major bundles modelled:

  * thalamus -> all cortical regions       (sensory relay)
  * cortex -> cortex within hemisphere     (short association)
  * L cortex <-> R cortex via corpus callosum (long inter-hem.)
  * cortex -> pontine nuclei -> cerebellum -> thalamus (cortico-cerebellar loop)
  * VTA/SN -> nucleus accumbens / striatum / cortex (dopaminergic)
  * locus coeruleus (pons) -> cortex       (noradrenergic broadcast)
  * raphe (medulla) -> cortex              (serotonergic broadcast)
  * basal forebrain -> cortex              (cholinergic broadcast)
  * amygdala -> hypothalamus + cortex      (threat/affect)
  * hippocampus <-> cortex                 (memory consolidation)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class ConnectomeEdge:
    src: str
    dst: str
    weight: float
    bundle: str  # 'thalamic' | 'callosal' | 'cortico-cerebellar' | ...


class Connectome:
    def __init__(self, atlas, default_intra: float = 0.4,
                 default_inter: float = 0.15, callosal: float = 0.5):
        self.atlas = atlas
        self.edges: List[ConnectomeEdge] = []
        self._adj: Dict[str, List[Tuple[str, float, str]]] = {}
        self._build(default_intra, default_inter, callosal)

    def _add(self, src: str, dst: str, w: float, bundle: str) -> None:
        self.edges.append(ConnectomeEdge(src=src, dst=dst,
                                         weight=float(w), bundle=bundle))
        self._adj.setdefault(src, []).append((dst, float(w), bundle))

    def _build(self, intra: float, inter: float, callosal: float) -> None:
        names = [r.name for r in self.atlas.regions]
        cortex_l = [n for n in names if n.startswith("lh_")]
        cortex_r = [n for n in names if n.startswith("rh_")]

        # 1) Thalamus -> segregated cortical relays (instead of blind global broadcast)
        if "thalamus" in names:
            for c in cortex_l + cortex_r:
                if "occipital" in c:
                    self._add("thalamus", c, 0.9, "thalamic-visual") # LGN
                elif "temporal" in c:
                    self._add("thalamus", c, 0.9, "thalamic-auditory") # MGN
                elif "parietal" in c:
                    self._add("thalamus", c, 0.9, "thalamic-somatosensory") # VPL
                else:
                    self._add("thalamus", c, 0.2, "thalamic-association") # Weak general relay
                self._add(c, "thalamus", 0.3, "cortico-thalamic")

        # 2) Within-hemisphere short-range cortical association
        def within(hemi):
            for i, a in enumerate(hemi):
                for b in hemi[i + 1:]:
                    self._add(a, b, intra, "ipsilateral-association")
                    self._add(b, a, intra, "ipsilateral-association")
        within(cortex_l)
        within(cortex_r)

        # 3) Corpus callosum: L<->R mirror regions
        for l in cortex_l:
            r = "rh_" + l[3:]
            if r in names:
                self._add(l, r, callosal, "callosal")
                self._add(r, l, callosal, "callosal")
        # general inter-hemispheric weak links
        for l in cortex_l:
            for r in cortex_r:
                if r != "rh_" + l[3:]:
                    self._add(l, r, inter, "callosal-weak")
                    self._add(r, l, inter, "callosal-weak")

        # 4) Cortico-cerebellar loops: cortex -> pons -> cerebellum -> thalamus -> cortex
        if "pons" in names:
            for c in cortex_l + cortex_r:
                self._add(c, "pons", 0.4, "cortico-pontine")
            for cb in ("cerebellum_left", "cerebellum_right", "vermis"):
                if cb in names:
                    self._add("pons", cb, 0.8, "ponto-cerebellar")
                    if "thalamus" in names:
                        self._add(cb, "thalamus", 0.7, "cerebello-thalamic")

        # 5) Dopaminergic broadcast: midbrain (VTA/SN) -> striatum, accumbens, cortex
        if "midbrain" in names:
            for tgt in ("striatum", "nucleus_accumbens"):
                if tgt in names:
                    self._add("midbrain", tgt, 0.9, "dopaminergic")
            for c in ("lh_frontal", "rh_frontal", "lh_cingulate", "rh_cingulate"):
                if c in names:
                    self._add("midbrain", c, 0.7, "dopaminergic")

        # 6) Noradrenergic (locus coeruleus in pons)
        if "pons" in names:
            for c in cortex_l + cortex_r:
                self._add("pons", c, 0.3, "noradrenergic")

        # 7) Serotonergic (raphe in medulla)
        if "medulla" in names:
            for c in cortex_l + cortex_r:
                self._add("medulla", c, 0.25, "serotonergic")
            if "amygdala" in names:
                self._add("medulla", "amygdala", 0.4, "serotonergic")

        # 8) Cholinergic (basal forebrain -> cortex)
        if "basal_forebrain" in names:
            for c in cortex_l + cortex_r:
                self._add("basal_forebrain", c, 0.4, "cholinergic")

        # 9) Amygdala -> hypothalamus + cortex, and Prefrontal Regulation
        if "amygdala" in names:
            if "hypothalamus" in names:
                self._add("amygdala", "hypothalamus", 0.8, "limbic")
            for c in ("lh_insula", "rh_insula", "lh_cingulate", "rh_cingulate"):
                if c in names:
                    self._add("amygdala", c, 0.5, "limbic")
                    
            # V29: Prefrontal Top-Down Inhibition (Emotion Regulation)
            for c in ("lh_frontal", "rh_frontal"):
                if c in names:
                    self._add(c, "amygdala", -0.8, "prefrontal-inhibition")
                    
            # V29: Flashbulb Memory Pathway
            if "hippocampus" in names:
                self._add("amygdala", "hippocampus", 0.9, "flashbulb-memory")
                self._add("hippocampus", "amygdala", 0.4, "flashbulb-memory")

        # 10) Hippocampus <-> cortex (especially temporal & cingulate)
        if "hippocampus" in names:
            for c in ("lh_temporal", "rh_temporal", "lh_cingulate", "rh_cingulate"):
                if c in names:
                    self._add("hippocampus", c, 0.7, "memory")
                    self._add(c, "hippocampus", 0.5, "memory")

        # 11) Striatum -> globus pallidus -> thalamus (canonical BG loop)
        if "striatum" in names and "globus_pallidus" in names:
            self._add("striatum", "globus_pallidus", 0.9, "basal-ganglia")
        if "globus_pallidus" in names and "thalamus" in names:
            self._add("globus_pallidus", "thalamus", -0.7, "basal-ganglia")

    # ---------- queries ----------
    def neighbours(self, region: str) -> List[Tuple[str, float, str]]:
        return list(self._adj.get(region, []))

    def adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        names = [r.name for r in self.atlas.regions]
        idx = {n: i for i, n in enumerate(names)}
        n = len(names)
        A = np.zeros((n, n), dtype=np.float32)
        for e in self.edges:
            if e.src in idx and e.dst in idx:
                A[idx[e.src], idx[e.dst]] = e.weight
        return A, names

    def stats(self) -> dict:
        bundles: Dict[str, int] = {}
        for e in self.edges:
            bundles[e.bundle] = bundles.get(e.bundle, 0) + 1
        return dict(
            n_edges=len(self.edges),
            bundles=bundles,
            avg_weight=float(np.mean([abs(e.weight) for e in self.edges])),
        )

    # ====================================================================
    # V14 Hebbian Plasticity — "Neurons that fire together, wire together"
    # ====================================================================
    def hebbian_update(self, active_regions: Dict[str, float],
                       lr: float = 0.01,
                       w_min: float = 0.01,
                       w_max: float = 1.5) -> int:
        """Strengthen edges between co-active regions, weaken edges to inactive ones.

        Parameters
        ----------
        active_regions : dict region_name -> activation_strength [0.0 - 1.0]
            Regions that fired this step and how strongly.
        lr : float
            Plasticity learning rate.  Kept low to avoid seizure-like runaway.
        w_min / w_max : float
            Hard clamps on edge weight magnitude to prevent degenerate wiring.

        Returns
        -------
        int  — number of edges updated.
        """
        n_updated = 0
        for edge in self.edges:
            src_act = active_regions.get(edge.src, 0.0)
            dst_act = active_regions.get(edge.dst, 0.0)

            # Hebbian term: both active → strengthen
            hebbian = src_act * dst_act

            # Anti-Hebbian / homeostatic term: one active, other silent → weaken
            anti_hebbian = abs(src_act - dst_act) * 0.5

            delta = lr * (hebbian - anti_hebbian)

            if abs(delta) > 1e-8:
                old_sign = 1.0 if edge.weight >= 0 else -1.0
                edge.weight += delta
                # Clamp magnitude but preserve original sign (inhibitory stays inhibitory)
                edge.weight = old_sign * max(w_min, min(w_max, abs(edge.weight)))
                n_updated += 1

        # Rebuild adjacency cache after weight changes
        self._adj.clear()
        for e in self.edges:
            self._adj.setdefault(e.src, []).append((e.dst, float(e.weight), e.bundle))

        return n_updated
