#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  B R A I N   S I M   —  Core Models                                        ║
║                                                                             ║
║  Multi-scale cortical simulation: neuron → synapse → column → network      ║
║                                                                             ║
║  Scales modeled:                                                            ║
║    • Ion channel (μs): HH gating, Ca²⁺ dynamics                             ║
║    • Synapse (ms): STDP, AMPA/NMDA/GABA conductance                         ║
║    • Neuron (ms): multi-compartment, spike generation                       ║
║    • Microcircuit (s): cortical column, 6 layers, 4 cell types              ║
║    • Network (min): thalamocortical, hippocampal loops                      ║
║    • Energy (continuous): ATP accounting per event                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import time
import numpy as np
from collections import defaultdict, deque
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple, Union

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — Biophysical Parameters
# ═══════════════════════════════════════════════════════════════════════════════

# Ionic concentrations (mM)
CONC_NA_OUT = 145.0   # Extracellular Na⁺
CONC_NA_IN = 15.0     # Intracellular Na⁺
CONC_K_OUT = 4.0      # Extracellular K⁺
CONC_K_IN = 140.0     # Intracellular K⁺
CONC_CA_OUT = 2.0     # Extracellular Ca²⁺
CONC_CA_IN = 1e-4     # Intracellular Ca²⁺ (100 nM resting)
CONC_MG_OUT = 1.0     # Extracellular Mg²⁺ (for NMDA block)

# Reversal potentials (mV) via Nernst equation at 37°C
E_NA = 58.0 * math.log10(CONC_NA_OUT / CONC_NA_IN)   # ~+58 mV
E_K = 58.0 * math.log10(CONC_K_OUT / CONC_K_IN)       # ~-90 mV
E_CA = 58.0 * math.log10(CONC_CA_OUT / CONC_CA_IN)    # ~+125 mV
E_L = -70.0   # Leak reversal
E_GABA = -75.0  # Cl⁻ reversal
E_AMPA = 0.0     # Cationic (non-selective)

# Temperature
TEMP_C = 37.0       # Physiological temperature (°C)
TEMP_BASE = 6.3     # Base temp for HH rate equations (°C)
TEMP_FACTOR = 3.0 ** ((TEMP_C - TEMP_BASE) / 10.0)  # Q10 temperature correction

# ═══════════════════════════════════════════════════════════════════════════════
#  ENERGY BUDGET — ATP Accounting
# ═══════════════════════════════════════════════════════════════════════════════

class EnergyBudget:
    """
    Realistic ATP accounting per neural event.

    Reference: Attwell & Laughlin (2001), Howarth et al. (2012)
    A single human brain consumes ~20W = ~3.3e25 ATP/s
    """

    # ATP molecules consumed per event
    ATP_PER_SPIKE = 3.9e8       # Single action potential (Na⁺/K⁺ pump restoration)
    ATP_PER_AMPA = 1.4e5        # Single AMPA receptor activation
    ATP_PER_NMDA = 2.1e5        # Single NMDA receptor activation (incl. Mg²⁺ unblock)
    ATP_PER_GABA_A = 1.1e5      # GABA_A receptor activation
    ATP_PER_GABA_B = 1.8e5      # GABA_B receptor (G-protein cascade)
    ATP_PER_VESICLE = 1.0e5     # Synaptic vesicle release (exocytosis + recycling)
    ATP_PER_REST_MS = 1.0e4     # Resting potential maintenance per ms per neuron
    ATP_PER_CASPASE = 5.0e6     # Apoptosis signaling event

    # Human brain scale
    HUMAN_NEURONS = 8.6e10      # ~86 billion
    HUMAN_ATP_PER_S = 3.3e25    # ~20W
    HUMAN_SPIKES_PER_S_AVG = 4e15  # Average firing rate encoding

    _instance = None
    _total_atp = 0.0
    _log = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._total_atp = 0.0
            cls._instance._log = []
        return cls._instance

    @classmethod
    def spend(cls, amount: float, category: str = "unknown"):
        """Record ATP expenditure."""
        cls._total_atp += amount
        cls._log.append((amount, category))
        return cls._total_atp

    @classmethod
    def spike_cost(cls):
        """Cost of one action potential and its propagation."""
        amt = cls.ATP_PER_SPIKE + cls.ATP_PER_VESICLE
        return cls.spend(amt, "spike")

    @classmethod
    def synapse_cost(cls, receptor_type: str = "ampa"):
        """Cost of activating one synapse."""
        costs = {
            "ampa": cls.ATP_PER_AMPA,
            "nmda": cls.ATP_PER_NMDA,
            "gaba_a": cls.ATP_PER_GABA_A,
            "gaba_b": cls.ATP_PER_GABA_B,
        }
        amt = costs.get(receptor_type, cls.ATP_PER_AMPA)
        return cls.spend(amt, f"synapse_{receptor_type}")

    @classmethod
    def rest_cost(cls, n_neurons: int, dt_ms: float):
        """Maintenance cost for n neurons over dt milliseconds."""
        amt = cls.ATP_PER_REST_MS * n_neurons * dt_ms
        return cls.spend(amt, "resting")

    @classmethod
    def total(cls):
        return cls._total_atp

    @classmethod
    def watts(cls, elapsed_seconds: float):
        """Convert ATP used to watts (20W = 3.3e25 ATP/s = 1 human brain)."""
        if elapsed_seconds <= 0:
            return 0.0
        atp_per_s = cls._total_atp / elapsed_seconds
        watts = atp_per_s / cls.HUMAN_ATP_PER_S * 20.0
        return watts

    @classmethod
    def report(cls, elapsed_seconds: float = 1.0):
        """Generate energy report."""
        if not cls._log:
            return "No energy data."
        categories = defaultdict(float)
        for amt, cat in cls._log:
            categories[cat] += amt
        total = cls._total_atp
        lines = [
            f"Energy Budget Report",
            f"{'='*50}",
            f"Total ATP: {total:.2e} molecules",
            f"Power:    {cls.watts(elapsed_seconds):.2f} W "
            f"(human brain: ~20W)",
            f"",
            f"Breakdown:",
        ]
        for cat, amt in sorted(categories.items(), key=lambda x: -x[1]):
            pct = amt / total * 100 if total > 0 else 0
            lines.append(f"  {cat:20s}: {amt:12.2e} ATP ({pct:5.1f}%)")
        return "\n".join(lines)

    @classmethod
    def reset(cls):
        """Reset energy accounting."""
        cls._total_atp = 0.0
        cls._log = []


# ═══════════════════════════════════════════════════════════════════════════════
#  HODGKIN-HUXLEY GATING KINETICS
# ═══════════════════════════════════════════════════════════════════════════════

class GatingKinetics:
    """
    Hodgkin-Huxley gating variable rate functions.

    Each gate has voltage-dependent alpha and beta rate functions.
    The gating variable x evolves as:
        dx/dt = alpha_x(V) * (1 - x) - beta_x(V) * x

    Temperature factor: Q10 ~ 3.0 (kinetics speed up ~3× per 10°C)
    """

    @staticmethod
    def alpha_m(V: float) -> float:
        """Na⁺ activation (fast)."""
        v_shift = V + 40.0
        if abs(v_shift) < 1e-6:
            return 1.0
        return 0.1 * v_shift / (1.0 - math.exp(-v_shift / 10.0))

    @staticmethod
    def beta_m(V: float) -> float:
        """Na⁺ activation (fast)."""
        return 4.0 * math.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def alpha_h(V: float) -> float:
        """Na⁺ inactivation (slower)."""
        return 0.07 * math.exp(-(V + 65.0) / 20.0)

    @staticmethod
    def beta_h(V: float) -> float:
        """Na⁺ inactivation (slower)."""
        return 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))

    @staticmethod
    def alpha_n(V: float) -> float:
        """K⁺ activation (slow)."""
        v_shift = V + 55.0
        if abs(v_shift) < 1e-6:
            return 0.01
        return 0.01 * v_shift / (1.0 - math.exp(-v_shift / 10.0))

    @staticmethod
    def beta_n(V: float) -> float:
        """K⁺ inactivation (slow)."""
        return 0.125 * math.exp(-(V + 65.0) / 80.0)

    @staticmethod
    def tau_m(V: float) -> float:
        """Na⁺ activation time constant (ms)."""
        a = GatingKinetics.alpha_m(V)
        b = GatingKinetics.beta_m(V)
        return 1.0 / (a + b) / TEMP_FACTOR

    @staticmethod
    def tau_h(V: float) -> float:
        """Na⁺ inactivation time constant (ms)."""
        a = GatingKinetics.alpha_h(V)
        b = GatingKinetics.beta_h(V)
        return 1.0 / (a + b) / TEMP_FACTOR

    @staticmethod
    def tau_n(V: float) -> float:
        """K⁺ activation time constant (ms)."""
        a = GatingKinetics.alpha_n(V)
        b = GatingKinetics.beta_n(V)
        return 1.0 / (a + b) / TEMP_FACTOR

    @staticmethod
    def m_inf(V: float) -> float:
        """Steady-state Na⁺ activation."""
        a = GatingKinetics.alpha_m(V)
        b = GatingKinetics.beta_m(V)
        return a / (a + b)

    @staticmethod
    def h_inf(V: float) -> float:
        """Steady-state Na⁺ inactivation."""
        a = GatingKinetics.alpha_h(V)
        b = GatingKinetics.beta_h(V)
        return a / (a + b)

    @staticmethod
    def n_inf(V: float) -> float:
        """Steady-state K⁺ activation."""
        a = GatingKinetics.alpha_n(V)
        b = GatingKinetics.beta_n(V)
        return a / (a + b)

    # Vectorized versions for array operations
    @staticmethod
    def m_inf_v(V: np.ndarray) -> np.ndarray:
        """Vectorized m_inf."""
        v_shift = V + 40.0
        alpha = np.where(np.abs(v_shift) < 1e-6, 1.0,
                         0.1 * v_shift / (1.0 - np.exp(-v_shift / 10.0)))
        beta = 4.0 * np.exp(-(V + 65.0) / 18.0)
        return alpha / (alpha + beta)

    @staticmethod
    def h_inf_v(V: np.ndarray) -> np.ndarray:
        """Vectorized h_inf."""
        alpha = 0.07 * np.exp(-(V + 65.0) / 20.0)
        beta = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
        return alpha / (alpha + beta)

    @staticmethod
    def n_inf_v(V: np.ndarray) -> np.ndarray:
        """Vectorized n_inf."""
        v_shift = V + 55.0
        alpha = np.where(np.abs(v_shift) < 1e-6, 0.01,
                         0.01 * v_shift / (1.0 - np.exp(-v_shift / 10.0)))
        beta = 0.125 * np.exp(-(V + 65.0) / 80.0)
        return alpha / (alpha + beta)


# ═══════════════════════════════════════════════════════════════════════════════
#  HODGKIN-HUXLEY SOMA — Single Compartment
# ═══════════════════════════════════════════════════════════════════════════════

class HodgkinHuxleySoma:
    """
    Single-compartment Hodgkin-Huxley soma with Na⁺, K⁺, leak, and Ca²⁺.

    Membrane equation:
        C_m * dV/dt = -g_Na * m³ * h * (V - E_Na) - g_K * n⁴ * (V - E_K)
                      - g_L * (V - E_L) - g_Ca * ... + I_syn + I_inj

    Capacitance: C_m = 1 μF/cm² (standard)
    """

    def __init__(self, g_na: float = 120.0, g_k: float = 36.0,
                 g_l: float = 0.3, g_ca: float = 1.0,
                 cm: float = 1.0, area: float = 1.0):
        """
        Args:
            g_na: Max Na⁺ conductance (mS/cm²). Default 120.
            g_k: Max K⁺ conductance (mS/cm²). Default 36.
            g_l: Leak conductance (mS/cm²). Default 0.3.
            g_ca: Ca²⁺ conductance (mS/cm²). Default 1.0.
            cm: Membrane capacitance (μF/cm²). Default 1.0.
            area: Membrane area relative to standard (1.0 = standard).
        """
        self.g_na = g_na
        self.g_k = g_k
        self.g_l = g_l
        self.g_ca = g_ca
        self.cm = cm
        self.area = area

        # State variables
        self.V = -65.0          # Membrane potential (mV)
        self.m = 0.05           # Na⁺ activation
        self.h = 0.6            # Na⁺ inactivation
        self.n = 0.32           # K⁺ activation
        self.ca_in = CONC_CA_IN  # Intracellular Ca²⁺ (mM)

        self._spiked = False    # Spike detection flag

    def reset(self):
        """Reset to resting state."""
        self.V = -65.0
        self.m = GatingKinetics.m_inf(-65.0)
        self.h = GatingKinetics.h_inf(-65.0)
        self.n = GatingKinetics.n_inf(-65.0)
        self.ca_in = CONC_CA_IN
        self._spiked = False

    def step(self, I_syn: float = 0.0, dt: float = 0.01) -> bool:
        """
        Advance one time step.

        Args:
            I_syn: Total synaptic current (nA).
            dt: Time step (ms). Default 0.01 (10 μs).

        Returns:
            True if a spike occurred in this step.
        """
        V = self.V
        m, h, n = self.m, self.h, self.n
        T = TEMP_FACTOR

        # Rate functions (temperature-corrected)
        a_m = GatingKinetics.alpha_m(V)
        b_m = GatingKinetics.beta_m(V)
        a_h = GatingKinetics.alpha_h(V)
        b_h = GatingKinetics.beta_h(V)
        a_n = GatingKinetics.alpha_n(V)
        b_n = GatingKinetics.beta_n(V)

        # Update gating variables (exact Euler)
        dm = T * (a_m * (1.0 - m) - b_m * m)
        dh = T * (a_h * (1.0 - h) - b_h * h)
        dn = T * (a_n * (1.0 - n) - b_n * n)

        self.m += dt * dm
        self.h += dt * dh
        self.n += dt * dn

        # Clamp gates to valid range
        self.m = np.clip(self.m, 0.0, 1.0)
        self.h = np.clip(self.h, 0.0, 1.0)
        self.n = np.clip(self.n, 0.0, 1.0)

        # Ionic currents
        I_na = self.g_na * (self.m ** 3) * self.h * (V - E_NA)
        I_k = self.g_k * (self.n ** 4) * (V - E_K)
        I_l = self.g_l * (V - E_L)

        # Ca²⁺ current (L-type, high-voltage activated)
        m_ca_inf = 1.0 / (1.0 + math.exp(-(V + 20.0) / 9.0))
        I_ca = self.g_ca * m_ca_inf * (V - E_CA)

        # Total ionic current
        I_ion = I_na + I_k + I_l + I_ca

        # Membrane potential update (clamped to prevent math overflow)
        dV = (-I_ion + I_syn) / (self.cm * self.area)
        self.V += dt * dV
        self.V = np.clip(self.V, -100.0, 60.0)

        # Ca²⁺ dynamics: influx during spikes, decay via pumps
        if self.V > -10.0:  # Ca²⁺ enters during depolarization
            ca_influx = -I_ca * dt * 5.18e-5  # Conversion factor
            self.ca_in += max(0, ca_influx)
            self.ca_in = min(self.ca_in, 5e-3)  # Max 5 μM
        # Ca²⁺ decay (pump)
        self.ca_in += dt * (CONC_CA_IN - self.ca_in) / 80.0  # τ ~80ms

        # Spike detection
        self._spiked = False
        if self.V > 0.0:
            self._spiked = True

        return self._spiked

    def has_spiked(self) -> bool:
        return self._spiked


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTI-COMPARTMENT NEURON — Soma + Dendrite + AIS
# ═══════════════════════════════════════════════════════════════════════════════

class MultiCompartmentNeuron:
    """
    Biologically realistic neuron with 3 compartments:

    1. **Soma** (somatic): HH Na⁺/K⁺ channels, spike initiation zone
    2. **Dendrite** (dendritic): NMDA/AMPA receptors, Ca²⁺ spikes, cable filtering
    3. **AIS** (axon initial segment): Low-threshold Na⁺ channels, spike propagation

    Compartments are coupled via axial resistance (gap junctions within cell).

    Architecture:
        Dendrite <──g_c──> Soma <──g_ais──> AIS

    Reference: Mainen & Sejnowski (1996), Stuart & Spruston (1998)
    """

    # Cell type parameters
    CELL_TYPES = {
        "pyramidal": {
            "g_na": 120.0, "g_k": 36.0, "g_l": 0.3, "g_ca": 2.0,
            "g_c": 0.5, "g_ais": 1.0, "cm": 1.0,
            "dend_g_na": 2.0, "dend_g_ca": 4.0,
        },
        "pv_interneuron": {
            "g_na": 200.0, "g_k": 60.0, "g_l": 0.2, "g_ca": 0.5,
            "g_c": 0.8, "g_ais": 2.0, "cm": 0.8,
            "dend_g_na": 0.5, "dend_g_ca": 1.0,
        },
        "som_interneuron": {
            "g_na": 150.0, "g_k": 45.0, "g_l": 0.25, "g_ca": 1.5,
            "g_c": 0.6, "g_ais": 1.5, "cm": 0.9,
            "dend_g_na": 1.0, "dend_g_ca": 3.0,
        },
        "vip_interneuron": {
            "g_na": 180.0, "g_k": 50.0, "g_l": 0.2, "g_ca": 0.8,
            "g_c": 0.7, "g_ais": 1.8, "cm": 0.85,
            "dend_g_na": 0.8, "dend_g_ca": 1.5,
        },
        "stellate": {
            "g_na": 140.0, "g_k": 40.0, "g_l": 0.3, "g_ca": 1.0,
            "g_c": 0.4, "g_ais": 1.2, "cm": 1.0,
            "dend_g_na": 1.5, "dend_g_ca": 2.0,
        },
    }

    def __init__(self, cell_type: str = "pyramidal", neuron_id: int = 0):
        params = self.CELL_TYPES.get(cell_type, self.CELL_TYPES["pyramidal"])
        self.cell_type = cell_type
        self.neuron_id = neuron_id

        # Soma (spike initiation)
        self.soma = HodgkinHuxleySoma(
            g_na=params["g_na"], g_k=params["g_k"],
            g_l=params["g_l"], g_ca=params["g_ca"],
            cm=params["cm"]
        )

        # Dendrite (receptor-rich compartment)
        self.dend_V = -65.0    # Dendritic membrane potential
        self.dend_g_na = params["dend_g_na"]
        self.dend_g_ca = params["dend_g_ca"]
        self.dend_cm = params["cm"]
        self.dend_ca_in = CONC_CA_IN

        # AIS (axon initial segment)
        self.ais_V = -65.0
        self.ais_g_na_high = 2.0  # High density of Na⁺ channels at AIS
        self.ais_cm = params["cm"]
        self.ais_ready = True   # Refractory state for propagation

        # Coupling conductances (mS)
        self.g_c = params["g_c"]       # Soma ↔ Dendrite
        self.g_ais = params["g_ais"]   # Soma ↔ AIS

        # Spike state
        self.spike_time = -1e9     # Time of last spike (ms)
        self._has_spiked = False   # Spike flag for current timestep
        self.refractory = 0.0      # Refractory counter (ms)
        self.refractory_period = 2.0  # Absolute refractory period (ms)

        # STDP trace (pre-synaptic tag for plasticity)
        self.stdp_trace = 0.0      # Spike trace for STDP (pre)
        self.stdp_trace_tau = 20.0  # STDP trace time constant (ms)

        # Incoming synaptic currents (accumulated each step)
        self.I_syn_soma = 0.0
        self.I_syn_dend = 0.0

    def reset(self):
        """Reset to resting state."""
        self.soma.reset()
        self.dend_V = -65.0
        self.ais_V = -65.0
        self.dend_ca_in = CONC_CA_IN
        self.ais_ready = True
        self.spike_time = -1e9
        self._has_spiked = False
        self.refractory = 0.0
        self.stdp_trace = 0.0
        self.I_syn_soma = 0.0
        self.I_syn_dend = 0.0

    def step(self, dt: float = 0.01) -> bool:
        """
        Advance one time step (all compartments).

        Args:
            dt: Time step in ms (default 0.01 = 10 μs).

        Returns:
            True if a spike was emitted (AIS propagated).
        """
        # Reset spike flag for this timestep
        self._has_spiked = False

        # Decrement refractory
        self.refractory = max(0.0, self.refractory - dt)

        # ── 1. Somatic compartment (HH + AIS coupling) ──
        I_soma_total = self.I_syn_soma
        # Coupling current from dendrite
        I_couple_soma = self.g_c * (self.dend_V - self.soma.V)
        I_soma_total += I_couple_soma
        # Coupling from AIS
        I_ais_soma = self.g_ais * (self.ais_V - self.soma.V)
        I_soma_total += I_ais_soma

        # Refractory shunt
        if self.refractory > 0:
            shunt = 1.0 + 10.0 * (self.refractory / self.refractory_period)
            g_l_shunt = self.soma.g_l * shunt
            I_leak_shunt = g_l_shunt * (self.soma.V - E_L)
            I_soma_total -= I_leak_shunt

        soma_spike = self.soma.step(I_soma_total, dt)

        # ── 2. Dendritic compartment (passive + Ca²⁺) ──
        I_dend_total = self.I_syn_dend
        I_couple_dend = self.g_c * (self.soma.V - self.dend_V)
        I_dend_total += I_couple_dend

        # Dendritic Na⁺ (reduced density)
        dend_m_inf = GatingKinetics.m_inf_v(np.array([self.dend_V]))[0]
        I_dend_na = self.dend_g_na * dend_m_inf * (self.dend_V - E_NA)
        # Dendritic Ca²⁺ (T-type, low-voltage activated)
        t_ca_inf = 1.0 / (1.0 + math.exp(-(self.dend_V + 55.0) / 7.0))
        I_dend_ca = self.dend_g_ca * t_ca_inf * (self.dend_V - E_CA)
        # Leak
        I_dend_l = self.soma.g_l * (self.dend_V - E_L)

        dV_dend = (-I_dend_na - I_dend_ca - I_dend_l - I_dend_total) / self.dend_cm
        self.dend_V += dt * dV_dend
        self.dend_V = np.clip(self.dend_V, -100.0, 60.0)

        # Dendritic Ca²⁺ dynamics
        if self.dend_V > -30.0:
            ca_influx = -I_dend_ca * dt * 5.18e-5
            self.dend_ca_in += max(0, ca_influx)
            self.dend_ca_in = min(self.dend_ca_in, 5e-3)
        self.dend_ca_in += dt * (CONC_CA_IN - self.dend_ca_in) / 120.0

        # ── 3. AIS compartment (axon initial segment, spike propagation) ──
        I_ais_total = 0.0
        I_couple_ais = self.g_ais * (self.soma.V - self.ais_V)
        I_ais_total += I_couple_ais

        # AIS has high Na⁺ density (low threshold for spike initiation)
        ais_m = GatingKinetics.m_inf(self.ais_V)
        ais_h = GatingKinetics.h_inf(self.ais_V)
        I_ais_na = self.ais_g_na_high * ais_m * ais_h * (self.ais_V - E_NA)
        I_ais_k = self.soma.g_k * GatingKinetics.n_inf(self.ais_V) * (self.ais_V - E_K)
        I_ais_l = self.soma.g_l * (self.ais_V - E_L)

        dV_ais = (-I_ais_na - I_ais_k - I_ais_l) / self.ais_cm
        self.ais_V += dt * dV_ais
        self.ais_V = np.clip(self.ais_V, -100.0, 60.0)

        # ── 4. Spike detection and propagation ──
        emitted_spike = False
        if soma_spike and self.refractory <= 0:
            self.spike_time = 0.0  # Will be updated by caller's time
            self.refractory = self.refractory_period
            self._has_spiked = True

            # AIS propagation (axon spike)
            self.ais_V = 30.0  # Force spike at AIS
            emitted_spike = True

            # STDP trace (pre-synaptic tag)
            self.stdp_trace = 1.0

        # STDP trace decay
        self.stdp_trace *= math.exp(-dt / self.stdp_trace_tau)

        # Reset accumulated currents
        self.I_syn_soma = 0.0
        self.I_syn_dend = 0.0

        return emitted_spike

    def add_synaptic_current(self, current: float, compartment: str = "soma"):
        """Add synaptic current to the specified compartment."""
        if compartment == "soma":
            self.I_syn_soma += current
        elif compartment == "dend":
            self.I_syn_dend += current

    def has_spiked(self) -> bool:
        """Check if neuron spiked in the current timestep."""
        return self._has_spiked

    def get_stdp_trace(self) -> float:
        """Get the current STDP trace value (for post-synaptic learning)."""
        return self.stdp_trace


# ═══════════════════════════════════════════════════════════════════════════════
#  SYNAPSE — STDP with AMPA/NMDA/GABA Receptors
# ═══════════════════════════════════════════════════════════════════════════════

class STDPSynapse:
    """
    Biologically realistic synapse with multiple receptor types and STDP.

    Receptors:
        • AMPA: Fast excitation (τ ~2-5ms). Drives fast EPSPs.
        • NMDA: Slow excitation (τ ~50-100ms). Voltage-dependent Mg²⁺ block.
                Coincidence detector for synaptic plasticity.
        • GABA_A: Fast inhibition (τ ~5-10ms). Cl⁻ channel (shunting).
        • GABA_B: Slow inhibition (τ ~100-200ms). G-protein coupled.

    STDP (Spike-Timing-Dependent Plasticity):
        Δw = A₊ * exp(-|Δt| / τ₊)    if Δt > 0 (pre▲ → post▲: LTP)
        Δw = -A₋ * exp(-|Δt| / τ₋)   if Δt < 0 (post▲ → pre▲: LTD)

        where Δt = t_post - t_pre.

        Three-factor STDP (dopamine-gated):
        Δw = [A₊ * exp(-|Δt|/τ₊) - A₋ * exp(-|Δt|/τ₋)] * D(t)

        where D(t) is dopamine concentration (0-1). No dopamine → no plasticity.

    Reference: Bi & Poo (1998), Markram et al. (1997), Sjöström et al. (2001)
    """

    RECEPTOR_PARAMS = {
        "ampa": {"g_max": 0.5e-3, "tau_r": 0.5, "tau_d": 3.0, "E_rev": E_AMPA},
        "nmda": {"g_max": 0.3e-3, "tau_r": 2.0, "tau_d": 80.0, "E_rev": E_AMPA},
        "gaba_a": {"g_max": 0.5e-3, "tau_r": 0.5, "tau_d": 6.0, "E_rev": E_GABA},
        "gaba_b": {"g_max": 0.2e-3, "tau_r": 20.0, "tau_d": 150.0, "E_rev": E_GABA},
    }

    # Default STDP parameters
    STDP_A_PLUS = 0.01    # LTP amplitude
    STDP_A_MINUS = 0.012  # LTD amplitude (larger = more LTD)
    STDP_TAU_PLUS = 20.0   # LTP time constant (ms)
    STDP_TAU_MINUS = 20.0  # LTD time constant (ms)
    W_MIN = 0.0
    W_MAX = 3.0

    def __init__(self, pre_neuron: MultiCompartmentNeuron,
                 post_neuron: MultiCompartmentNeuron,
                 receptor_type: str = "ampa",
                 weight: float = 1.0,
                 compartment: str = "dend",
                 delay: float = 1.0):
        """
        Args:
            pre_neuron: Pre-synaptic neuron.
            post_neuron: Post-synaptic neuron.
            receptor_type: 'ampa', 'nmda', 'gaba_a', or 'gaba_b'.
            weight: Initial synaptic weight (multiplicative factor).
            compartment: 'soma' or 'dend' (where synapse connects).
            delay: Conduction delay (ms).
        """
        self.pre = pre_neuron
        self.post = post_neuron
        self.receptor_type = receptor_type
        self.weight = weight
        self.compartment = compartment
        self.delay = delay

        params = self.RECEPTOR_PARAMS[receptor_type]
        self.g_max = params["g_max"]
        self.tau_r = params["tau_r"]
        self.tau_d = params["tau_d"]
        self.E_rev = params["E_rev"]

        # Synaptic gating variables (double exponential)
        self.s = 0.0     # Synaptic activation
        self.r = 0.0     # Rising component

        # STDP eligibility trace
        self.pre_trace = 0.0
        self.post_trace = 0.0
        self.tau_pre = 20.0
        self.tau_post = 20.0

        # History for STDP
        self._pre_spike_times = deque(maxlen=50)
        self._post_spike_times = deque(maxlen=50)

        # Last weight change (for monitoring)
        self.last_dw = 0.0

    def reset(self):
        self.s = 0.0
        self.r = 0.0
        self.pre_trace = 0.0
        self.post_trace = 0.0
        self._pre_spike_times.clear()
        self._post_spike_times.clear()
        self.last_dw = 0.0

    def _mg_block(self, V_post: float) -> float:
        """
        Voltage-dependent Mg²⁺ block for NMDA receptors.

        B(V) = 1 / (1 + exp(-0.062 * V / 3.57) * [Mg²⁺]_o / 3.57)

        At resting (-65mV): ~95% blocked
        At depolarized (0mV): ~50% blocked
        At +30mV: ~10% blocked (fully open)
        """
        if self.receptor_type != "nmda":
            return 1.0
        mg_block = 1.0 / (1.0 + math.exp(-0.062 * V_post / 3.57) * CONC_MG_OUT / 3.57)
        return mg_block

    def trigger_pre_spike(self):
        """Called when pre-synaptic neuron fires."""
        self._pre_spike_times.append(0.0)  # Relative time
        # Activate synaptic gating
        self.r += 1.0

    def trigger_post_spike(self):
        """Called when post-synaptic neuron fires."""
        self._post_spike_times.append(0.0)

    def compute_current(self, V_post: float) -> float:
        """
        Compute post-synaptic current.

        I_syn = g_max * w * s * B(V) * (V - E_rev)

        For GABA: I_syn = g_max * w * s * (V - E_GABA)
        (GABA is NOT voltage-dependent except for driving force)
        """
        # Mg²⁺ block (NMDA only)
        block = self._mg_block(V_post)

        # Synaptic current
        if self.receptor_type in ("gaba_a", "gaba_b"):
            I = self.g_max * self.weight * self.s * (V_post - self.E_rev)
        else:
            # AMPA and NMDA: excitatory, plus NMDA has Mg block
            I = self.g_max * self.weight * self.s * block * (V_post - self.E_rev)

        return I

    def step(self, dt: float, dopamine: float = 0.5):
        """
        Advance synapse state and apply STDP.

        Args:
            dt: Time step (ms).
            dopamine: Dopamine concentration (0-1). Modulates plasticity.
        """
        # ── Double exponential synaptic kinetics ──
        # dr/dt = -r / tau_r  (rise)
        # ds/dt = (r - s) / tau_d  (decay)
        dr = -self.r / self.tau_r
        ds = (self.r - self.s) / self.tau_d

        self.r += dt * dr
        self.s += dt * ds
        self.r = max(0.0, self.r)
        self.s = max(0.0, self.s)

        # ── STDP: detect spike pairs and update weight ──
        # Check for pre-before-post pairs (LTP)
        pre_times_to_remove = []
        for pre_t in list(self._pre_spike_times):
            pre_t_adj = pre_t + dt  # Advance time
            # Check if any post spike occurred after pre
            for post_t in list(self._post_spike_times):
                delta_t = post_t - pre_t_adj  # t_post - t_pre
                if delta_t > 0 and delta_t < 50.0:  # 50ms window
                    # LTP: pre before post
                    dw = self.STDP_A_PLUS * math.exp(-delta_t / self.STDP_TAU_PLUS)
                    dw *= dopamine  # Three-factor: dopamine gates plasticity
                    self.weight += dw
                    self.last_dw = dw
                    pre_times_to_remove.append(pre_t)
                    break

        for pt in pre_times_to_remove:
            if pt in self._pre_spike_times:
                self._pre_spike_times.remove(pt)

        # Check for post-before-pre pairs (LTD)
        post_times_to_remove = []
        for post_t in list(self._post_spike_times):
            post_t_adj = post_t + dt
            for pre_t in list(self._pre_spike_times):
                delta_t = post_t_adj - pre_t  # t_post - t_pre
                if delta_t < 0 and abs(delta_t) < 50.0:
                    # LTD: post before pre (negative delta_t)
                    dw = -self.STDP_A_MINUS * math.exp(-abs(delta_t) / self.STDP_TAU_MINUS)
                    dw *= dopamine
                    self.weight += dw
                    self.last_dw = dw
                    post_times_to_remove.append(post_t)
                    break

        for pt in post_times_to_remove:
            if pt in self._post_spike_times:
                self._post_spike_times.remove(pt)

        # Clamp weight
        self.weight = np.clip(self.weight, self.W_MIN, self.W_MAX)

        # Update spike time offsets for all remaining events
        updated_pre = deque(maxlen=50)
        for t in self._pre_spike_times:
            updated_pre.append(t + dt)
        self._pre_spike_times = updated_pre

        updated_post = deque(maxlen=50)
        for t in self._post_spike_times:
            updated_post.append(t + dt)
        self._post_spike_times = updated_post

    def energy_cost(self) -> float:
        """Return ATP cost for this synapse's activation."""
        return EnergyBudget.synapse_cost(self.receptor_type)


# ═══════════════════════════════════════════════════════════════════════════════
#  CELL TYPES — Cortical Neuron Classification
# ═══════════════════════════════════════════════════════════════════════════════

class CellType(IntEnum):
    """Cortical neuron types (based on transcriptomic and morphological data)."""
    PYRAMIDAL = 0    # Excitatory (glutamatergic) — principal cells
    PV = 1           # Parvalbumin+ fast-spiking interneuron (perisomatic)
    SOM = 2          # Somatostatin+ interneuron (dendritic targeting)
    VIP = 3          # VIP+ interneuron (disinhibition, targets SOM/PV)
    STELLATE = 4     # Excitatory spiny stellate (layer 4, thalamic input)

    @classmethod
    def is_excitatory(cls, ct: 'CellType') -> bool:
        return ct in (cls.PYRAMIDAL, cls.STELLATE)

    @classmethod
    def is_inhibitory(cls, ct: 'CellType') -> bool:
        return ct in (cls.PV, cls.SOM, cls.VIP)

    @classmethod
    def type_name(cls, ct: 'CellType') -> str:
        """Get human-readable name for a cell type."""
        names = {
            cls.PYRAMIDAL: "Pyramidal",
            cls.PV: "PV+ Basket",
            cls.SOM: "SOM+ Martinotti",
            cls.VIP: "VIP+ Bipolar",
            cls.STELLATE: "Spiny Stellate",
        }
        return names.get(ct, "Unknown")


class Layer(IntEnum):
    """Cortical layers (1-6)."""
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5
    L6 = 6


# ═══════════════════════════════════════════════════════════════════════════════
#  NEURON GROUP — A Population of Neurons
# ═══════════════════════════════════════════════════════════════════════════════

class NeuronGroup:
    """
    A homogeneous population of neurons with the same cell type and layer.

    Manages the synchronous stepping of all neurons in the group.
    """

    def __init__(self, n_neurons: int, cell_type: CellType, layer: Layer,
                 name: str = ""):
        self.cell_type = cell_type
        self.layer = layer
        self.name = name or f"{CellType.type_name(cell_type)}_L{layer}"
        self.n_neurons = n_neurons

        # Create neurons
        ct_str = cell_type.name.lower()
        neuron_type = {
            CellType.PYRAMIDAL: "pyramidal",
            CellType.PV: "pv_interneuron",
            CellType.SOM: "som_interneuron",
            CellType.VIP: "vip_interneuron",
            CellType.STELLATE: "stellate",
        }[cell_type]

        self.neurons = [
            MultiCompartmentNeuron(cell_type=neuron_type, neuron_id=i)
            for i in range(n_neurons)
        ]

        # Spike times for visualization
        self.spike_times = [[] for _ in range(n_neurons)]
        self.current_time = 0.0

    def step(self, dt: float = 0.01):
        """Step all neurons."""
        self.current_time += dt
        for i, neuron in enumerate(self.neurons):
            spiked = neuron.step(dt)
            if spiked:
                self.spike_times[i].append(self.current_time)

    def reset(self):
        """Reset all neurons."""
        self.current_time = 0.0
        self.spike_times = [[] for _ in range(self.n_neurons)]
        for neuron in self.neurons:
            neuron.reset()

    def get_firing_rates(self, window_ms: float = 100.0) -> np.ndarray:
        """Get firing rates for each neuron over the last window_ms."""
        rates = np.zeros(self.n_neurons)
        t_cutoff = self.current_time - window_ms
        for i, times in enumerate(self.spike_times):
            recent = [t for t in times if t > t_cutoff]
            rates[i] = len(recent) / (window_ms / 1000.0)
        return rates

    def mean_firing_rate(self, window_ms: float = 100.0) -> float:
        """Mean firing rate across all neurons."""
        return float(self.get_firing_rates(window_ms).mean())


# ═══════════════════════════════════════════════════════════════════════════════
#  SYNAPSE MATRIX — Connections Between Groups
# ═══════════════════════════════════════════════════════════════════════════════

class SynapseMatrix:
    """
    A sparse connection matrix between two NeuronGroups.

    Supports:
        • Random connectivity with specified probability
        • Distance-dependent connectivity
        • Cell-type-specific connection rules
        • STDP learning across all synapses
    """

    def __init__(self, pre_group: NeuronGroup, post_group: NeuronGroup,
                 connection_prob: float = 0.1,
                 receptor_type: str = "ampa",
                 weight_mean: float = 1.0, weight_std: float = 0.3,
                 delay_mean: float = 1.0, delay_std: float = 0.3,
                ):
        self.pre = pre_group
        self.post = post_group
        self.n_pre = pre_group.n_neurons
        self.n_post = post_group.n_neurons

        # Determine synapse compartment
        # Distal dendrite for excitatory → excitatory (NMDA-rich)
        # Soma for inhibitory → excitatory (PV targets perisomatic)
        if receptor_type in ("gaba_a",):
            compartment = "soma"
        elif receptor_type in ("gaba_b",):
            compartment = "dend"
        else:
            compartment = "dend"

        # Create synapses
        self.synapses = []  # List of (pre_idx, post_idx, synapse)
        rng = np.random.RandomState(42 + hash((pre_group.name, post_group.name)) % 10000)

        for i in range(self.n_pre):
            for j in range(self.n_post):
                if rng.random() < connection_prob:
                    weight = max(0.1, rng.normal(weight_mean, weight_std))
                    delay = max(0.5, rng.normal(delay_mean, delay_std))
                    syn = STDPSynapse(
                        pre_group.neurons[i],
                        post_group.neurons[j],
                        receptor_type=receptor_type,
                        weight=weight,
                        compartment=compartment,
                        delay=delay,
                    )
                    self.synapses.append((i, j, syn))

        self.n_synapses = len(self.synapses)
        self.receptor_type = receptor_type

    def step(self, dt: float, dopamine: float = 0.5):
        """Step all synapses: compute currents and apply STDP."""
        for pre_idx, post_idx, syn in self.synapses:
            pre_neuron = self.pre.neurons[pre_idx]
            post_neuron = self.post.neurons[post_idx]

            # Detect pre-synaptic spike
            if pre_neuron.has_spiked():
                syn.trigger_pre_spike()

            # Detect post-synaptic spike
            if post_neuron.has_spiked():
                syn.trigger_post_spike()

            # Step synapse (STDP + gating)
            syn.step(dt, dopamine=dopamine)

            # Compute current and add to post-synaptic neuron
            V_post = post_neuron.soma.V
            I = syn.compute_current(V_post)
            post_neuron.add_synaptic_current(I, syn.compartment)

    def get_weights(self) -> np.ndarray:
        """Get weight matrix (n_pre × n_post)."""
        W = np.zeros((self.n_pre, self.n_post), dtype=np.float32)
        for pre_idx, post_idx, syn in self.synapses:
            W[pre_idx, post_idx] = syn.weight
        return W

    def mean_weight(self) -> float:
        """Mean synaptic weight."""
        if not self.synapses:
            return 0.0
        weights = [syn.weight for _, _, syn in self.synapses]
        return sum(weights) / len(weights)

    def stats(self) -> dict:
        """Summary statistics."""
        weights = [syn.weight for _, _, syn in self.synapses]
        return {
            "n_synapses": self.n_synapses,
            "mean_weight": float(np.mean(weights)),
            "std_weight": float(np.std(weights)),
            "min_weight": float(np.min(weights)),
            "max_weight": float(np.max(weights)),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  CORTICAL COLUMN — 6-Layer Microcircuit
# ═══════════════════════════════════════════════════════════════════════════════

class CorticalColumn:
    """
    Biologically realistic 6-layer cortical column.

    Layer architecture (from pia to white matter):
        L1:  Mostly dendrites, sparse GABAergic (VIP, SOM) — ~5% of neurons
        L2:  Small pyramidal (cortico-cortical) + PV/SOM interneurons — ~15%
        L3:  Medium pyramidal (callosal) + PV/SOM — ~15%
        L4:  Spiny stellate (thalamocortical input) + PV/SOM — ~20%
        L5:  Large pyramidal (subcortical output) + PV/SOM — ~25%
        L6:  Corticothalamic pyramidal + PV/SOM — ~20%

    Cell type ratios (inhibitory):
        PV:  ~40% of interneurons (fast-spiking, perisomatic)
        SOM: ~30% of interneurons (dendritic-targeting)
        VIP: ~30% of interneurons (disinhibition)

    Local connectivity rules (canonical microcircuit):
        • L4 stellate → L2/3 pyramidal (feedforward)
        • L2/3 pyramidal → L5 pyramidal (feedforward)
        • L5 pyramidal → L6 pyramidal (feedforward)
        • L6 pyramidal → L4 stellate (feedback)
        • L5/L6 → thalamus (subcortical output)
        • PV targets perisomatic of pyramidal (strong inhibition)
        • SOM targets dendrites of pyramidal (dendritic inhibition)
        • VIP targets SOM (disinhibition)

    Reference: Douglas & Martin (2004), Markram et al. (2015, Blue Brain),
               Thomson & Bannister (2003)
    """

    # Layer composition: (n_excitatory, n_inhibitory) for a ~1mm² column
    LAYER_COMPOSITION = {
        Layer.L1: {"n_pyramidal": 0, "n_pv": 10, "n_som": 15, "n_vip": 5,
                   "n_stellate": 0, "name": "L1 (Molecular)"},
        Layer.L2: {"n_pyramidal": 80, "n_pv": 15, "n_som": 10, "n_vip": 8,
                   "n_stellate": 0, "name": "L2 (External Granular)"},
        Layer.L3: {"n_pyramidal": 100, "n_pv": 20, "n_som": 15, "n_vip": 10,
                   "n_stellate": 0, "name": "L3 (External Pyramidal)"},
        Layer.L4: {"n_pyramidal": 20, "n_pv": 25, "n_som": 15, "n_vip": 10,
                   "n_stellate": 80, "name": "L4 (Internal Granular)"},
        Layer.L5: {"n_pyramidal": 120, "n_pv": 30, "n_som": 20, "n_vip": 15,
                   "n_stellate": 0, "name": "L5 (Internal Pyramidal)"},
        Layer.L6: {"n_pyramidal": 60, "n_pv": 20, "n_som": 15, "n_vip": 10,
                   "n_stellate": 0, "name": "L6 (Multiform)"},
    }

    # Canonical connection probabilities
    # Format: (pre_layer, pre_type, post_layer, post_type) -> prob
    CANONICAL_CONNECTIONS = [
        # L4 stellate → L2/3 pyramidal (feedforward excitatory)
        (Layer.L4, CellType.STELLATE, Layer.L2, CellType.PYRAMIDAL, 0.15),
        (Layer.L4, CellType.STELLATE, Layer.L3, CellType.PYRAMIDAL, 0.12),
        # L2/3 pyramidal → L5 pyramidal (feedforward)
        (Layer.L2, CellType.PYRAMIDAL, Layer.L5, CellType.PYRAMIDAL, 0.10),
        (Layer.L3, CellType.PYRAMIDAL, Layer.L5, CellType.PYRAMIDAL, 0.12),
        # L5 pyramidal → L6 (feedforward)
        (Layer.L5, CellType.PYRAMIDAL, Layer.L6, CellType.PYRAMIDAL, 0.08),
        # L6 → L4 (feedback)
        (Layer.L6, CellType.PYRAMIDAL, Layer.L4, CellType.STELLATE, 0.06),
        # L2/3 → L2/3 (recurrent)
        (Layer.L2, CellType.PYRAMIDAL, Layer.L2, CellType.PYRAMIDAL, 0.08),
        (Layer.L3, CellType.PYRAMIDAL, Layer.L3, CellType.PYRAMIDAL, 0.10),
        # L5 → L5 (recurrent)
        (Layer.L5, CellType.PYRAMIDAL, Layer.L5, CellType.PYRAMIDAL, 0.08),
        # PV → Pyramidal (perisomatic inhibition) — high prob
        (Layer.L2, CellType.PV, Layer.L2, CellType.PYRAMIDAL, 0.30),
        (Layer.L3, CellType.PV, Layer.L3, CellType.PYRAMIDAL, 0.35),
        (Layer.L4, CellType.PV, Layer.L4, CellType.PYRAMIDAL, 0.30),
        (Layer.L5, CellType.PV, Layer.L5, CellType.PYRAMIDAL, 0.25),
        (Layer.L6, CellType.PV, Layer.L6, CellType.PYRAMIDAL, 0.25),
        # PV → PV (lateral inhibition)
        (Layer.L2, CellType.PV, Layer.L2, CellType.PV, 0.15),
        (Layer.L3, CellType.PV, Layer.L3, CellType.PV, 0.15),
        # SOM → Pyramidal (dendritic inhibition)
        (Layer.L2, CellType.SOM, Layer.L2, CellType.PYRAMIDAL, 0.20),
        (Layer.L3, CellType.SOM, Layer.L3, CellType.PYRAMIDAL, 0.20),
        (Layer.L5, CellType.SOM, Layer.L5, CellType.PYRAMIDAL, 0.15),
        # VIP → SOM (disinhibition)
        (Layer.L2, CellType.VIP, Layer.L2, CellType.SOM, 0.25),
        (Layer.L3, CellType.VIP, Layer.L3, CellType.SOM, 0.25),
    ]

    def __init__(self, name: str = "A1", scale: float = 1.0):
        """
        Args:
            name: Column name/identifier.
            scale: Scaling factor for neuron counts (1.0 = ~500 neurons).
        """
        self.name = name
        self.scale = max(0.1, scale)

        # ── Create neuron groups for each layer ──
        self.groups = {}  # (Layer, CellType) -> NeuronGroup
        self.all_neurons = []

        for layer, comp in self.LAYER_COMPOSITION.items():
            for ct_name, n_cells in comp.items():
                if ct_name == "name" or n_cells == 0:
                    continue
                n = max(1, int(n_cells * scale))
                cell_type = {
                    "n_pyramidal": CellType.PYRAMIDAL,
                    "n_pv": CellType.PV,
                    "n_som": CellType.SOM,
                    "n_vip": CellType.VIP,
                    "n_stellate": CellType.STELLATE,
                }[ct_name]

                group = NeuronGroup(n, cell_type, layer,
                                    name=f"{name}_{CellType.type_name(cell_type)}_L{layer}")
                self.groups[(layer, cell_type)] = group
                self.all_neurons.extend(group.neurons)

        self.total_neurons = len(self.all_neurons)

        # ── Create synapse matrices ──
        self.synapse_matrices = []
        for pre_layer, pre_type, post_layer, post_type, prob in self.CANONICAL_CONNECTIONS:
            pre_group = self.groups.get((pre_layer, pre_type))
            post_group = self.groups.get((post_layer, post_type))
            if pre_group is None or post_group is None:
                continue

            # Determine receptor type based on pre-synaptic cell type
            if CellType.is_excitatory(pre_type):
                # Excitatory → excitatory: AMPA + NMDA
                # Excitatory → inhibitory: AMPA
                receptor = "ampa"
                # Add NMDA for E→E connections
                if CellType.is_excitatory(post_type):
                    receptor_extra = "nmda"
                    sm_extra = SynapseMatrix(
                        pre_group, post_group,
                        connection_prob=prob * 0.8,  # NMDA has slightly broader
                        receptor_type=receptor_extra,
                        weight_mean=0.5, weight_std=0.2,
                    )
                    self.synapse_matrices.append(sm_extra)
            else:
                # Inhibitory: GABA_A or GABA_B
                if pre_type == CellType.SOM:
                    receptor = "gaba_b"  # SOM targets dendrites (GABA_B)
                else:
                    receptor = "gaba_a"  # PV targets soma (GABA_A)

            sm = SynapseMatrix(
                pre_group, post_group,
                connection_prob=prob,
                receptor_type=receptor,
                weight_mean=1.0, weight_std=0.3,
            )
            self.synapse_matrices.append(sm)

        # ── Intra-layer excitatory connections ──
        # Pyramidal → Pyramidal within same layer (recurrent excitation)
        for layer in [Layer.L2, Layer.L3, Layer.L4, Layer.L5, Layer.L6]:
            pyr_group = self.groups.get((layer, CellType.PYRAMIDAL))
            if pyr_group and pyr_group.n_neurons > 1:
                sm = SynapseMatrix(
                    pyr_group, pyr_group,
                    connection_prob=0.05,
                    receptor_type="ampa",
                    weight_mean=0.3, weight_std=0.15,
                )
                self.synapse_matrices.append(sm)

        # ── Local inhibition within layers ──
        # PV→Pyramidal (already in CANONICAL). Also PV→PV, SOM→PV, etc.
        for layer in [Layer.L2, Layer.L3, Layer.L4, Layer.L5, Layer.L6]:
            pv_group = self.groups.get((layer, CellType.PV))
            som_group = self.groups.get((layer, CellType.SOM))
            pyr_group = self.groups.get((layer, CellType.PYRAMIDAL))

            # PV → PV (lateral inhibition among fast-spiking)
            if pv_group and pv_group.n_neurons > 1:
                sm = SynapseMatrix(pv_group, pv_group, 0.10,
                                   receptor_type="gaba_a", weight_mean=0.8)
                self.synapse_matrices.append(sm)

            # SOM → PV
            if som_group and pv_group:
                sm = SynapseMatrix(som_group, pv_group, 0.12,
                                   receptor_type="gaba_b", weight_mean=0.6)
                self.synapse_matrices.append(sm)

            # VIP → SOM (disinhibition)
            vip_group = self.groups.get((layer, CellType.VIP))
            if vip_group and som_group:
                sm = SynapseMatrix(vip_group, som_group, 0.20,
                                   receptor_type="gaba_a", weight_mean=0.7)
                self.synapse_matrices.append(sm)

        self.n_synapses = sum(sm.n_synapses for sm in self.synapse_matrices)

    def step(self, dt: float = 0.01, dopamine: float = 0.5):
        """Step all neurons and synapses in the column."""
        # Step synapses (accumulates currents)
        for sm in self.synapse_matrices:
            sm.step(dt, dopamine=dopamine)

        # Step all neurons (uses accumulated currents)
        for group in self.groups.values():
            group.step(dt)

        # ATP accounting
        n_spikes = sum(1 for n in self.all_neurons if n.has_spiked())
        for _ in range(n_spikes):
            EnergyBudget.spike_cost()

    def reset(self):
        """Reset all neurons and synapses."""
        for group in self.groups.values():
            group.reset()
        for sm in self.synapse_matrices:
            for _, _, syn in sm.synapses:
                syn.reset()

    def get_spike_data(self) -> Tuple[List[int], List[float]]:
        """
        Get spike raster data for visualization.

        Returns:
            (neuron_indices, times) where neuron_indices are global indices.
        """
        indices = []
        times = []
        for i, neuron in enumerate(self.all_neurons):
            if neuron.has_spiked():
                indices.append(i)
                times.append(neuron.spike_time)
        return indices, times

    def summary(self) -> str:
        """Text summary of column structure."""
        lines = [
            f"Cortical Column: {self.name}",
            f"{'='*50}",
            f"  Neurons: {self.total_neurons}",
            f"  Synapses: {self.n_synapses}",
            f"",
            f"  Layer composition:",
        ]
        for layer in [Layer.L1, Layer.L2, Layer.L3, Layer.L4, Layer.L5, Layer.L6]:
            comp = self.LAYER_COMPOSITION[layer]
            grp = lambda ct: self.groups.get((layer, ct))
            pyr = grp(CellType.PYRAMIDAL).n_neurons if grp(CellType.PYRAMIDAL) else 0
            pv = grp(CellType.PV).n_neurons if grp(CellType.PV) else 0
            som = grp(CellType.SOM).n_neurons if grp(CellType.SOM) else 0
            vip = grp(CellType.VIP).n_neurons if grp(CellType.VIP) else 0
            st = grp(CellType.STELLATE).n_neurons if grp(CellType.STELLATE) else 0
            lines.append(
                f"    {comp['name']:30s}: Pyr={pyr:3d} PV={pv:2d} "
                f"SOM={som:2d} VIP={vip:2d} St={st:2d}"
            )
        lines.append("")
        lines.append(f"  Synapse matrices: {len(self.synapse_matrices)}")
        for sm in self.synapse_matrices:
            lines.append(
                f"    {sm.pre.name:20s} → {sm.post.name:20s}: "
                f"{sm.n_synapses:4d} synapses, {sm.receptor_type:6s}, "
                f"w={sm.mean_weight():.2f}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  NEUROMODULATION — Dopamine, Acetylcholine, Serotonin, Norepinephrine
# ═══════════════════════════════════════════════════════════════════════════════

class Neuromodulator:
    """
    A global neuromodulatory signal (diffuse projection system).

    Each neuromodulator has:
        • Baseline concentration (tonic)
        • Phasic release (event-triggered)
        • Exponential decay back to baseline
        • Effect on neural excitability, plasticity, and network state

    Systems modeled:
        • Dopamine (VTA/SNc): Reward prediction error, motor gating
        • Acetylcholine (NBM): Attention, learning rate, cortical activation
        • Serotonin (Raphe): Mood, time perception, behavioral inhibition
        • Norepinephrine (LC): Arousal, gain modulation, alertness
    """

    def __init__(self, name: str, baseline: float = 0.3,
                 tau_rise: float = 10.0, tau_decay: float = 200.0,
                 phasic_amplitude: float = 0.5):
        self.name = name
        self.baseline = baseline
        self.tau_rise = tau_rise       # Time constant for rise (ms)
        self.tau_decay = tau_decay     # Time constant for decay (ms)
        self.phasic_amplitude = phasic_amplitude
        self.concentration = baseline
        self._target = baseline

    def phasic_burst(self, amplitude: float = None):
        """Trigger a phasic release event."""
        amp = amplitude if amplitude is not None else self.phasic_amplitude
        self._target = min(1.0, self.concentration + amp)

    def step(self, dt: float):
        """Update concentration toward target, then decay to baseline."""
        # Rise toward target (if target > concentration)
        if self._target > self.concentration:
            self.concentration += dt * (self._target - self.concentration) / self.tau_rise
        # Decay toward baseline — target also decays to baseline
        self._target += dt * (self.baseline - self._target) / self.tau_decay
        if self.concentration > self.baseline:
            self.concentration -= dt * (self.concentration - self.baseline) / self.tau_decay
        else:
            self.concentration += dt * (self.baseline - self.concentration) / (self.tau_decay * 2)

        self.concentration = np.clip(self.concentration, 0.0, 1.0)

    def get(self) -> float:
        return self.concentration


class NeuromodulationSystem:
    """
    Integrated neuromodulatory control.

    Manages the four major neuromodulators and their effects on the network.
    """

    def __init__(self):
        self.dopamine = Neuromodulator("Dopamine", baseline=0.3,
                                        tau_rise=10.0, tau_decay=300.0,
                                        phasic_amplitude=0.6)
        self.acetylcholine = Neuromodulator("Acetylcholine", baseline=0.4,
                                             tau_rise=50.0, tau_decay=500.0,
                                             phasic_amplitude=0.4)
        self.serotonin = Neuromodulator("Serotonin", baseline=0.5,
                                         tau_rise=100.0, tau_decay=2000.0,
                                         phasic_amplitude=0.3)
        self.norepinephrine = Neuromodulator("Norepinephrine", baseline=0.2,
                                              tau_rise=5.0, tau_decay=100.0,
                                              phasic_amplitude=0.7)

    def step(self, dt: float, reward: float = 0.0, novelty: float = 0.0,
             arousal: float = 0.0):
        """
        Step all neuromodulators.

        Args:
            dt: Time step (ms).
            reward: Reward signal (0-1). Triggers dopamine phasic.
            novelty: Novelty/surprise signal. Triggers ACh + NE.
            arousal: Global arousal level. Modulates NE baseline.
        """
        # Dopamine: reward prediction error
        if reward > 0.3:
            self.dopamine.phasic_burst(reward * 0.6)
        self.dopamine.step(dt)

        # Acetylcholine: novelty and attention
        if novelty > 0.2:
            self.acetylcholine.phasic_burst(novelty * 0.4)
        self.acetylcholine.step(dt)

        # Serotonin: slow baseline drift
        self.serotonin.step(dt)

        # Norepinephrine: arousal
        if arousal > 0.3:
            self.norepinephrine.phasic_burst(arousal * 0.7)
        self.norepinephrine.baseline = 0.2 + arousal * 0.3
        self.norepinephrine.step(dt)

    def get_state(self) -> dict:
        """Get current neuromodulator concentrations."""
        return {
            "dopamine": self.dopamine.get(),
            "acetylcholine": self.acetylcholine.get(),
            "serotonin": self.serotonin.get(),
            "norepinephrine": self.norepinephrine.get(),
        }

    def get_dopamine(self) -> float:
        return self.dopamine.get()

    def report(self) -> str:
        state = self.get_state()
        lines = [
            f"Neuromodulatory State",
            f"{'='*50}",
        ]
        for name, val in state.items():
            level = "⬆ High" if val > 0.6 else "↗ Moderate" if val > 0.3 else "→ Low"
            lines.append(f"  {name:20s}: {val:.2f}  {level}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  THALAMOCORTICAL RELAY
# ═══════════════════════════════════════════════════════════════════════════════

class ThalamocorticalRelay:
    """
    Simplified thalamocortical relay nucleus.

    The thalamus has two main cell types:
        1. Thalamocortical (TC) relay cells: Send sensory info to cortex
        2. Thalamic reticular nucleus (TRN): Inhibitory, gates thalamic output

    Functions:
        • Sensory gating (attention filter)
        • Cortical synchronization (spindle oscillations)
        • Information routing (via TRN inhibition)
    """

    def __init__(self, n_relay: int = 20, n_reticular: int = 10):
        self.tc_group = NeuronGroup(n_relay, CellType.PYRAMIDAL, Layer.L4,
                                     name="Thalamocortical")
        self.trn_group = NeuronGroup(n_reticular, CellType.PV, Layer.L1,
                                      name="TRN")

        # Local connections
        self.tc_trn = SynapseMatrix(self.tc_group, self.trn_group, 0.15,
                                     receptor_type="ampa", weight_mean=1.0)
        self.trn_tc = SynapseMatrix(self.trn_group, self.tc_group, 0.20,
                                     receptor_type="gaba_a", weight_mean=1.5)

        self.neurons = self.tc_group.neurons + self.trn_group.neurons

    def step(self, dt: float = 0.01, input_current: float = 0.0):
        """Step thalamus. input_current drives TC cells."""
        # Add external input to TC cells
        if input_current > 0:
            for neuron in self.tc_group.neurons:
                neuron.add_synaptic_current(input_current, "soma")

        self.tc_trn.step(dt, dopamine=0.5)
        self.trn_tc.step(dt, dopamine=0.5)
        self.tc_group.step(dt)
        self.trn_group.step(dt)

    def get_output_spikes(self) -> List[int]:
        """Get TC cells that spiked (output to cortex)."""
        return [i for i, n in enumerate(self.tc_group.neurons) if n.has_spiked()]


# ═══════════════════════════════════════════════════════════════════════════════
#  HIPPOCAMPAL FORMATION
# ═══════════════════════════════════════════════════════════════════════════════

class HippocampalFormation:
    """
    Simplified hippocampal formation for memory consolidation.

    Regions:
        • Dentate Gyrus (DG): Pattern separation, sparse coding
        • CA3: Auto-associative memory (recurrent collaterals)
        • CA1: Comparison (input from CA3 + entorhinal cortex)
        • Subiculum: Output to cortex

    Memory operations:
        • Encoding: Novel patterns → DG → CA3 (via mossy fibers)
        • Recall: Partial cue → CA3 → pattern completion → CA1
        • Consolidation: Sharp-wave ripples during sleep
    """

    def __init__(self, n_dg: int = 30, n_ca3: int = 40, n_ca1: int = 30,
                 n_sub: int = 20):
        self.dg = NeuronGroup(n_dg, CellType.STELLATE, Layer.L4, name="DG")
        self.ca3 = NeuronGroup(n_ca3, CellType.PYRAMIDAL, Layer.L5, name="CA3")
        self.ca1 = NeuronGroup(n_ca1, CellType.PYRAMIDAL, Layer.L5, name="CA1")
        self.sub = NeuronGroup(n_sub, CellType.PYRAMIDAL, Layer.L6, name="Sub")

        # DG → CA3 (mossy fibers, strong sparse connections)
        self.mossy = SynapseMatrix(self.dg, self.ca3, 0.05,
                                    receptor_type="ampa", weight_mean=3.0)
        # CA3 → CA3 (recurrent collaterals, auto-association)
        self.ca3_rec = SynapseMatrix(self.ca3, self.ca3, 0.08,
                                      receptor_type="ampa", weight_mean=0.5)
        # CA3 → CA1 (Schaffer collaterals)
        self.schaffer = SynapseMatrix(self.ca3, self.ca1, 0.10,
                                       receptor_type="ampa", weight_mean=1.0)
        # CA1 → Subiculum
        self.ca1_sub = SynapseMatrix(self.ca1, self.sub, 0.10,
                                      receptor_type="ampa", weight_mean=1.0)

        self.neurons = (self.dg.neurons + self.ca3.neurons +
                        self.ca1.neurons + self.sub.neurons)

        # Memory buffers for replay
        self.memory_buffer = deque(maxlen=50)
        self.replay_active = False

    def encode_pattern(self, pattern: np.ndarray, dt: float = 0.01):
        """
        Encode a pattern into the hippocampal circuit.
        Pattern drives DG → CA3 via mossy fibers (pattern separation).
        """
        for i, neuron in enumerate(self.dg.neurons):
            if i < len(pattern) and pattern[i] > 0:
                neuron.add_synaptic_current(pattern[i] * 0.5, "soma")

        self.step(dt)
        self.memory_buffer.append((
            [n.has_spiked() for n in self.dg.neurons],
            [n.has_spiked() for n in self.ca3.neurons],
        ))

    def replay(self, n_cycles: int = 5, dt: float = 0.01):
        """
        Memory replay (sharp-wave ripples during sleep).
        Randomly reactivates stored CA3 patterns.
        """
        if not self.memory_buffer:
            return

        self.replay_active = True
        for _ in range(n_cycles):
            # Pick a random memory
            dg_spikes, ca3_spikes = self.memory_buffer[
                np.random.randint(len(self.memory_buffer))]

            # Drive CA3 from the stored pattern
            for i, (neuron, spiked) in enumerate(zip(self.ca3.neurons, ca3_spikes)):
                if spiked and i < len(ca3_spikes):
                    neuron.add_synaptic_current(0.3, "soma")

            self.step(dt)
        self.replay_active = False

    def step(self, dt: float = 0.01):
        """Step all hippocampal synapses and neurons."""
        self.mossy.step(dt, dopamine=0.5)
        self.ca3_rec.step(dt, dopamine=0.5)
        self.schaffer.step(dt, dopamine=0.5)
        self.ca1_sub.step(dt, dopamine=0.5)

        self.dg.step(dt)
        self.ca3.step(dt)
        self.ca1.step(dt)
        self.sub.step(dt)

    def reset(self):
        """Reset all hippocampal neurons."""
        self.dg.reset()
        self.ca3.reset()
        self.ca1.reset()
        self.sub.reset()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_step(column: CorticalColumn, modulator: NeuromodulationSystem,
                  dt: float = 0.01, input_currents: Optional[dict] = None,
                  reward: float = 0.0, novelty: float = 0.0):
    """
    Single simulation step for the entire brain system.

    Args:
        column: CorticalColumn to simulate.
        modulator: NeuromodulationSystem for the step.
        dt: Time step (ms).
        input_currents: Dict of (layer, cell_type) -> current array.
        reward: Reward signal for dopamine (0-1).
        novelty: Novelty signal for ACh/NE (0-1).
    """
    # Apply external input currents
    if input_currents:
        for (layer, cell_type), current in input_currents.items():
            group = column.groups.get((layer, cell_type))
            if group:
                for i, neuron in enumerate(group.neurons):
                    if i < len(current):
                        neuron.add_synaptic_current(current[i], "soma")

    # Step column (synapses first, then neurons)
    dopamine = modulator.get_dopamine()
    column.step(dt, dopamine=dopamine)

    # Step neuromodulators
    modulator.step(dt, reward=reward, novelty=novelty)

    # ATP accounting for resting maintenance
    total_neurons = column.total_neurons
    EnergyBudget.rest_cost(total_neurons, dt)


def run_simulation(column: CorticalColumn, modulator: NeuromodulationSystem,
                   duration_ms: float = 1000.0, dt: float = 0.01,
                   input_schedule: Optional[list] = None,
                   record_spikes: bool = True,
                   verbose: bool = True) -> dict:
    """
    Run a full simulation with pacing and input scheduling.

    Args:
        column: CorticalColumn to simulate.
        modulator: NeuromodulationSystem.
        duration_ms: Total simulation time (ms).
        dt: Time step (ms).
        input_schedule: List of (start_ms, end_ms, input_dict) tuples.
        record_spikes: Whether to record spike times for raster.
        verbose: Whether to print progress.

    Returns:
        dict with simulation results.
    """
    EnergyBudget.reset()

    n_steps = int(duration_ms / dt)
    spike_raster = []  # (neuron_idx, time_ms) for each spike

    t0 = time.time()

    for step_idx in range(n_steps):
        t_ms = step_idx * dt

        # Check input schedule
        input_currents = None
        reward = 0.0
        novelty = 0.0
        if input_schedule:
            for start, end, inp in input_schedule:
                if start <= t_ms < end:
                    if "currents" in inp:
                        input_currents = inp["currents"]
                    reward = inp.get("reward", 0.0)
                    novelty = inp.get("novelty", 0.0)
                    break

        # Simulation step
        simulate_step(column, modulator, dt, input_currents, reward, novelty)

        # Record spikes
        if record_spikes:
            for i, neuron in enumerate(column.all_neurons):
                if neuron.has_spiked():
                    spike_raster.append((i, t_ms))

    elapsed = time.time() - t0
    total_atp = EnergyBudget.total()

    result = {
        "duration_ms": duration_ms,
        "dt": dt,
        "n_steps": n_steps,
        "wall_time_s": elapsed,
        "speedup": (duration_ms / 1000.0) / max(elapsed, 1e-6),
        "spike_raster": spike_raster,
        "n_neurons": column.total_neurons,
        "n_synapses": column.n_synapses,
        "total_spikes": len(spike_raster),
        "mean_firing_rate_hz": len(spike_raster) / column.total_neurons / (duration_ms / 1000.0),
        "energy": {
            "total_atp": total_atp,
            "watts": EnergyBudget.watts(elapsed),
            "atp_per_spike": total_atp / max(len(spike_raster), 1),
        },
        "neuromod_state": modulator.get_state(),
    }

    if verbose:
        print(f"Simulation complete: {duration_ms:.0f}ms simulated in "
              f"{elapsed:.1f}s ({result['speedup']:.1f}× realtime)")
        print(f"  Neurons: {column.total_neurons}, Synapses: {column.n_synapses}")
        print(f"  Total spikes: {len(spike_raster)}")
        print(f"  Mean firing rate: {result['mean_firing_rate_hz']:.1f} Hz")
        print(f"  Energy: {total_atp:.2e} ATP ({result['energy']['watts']:.2e} W)")
        print(f"  ATP per spike: {result['energy']['atp_per_spike']:.2e}")

    return result
