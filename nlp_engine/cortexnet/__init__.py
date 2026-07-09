"""
CortexNet patch package — Phase 2 / Omega Edition.

Phase-1 components (from the original arc_proposal):
    - DendriticNeuron  (V6 dual-gate, fast/slow EMA, NMDA write-gate)
    - MANARAttention   (linear-time memory + local window, replaces O(n²) MHA)
                       — now with streaming KV cache + vectorized local attn
    - CfCLayer         (closed-form continuous-time RNN)
    - FTPTrainer       (Forward Target Propagation, no backprop)
    - ComplementPruner (C1q/C3-style developmental pruning)
    - SurrogateSpike   (trainable SNN layer with surrogate gradient)
    - CorticalColumn   (L4 / L2-3 / L5 laminar stack)
                       — now with 4-channel neuromodulation + streaming
    - CortexNetLite    (top-level Phase-1 model, kept for backward compat)

Phase-2 additions (this patch):
    - UniversalTokenizer (byte-level, role/modality/language tags)
    - HippocampalReplay  (sleep consolidation loop)
    - ThinkingLoop       (deliberative pre-pass, internalised CoT)
    - CortexNetOmega     (top-level Phase-2 model with generate / think_then_speak / sleep)

Everything is pure NumPy — same stack as the rest of the repo.
"""
from .dendritic_neuron import DendriticNeuron
from .manar_attention import MANARAttention, MANARCache
from .cfc_layer import CfCLayer
from .ftp_trainer import FTPTrainer
from .complement_pruner import ComplementPruner
from .surrogate_spike import SurrogateSpikeLayer
from .cortical_column import CorticalColumn, NeuromodulatoryState
from .cortexnet_lite import CortexNetLite

# Phase-2 additions
from .universal_tokenizer import (
    UniversalTokenizer, PromptBlock, VOCAB_SIZE,
    TOK_PAD, TOK_BOS, TOK_EOS, TOK_ROLE_SYSTEM, TOK_ROLE_USER,
    TOK_ROLE_ASSISTANT, TOK_ROLE_THINKING, TOK_MOD_TEXT, TOK_MOD_CODE,
    TOK_MOD_MATH, TOK_MOD_INSTRUCTION, TOK_MOD_KNOWLEDGE, TOK_MOD_QUERY,
)
from .hippocampal_replay import HippocampalReplay, Episode
from .thinking_loop import PrefrontalCortex, ThinkingLoop
from .neuromodulators import NeuromodulatorState
from .cortexnet_omega import CortexNetOmega

__all__ = [
    # Phase-1
    "DendriticNeuron", "MANARAttention", "MANARCache", "CfCLayer",
    "FTPTrainer", "ComplementPruner", "SurrogateSpikeLayer",
    "CorticalColumn", "NeuromodulatoryState", "CortexNetLite",
    # Phase-2
    "UniversalTokenizer", "PromptBlock", "VOCAB_SIZE",
    "TOK_PAD", "TOK_BOS", "TOK_EOS", "TOK_ROLE_SYSTEM", "TOK_ROLE_USER",
    "TOK_ROLE_ASSISTANT", "TOK_ROLE_THINKING", "TOK_MOD_TEXT", "TOK_MOD_CODE",
    "TOK_MOD_MATH", "TOK_MOD_INSTRUCTION", "TOK_MOD_KNOWLEDGE", "TOK_MOD_QUERY",
    "HippocampalReplay", "Episode",
    "deliberative_think", "ThinkingTrace",
    "NeuromodulatorState",
    "CortexNetOmega",
]
