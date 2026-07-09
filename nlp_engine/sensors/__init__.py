"""
nlp_engine.sensors — Hyperion's sensory subsystem.

Phase-4 (legacy):
    - ChemotaxisEngine      (active foraging / run-and-tumble)
    - PeripheralSniffer     (low-energy broad sensor)
    - FovealReader          (high-acuity targeted sensor)

Phase-5 (NEW — Universal Multimodal Embodiment):
    - OmniSensoryCortex     (the eyes of Hyperion: pixels, text, symbols,
                             code in any language, raw binary — all flattened
                             into the universal 2048-bit SDR)
    - SensoryPercept        (a unified "moment of seeing")
    - Modality              (PIXEL / TEXT / SYMBOL / CODE / BINARY / UNKNOWN)
"""
from .chemotaxis import ChemotaxisEngine, PeripheralSniffer, FovealReader
from .omni_sensory_cortex import (
    OmniSensoryCortex,
    SensoryPercept,
    Modality,
    PixelCortex,
    TextCortex,
    SymbolCortex,
    CodeCortex,
    BinaryCortex,
    ModalitySniffer,
    SDR_DIM,
    SDR_ACTIVE_BITS,
)

__all__ = [
    "ChemotaxisEngine", "PeripheralSniffer", "FovealReader",
    "OmniSensoryCortex", "SensoryPercept", "Modality",
    "PixelCortex", "TextCortex", "SymbolCortex", "CodeCortex", "BinaryCortex",
    "ModalitySniffer", "SDR_DIM", "SDR_ACTIVE_BITS",
]
