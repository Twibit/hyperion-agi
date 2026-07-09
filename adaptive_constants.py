"""
Hyperion Adaptive Constants
===========================
A single source of truth for every tunable parameter in the AGI.
No magic numbers anywhere else. Every module imports from here.

Values auto-adapt based on detected hardware (RAM, CPU cores, disk speed).
"""

import os
import multiprocessing

# ── Hardware Detection ──────────────────────────────────────────────────
def _detect_ram_mb():
    """Detect available RAM in MB. Works on Linux/Android/Termux."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except Exception:
        pass
    return 4096  # Conservative fallback: 4GB

def _detect_cpu_cores():
    """Detect available CPU cores."""
    try:
        return multiprocessing.cpu_count() or 2
    except Exception:
        return 2

DETECTED_RAM_MB = _detect_ram_mb()
DETECTED_CPU_CORES = _detect_cpu_cores()

# ── Rust Vision Core ────────────────────────────────────────────────────
# Buffer size for native file I/O. Scales with available RAM.
# Formula: min(64MB, RAM / 128). On 8GB device -> 64MB. On 2GB -> 16MB.
VISION_BUFFER_BYTES = min(64 * 1024 * 1024, int((DETECTED_RAM_MB * 1024 * 1024) / 128))

# How often the Rust scanner prints a progress update (in lines).
# Scales inversely with verbosity need. Smaller RAM = more frequent checks.
VISION_TELEMETRY_INTERVAL = 10_000_000 if DETECTED_RAM_MB > 4096 else 5_000_000

# ── Calcium / Synaptic Biology ──────────────────────────────────────────
CALCIUM_DECAY_RATE = 0.95        # Astrocyte calcium decay per tick
CALCIUM_INFLUX_SCALE = 0.1      # Pre*Post spike contribution
CALCIUM_MAX = 1.0               # Biological ceiling

# STDP (Spike-Timing-Dependent Plasticity)
STDP_CALCIUM_MIDPOINT = 0.2     # Sigmoid center for calcium gate
STDP_CALCIUM_STEEPNESS = -8.0   # Sigmoid steepness

# ── Predictive Coding (Cortex) ──────────────────────────────────────────
PREDICTIVE_ERR_DECAY = 0.99     # Error variance EMA decay
PREDICTIVE_ERR_LR = 0.01        # Error variance learning rate
PREDICTIVE_STATE_MOMENTUM = 0.9 # State update momentum
PREDICTIVE_STATE_LR = 0.1       # State update learning rate

# ── Efficiency Gate ─────────────────────────────────────────────────────
EFFICIENCY_EASY_THRESHOLD = 0.05
EFFICIENCY_HARD_THRESHOLD = 0.3
EFFICIENCY_EMA_ALPHA = 0.01
EFFICIENCY_EMA_INIT = 0.15      # Starting mid-range EMA

# ── Stigmergic Graph ───────────────────────────────────────────────────
STIGMERGIC_EDGE_PRUNE_THRESHOLD = 0.01  # Edges below this are pruned

# ── PFIT Highway ────────────────────────────────────────────────────────
PFIT_INITIAL_MYELINATION = 0.1  # Starting myelination level

# ── MoE Cortex ──────────────────────────────────────────────────────────
MOE_LOAD_BALANCE_WEIGHT = 0.1   # Weight of load-balance penalty in FE

# ── Neurogenesis ────────────────────────────────────────────────────────
NEUROGENESIS_NOISE_SCALE = 0.01 # Random noise added to new neurons

# ── SNN (Spiking Neural Network) ───────────────────────────────────────
SNN_DEFAULT_THRESHOLD = 1.0     # Default firing threshold

# ── Substrate ───────────────────────────────────────────────────────────
# k_active scales with available RAM. More RAM = more concurrent neurons.
# Formula: min(2048, RAM_MB / 16). On 8GB -> 512. On 32GB -> 2048.
# In testing mode (e.g. Gridworld simulation), we override this to 32 to run 20x faster.
if os.environ.get("HYPERION_TESTING") == "1":
    SUBSTRATE_K_ACTIVE = 32
else:
    SUBSTRATE_K_ACTIVE = min(2048, max(128, int(DETECTED_RAM_MB / 16)))

# Adenosine RAM limit (how much RAM the substrate is allowed to consume).
# Formula: 90% of detected RAM, leaving headroom for OS.
SUBSTRATE_RAM_LIMIT_MB = int(DETECTED_RAM_MB * 0.9)

# ── Omni Optical Nerve (Eye) ───────────────────────────────────────────
# EMA alpha for adaptive streaming signal detection
EYE_STREAM_EMA_ALPHA = 0.01

# Progress logging interval for Python fallback reader
EYE_PROGRESS_INTERVAL = 500_000

# ── Wiktionary Ingestion ───────────────────────────────────────────────
# REM sleep consolidation interval (every N concepts)
INGESTION_SLEEP_INTERVAL = 500

# Default drive keywords (can be overridden at runtime)
INGESTION_DEFAULT_DRIVE = ["animal", "biology", "creature", "life", "organism"]

# ── Adaptive Brain Tier ────────────────────────────────────────────────
# Auto-selects the largest brain tier that fits comfortably in RAM.
# Each tier's approximate RAM footprint (empirical):
#   tiny:  ~200 MB    base:  ~1.5 GB    large: ~6 GB
#   brain/cerebrum+: ~32 GB (requires disk eviction / aether mode)
def _detect_tier():
    if DETECTED_RAM_MB >= 24000:
        return "large"
    elif DETECTED_RAM_MB >= 4000:
        return "base"
    else:
        return "tiny"

ADAPTIVE_TIER = _detect_tier()

# The user wants "high tier rate but adaptive"
# The default safe rate is 0.05.
# We scale based on latent dimension width.
# tiny(128)->0.05, base(512)->0.01, large(2048)->0.002
def _detect_lr_cortex():
    if ADAPTIVE_TIER == "large":
        return 0.002
    elif ADAPTIVE_TIER == "base":
        return 0.01
    else:
        return 0.05
# Keep alias for backward compatibility
ADAPTIVE_LR_CORTEX = _detect_lr_cortex()

# ── File Path Detection ────────────────────────────────────────────────
# Auto-detect wiktionary dump location
def detect_wiktionary_path():
    """Search common locations for the wiktionary dump."""
    candidates = [
        "/sdcard/Download/enwiktionary-latest-pages-meta-current.xml",
        os.path.expanduser("~/Downloads/enwiktionary-latest-pages-meta-current.xml"),
        "/data/enwiktionary-latest-pages-meta-current.xml",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]  # Return first candidate as default

WIKTIONARY_PATH = detect_wiktionary_path()

# ── Interactive UI Kind Mapping ─────────────────────────────────────────
# Which UI element kinds are treated as interactive (not hardcoded in eye)
INTERACTIVE_UI_KINDS = {"Button", "Link", "Input", "Checkbox"}

# ── HTML Tags recognized by the Web Saccade Parser ──────────────────────
SEMANTIC_HTML_TAGS = {'p', 'h1', 'h2', 'h3', 'span', 'div', 'article'}
INTERACTIVE_HTML_TAGS = {'a', 'button'}

# ── Code file extensions recognized by the Eye ──────────────────────────
CODE_EXTENSIONS = {'.py', '.js', '.ts', '.rs', '.go', '.java', '.cpp', '.c', '.rb'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov'}

# ── Print detected hardware on import ───────────────────────────────────
print(f"[Adaptive Constants] Detected {DETECTED_RAM_MB:.0f} MB RAM, {DETECTED_CPU_CORES} CPU cores.")
print(f"[Adaptive Constants] Substrate k_active={SUBSTRATE_K_ACTIVE}, Vision buffer={VISION_BUFFER_BYTES // (1024*1024)}MB, RAM limit={SUBSTRATE_RAM_LIMIT_MB}MB.")
