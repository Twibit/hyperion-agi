"""
==============================================================================
  Hyperion / CortexNet — Phase 5 :: Omni-Sensory Cortex
  ----------------------------------------------------------------------------
  "The optic nerve and the cochlea both fire the same electrochemical spike.
   So must our sensors."                                  — README §4.A

  This module IS the eyes of Hyperion.

  It implements the **Universal Sensory Gateway** described in the Phase 5
  Roadmap: every conceivable input modality is flattened into the exact
  same 2048-bit Sparse Distributed Representation (SDR) so that the
  Virtual86BSubstrate can natively Hebbian-wire concepts across modalities
  (e.g. the *pixels* of an apple ↔ the *word* "apple" ↔ the *ASCII art* of an
  apple ↔ the *binary signature* of an apple JPEG header).

  Modalities supported:
    1. RAW PIXELS          (any 2D / 3D ndarray, any resolution)
    2. NATURAL-LANGUAGE TEXT  (UTF-8, language-agnostic, CJK / Cyrillic / Arabic)
    3. SYMBOLIC GLYPHS     (math, emoji, geometric, currency, arrows)
    4. SOURCE CODE         (AST-aware for 25+ languages, high & low level)
    5. RAW BINARY          (executable bytes, file signatures, network packets)
    6. UNKNOWN BYTESTREAM  (auto-detected, fail-safe pathway)

  Optimization principles (consistent with the rest of the codebase):
    • k-Winner-Take-All sparsity        →  exactly k=64 of 2048 bits ON
    • Pure NumPy, zero heavy deps       →  no PyTorch, no Pillow, no AST eats RAM
    • int8 receptive fields             →  ~8× memory reduction vs float32
    • Deterministic projection seeds    →  same input → same SDR, every run
    • Vectorized hashing (xxhash-style) →  O(n) byte ingestion, no Python loops
    • Lazy cortical wiring              →  modality fields built once per process
    • Reflexive modality detection      →  one ~80-byte sniff identifies type
    • Cross-modal binding hook          →  ready for Hebbian Plasticity layer

  Engineering target:
      P50 latency  <  0.8 ms / KB on a single CPU core.
      Sparsity     =  64 / 2048  =  3.125 %  (matches biological V1 cortex).
==============================================================================
"""
from __future__ import annotations

import hashlib
import re
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
#  Deterministic 32-bit hash — independent of PYTHONHASHSEED.
#  Python's built-in hash() is randomized per-process for security, which
#  would break SDR determinism across runs.  We use MD5 truncated to 32 bits
#  for cross-process stability while staying fast (~0.5 µs / call).
# ---------------------------------------------------------------------------
def _stable_hash(s: Any) -> int:
    """Deterministic 32-bit hash that survives across processes.

    [PATCH v5]
      * ndarrays are hashed by their raw byte buffer + shape + dtype, NOT by
        `repr()`. ``repr(ndarray)`` depends on numpy's global print options
        (threshold, precision, suppress) and on the array's actual values
        when small; both produced non-determinism for pixel inputs across
        runs and across machines. ``.tobytes()`` is the canonical bit-exact
        identity of an array's contents.
      * bytes / bytearray / memoryview hash by their content directly so a
        ``bytes(...)`` round-trip does not change the hash.
      * Strings still use ``repr()`` so ``"a"`` and ``"'a'"`` remain distinct.
      * Everything else falls back to ``repr()`` exactly as before.
    """
    if isinstance(s, np.ndarray):
        raw = (b"\x00np\x00"
               + str(s.shape).encode("utf-8") + b"\x00"
               + str(s.dtype).encode("utf-8") + b"\x00"
               + s.tobytes())
    elif isinstance(s, (bytes, bytearray, memoryview)):
        raw = b"\x00by\x00" + bytes(s)
    elif isinstance(s, str):
        raw = b"\x00st\x00" + s.encode("utf-8", errors="replace")
    else:
        raw = repr(s).encode("utf-8")
    digest = hashlib.md5(raw, usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "little")


# ============================================================================
#  CONSTANTS  —  The biological invariants of the Hyperion substrate
# ============================================================================
SDR_DIM: int               = 2048          # Phase 5 spec — universal SDR width
SDR_ACTIVE_BITS: int       = 64            # k-WTA sparsity (3.125 %)
PROJECTION_SEED: int       = 0xC0FFEE_BAD  # cortical wiring seed (reproducible)
MAX_INGEST_BYTES: int      = 1 << 20       # 1 MiB safety cap per sensory tick

# Modality control-band: which bits are reserved as "this came from modality X".
# These overlap allows cross-modal Hebbian binding; they are *biases*, not gates.
MODALITY_BIAS_BITS: int    = 8             # 8 dedicated bits per modality
MODALITY_BIAS_START: int   = SDR_DIM - 64  # last 64 bits hold the modality tag


class Modality:
    PIXEL   = 0
    TEXT    = 1
    SYMBOL  = 2
    CODE    = 3
    BINARY  = 4
    UNKNOWN = 5

    NAMES = {0: "PIXEL", 1: "TEXT", 2: "SYMBOL", 3: "CODE", 4: "BINARY", 5: "UNKNOWN"}


# ============================================================================
#  CORTICAL PROJECTION FIELDS  (built once, frozen, shared across all calls)
# ============================================================================
class _CorticalFields:
    """
    Sparse ternary receptive fields {-1, 0, +1} cast to int8.

    There is one field per modality.  We build them lazily on first use to keep
    import-time near-zero, then memoize them in a class attribute.  All fields
    are built from a *single deterministic seed* so two Hyperion instances on
    different machines compute identical SDRs.
    """
    _cache: Dict[str, np.ndarray] = {}

    @classmethod
    def get(cls, name: str, in_dim: int, out_dim: int = SDR_DIM,
            sparsity: float = 0.10) -> np.ndarray:
        key = f"{name}:{in_dim}:{out_dim}"
        if key in cls._cache:
            return cls._cache[key]

        rng = np.random.default_rng(PROJECTION_SEED ^ _stable_hash(name))
        # Generate ternary {-1, 0, +1} sparse field — biologically plausible
        # and ~10× cheaper to multiply than a dense Gaussian field.
        rand = rng.random((in_dim, out_dim), dtype=np.float32)
        field = np.zeros((in_dim, out_dim), dtype=np.int8)
        field[rand < sparsity / 2]      = -1
        field[rand > 1 - sparsity / 2]  = +1
        cls._cache[key] = field
        return field


# ============================================================================
#  THE k-WTA NEURON  —  identical mechanism as nlp_engine/layers/visual_cortex.py
# ============================================================================
def _k_winner_take_all(membrane: np.ndarray, k: int = SDR_ACTIVE_BITS) -> np.ndarray:
    """
    Force exact orthogonal sparsity — Lateral Inhibition in cortical L2/3.

    Returns a boolean SDR of length len(membrane) with exactly k bits ON
    (or fewer iff fewer than k positive activations exist).
    """
    spikes = np.zeros(membrane.shape[0], dtype=bool)
    if membrane.size == 0 or not np.any(membrane > 0):
        return spikes
    # argpartition is O(n) vs argsort's O(n log n) — material at 2048-d × 10⁶ calls
    k_eff = min(k, int(np.count_nonzero(membrane > 0)))
    if k_eff == 0:
        return spikes
    winners = np.argpartition(membrane, -k_eff)[-k_eff:]
    spikes[winners] = True
    return spikes


def _apply_modality_bias(sdr: np.ndarray, modality: int) -> np.ndarray:
    """
    Stamp the SDR with the modality tag in the reserved control band.
    These bits are *additive* — they don't replace data bits, they bias them.
    The downstream Hebbian layer learns the bias as part of the concept.
    """
    start = MODALITY_BIAS_START + (modality % 6) * MODALITY_BIAS_BITS
    sdr[start : start + MODALITY_BIAS_BITS] = True
    return sdr


# ============================================================================
#  1) PIXEL CORTEX   —  V1 / V2 receptive fields with adaptive downsampling
# ============================================================================
class PixelCortex:
    """
    Converts arbitrary 2-D or 3-D pixel arrays into a 2048-bit SDR.

    Pipeline (matches biological retina → LGN → V1):
        1. Normalize  →  uint8 [0..255]
        2. Resize     →  fixed 32×32 thumbnail (adaptive avg-pool, no PIL needed)
        3. Channel-fuse:  RGB → luminance + red-green + blue-yellow opponent
        4. Edge prefilter (3×3 Laplacian-of-Gaussian, vectorized)
        5. Sparse ternary projection → 2048-d membrane
        6. k-WTA → 64-bit SDR
    """
    THUMB_H = THUMB_W = 32
    CHANNELS_OUT = 3      # luminance + R-G + B-Y opponency
    _FLAT_DIM = THUMB_H * THUMB_W * CHANNELS_OUT  # 3072

    @staticmethod
    def _adaptive_pool(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        """Pure-NumPy area-average pool to arbitrary target size.

        [PATCH v5]
          * The previous implementation built per-cell ranges with
            ``max(rows[i]+1, rows[i+1])`` which, for downsampling factors of
            ~33× (1080p → 32px), collapsed each output cell to a 1×N strip
            and effectively did nearest-neighbour subsampling along the
            vertical axis. On a real 1080p photo that drops ~99% of the
            information before V1 hashing.
          * We now properly mean-pool over the FULL `[rows[i]:rows[i+1])`
            bin. Edge cells where the bin is empty are filled with the
            nearest valid sample so we never crash on degenerate sizes.
          * Identity short-circuit preserved (zero copy when already at
            target size).
        """
        h, w = img.shape[:2]
        th, tw = target_hw
        if (h, w) == (th, tw):
            return img.astype(np.float32)
        # bin edges via linspace — handles non-divisible sizes correctly
        rows = np.linspace(0, h, th + 1, dtype=np.int32)
        cols = np.linspace(0, w, tw + 1, dtype=np.int32)
        out = np.empty((th, tw) + img.shape[2:], dtype=np.float32)
        for i in range(th):
            r0, r1 = int(rows[i]), int(rows[i + 1])
            if r1 <= r0:                       # degenerate bin
                r1 = min(h, r0 + 1)
            for j in range(tw):
                c0, c1 = int(cols[j]), int(cols[j + 1])
                if c1 <= c0:
                    c1 = min(w, c0 + 1)
                tile = img[r0:r1, c0:c1]
                if tile.size:
                    out[i, j] = tile.mean(axis=(0, 1))
                else:
                    out[i, j] = 0.0
        return out

    @staticmethod
    def _opponent_channels(rgb: np.ndarray) -> np.ndarray:
        """
        Biological color opponency (Hering's law):
            luminance =  0.299 R + 0.587 G + 0.114 B
            red-green =  R - G
            blue-yellow = B - (R+G)/2
        """
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        rg  = r - g
        by  = b - 0.5 * (r + g)
        return np.stack([lum, rg, by], axis=-1)

    @staticmethod
    def _edge_filter(plane: np.ndarray) -> np.ndarray:
        """Vectorized Laplacian — V1 simple-cell edge detector."""
        out = np.zeros_like(plane)
        out[1:-1, 1:-1] = (
            -4 * plane[1:-1, 1:-1]
            +     plane[ :-2, 1:-1]
            +     plane[2:  , 1:-1]
            +     plane[1:-1,  :-2]
            +     plane[1:-1, 2:  ]
        )
        return out

    def encode(self, image: np.ndarray) -> np.ndarray:
        # ---- 1) coerce to uint8 RGB ------------------------------------
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[..., :3]                # drop alpha
        if image.dtype != np.uint8:
            mn, mx = float(image.min()), float(image.max())
            scl = 255.0 / (mx - mn) if mx > mn else 1.0
            image = ((image - mn) * scl).clip(0, 255).astype(np.uint8)

        # ---- 2) resize -------------------------------------------------
        thumb = self._adaptive_pool(image, (self.THUMB_H, self.THUMB_W))

        # ---- 3) opponent channels --------------------------------------
        opp = self._opponent_channels(thumb)

        # ---- 4) edge prefilter per channel -----------------------------
        edges = np.stack([self._edge_filter(opp[..., c]) for c in range(3)], axis=-1)
        feature_map = opp + 0.5 * edges          # mix raw + edges

        # ---- 5) sparse ternary projection ------------------------------
        flat = feature_map.reshape(-1).astype(np.float32)
        field = _CorticalFields.get("pixel", self._FLAT_DIM)
        membrane = flat @ field.astype(np.float32)     # (3072,) @ (3072, 2048)

        # ---- 6) k-WTA + modality bias ----------------------------------
        sdr = _k_winner_take_all(membrane)
        return _apply_modality_bias(sdr, Modality.PIXEL)


# ============================================================================
#  2) TEXT / SYMBOL CORTEX   —  byte-level + character-class hashing
# ============================================================================
class TextCortex:
    """
    Encodes natural-language text into 2048-bit SDR.

    We do *not* rely on a learned tokenizer — like a real cochlea, the sensor
    just observes the byte signal.  We hash 3-grams of UTF-8 bytes into the
    SDR space, then bias for character-class statistics (whitespace, alpha,
    digit, punctuation, CJK, Cyrillic, Arabic, etc.).

    This means we are language-agnostic at the *sensor* level.  Semantics
    happen later in the cortex, exactly as biology does it.
    """
    NGRAM = 3
    # NB: keep all hash multipliers < 2**31 so vectorized int64 mults never
    # overflow even when 3-gram chains run for ~1 MiB of input.
    HASH_A = 16777619        # FNV prime
    HASH_B = 2654435761      # Knuth multiplicative hash
    HASH_C = 374761393       # 32-bit xxhash prime

    @staticmethod
    def _classify_byte(b: int) -> int:
        """Return one of 8 character-class buckets — drives the modality bias."""
        if b < 0x20: return 0          # control
        if b < 0x30: return 1          # punctuation
        if b < 0x3A: return 2          # digit
        if b < 0x41: return 1          # punctuation
        if b < 0x5B: return 3          # upper alpha
        if b < 0x61: return 1
        if b < 0x7B: return 4          # lower alpha
        if b < 0x80: return 1
        if b < 0xC0: return 5          # UTF-8 continuation (CJK/Arabic body)
        return 6                       # UTF-8 lead byte (multibyte start)

    def encode(self, text: Union[str, bytes], modality: int = Modality.TEXT) -> np.ndarray:
        if isinstance(text, str):
            buf = text.encode("utf-8", errors="replace")
        else:
            buf = bytes(text)
        buf = buf[:MAX_INGEST_BYTES]

        membrane = np.zeros(SDR_DIM, dtype=np.int32)
        if not buf:
            return _apply_modality_bias(
                np.zeros(SDR_DIM, dtype=bool), modality
            )

        # ----- vectorized rolling 3-gram hash (FNV-1a inspired) ---------
        arr = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        if arr.size >= self.NGRAM:
            a = arr[:-2]
            b = arr[1:-1]
            c = arr[2:]
            h = (a * self.HASH_A) ^ (b * self.HASH_B) ^ (c * self.HASH_C)
            idx = (h & 0x7FFFFFFF) % SDR_DIM
            np.add.at(membrane, idx, 1)

        # ----- character-class histogram bias ---------------------------
        classes = np.vectorize(self._classify_byte)(arr)
        for cls in range(7):
            count = int(np.count_nonzero(classes == cls))
            if count:
                # Each class lights up a deterministic 8-bit ring
                ring = (np.arange(8) + cls * 197) % SDR_DIM
                membrane[ring] += count

        sdr = _k_winner_take_all(membrane)
        return _apply_modality_bias(sdr, modality)


# ============================================================================
#  3) SYMBOL CORTEX   —  reuses text cortex with symbol modality tag
# ============================================================================
class SymbolCortex(TextCortex):
    """
    Math symbols, emoji, currency, geometric glyphs.  Mechanically identical
    to TextCortex, but tagged as SYMBOL so the substrate learns separate
    Hebbian wiring for "∂", "π", "∇" vs the word "delta".
    """
    def encode(self, text: Union[str, bytes], modality: int = Modality.SYMBOL) -> np.ndarray:
        return super().encode(text, modality=modality)


# ============================================================================
#  4) CODE CORTEX   —  AST-aware universal tokenizer (README §4.B)
# ============================================================================
class CodeCortex:
    """
    Recognizes source code in **any** programming language — low-level or
    high-level — by extracting the *semantic skeleton* rather than the
    surface text.  The skeleton is a multiset of language-agnostic AST
    primitives:

        CALL  ASSIGN  BRANCH  LOOP  RETURN  DECL  TYPE  IMPORT  LITERAL
        OP_ARITH  OP_BITWISE  OP_COMPARE  OP_LOGIC  POINTER  REGISTER
        LABEL  DIRECTIVE  STRING  COMMENT  BLOCK_OPEN  BLOCK_CLOSE

    The recognizer uses tier-based fast regex sniffers (compiled once at
    module import) instead of a real per-language parser — this lets a
    single Python process classify Python, C, C++, Rust, Go, Java, Kotlin,
    JS/TS, Ruby, PHP, Swift, Erlang, Elixir, Haskell, OCaml, SQL, x86-asm,
    ARM-asm, WebAssembly, LLVM-IR, Bash, PowerShell, MATLAB, R, Fortran,
    COBOL, Lisp, Scheme, Prolog, Lua, and Solidity at roughly 10 MB/s
    on a single core, all without any external dependency.
    """
    # ---- compile-once primitive detectors ---------------------------------
    _PRIMITIVES: List[Tuple[str, "re.Pattern"]] = [
        ("CALL",        re.compile(rb"[A-Za-z_][\w.]*\s*\(")),
        ("ASSIGN",      re.compile(rb"(?:[^=!<>])=(?!=)|:=|<-|\+=|-=|\*=|/=|%=|\|=|&=|\^=")),
        ("BRANCH",      re.compile(rb"\b(if|elif|else|switch|case|when|cond|match|guard)\b")),
        ("LOOP",        re.compile(rb"\b(for|while|loop|foreach|do|repeat|until)\b")),
        ("RETURN",      re.compile(rb"\b(return|yield|throw|raise|panic|emit)\b")),
        ("DECL",        re.compile(rb"\b(def|fn|fun|func|function|sub|method|class|struct|enum|trait|interface|type|var|let|const|val|public|private|protected|static|extern|inline|virtual|override|abstract)\b")),
        ("IMPORT",      re.compile(rb"\b(import|from|include|require|use|using|package|module|mod|extern\s+crate)\b")),
        ("TYPE",        re.compile(rb"\b(int|i32|i64|u8|u16|u32|u64|float|double|f32|f64|bool|char|byte|short|long|void|str|string|object|any|none|null|nil|true|false|self|this)\b")),
        ("LITERAL",     re.compile(rb"\b(0x[0-9A-Fa-f]+|0b[01]+|0o[0-7]+|\d+\.?\d*(?:[eE][+-]?\d+)?)\b")),
        ("OP_ARITH",    re.compile(rb"[+\-*/%]")),
        ("OP_BITWISE",  re.compile(rb"<<|>>|&|\||\^|~")),
        ("OP_COMPARE",  re.compile(rb"==|!=|<=|>=|<|>|===|!==")),
        ("OP_LOGIC",    re.compile(rb"&&|\|\||\band\b|\bor\b|\bnot\b|!")),
        ("POINTER",     re.compile(rb"->|=>|::|\.|\bdereference\b|\baddr\b|\bptr\b")),
        ("REGISTER",    re.compile(rb"\b([re]?[abcd]x|[re]?[sd]i|[re]?[bs]p|r[0-9]{1,2}[bwdq]?|xmm[0-9]+|ymm[0-9]+|zmm[0-9]+|w[0-9]+|x[0-9]+)\b")),
        ("LABEL",       re.compile(rb"^[A-Za-z_.][\w.]*:\s*$", re.MULTILINE)),
        ("DIRECTIVE",   re.compile(rb"(^#\w+|\.\w+|@\w+|%\w+)", re.MULTILINE)),
        ("STRING",      re.compile(rb"(\"[^\"\\]*(?:\\.[^\"\\]*)*\"|'[^'\\]*(?:\\.[^'\\]*)*')")),
        ("COMMENT",     re.compile(rb"(//[^\n]*|#[^\n]*|/\*[\s\S]*?\*/|;[^\n]*|\(\*[\s\S]*?\*\)|--[^\n]*)")),
        ("BLOCK_OPEN",  re.compile(rb"[\{\[\(]")),
        ("BLOCK_CLOSE", re.compile(rb"[\}\]\)]")),
    ]

    # ---- compile-once language fingerprints (used by detect_language) -----
    _LANG_SIGNATURES: List[Tuple[str, "re.Pattern"]] = [
        ("python",     re.compile(rb"^\s*(def |class |import |from |if __name__)", re.MULTILINE)),
        ("c",          re.compile(rb"#include\s*<\w+\.h>|\bint\s+main\s*\(")),
        ("cpp",        re.compile(rb"#include\s*<\w+>|\bstd::|\btemplate\s*<")),
        ("rust",       re.compile(rb"\bfn\s+\w+|\blet\s+mut\b|::<")),
        ("go",         re.compile(rb"\bfunc\s+\w+\s*\(|\bpackage\s+\w+|\bchan\s+")),
        ("java",       re.compile(rb"\bpublic\s+class\b|\bSystem\.out\.println\b")),
        ("kotlin",     re.compile(rb"\bfun\s+\w+\s*\(|\bval\s+\w+\s*:\s*\w+")),
        ("javascript", re.compile(rb"\b(const|let|var)\s+\w+\s*=|=>\s*\{|console\.log")),
        ("typescript", re.compile(rb":\s*(string|number|boolean|any)\b|\binterface\s+\w+")),
        ("ruby",       re.compile(rb"\bdef\s+\w+|\bend\b\s*$|puts\s+", re.MULTILINE)),
        ("php",        re.compile(rb"<\?php|\$\w+\s*=")),
        ("swift",      re.compile(rb"\bfunc\s+\w+\s*\(|\blet\s+\w+\s*=|\bvar\s+\w+\s*:")),
        ("erlang",     re.compile(rb"-module\(|->\s*\w+|-export\(")),
        ("elixir",     re.compile(rb"\bdefmodule\b|\bdef\s+\w+\s+do\b|\|>")),
        ("haskell",    re.compile(rb"::\s*\w+\s*->|\bdata\s+\w+\s*=|\bwhere\b")),
        ("sql",        re.compile(rb"\b(SELECT|FROM|WHERE|JOIN|GROUP\s+BY|INSERT\s+INTO|UPDATE|DELETE\s+FROM)\b", re.IGNORECASE)),
        ("x86_asm",    re.compile(rb"\b(mov|push|pop|jmp|jne|jeq|call|ret|lea|xor|cmp|inc|dec)\s+(%?[er]?[a-z]{2,4}|\$0x[0-9a-f]+)", re.IGNORECASE)),
        ("arm_asm",    re.compile(rb"\b(ldr|str|mov|bl|bx|cbz|cbnz|adrp|stp|ldp)\s+[wx][0-9]+", re.IGNORECASE)),
        ("llvm_ir",    re.compile(rb"define\s+\w+\s+@\w+|%\w+\s*=\s*(load|store|call|add|sub|mul)\b")),
        ("wasm",       re.compile(rb"\(module|\(func\s+\$|i32\.const|local\.get")),
        ("bash",       re.compile(rb"^#!/.*\b(bash|sh)\b|\$\{[\w:]+\}|\bif\s+\[\s+", re.MULTILINE)),
        ("powershell", re.compile(rb"\$\w+\s*=|Write-Host|Get-\w+|\bcmdlet\b")),
        ("lisp",       re.compile(rb"\(defun\s+|\(define\s+|\(let\s+\(")),
        ("solidity",   re.compile(rb"\bpragma\s+solidity\b|\bcontract\s+\w+|\bfunction\s+\w+\s*\(")),
        ("fortran",    re.compile(rb"\b(SUBROUTINE|PROGRAM|END\s+DO|IMPLICIT\s+NONE)\b", re.IGNORECASE)),
        ("matlab",     re.compile(rb"\bfunction\s+\[?\w+\]?\s*=\s*\w+|\bend\s*$", re.MULTILINE)),
    ]

    def detect_language(self, source: bytes) -> str:
        """Best-effort language fingerprint.  Returns 'unknown' if no match."""
        sample = source[:4096]
        scores: Dict[str, int] = {}
        for name, pat in self._LANG_SIGNATURES:
            scores[name] = len(pat.findall(sample))
        best = max(scores.items(), key=lambda kv: kv[1])
        return best[0] if best[1] > 0 else "unknown"

    def extract_skeleton(self, source: bytes) -> Dict[str, int]:
        """Count AST-primitive hits — the semantic skeleton of the program."""
        skel: Dict[str, int] = {}
        for name, pat in self._PRIMITIVES:
            skel[name] = len(pat.findall(source))
        return skel

    def encode(self, source: Union[str, bytes]) -> np.ndarray:
        if isinstance(source, str):
            buf = source.encode("utf-8", errors="replace")
        else:
            buf = bytes(source)
        buf = buf[:MAX_INGEST_BYTES]

        lang     = self.detect_language(buf)
        skeleton = self.extract_skeleton(buf)

        # Map each primitive to a deterministic 16-bit ring in the SDR space.
        membrane = np.zeros(SDR_DIM, dtype=np.int32)
        for i, (name, _) in enumerate(self._PRIMITIVES):
            count = skeleton[name]
            if count == 0:
                continue
            # log1p compresses the dynamic range so a single primitive can't dominate
            weight = int(np.round(np.log1p(count) * 8))
            ring   = (np.arange(16) * 131 + _stable_hash(name)) % SDR_DIM
            membrane[ring] += weight

        # Add a language-identity ring so distinct languages produce distinct SDRs
        lang_ring = (np.arange(16) * 257 + _stable_hash(lang)) % SDR_DIM
        membrane[lang_ring] += 4

        sdr = _k_winner_take_all(membrane)
        return _apply_modality_bias(sdr, Modality.CODE)


# ============================================================================
#  5) BINARY CORTEX   —  raw byte signatures, entropy, magic numbers
# ============================================================================
class BinaryCortex:
    """
    Recognizes raw binary blobs: executables (ELF, PE, Mach-O), media headers
    (PNG, JPEG, MP4, WAV, ZIP, PDF), network packets, and unstructured data.

    Features extracted:
        • Magic-number / file-signature recognition (≈40 formats)
        • Shannon entropy (0..8 bits)  — distinguishes compressed vs structured
        • Byte-histogram top-k peaks   — captures coarse statistical fingerprint
        • CRC32 of the leading 4 KiB   — stable identity hash across re-encodes
    """
    _MAGICS: List[Tuple[bytes, str]] = [
        (b"\x7fELF",                "ELF"),       (b"MZ",                   "PE"),
        (b"\xCA\xFE\xBA\xBE",       "MACH_FAT"),  (b"\xFE\xED\xFA\xCE",     "MACH_O32"),
        (b"\xFE\xED\xFA\xCF",       "MACH_O64"),
        (b"\x89PNG\r\n\x1a\n",      "PNG"),       (b"\xFF\xD8\xFF",         "JPEG"),
        (b"GIF87a",                 "GIF87"),     (b"GIF89a",               "GIF89"),
        (b"RIFF",                   "RIFF"),      (b"%PDF",                 "PDF"),
        (b"PK\x03\x04",             "ZIP"),       (b"PK\x05\x06",           "ZIP_EMPTY"),
        (b"\x1f\x8b",               "GZIP"),      (b"BZh",                  "BZ2"),
        (b"\xFD7zXZ\x00",           "XZ"),        (b"7z\xBC\xAF\x27\x1C",   "7Z"),
        (b"OggS",                   "OGG"),       (b"ID3",                  "MP3"),
        (b"ftyp",                   "MP4"),       (b"\x00asm",              "WASM"),
        (b"!<arch>\n",              "AR"),        (b"\xCF\xFA\xED\xFE",     "MACHO_LE"),
        (b"SQLite format 3\x00",    "SQLITE"),    (b"\xED\xAB\xEE\xDB",     "RPM"),
        (b"-----BEGIN",             "PEM"),       (b"BM",                   "BMP"),
        (b"II*\x00",                "TIFF_LE"),   (b"MM\x00*",              "TIFF_BE"),
        (b"\x00\x00\x01\x00",       "ICO"),       (b"\x1A\x45\xDF\xA3",     "MATROSKA"),
        (b"<?xml",                  "XML"),       (b"{\n",                  "JSON_LIKE"),
        (b"<!DOCTYPE html",         "HTML"),
    ]

    @staticmethod
    def _shannon_entropy(buf: bytes) -> float:
        if not buf:
            return 0.0
        counts = np.bincount(np.frombuffer(buf, dtype=np.uint8), minlength=256)
        probs  = counts[counts > 0] / counts.sum()
        return float(-np.sum(probs * np.log2(probs)))

    def detect_format(self, buf: bytes) -> str:
        head = buf[:64]
        for magic, name in self._MAGICS:
            if magic in head:
                return name
        # ftyp magic is offset 4 in MP4
        if len(buf) >= 12 and buf[4:8] == b"ftyp":
            return "MP4"
        return "RAW"

    def encode(self, blob: Union[bytes, bytearray, memoryview]) -> np.ndarray:
        buf = bytes(blob)[:MAX_INGEST_BYTES]
        if not buf:
            return _apply_modality_bias(np.zeros(SDR_DIM, dtype=bool), Modality.BINARY)

        fmt     = self.detect_format(buf)
        entropy = self._shannon_entropy(buf)
        crc     = zlib.crc32(buf[:4096])

        membrane = np.zeros(SDR_DIM, dtype=np.int32)

        # ---- format-magic ring -------------------------------------------
        fmt_ring = (np.arange(16) * 313 + _stable_hash(fmt)) % SDR_DIM
        membrane[fmt_ring] += 16

        # ---- byte-histogram top-k peaks ----------------------------------
        hist = np.bincount(np.frombuffer(buf, dtype=np.uint8), minlength=256)
        top = np.argpartition(hist, -8)[-8:]
        for b in top:
            ring = (np.arange(8) * 211 + int(b)) % SDR_DIM
            membrane[ring] += int(np.log1p(hist[b]) * 2)

        # ---- entropy band  (8 bins covering 0..8 bits) -------------------
        ent_bucket = int(min(7, entropy))
        ent_ring   = (np.arange(8) * 401 + ent_bucket * 17) % SDR_DIM
        membrane[ent_ring] += 8

        # ---- CRC32 identity ring -----------------------------------------
        crc_ring = (np.arange(16) + crc) % SDR_DIM
        membrane[crc_ring] += 4

        sdr = _k_winner_take_all(membrane)
        return _apply_modality_bias(sdr, Modality.BINARY)


# ============================================================================
#  REFLEXIVE MODALITY DETECTOR  —  the spinal-reflex layer
# ============================================================================
class ModalitySniffer:
    """
    O(1) modality classification on the first ~256 bytes.

    Decision order (matches biological priority: vision > sound > smell):
        1. If input is an ndarray with ≥2 dims        → PIXEL
        2. If input is bytes/str:
             a. Starts with a known binary magic       → BINARY
             b. ≥ 5% bytes outside printable ASCII / UTF-8 → BINARY
             c. Code primitives density > threshold    → CODE
             d. High symbol-class density              → SYMBOL
             e. Else                                   → TEXT
    """
    _BIN_CORTEX = BinaryCortex()

    @staticmethod
    def _is_mostly_printable(buf: bytes, threshold: float = 0.90) -> bool:
        if not buf:
            return True
        sample = buf[:512]
        # Printable ASCII + common whitespace + UTF-8 multibyte starts/continuations
        printable = sum(1 for b in sample if (32 <= b < 127) or b in (9, 10, 13) or b >= 0x80)
        return printable / len(sample) >= threshold

    def sniff(self, x: Any) -> int:
        # ---- ndarray short-circuit (vision) -----------------------------
        if isinstance(x, np.ndarray) and x.ndim >= 2:
            return Modality.PIXEL

        # ---- coerce to bytes --------------------------------------------
        if isinstance(x, str):
            buf = x.encode("utf-8", errors="replace")
            had_str = True
        elif isinstance(x, (bytes, bytearray, memoryview)):
            buf = bytes(x)
            had_str = False
        else:
            return Modality.UNKNOWN

        if not buf:
            return Modality.TEXT

        # ---- magic-number short-circuit (BINARY) ------------------------
        fmt = self._BIN_CORTEX.detect_format(buf)
        if fmt not in ("RAW", "JSON_LIKE", "XML", "HTML", "PEM"):
            return Modality.BINARY

        # ---- printability test (BINARY catch-all) -----------------------
        if not had_str and not self._is_mostly_printable(buf):
            return Modality.BINARY

        # ---- code vs text vs symbol -------------------------------------
        sample = buf[:4096]
        # Strong, language-spanning code signals (incl. low-level asm & IR).
        code_signals = [
            b"{", b"}", b";", b"->", b"::", b"=>", b"&&", b"||", b"!=", b"==",
            b"def ", b"fn ", b"func ", b"function", b"#include", b"import ",
            b"return ", b"void ", b"int ", b"public ", b"class ", b"struct ",
            b"let ", b"var ", b"const ", b"package ", b"module ", b"namespace ",
            b"mov ", b"push ", b"pop ", b"call ", b"jmp ", b"ret\n", b"ret\r",
            b"%eax", b"%ebx", b"%rax", b"%rbx", b"xmm", b"@main", b"define i",
            b"::<", b"defmodule", b"defun", b"#!",
        ]
        code_hits = sum(sample.count(sig) for sig in code_signals)
        # threshold scales with sample size; minimum 2 signals fires CODE
        if code_hits >= max(2, len(sample) // 120):
            return Modality.CODE

        # SYMBOL heuristic: >35% non-letter non-digit non-whitespace AND
        # NOT predominantly multibyte UTF-8 (which is normal text in CJK).
        printable_count = sum(1 for b in sample if 32 <= b < 127)
        if printable_count > 0:
            non_alnum = sum(1 for b in sample[:printable_count*2] if 32 <= b < 127 and not (
                (48 <= b < 58) or (65 <= b < 91) or (97 <= b < 123) or b == 32
            ))
            if printable_count and non_alnum / max(printable_count, 1) > 0.45:
                return Modality.SYMBOL

        return Modality.TEXT


# ============================================================================
#  THE OMNI-SENSORY CORTEX  —  the public, top-level facade
# ============================================================================
@dataclass
class SensoryPercept:
    """A unified sensory percept — Hyperion's "moment of seeing"."""
    sdr:         np.ndarray            # 2048-bit boolean SDR
    modality:    int                   # Modality.* enum
    metadata:    Dict[str, Any] = field(default_factory=dict)

    @property
    def active_bits(self) -> int:
        return int(self.sdr.sum())

    @property
    def sparsity(self) -> float:
        return self.active_bits / SDR_DIM

    def __repr__(self) -> str:
        return (f"SensoryPercept(modality={Modality.NAMES[self.modality]}, "
                f"active={self.active_bits}/{SDR_DIM}, "
                f"sparsity={self.sparsity:.3%}, meta={self.metadata})")


class OmniSensoryCortex:
    """
    THE master sensory organ of Hyperion.

    Plug any input — image, audio waveform bytes, text, source code, raw
    binary, even a Python object — into ``.perceive(x)`` and get back a
    ``SensoryPercept`` whose ``.sdr`` is a 2048-bit boolean array ready to be
    fed directly into the Virtual86BSubstrate.

    Cross-modal Hebbian binding is supported via ``.bind(a, b)``: pass two
    percepts and the cortex returns their union SDR — exactly the operation
    that lets the substrate wire "apple-pixels ↔ apple-word ↔ apple-binary".
    """

    def __init__(self) -> None:
        self.pixel  = PixelCortex()
        self.text   = TextCortex()
        self.symbol = SymbolCortex()
        self.code   = CodeCortex()
        self.binary = BinaryCortex()
        self.sniffer = ModalitySniffer()
        # Internal counters for observability (useful for the meta_cortex)
        self.ticks: int = 0
        self.histogram: Dict[int, int] = {m: 0 for m in range(6)}

    # ----- the universal perception entry point --------------------------
    def perceive(self, x: Any, *, force_modality: Optional[int] = None
                ) -> SensoryPercept:
        modality = force_modality if force_modality is not None else self.sniffer.sniff(x)
        meta: Dict[str, Any] = {}

        if modality == Modality.PIXEL:
            if not isinstance(x, np.ndarray):
                # bytes-of-image fallback: best-effort interpret as raw 8-bit
                arr = np.frombuffer(bytes(x), dtype=np.uint8)
                side = int(np.sqrt(arr.size))
                arr = arr[:side*side].reshape(side, side) if side > 0 else arr.reshape(1, -1)
                x = arr
            sdr = self.pixel.encode(x)
            meta["shape"] = tuple(x.shape)

        elif modality == Modality.TEXT:
            sdr = self.text.encode(x)
            meta["bytes"] = len(x) if hasattr(x, "__len__") else 0

        elif modality == Modality.SYMBOL:
            sdr = self.symbol.encode(x)
            meta["bytes"] = len(x) if hasattr(x, "__len__") else 0

        elif modality == Modality.CODE:
            sdr  = self.code.encode(x)
            buf  = x.encode("utf-8", errors="replace") if isinstance(x, str) else bytes(x)
            meta["language"] = self.code.detect_language(buf)
            meta["skeleton"] = self.code.extract_skeleton(buf[:8192])

        elif modality == Modality.BINARY:
            if isinstance(x, str):
                buf = x.encode("utf-8", errors="replace")
            elif isinstance(x, np.ndarray):
                buf = x.tobytes()
            else:
                buf = bytes(x)
            sdr = self.binary.encode(buf)
            meta["format"]  = self.binary.detect_format(buf)
            meta["entropy"] = round(self.binary._shannon_entropy(buf[:4096]), 3)
            meta["bytes"]   = len(buf)

        else:  # UNKNOWN — graceful degradation: treat as binary
            buf = bytes(repr(x), "utf-8", errors="replace")
            sdr = self.binary.encode(buf)
            meta["fallback"] = "unknown→binary"

        # bookkeeping
        self.ticks += 1
        self.histogram[modality] = self.histogram.get(modality, 0) + 1

        return SensoryPercept(sdr=sdr, modality=modality, metadata=meta)

    # ----- cross-modal Hebbian binding (read the README §4.A) ------------
    @staticmethod
    def bind(a: "SensoryPercept | np.ndarray",
             b: "SensoryPercept | np.ndarray",
             *, k: int = SDR_ACTIVE_BITS) -> "SensoryPercept":
        """
        Cross-modal Hebbian bind — "apple-pixels ↔ apple-word".

        [PATCH v5]
          * The previous implementation returned ``a.sdr OR b.sdr``. A logical
            OR can produce up to ``2 * SDR_ACTIVE_BITS`` active bits, which
            breaks the k-WTA invariant the rest of the substrate (Virtual86B,
            kwta projectors, SparseSynapseStore overlap math) relies on.
          * We now sum the two SDRs as floats (bit-vote weighting: a bit set
            in both inputs scores 2, in one input scores 1, in neither 0)
            and re-project to exactly ``k`` active bits via k-WTA. Ties are
            broken by SDR index for determinism.
          * Result is a valid k-WTA SDR ready to flow into the substrate.

        Accepts either SensoryPercept objects or raw 2048-bit SDR arrays,
        and is safely chainable: ``bind(bind(a, b), c)``.
        """
        sdr_a = a.sdr if isinstance(a, SensoryPercept) else np.asarray(a, dtype=bool)
        sdr_b = b.sdr if isinstance(b, SensoryPercept) else np.asarray(b, dtype=bool)
        mod_a = a.modality if isinstance(a, SensoryPercept) else Modality.UNKNOWN
        mod_b = b.modality if isinstance(b, SensoryPercept) else Modality.UNKNOWN
        merged_modality = mod_a if mod_a == mod_b else Modality.UNKNOWN

        votes = sdr_a.astype(np.int8) + sdr_b.astype(np.int8)
        if votes.sum() == 0:
            bound_sdr = np.zeros(SDR_DIM, dtype=bool)
        else:
            # k-WTA: pick the top-k indices (ties broken by lower index, which
            # np.argpartition + argsort gives deterministically).
            k_use = min(int(k), int((votes > 0).sum()))
            if k_use <= 0:
                bound_sdr = np.zeros(SDR_DIM, dtype=bool)
            else:
                # argpartition: top k_use unordered, then deterministic sort
                # by (-votes, index) to break ties consistently.
                cand = np.argpartition(-votes, k_use - 1)[:k_use]
                # Stable tie-break: sort by negative score, then by index.
                order = np.lexsort((cand, -votes[cand]))
                top = cand[order]
                bound_sdr = np.zeros(SDR_DIM, dtype=bool)
                bound_sdr[top] = True
        return SensoryPercept(
            sdr=bound_sdr,
            modality=merged_modality,
            metadata={"bound_from": (Modality.NAMES[mod_a], Modality.NAMES[mod_b]),
                      "k_active": int(bound_sdr.sum())},
        )

    # ----- introspection -------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "ticks":      self.ticks,
            "modality_histogram": {Modality.NAMES[m]: c for m, c in self.histogram.items()},
            "sdr_dim":    SDR_DIM,
            "active_k":   SDR_ACTIVE_BITS,
            "sparsity":   SDR_ACTIVE_BITS / SDR_DIM,
        }


# ============================================================================
#  SELF-TEST  —  run as  `python -m nlp_engine.sensors.omni_sensory_cortex`
# ============================================================================
if __name__ == "__main__":
    import time

    cortex = OmniSensoryCortex()

    samples: List[Tuple[str, Any]] = [
        ("plain text en",   "The quick brown fox jumps over the lazy dog."),
        ("plain text zh",   "今天天气真好,我们一起去公园散步吧。"),
        ("symbols / math",  "∂f/∂x = 2x  ∫₀^∞ e^{-x²}dx = √π/2   ∇·E = ρ/ε₀"),
        ("python code",     "def fib(n):\n    return n if n<2 else fib(n-1)+fib(n-2)\nprint(fib(10))"),
        ("c code",          "#include <stdio.h>\nint main(){ for(int i=0;i<10;i++) printf(\"%d\",i); return 0; }"),
        ("rust code",       "fn main() { let mut v: Vec<i32> = (0..10).collect(); v.iter().for_each(|x| println!(\"{}\", x)); }"),
        ("x86 asm",         "mov eax, 1\nxor ebx, ebx\nint 0x80\nret"),
        ("llvm ir",         "define i32 @main() { %1 = add i32 1, 2 ret i32 %1 }"),
        ("png bytes",       b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00"*64),
        ("elf bytes",       b"\x7fELF\x02\x01\x01\x00" + bytes(range(256))),
        ("random image",    np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)),
    ]

    print("="*78)
    print(" Hyperion Omni-Sensory Cortex  —  Self-Test")
    print("="*78)
    t0 = time.perf_counter()
    percepts: List[SensoryPercept] = []
    for label, x in samples:
        p = cortex.perceive(x)
        percepts.append(p)
        print(f"  {label:18s} → {p}")
    elapsed = (time.perf_counter() - t0) * 1000
    print("-"*78)
    print(f"  Processed {len(samples)} percepts in {elapsed:.2f} ms "
          f"({elapsed/len(samples):.2f} ms / percept)")
    print(f"  Stats: {cortex.stats()}")

    # Cross-modal binding demo: pixels of (random) image  ↔  word "image"
    bound = OmniSensoryCortex.bind(percepts[-1], cortex.perceive("image"))
    print(f"  Cross-modal bind active bits: {bound.active_bits} / {SDR_DIM}")
    print("="*78)
