"""
UniversalTokenizer — byte-level encoder for *any* prompt modality.

Design goals (rivalling modern LLM frontends):
    - Language-agnostic: works on English, CJK, Cyrillic, Arabic, ... because
      UTF-8 bytes are the universal lowest common denominator.
    - Code-aware: doesn't choke on punctuation, whitespace, or unicode in
      identifiers. (BPE tokenizers often fragment code awkwardly.)
    - Instruction-aware: special "control" byte ranges encode role tags
      (system / user / assistant / thinking) without polluting the data vocab.
    - Knowledge-source-aware: similar control bytes mark <doc> / <web> /
      <code> / <math> regions so the model can route different modalities
      to different cortical sub-circuits.

Vocabulary layout (vocab_size = 320):
       0 ..  255 : raw UTF-8 bytes
     256 ..  271 : control tokens (BOS, EOS, PAD, role tags, modality tags, ...)
     272 ..  319 : reserved for future modality / language tags
                   (we expose the symbolic constants below).

The encoder is intentionally simple — no merges, no learned table. All the
intelligence happens *downstream* inside CortexNet, exactly as biological
sensory cortex doesn't learn its photoreceptor layout but its V1 codebook.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


# ---- vocab layout ---------------------------------------------------------
VOCAB_SIZE = 320

# Control tokens (256..271)
TOK_PAD              = 256
TOK_BOS              = 257
TOK_EOS              = 258
TOK_SEP              = 259
TOK_ROLE_SYSTEM      = 260
TOK_ROLE_USER        = 261
TOK_ROLE_ASSISTANT   = 262
TOK_ROLE_THINKING    = 263
TOK_MOD_TEXT         = 264
TOK_MOD_CODE         = 265
TOK_MOD_MATH         = 266
TOK_MOD_INSTRUCTION  = 267
TOK_MOD_KNOWLEDGE    = 268
TOK_MOD_QUERY        = 269
TOK_LANG_TAG         = 270   # followed by a single language ID byte
TOK_END_BLOCK        = 271

CONTROL_NAMES = {
    TOK_PAD: "<pad>", TOK_BOS: "<bos>", TOK_EOS: "<eos>", TOK_SEP: "<sep>",
    TOK_ROLE_SYSTEM: "<system>", TOK_ROLE_USER: "<user>",
    TOK_ROLE_ASSISTANT: "<assistant>", TOK_ROLE_THINKING: "<thinking>",
    TOK_MOD_TEXT: "<text>", TOK_MOD_CODE: "<code>",
    TOK_MOD_MATH: "<math>", TOK_MOD_INSTRUCTION: "<instr>",
    TOK_MOD_KNOWLEDGE: "<knowledge>", TOK_MOD_QUERY: "<query>",
    TOK_LANG_TAG: "<lang>", TOK_END_BLOCK: "<end>",
}


@dataclass
class PromptBlock:
    """One semantic segment of a prompt."""
    text: str
    role: str = "user"          # "system", "user", "assistant", "thinking"
    modality: str = "text"      # "text", "code", "math", "instruction", "knowledge", "query"
    lang: str | None = None     # 2-letter ISO code, optional

    def role_token(self) -> int:
        return {
            "system":    TOK_ROLE_SYSTEM,
            "user":      TOK_ROLE_USER,
            "assistant": TOK_ROLE_ASSISTANT,
            "thinking":  TOK_ROLE_THINKING,
        }.get(self.role, TOK_ROLE_USER)

    def modality_token(self) -> int:
        return {
            "text":        TOK_MOD_TEXT,
            "code":        TOK_MOD_CODE,
            "math":        TOK_MOD_MATH,
            "instruction": TOK_MOD_INSTRUCTION,
            "knowledge":   TOK_MOD_KNOWLEDGE,
            "query":       TOK_MOD_QUERY,
        }.get(self.modality, TOK_MOD_TEXT)


_LANG_TABLE = {
    "en": 0, "es": 1, "fr": 2, "de": 3, "it": 4, "pt": 5,
    "ru": 6, "ar": 7, "hi": 8, "zh": 9, "ja": 10, "ko": 11,
    "tr": 12, "vi": 13, "th": 14, "id": 15, "nl": 16, "pl": 17,
    "sv": 18, "fi": 19, "no": 20, "da": 21, "el": 22, "he": 23,
    "fa": 24, "uk": 25, "cs": 26, "hu": 27, "ro": 28, "bg": 29,
    "ms": 30, "bn": 31,
}


class UniversalTokenizer:
    """Byte-level tokenizer with role / modality / language control tokens."""

    vocab_size = VOCAB_SIZE
    pad_id = TOK_PAD
    bos_id = TOK_BOS
    eos_id = TOK_EOS

    # ------------------------------------------------------------------- encode
    def encode_text(self, s: str) -> List[int]:
        return list(s.encode("utf-8"))

    def encode_block(self, block: PromptBlock) -> List[int]:
        ids: List[int] = [block.role_token(), block.modality_token()]
        if block.lang and block.lang in _LANG_TABLE:
            ids.extend([TOK_LANG_TAG, _LANG_TABLE[block.lang]])
        ids.extend(self.encode_text(block.text))
        ids.append(TOK_END_BLOCK)
        return ids

    def encode(self, prompt: str | Sequence[PromptBlock],
               add_bos: bool = True,
               add_eos: bool = False) -> np.ndarray:
        """Universal entry point. Accepts either a raw string or a list of
        PromptBlocks. Always returns a 1-D int array of token ids."""
        ids: List[int] = []
        if add_bos:
            ids.append(TOK_BOS)
        if isinstance(prompt, str):
            # Auto-detect a small set of obvious modalities so the model gets
            # *some* signal even when the caller doesn't structure the prompt.
            modality = _autodetect_modality(prompt)
            ids.extend(self.encode_block(PromptBlock(prompt, modality=modality)))
        else:
            for block in prompt:
                ids.extend(self.encode_block(block))
        if add_eos:
            ids.append(TOK_EOS)
        return np.asarray(ids, dtype=np.int64)

    # ------------------------------------------------------------------- decode
    def decode(self, ids: Sequence[int],
               keep_control: bool = False) -> str:
        """Best-effort UTF-8 decode. Skips control tokens by default."""
        byte_buf: List[int] = []
        out: List[str] = []
        for tok in ids:
            tok = int(tok)
            if tok < 256:
                byte_buf.append(tok)
            else:
                if byte_buf:
                    out.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                    byte_buf.clear()
                if keep_control and tok in CONTROL_NAMES:
                    out.append(CONTROL_NAMES[tok])
        if byte_buf:
            out.append(bytes(byte_buf).decode("utf-8", errors="replace"))
        return "".join(out)


def _autodetect_modality(s: str) -> str:
    """Cheap heuristic — we don't have to be perfect; downstream cortical
    columns also receive the raw bytes."""
    if not s:
        return "text"
    sample = s[:200]
    # code markers
    code_markers = ("def ", "function ", "class ", "import ", "#include",
                    "println", "console.log", "->", "::", "=>", "{", "}",
                    "</", "<?", "int main", "fn ", "pub ", "lambda ", "var ")
    if any(m in sample for m in code_markers):
        return "code"
    # math markers
    math_markers = ("∑", "∫", "∂", "√", "≈", "≥", "≤", "≠", "lim_", "\\frac")
    if any(m in sample for m in math_markers):
        return "math"
    # instruction markers
    if sample.lower().startswith(("write ", "explain ", "list ", "summarize ",
                                  "translate ", "give me ", "how to ",
                                  "show me ", "create ", "build ", "implement ")):
        return "instruction"
    # question
    if "?" in sample[:100]:
        return "query"
    return "text"
