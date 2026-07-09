#!/usr/bin/env python3
"""
WordTokenizer — upgraded multilingual, Unicode-aware, trainable tokenizer.

What changed in this patched version
-----------------------------------
- Replaced the old regex-only splitter with a Unicode-aware state machine.
- Better multilingual coverage for Latin, Cyrillic, Greek, Arabic, Hebrew,
  Devanagari, Thai, Hangul, Hiragana, Katakana, and Han text.
- Preserves contractions and code identifiers (don't, l'esprit, snake_case,
  state-of-the-art).
- Uses NFKC normalization by default for more stable multilingual training.
- Improves decode spacing for CJK and Japanese mixed-script text.
- Adds save/load helpers so a trained vocabulary can be reused.
- Keeps the original public API (`fit`, `encode`, `decode`, `build_sequences`,
  `coverage`, `stats`) so existing benchmarks keep working.

This is still a *word/subword-ish* tokenizer, not a learned BPE/SentencePiece
model. The goal here is a robust, dependency-free multilingual tokenizer that
fits the current repository cleanly and deterministically.
"""

from __future__ import annotations

import json
import os
import unicodedata
from collections import Counter
from typing import List, Optional, Sequence, Union

import numpy as np


def _script_of(ch: str) -> str:
    """Best-effort Unicode script grouping without external dependencies."""
    cp = ord(ch)
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
        return "han"
    if 0x3040 <= cp <= 0x309F:
        return "hiragana"
    if 0x30A0 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
        return "katakana"
    if 0xAC00 <= cp <= 0xD7A3:
        return "hangul"
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF:
        return "arabic"
    if 0x0590 <= cp <= 0x05FF:
        return "hebrew"
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"
    if 0x0E00 <= cp <= 0x0E7F:
        return "thai"
    if 0x0400 <= cp <= 0x052F:
        return "cyrillic"
    if 0x0370 <= cp <= 0x03FF:
        return "greek"
    if (0x0041 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
        return "latin"
    return "other"


class WordTokenizer:
    """
    Multilingual tokenizer with fixed-length training sequence generation.

    Parameters
    ----------
    vocab_size : int
        Number of learned vocabulary entries excluding special tokens.
    seq_len : int
        Returned sequence length is always `seq_len + 1`.
    case_sensitive : bool
        If False, lowercases text before tokenization.
    normalize_form : str
        Unicode normalization form, default NFKC.
    split_han : bool
        If True, tokenizes Han characters one-by-one, which is a safer default
        for Chinese/Japanese text without a dictionary segmenter.
    """

    UNK_SYMBOL = "<UNK>"
    PAD_SYMBOL = "<PAD>"
    UNK_ID = 0
    PAD_ID = 1
    _SPECIAL = {UNK_SYMBOL, PAD_SYMBOL}

    _OPEN_PUNCT = {"(", "[", "{", "«", "“", "‘", '"'}
    _CLOSE_PUNCT = {")", "]", "}", ".", ",", "!", "?", ";", ":", "%", "»", "”", "’"}
    _NO_SPACE_BEFORE = _CLOSE_PUNCT | {",", ".", "!", "?", ":", ";", "%", "…", "、", "。", "，", "！", "？", "：", "；"}
    _NO_SPACE_AFTER = _OPEN_PUNCT | {"$", "#", "@", "£", "€", "¥"}
    _JOINER_CHARS = {"'", "’", "`", "-", "_"}
    _CJKISH = {"han", "hiragana", "katakana"}

    def __init__(
        self,
        vocab_size: int = 5000,
        seq_len: int = 128,
        case_sensitive: bool = False,
        normalize_form: str = "NFKC",
        split_han: bool = True,
    ):
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")

        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.case_sensitive = bool(case_sensitive)
        self.normalize_form = normalize_form
        self.split_han = bool(split_han)
        self.effective_vocab_size = self.vocab_size + 2

        self.word_to_id: dict[str, int] = {
            self.UNK_SYMBOL: self.UNK_ID,
            self.PAD_SYMBOL: self.PAD_ID,
        }
        self.id_to_word: dict[int, str] = {
            self.UNK_ID: self.UNK_SYMBOL,
            self.PAD_ID: self.PAD_SYMBOL,
        }
        self.token_counts: Counter[str] = Counter()

    # ------------------------------------------------------------------ helpers
    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize(self.normalize_form, text)
        return text if self.case_sensitive else text.lower()

    @staticmethod
    def _category(ch: str) -> str:
        return unicodedata.category(ch)

    @classmethod
    def _is_word_char(cls, ch: str) -> bool:
        cat = cls._category(ch)
        return cat[0] in {"L", "N", "M"}

    @classmethod
    def _is_punct_or_symbol(cls, ch: str) -> bool:
        cat = cls._category(ch)
        return cat[0] in {"P", "S"}

    @classmethod
    def _is_joiner(cls, ch: str) -> bool:
        return ch in cls._JOINER_CHARS

    @classmethod
    def _can_join_around(cls, left: str | None, joiner: str, right: str | None) -> bool:
        if not left or not right:
            return False
        if not cls._is_word_char(left) or not cls._is_word_char(right):
            return False
        if joiner == "_":
            return True
        return True

    @classmethod
    def _token_script(cls, token: str) -> str:
        for ch in token:
            if cls._is_word_char(ch):
                return _script_of(ch)
        return "other"

    @classmethod
    def _is_punct_token(cls, token: str) -> bool:
        return bool(token) and all(cls._is_punct_or_symbol(ch) for ch in token)

    @classmethod
    def _no_space_between(cls, prev: str, cur: str) -> bool:
        if not prev or not cur:
            return False
        prev_script = cls._token_script(prev)
        cur_script = cls._token_script(cur)
        if prev_script in cls._CJKISH and cur_script in cls._CJKISH:
            return True
        if prev in {",", "，", "、", "。", ".", "!", "！", "?", "？"} and cur_script in cls._CJKISH:
            return True
        if cls._is_punct_token(cur) and cur in cls._NO_SPACE_BEFORE:
            return True
        if cls._is_punct_token(prev) and prev in cls._NO_SPACE_AFTER:
            return True
        return False

    # ---------------------------------------------------------------- tokenizing
    def split(self, text: str) -> List[str]:
        """Split text into multilingual tokens without external libraries."""
        text = self._normalize(text)
        if not text:
            return []

        tokens: List[str] = []
        buf: List[str] = []
        buf_kind: str | None = None
        buf_script: str | None = None

        def flush() -> None:
            nonlocal buf, buf_kind, buf_script
            if buf:
                tokens.append("".join(buf))
            buf = []
            buf_kind = None
            buf_script = None

        i = 0
        n = len(text)
        while i < n:
            ch = text[i]

            if ch.isspace():
                flush()
                i += 1
                continue

            if self._is_joiner(ch):
                left = buf[-1] if buf else None
                right = text[i + 1] if i + 1 < n else None
                if buf and self._can_join_around(left, ch, right):
                    buf.append(ch)
                    i += 1
                    continue
                flush()
                tokens.append(ch)
                i += 1
                continue

            if self._is_word_char(ch):
                script = _script_of(ch)
                kind = "number" if ch.isdigit() else "word"

                if script == "han" and self.split_han:
                    flush()
                    tokens.append(ch)
                    i += 1
                    continue

                if not buf:
                    buf = [ch]
                    buf_kind = kind
                    buf_script = script
                else:
                    compatible = (
                        (buf_kind == kind) and
                        (buf_script == script or kind == "number" or script == "other" or buf_script == "other")
                    )
                    if compatible:
                        buf.append(ch)
                    else:
                        flush()
                        buf = [ch]
                        buf_kind = kind
                        buf_script = script
                i += 1
                continue

            flush()
            tokens.append(ch)
            i += 1

        flush()
        return tokens

    # ------------------------------------------------------------- vocabulary io
    def fit(self, text: Union[str, Sequence[str]]) -> "WordTokenizer":
        texts = [text] if isinstance(text, str) else list(text)
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(self.split(t))

        self.token_counts = counter.copy()
        self.word_to_id = {
            self.UNK_SYMBOL: self.UNK_ID,
            self.PAD_SYMBOL: self.PAD_ID,
        }
        self.id_to_word = {
            self.UNK_ID: self.UNK_SYMBOL,
            self.PAD_ID: self.PAD_SYMBOL,
        }

        next_idx = 2
        for token, _count in counter.most_common():
            if token in self._SPECIAL:
                continue
            self.word_to_id[token] = next_idx
            self.id_to_word[next_idx] = token
            next_idx += 1
            if next_idx >= self.vocab_size + 2:
                break

        # [PATCH v5] effective_vocab_size MUST equal the true number of
        # populated ids, not vocab_size + 2. Otherwise downstream embedding
        # tables (cain_brain.E) allocate dead rows for token ids that can
        # never be emitted by encode(), wasting parameters and biasing
        # softmax denominators. Cap matches the actual id_to_word table.
        self.effective_vocab_size = (max(self.id_to_word) + 1
                                     if self.id_to_word else 2)
        return self

    train = fit

    def fit_files(self, paths: Sequence[str], encoding: str = "utf-8") -> "WordTokenizer":
        texts = []
        for path in paths:
            with open(path, "r", encoding=encoding) as f:
                texts.append(f.read())
        return self.fit(texts)

    def save_vocab(self, path: str) -> str:
        payload = {
            "config": {
                "vocab_size": self.vocab_size,
                "seq_len": self.seq_len,
                "case_sensitive": self.case_sensitive,
                "normalize_form": self.normalize_form,
                "split_han": self.split_han,
            },
            "word_to_id": self.word_to_id,
            "token_counts": self.token_counts,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load_vocab(cls, path: str) -> "WordTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cfg = payload["config"]
        tok = cls(**cfg)
        tok.word_to_id = {k: int(v) for k, v in payload["word_to_id"].items()}
        tok.id_to_word = {int(v): k for k, v in tok.word_to_id.items()}
        tok.token_counts = Counter(payload.get("token_counts", {}))
        tok.effective_vocab_size = max(tok.id_to_word) + 1 if tok.id_to_word else tok.vocab_size + 2
        return tok

    # --------------------------------------------------------------- encode/decode
    def encode(self, text: str) -> List[int]:
        return [self.word_to_id.get(tok, self.UNK_ID) for tok in self.split(text)]

    @classmethod
    def _join_words(cls, words: List[str]) -> str:
        if not words:
            return ""
        out = [words[0]]
        for w in words[1:]:
            prev = out[-1]
            prev_text = prev if isinstance(prev, str) else str(prev)
            if cls._no_space_between(prev_text, w):
                out.append(w)
            else:
                out.append(" " + w)
        return "".join(out)

    def decode(self, ids: Union[Sequence[int], np.ndarray]) -> str:
        words: List[str] = []
        for idx in ids:
            token = self.id_to_word.get(int(idx), self.UNK_SYMBOL)
            if token == self.PAD_SYMBOL:
                continue
            words.append(token)
        return self._join_words(words)

    # ----------------------------------------------------------- training sequences
    def _pad_crop(self, ids: List[int]) -> np.ndarray:
        target_len = self.seq_len + 1
        if len(ids) >= target_len:
            return np.array(ids[:target_len], dtype=np.int32)
        return np.array(ids + [self.PAD_ID] * (target_len - len(ids)), dtype=np.int32)

    def build_sequences(self, text: str, stride: Optional[int] = None) -> List[np.ndarray]:
        ids = self.encode(text)
        if not ids:
            return [self._pad_crop([])]
        if stride is None:
            stride = self.seq_len
        sequences: List[np.ndarray] = []
        i = 0
        while i + self.seq_len < len(ids):
            sequences.append(self._pad_crop(ids[i:i + self.seq_len + 1]))
            i += stride
        return sequences

    # -------------------------------------------------------------------- analytics
    @property
    def stats(self) -> dict:
        actual = len([k for k in self.word_to_id if k not in self._SPECIAL])
        top_tokens = self.token_counts.most_common(10)
        return {
            "vocab_size": self.vocab_size,
            "effective_vocab_size": self.effective_vocab_size,
            "seq_len": self.seq_len,
            "actual_word_count": actual,
            "case_sensitive": self.case_sensitive,
            "normalize_form": self.normalize_form,
            "split_han": self.split_han,
            "unk_id": self.UNK_ID,
            "pad_id": self.PAD_ID,
            "unk_token": self.UNK_SYMBOL,
            "pad_token": self.PAD_SYMBOL,
            "trained_token_count": int(sum(self.token_counts.values())),
            "top_tokens": top_tokens,
        }

    def coverage(self, text: str) -> dict:
        tokens = self.split(text)
        total = len(tokens)
        if total == 0:
            return {
                "total_tokens": 0,
                "known_tokens": 0,
                "unknown_tokens": 0,
                "coverage_pct": 0.0,
                "unique_types": 0,
                "known_types": 0,
                "type_token_ratio": 0.0,
                "unknown_examples": [],
            }

        known = [t for t in tokens if t in self.word_to_id]
        unknown = [t for t in tokens if t not in self.word_to_id]
        unique = set(tokens)
        return {
            "total_tokens": total,
            "known_tokens": len(known),
            "unknown_tokens": len(unknown),
            "coverage_pct": (len(known) / total) * 100.0,
            "unique_types": len(unique),
            "known_types": sum(1 for t in unique if t in self.word_to_id),
            "type_token_ratio": len(unique) / total,
            "unknown_examples": unknown[:10],
        }


MultilingualTokenizer = WordTokenizer


# ============================================================================
# Self-tests
# ============================================================================

def _test_core() -> bool:
    tok = WordTokenizer(vocab_size=100, seq_len=8, case_sensitive=False)

    assert tok.split("Hello, world!") == ["hello", ",", "world", "!"], "latin punctuation"
    assert tok.split("it's a dog's life") == ["it's", "a", "dog's", "life"], "apostrophes"
    assert tok.split("snake_case variable") == ["snake_case", "variable"], "identifier preservation"
    assert tok.split("state-of-the-art") == ["state-of-the-art"], "hyphen preservation"
    assert tok.split("你好，世界") == ["你", "好", ",", "世", "界"], "han splitting"
    assert tok.split("Привет, мир!") == ["привет", ",", "мир", "!"], "cyrillic"
    assert tok.split("مرحبا بالعالم") == ["مرحبا", "بالعالم"], "arabic"
    assert tok.split("नमस्ते दुनिया") == ["नमस्ते", "दुनिया"], "devanagari"

    corpus = [
        "Hello, world!",
        "Привет, мир!",
        "你好，世界。",
        "مرحبا بالعالم",
        "नमस्ते दुनिया",
        "state-of-the-art tokenizer",
        "snake_case and camelCase",
    ]
    tok.fit(corpus)

    ids = tok.encode("Hello, world!")
    assert all(isinstance(i, int) for i in ids), "encode ints"
    assert "hello" in tok.decode(ids), "decode roundtrip latin"

    zh_ids = tok.encode("你好，世界")
    assert tok.decode(zh_ids) == "你好,世界", "decode CJK spacing"

    pad = tok._pad_crop([2, 3])
    assert len(pad) == 9 and pad[-1] == tok.PAD_ID, "pad_crop"

    seqs = tok.build_sequences("hello world hello world hello world hello world hello world", stride=2)
    assert len(seqs) > 0 and all(s.shape == (9,) for s in seqs), "build_sequences"

    cov = tok.coverage("مرحبا بالعالم و أهلا")
    assert cov["total_tokens"] >= 2, "coverage nonzero"

    tmp_path = os.path.join(os.path.dirname(__file__), "_tmp_tokenizer_vocab.json")
    tok.save_vocab(tmp_path)
    tok2 = WordTokenizer.load_vocab(tmp_path)
    os.remove(tmp_path)
    assert tok2.decode(tok2.encode("Привет, мир!")).startswith("привет"), "save/load"

    print("  All upgraded multilingual tokenizer tests PASSED")
    return True


def demo() -> None:
    corpus = [
        "Artificial intelligence changes everything.",
        "L'intelligence artificielle transforme le monde.",
        "La inteligencia artificial cambia todo.",
        "Искусственный интеллект меняет всё.",
        "الذكاء الاصطناعي يغير كل شيء.",
        "人工智能正在改变世界。",
        "ニューラルネットワークは面白いです。",
        "인공지능은 세상을 바꾸고 있습니다.",
        "कृत्रिम बुद्धिमत्ता दुनिया बदल रही है।",
        "ระบบปัญญาประดิษฐ์กำลังเปลี่ยนโลก",
        "def translate(text): return text.lower()",
    ]
    tok = WordTokenizer(vocab_size=256, seq_len=16)
    tok.fit(corpus)
    print("╔═══ Upgraded Multilingual Tokenizer Demo ═══╗")
    print(f"Vocab: {tok.stats['actual_word_count']} learned tokens")
    print("Top tokens:", tok.stats["top_tokens"][:8])
    sample = "Hello 世界! مرحبا"
    ids = tok.encode(sample)
    print("Sample tokens:", tok.split(sample))
    print("Encoded:", ids)
    print("Decoded:", tok.decode(ids))


if __name__ == "__main__":
    _test_core()
    demo()
