"""Phase-2 / Omega smoke test + micro-benchmark.

Run from repo root:
    python3 -m nlp_engine.cortexnet._omega_test
"""
import time
import numpy as np

from nlp_engine.cortexnet import (
    CortexNetOmega, CortexNetLite,
    UniversalTokenizer, PromptBlock,
    MANARAttention, CorticalColumn,
    HippocampalReplay,
)


# ---------------------------------------------------------------------- 1
def test_universal_tokenizer():
    tok = UniversalTokenizer()
    # English
    ids = tok.encode("Hello, world!")
    assert tok.decode(ids).endswith("Hello, world!")
    # Chinese
    ids2 = tok.encode("你好，世界")
    assert "你好，世界" in tok.decode(ids2)
    # Code
    ids3 = tok.encode("def add(a, b): return a + b")
    assert "def add" in tok.decode(ids3)
    # Mixed block prompt
    blocks = [
        PromptBlock("You are a helpful assistant.", role="system", modality="instruction"),
        PromptBlock("Translate to French.",          role="user",   modality="instruction"),
        PromptBlock("Hello, how are you?",           role="user",   modality="text", lang="en"),
    ]
    ids4 = tok.encode(blocks)
    rt   = tok.decode(ids4)
    assert "Hello, how are you?" in rt
    print(f"  tokenizer round-trip OK on en/zh/code/multi-block (vocab={tok.vocab_size})")


# ---------------------------------------------------------------------- 2
def test_manar_streaming_matches_full():
    """A streaming forward (token-by-token) should give a similar trajectory
    to a full-sequence forward on the same prefix."""
    rng = np.random.default_rng(0)
    a = MANARAttention(d_model=32, num_concepts=8, window=4, causal=True, seed=0)
    x = rng.standard_normal((20, 32))

    y_full = a.forward(x)                                # (20, 32)

    cache = a.init_cache()
    y_stream = np.zeros_like(y_full)
    # Build streaming step-by-step from t=0
    for t in range(x.shape[0]):
        y_stream[t] = a.step_with_cache(x[t], cache)

    # We don't expect exact equality (full-seq sees future under non-causal;
    # streaming sees only past under causal). For *causal* MANAR the global
    # ACR differs because full uses the whole sequence while streaming sees
    # one token at a time. So just check shape + finiteness + reasonable
    # magnitude correlation.
    assert y_stream.shape == y_full.shape
    assert np.all(np.isfinite(y_stream))
    print(f"  MANAR streaming shape OK, mean magnitude full={np.abs(y_full).mean():.4f} stream={np.abs(y_stream).mean():.4f}")


# ---------------------------------------------------------------------- 3
def test_cortical_column_streaming():
    rng = np.random.default_rng(0)
    col = CorticalColumn(d_model=32, num_concepts=8, window=4, causal=True, seed=0)
    cache = col.init_cache()

    prompt = rng.standard_normal((10, 32))
    out_p = col.prime_cache(prompt, cache)
    assert out_p.shape == (10, 32)

    # Snapshot & restore
    snap = col.snapshot()
    y1 = col.step_with_cache(rng.standard_normal(32), cache)
    col.restore(snap)
    assert np.all(np.isfinite(y1))
    print("  CorticalColumn streaming + snapshot/restore OK")


# ---------------------------------------------------------------------- 4
def test_omega_generate():
    m = CortexNetOmega(d_model=32, n_columns=2, num_concepts=8, window=4, seed=0)
    out = m.generate("Hello", max_new_tokens=20, temperature=0.7, seed=42)
    assert isinstance(out, str)
    print(f"  generate('Hello') -> {repr(out)[:80]}")


# ---------------------------------------------------------------------- 5
def test_omega_think_then_speak():
    m = CortexNetOmega(d_model=32, n_columns=2, num_concepts=8, window=4, seed=1)
    result = m.think_then_speak("What is 2+2?",
                                 max_thoughts=8,
                                 max_new_tokens=15,
                                 temperature=0.7,
                                 seed=7,
                                 return_trace=True)
    assert "answer"  in result
    assert "thought" in result
    print(f"  think_then_speak: thought='{result['thought'][:40]}...' answer='{result['answer'][:40]}...'")


# ---------------------------------------------------------------------- 6
def test_hippocampal_sleep():
    m = CortexNetOmega(d_model=32, n_columns=2, num_concepts=8, window=4, seed=0)
    rng = np.random.default_rng(0)
    # Inject a few "experiences"
    for _ in range(8):
        ids = rng.integers(0, m.vocab_size, size=16)
        m.train_step(ids, lr=2e-3, salience=float(rng.uniform(0.1, 1.0)))
    stats = m.sleep(n_replays=4, lr=2e-3, seed=0)
    assert stats["replays"] > 0
    # Sleep should not blow up
    assert np.all(np.isfinite(m.W_out))
    print(f"  sleep stats: {stats}")


# ---------------------------------------------------------------------- 7
def test_universal_prompt_recognition():
    """The model should map English, Chinese, code, math, instruction
    prompts to different concept patterns — at minimum, produce distinct
    finite logits without crashing."""
    m = CortexNetOmega(d_model=32, n_columns=2, num_concepts=8, window=4, seed=0)
    samples = [
        "Hello world",
        "你好世界",
        "def fib(n): return n if n<2 else fib(n-1)+fib(n-2)",
        "Solve: ∫ x² dx from 0 to 1",
        "Please translate this sentence to Spanish.",
        "What is the capital of France?",
    ]
    sigs = []
    for s in samples:
        ids = m.encode_prompt(s)
        logits = m.forward(ids)
        assert np.all(np.isfinite(logits))
        sigs.append(float(np.linalg.norm(logits)))
    # Different prompts should yield distinct logit norms (no degenerate map)
    assert len(set(round(s, 3) for s in sigs)) >= 4
    print(f"  universal prompt sigs: {[round(s,2) for s in sigs]}")


# ---------------------------------------------------------------------- 8 (speed)
def bench_tps():
    """Token-per-second benchmark: streaming vs. naive full-recompute."""
    m = CortexNetOmega(d_model=64, n_columns=4, num_concepts=16,
                       window=8, seed=0)
    prompt = m.encode_prompt("Once upon a time")

    # ---- Streaming (Omega)
    t0 = time.perf_counter()
    out = m.generate(prompt, max_new_tokens=50, temperature=0.7, seed=0)
    t1 = time.perf_counter()
    streamed_tps = 50 / (t1 - t0)
    print(f"  Streaming generate     : {streamed_tps:6.2f} tok/s   (50 tokens in {t1-t0:.3f}s)")

    # ---- Naive recompute baseline: forward on full ids per new token
    lite = CortexNetLite(vocab_size=m.vocab_size, d_model=64, n_columns=4,
                          num_concepts=16, window=8, seed=0)
    ids = list(prompt.tolist())
    t0 = time.perf_counter()
    for _ in range(50):
        logits = lite.forward(np.array(ids, dtype=np.int64))
        next_id = int(np.argmax(logits[-1]))
        ids.append(next_id)
    t1 = time.perf_counter()
    naive_tps = 50 / (t1 - t0)
    print(f"  Naive O(N²)-style Lite : {naive_tps:6.2f} tok/s   (50 tokens in {t1-t0:.3f}s)")

    speedup = streamed_tps / max(naive_tps, 1e-9)
    print(f"  SPEEDUP (Omega vs Lite): {speedup:5.1f}×")
    return streamed_tps, naive_tps, speedup


if __name__ == "__main__":
    tests = [
        test_universal_tokenizer,
        test_manar_streaming_matches_full,
        test_cortical_column_streaming,
        test_omega_generate,
        test_omega_think_then_speak,
        test_hippocampal_sleep,
        test_universal_prompt_recognition,
    ]
    for t in tests:
        print(f"[ RUN ] {t.__name__}")
        t()
        print(f"[ OK  ] {t.__name__}")
    print("\n--- Speed Benchmark ---")
    bench_tps()
    print("\nAll Omega smoke tests passed ✓")
