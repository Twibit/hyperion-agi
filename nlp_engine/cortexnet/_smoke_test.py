"""Quick smoke test for the CortexNet patch. Run from repo root:

    python3 -m nlp_engine.cortexnet._smoke_test
"""
import numpy as np

from nlp_engine.cortexnet import (
    DendriticNeuron, MANARAttention, CfCLayer, FTPTrainer,
    ComplementPruner, SurrogateSpikeLayer, CorticalColumn, CortexNetLite,
)


def test_dendritic_neuron():
    dn = DendriticNeuron(8, 16, seed=0)
    y = dn.forward(np.random.randn(20, 8))
    assert y.shape == (20, 16)
    # α_slow should be initialised higher than α_fast (biological prior)
    assert float(dn.alpha_slow.mean()) > float(dn.alpha_fast.mean())


def test_manar():
    a = MANARAttention(d_model=16, num_concepts=8, window=4, seed=0)
    y = a.forward(np.random.randn(40, 16))
    assert y.shape == (40, 16)


def test_cfc():
    c = CfCLayer(8, 16, seed=0)
    y = c.forward(np.random.randn(30, 8))
    assert y.shape == (30, 16)


def test_ftp_learns_xor():
    rng = np.random.default_rng(0)
    X = rng.choice([-1.0, 1.0], size=(256, 2))
    Y = ((X[:, 0] * X[:, 1]) > 0).astype(np.float32).reshape(-1, 1)

    ftp = FTPTrainer([2, 32, 32, 1], activations=["tanh", "tanh", "id"], seed=0)
    losses = []
    for epoch in range(400):
        loss = ftp.train_step(X, Y, lr=2e-2, eta=1.0)
        losses.append(loss)
    final = float(np.mean(losses[-20:]))
    initial = float(np.mean(losses[:20]))
    assert final < initial * 0.8, f"FTP did not decrease loss enough: {initial:.4f} -> {final:.4f}"
    print(f"  FTP XOR pred-MSE: {initial:.4f} -> {final:.4f}")


def test_complement_pruner():
    W = np.random.randn(64, 64) * 0.1
    p = ComplementPruner(c1q_threshold=0.05, c3_threshold=0.2)
    for _ in range(40):
        p.observe(W)
        W, _ = p.step(W)
    s = p.sparsity(W)
    assert s > 0.05, f"pruner failed to produce sparsity, got {s}"
    print(f"  Pruner sparsity reached: {s:.2%}")


def test_surrogate_spike():
    s = SurrogateSpikeLayer(8, 16, seed=0)
    spk, sg = s.forward(np.random.randn(50, 8) * 3.0)
    assert spk.shape == (50, 16) and sg.shape == (50, 16)


def test_cortical_column():
    col = CorticalColumn(d_model=16, num_concepts=8, window=4, seed=0)
    y = col.forward(np.random.randn(20, 16))
    assert y.shape == (20, 16)


def test_cortexnet_lite():
    m = CortexNetLite(vocab_size=50, d_model=16, n_columns=2,
                      num_concepts=8, window=4, seed=0)
    ids = np.random.randint(0, 50, size=20)
    logits = m.forward(ids)
    assert logits.shape == (20, 50)
    probs = m.predict(ids)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
    m.prune_step()
    print(f"  Model info: {m.info()}")


if __name__ == "__main__":
    tests = [
        test_dendritic_neuron, test_manar, test_cfc,
        test_ftp_learns_xor, test_complement_pruner, test_surrogate_spike,
        test_cortical_column, test_cortexnet_lite,
    ]
    for t in tests:
        print(f"[ RUN ] {t.__name__}")
        t()
        print(f"[ OK  ] {t.__name__}")
    print("\nAll CortexNet smoke tests passed ✓")
