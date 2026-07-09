#!/usr/bin/env python3
"""
cli.py — single coherent entry point for the Hyperion codebase.

The archive shipped three loosely-connected personalities:

  * dual_planner.py           — dual-brain (right = generative noise, left =
                            CAIN Markov blanket + tree search) toy.
   * nlp_engine/layers/cain_brain.py + main.py — chat loop driven by the PredictiveDecoder (no N-gram)
                            language model, a Hebbian embedding cortex,
                            an HTM predictor and a SQLite episodic memory.
  * embodied_agent/         — the most serious subsystem: an embodied loop
                            with predictive-coding cortex stacks, global
                            workspace, cerebellum, hippocampus, active
                            inference, thalamic routing, etc.

`cli` exposes all three behind one CLI so a first-time
reader can actually see what the project does without hunting for the
right script and the right combination of flags.

Usage
-----
    python3 cli.py --help
    python3 cli.py dual-brain
    python3 cli.py aurora     --steps 20
    python3 cli.py inspect
    python3 cli.py chat        # if data/ exists
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# ---------------------------------------------------------------------------
def cmd_dual_brain(args):
    """Runs the dual_planner.py Right/Left brain thought resolution."""
    import numpy as np
    from dual_planner import HyperionOrganism

    dim = int(args.dim)
    n_proposals = int(args.proposals)

    print(f"[unified] Booting dual-brain organism (dim={dim}, proposals={n_proposals})")
    org = HyperionOrganism(dim=dim)
    np.random.seed(int(args.seed))
    truth = np.random.randn(dim)
    org.process_thought(target_concept=truth, n_proposals=n_proposals)


# ---------------------------------------------------------------------------
def cmd_aurora(args):
    """Runs the embodied_agent embodied loop."""
    from embodied_agent import AuroraOmnia, AuroraOmniaConfig
    from embodied_agent.cortex.snn_engine import USING_RUST_CORE

    print(f"[unified] Aurora-Omnia embodied loop (rust_backend={USING_RUST_CORE})")
    cfg = AuroraOmniaConfig(
        tier=args.tier,
        log_every=max(1, int(args.steps) // 5),
        consolidation_every=999,
        anatomy_enable=True,
        neuromod_bus_enable=True,
        metabolic_budget_enable=True,
        thalamic_router_enable=True,
    )
    org = AuroraOmnia(cfg)

    modalities = ["text", "vision", "proprio", "reward"]
    t0 = time.time()
    for t in range(int(args.steps)):
        # Rotate modality so the thalamic router has something to do.
        m = modalities[t % len(modalities)]
        obs = {m: f"observation-{t}", "reward": 0.5 - abs((t % 20) - 10) / 20.0}
        r = org.step(obs)
        if (t + 1) % max(1, int(args.steps) // 5) == 0:
            keys = ", ".join(sorted(r.keys())[:6]) + "..."
            affect = r.get("affect", 0.0)
            if isinstance(affect, dict):
                affect_str = ", ".join(f"{k}={v:.2f}"
                                        for k, v in list(affect.items())[:2]
                                        if isinstance(v, (int, float)))
            else:
                affect_str = f"{float(affect):.3f}"
            print(f"  step {t+1:>4d} | keys=[{keys}] | affect=({affect_str})")
    dt = time.time() - t0
    print(f"[unified] {args.steps} steps in {dt:.2f}s "
          f"({args.steps / max(dt, 1e-6):.1f} steps/s).")


# ---------------------------------------------------------------------------
def cmd_chat(args):
    """Boots the CainBrain conversational interface (needs data/)."""
    data_dir = os.path.join(HERE, "data")
    if not os.path.isdir(data_dir):
        print(f"[unified] chat mode requires the ./data/ directory (SQLite memories, "
              f"pickled weights). Not found at {data_dir}. Skipping.")
        return
    from main import boot_interface
    boot_interface()


# ---------------------------------------------------------------------------
def cmd_inspect(_args):
    """Runs a quick self-test and prints what actually works in this checkout."""
    report = []

    def check(name, fn):
        try:
            fn()
            report.append((name, "OK", ""))
        except Exception as e:
            report.append((name, "FAIL", f"{type(e).__name__}: {e}"))

    # 1) dual_planner.py dual brain
    def _try_dual():
        import numpy as np
        from dual_planner import HyperionOrganism
        HyperionOrganism(dim=8).process_thought(np.zeros(8), n_proposals=100)
    check("dual_planner.py dual-brain", _try_dual)

    # 2) embodied_agent end-to-end
    def _try_aurora():
        from embodied_agent import AuroraOmnia, AuroraOmniaConfig
        org = AuroraOmnia(AuroraOmniaConfig(tier="tiny", log_every=999,
                                             consolidation_every=999))
        for _ in range(3):
            org.step({"text": "x"})
    check("embodied_agent embodied loop", _try_aurora)

    # 3) SNN memristor layer (rust or fallback)
    def _try_snn():
        import numpy as np
        from embodied_agent.cortex.snn_engine import MemristorLayer, USING_RUST_CORE
        layer = MemristorLayer(16, 4, threshold=2.0, leak=0.8)
        for _ in range(10):
            layer.forward((np.random.rand(16) > 0.5).astype("uint8"))
        return USING_RUST_CORE
    check("SNN MemristorLayer (rust or numpy fallback)", _try_snn)

    # 4) HTM sequence memory
    def _try_htm():
        import numpy as np
        from embodied_agent.praxis.htm_predictor import HTMSequenceMemory
        htm = HTMSequenceMemory(latent_dim=64)
        for _ in range(5):
            sdr = np.zeros(64, dtype=np.uint8)
            sdr[np.random.choice(64, 3, replace=False)] = 1
            htm.step(sdr)
    check("HTM sequence memory", _try_htm)

    # 5) CainBrain (large; may fail on missing rust_backend_shim PyO3 module)
    def _try_cain():
        from nlp_engine.layers.cain_brain import CainBrain
        CainBrain()
    check("CainBrain (chat)", _try_cain)

    print("\n" + "=" * 70)
    print(" Hyperion self-test report")
    print("=" * 70)
    width = max(len(n) for n, _, _ in report)
    for name, status, detail in report:
        pad = " " * (width - len(name))
        line = f"  {name}{pad}  {status}"
        print(line)
        if detail:
            print(f"      -> {detail}")

    # Print concise honest verdict
    print()
    print(textwrap.dedent("""\
        Verdict
        -------
        Hyperion is a rich bio-inspired research substrate. It is NOT an
        AGI: nothing in this checkout can pass an open-ended reasoning
        benchmark, hold a coherent conversation without heavy scripted
        scaffolding, or transfer skills across tasks. What it DOES have
        is a legitimate collection of components (predictive coding,
        STDP, global workspace, active inference, Hebbian embeddings,
        HTM sequence memory) wired together in a single Python package,
        which makes it a solid *substrate* for further research.
        See AGI_ASSESSMENT.md for the full write-up.
        """))


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("dual-brain", help="Run dual_planner.py's Right/Left brain thought.")
    sp.add_argument("--dim", default=16, type=int)
    sp.add_argument("--proposals", default=5000, type=int)
    sp.add_argument("--seed", default=42, type=int)
    sp.set_defaults(func=cmd_dual_brain)

    sp = sub.add_parser("aurora", help="Run the embodied_agent embodied loop.")
    sp.add_argument("--steps", default=50, type=int)
    sp.add_argument("--tier", default="tiny",
                    choices=("tiny", "small", "medium", "large"))
    sp.set_defaults(func=cmd_aurora)

    sp = sub.add_parser("chat", help="Boot the CainBrain conversational interface.")
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("inspect", help="Self-test every major subsystem and print a report.")
    sp.set_defaults(func=cmd_inspect)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
