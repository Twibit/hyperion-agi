#!/usr/bin/env python3
"""
preregistration_harness.py — Benchmark suite for sensorimotor round-trip latency.

Measures latency profiles across all active biophysical layers and checks for 
mathematical stability (NaN/Inf overflows) during continuous execution.
"""
import os
import sys
import time
import resource
import numpy as np

# Ensure root paths are registered
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from embodied_agent import AuroraOmnia, AuroraOmniaConfig
from embodied_agent.organism import Episode

class ProfiledAuroraOmnia(AuroraOmnia):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.telemetry = {
            "sensory_encoding": [],
            "thalamic_routing": [],
            "cortex_forward": [],
            "cerebral_step": [],
            "workspace_integration": [],
            "action_selection": [],
            "learning_telemetry": [],
            "total_round_trip": []
        }

    def step(self, observations: dict, reward: float = 0.0, done: bool = False) -> dict:
        t_start = time.perf_counter()
        
        # 0) Interoception
        obs = dict(observations)
        obs.setdefault("interoception", self.nocic.snapshot())
        if (self.somatic is not None
                and getattr(self.cfg, "somatosensory_inject", False)
                and self.hvsa is not None
                and "somatic" not in obs):
            obs["somatic"] = self.somatic.readout()

        symbiosis_stats = None
        if self.symbiosis is not None and self.endocrine is not None:
            alignment = float(np.clip(reward, -1.0, 1.0))
            symbiosis_stats = self.symbiosis.process_interaction(
                user_alignment_score=alignment,
                nociceptor=self.nocic,
                endocrine=self.endocrine,
            )
            reward += float(np.clip(symbiosis_stats["symbiotic_reward"], -0.5, 0.5))

        # 1) Encode (Sensory Encoding)
        t0 = time.perf_counter()
        candidates = {m: self.perceive(v, m) for m, v in obs.items()}
        t_enc = time.perf_counter() - t0
        self.telemetry["sensory_encoding"].append(t_enc)

        # 2) Thalamic broadcast (Thalamic Routing)
        t0 = time.perf_counter()
        if hasattr(self.cortex, "precision_profile"):
            prof = self.cortex.precision_profile()
            for s in candidates:
                self.thalamus.set_precision(s, float(np.mean(prof)))
        z_in, attn = self.thalamus.broadcast(
            candidates,
            acetylcholine=self.modulators.acetylcholine,
            norepinephrine=self.modulators.norepinephrine,
        )
        t_thal = time.perf_counter() - t0
        self.telemetry["thalamic_routing"].append(t_thal)

        # 3) Cortex forward (Cortex Forward)
        t0 = time.perf_counter()
        lr_base = self.modulators.effective_lr(self.cfg.lr_cortex, cortisol=self.nocic.state.cortisol)
        lr = lr_base
        if self.astro is not None:
            g = self.astro.gain(0)
            lr = lr_base * g
        z_top, sensory_err = self.cortex.forward(z_in, lr=lr)
        
        if np.isnan(sensory_err) or np.isinf(sensory_err):
            sensory_err = 1.0
        z_top = np.nan_to_num(z_top, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self._ema_error = 0.99 * self._ema_error + 0.01 * sensory_err
        if self.astro is not None:
            self.astro.report_error(0, sensory_err)
        t_cortex = time.perf_counter() - t0
        self.telemetry["cortex_forward"].append(t_cortex)

        # 4) Cerebrum step (Cerebral Step)
        t0 = time.perf_counter()
        anatomy_states = None
        if self.cerebrum is not None and (self._step % self.cfg.anatomy_step_every == 0):
            neuromod_gains = None
            if self.cerebrum.neuromod_bus is not None:
                neuromod_gains = self.cerebrum.neuromod_bus.gains(
                    self.modulators,
                    cortisol=self.nocic.state.cortisol,
                )
            priorities = None
            if self.cfg.thalamic_router_enable:
                priorities = self._thalamic_priority_map(observations, attn)
            anatomy_states = self.cerebrum.step(
                z_top, lr=lr * 0.5,
                modulator_gains=neuromod_gains,
                priorities=priorities,
            )
            
            if self._ema_error > 0.5 or self.nocic.state.cortisol > 0.5:
                thinking_steps = min(10, int((self._ema_error + self.nocic.state.cortisol) * 5))
                for _ in range(thinking_steps):
                    inner_z = z_top
                    if self.workspace is not None:
                        precs = self.cerebrum.regional_precisions()
                        pairs = list(zip(list(anatomy_states.values()), precs))
                        active_pairs = [(st, pr) for st, pr in pairs if float(pr) > 0.05]
                        states_in = [inner_z] + [st for st, pr in active_pairs]
                        precin = [1.0] + [float(pr) for st, pr in active_pairs]
                        inner_z = self.workspace.integrate(states_in, precin)
                    
                    anatomy_states = self.cerebrum.step(
                        inner_z, lr=lr * 0.1,
                        modulator_gains=neuromod_gains,
                        priorities=priorities,
                    )
                z_top = inner_z
                
            self._last_anatomy_step = self._step
        t_cerebrum = time.perf_counter() - t0
        self.telemetry["cerebral_step"].append(t_cerebrum)

        # 5) Global workspace integration (Workspace Integration)
        t0 = time.perf_counter()
        workspace_state = None
        if self.workspace is not None:
            states_in = [z_top]
            precisions = [1.0]
            if anatomy_states is not None:
                precs = self.cerebrum.regional_precisions()
                pairs = list(zip(list(anatomy_states.values()), precs))
                active_pairs = [(st, pr) for st, pr in pairs if float(pr) > 0.05]
                for st, pr in active_pairs:
                    states_in.append(st)
                    precisions.append(float(pr))
            workspace_state = self.workspace.integrate(states_in, precisions)
            z_top = (0.7 * z_top + 0.3 * workspace_state).astype(np.float32)
            n = np.linalg.norm(z_top) + 1e-9
            z_top = z_top / n
        t_workspace = time.perf_counter() - t0
        self.telemetry["workspace_integration"].append(t_workspace)

        # Stigmergy & SDM
        if self.stig is not None:
            self.stig.deposit_step(z_top)
        if self.sdm is not None and self._prev_z is not None:
            self.sdm.write(self._prev_z, z_top)

        # Alignment imagine
        align_penalty_vec = None
        if self.cfg.safety_enable:
            imag = self.world_model.rollout(
                z_top,
                policy_fn=lambda z: self.basal.policy(z, temp=1.5)[0],
                horizon=min(4, self.cfg.imagination_horizon),
                uncertainty_stop=0.6,
            )
            pen = self.alignment.penalty(imag)
            if pen > 0.0:
                align_penalty_vec = np.full(self.cfg.n_actions, pen, dtype=np.float32)

        # 6) Action selection (Action Selection)
        t0 = time.perf_counter()
        temp = self.modulators.exploration_temp(self.cfg.action_temperature)
        habit_action, action_probs = self.basal.policy(z_top, temp=temp,
                                                        alignment_penalty=align_penalty_vec)
        _zp, _rp, uncertainty_now = self.world_model.forward(z_top, habit_action)

        action = habit_action
        action_mode = "habit"
        if self.ai_selector is not None:
            action, action_mode = self.ai_selector.select(
                z_top, habit_action, action_probs, float(uncertainty_now),
                world_forward=self.world_model.forward,
                value_fn=self.basal.value,
                cortisol=self.nocic.state.cortisol)
        t_action = time.perf_counter() - t0
        self.telemetry["action_selection"].append(t_action)

        # 7) Learning & Telemetry (Learning & Telemetry)
        t0 = time.perf_counter()
        z_pred, r_pred, uncertainty = self.world_model.forward(z_top, action)

        cb_err = 0.0
        if self.cerebellum_module is not None and self._prev_z is not None and self._prev_a is not None:
            cb_err = self.cerebellum_module.learn(self._prev_z, self._prev_a, z_top)

        td_err = 0.0
        if self._prev_z is not None:
            noc = self.nocic.ingest(reward)
            lr_world_eff = self.cfg.lr_world * noc["lr_scale"]
            self.world_model.learn(self._prev_z, self._prev_a,
                                   z_top, noc["pain"], lr=lr_world_eff)
            td_err = self.basal.learn_td(self._prev_z, self._prev_a, noc["pain"], z_top,
                                         done=done,
                                         dopamine_gain=self.modulators.td_gain(),
                                         discount=self.modulators.discount())
            if self.elig_critic is not None:
                self.elig_critic.step(z_top, self._prev_z)
            if self.ewc is not None and self.bfc is not None:
                grad_like = td_err * self._prev_z
                self.ewc.tag(grad_like, reward_burst=max(0.0, td_err))
                self.bfc.inject(0.01 * grad_like)
                self.bfc.relax()

            self.hippocampus.store(Episode(
                z=self._prev_z, a=self._prev_a, r=noc["pain"], z_next=z_top,
                mod=self.modulators.snapshot(), priority=abs(td_err) + 1e-3,
                tag=",".join(observations.keys()),
                quarantine=noc["quarantine"],
                affect=self.nocic.snapshot(),
            ))

        progress = self._last_ema - self._ema_error
        self._last_ema = self._ema_error
        self.modulators.update(
            reward_prediction_error=td_err,
            sensory_surprise=sensory_err,
            task_progress=progress,
            attention_load=max(attn.values()) if attn else 0.0,
        )

        if attn:
            for src, w in attn.items():
                self.thalamus.reinforce(src, signal=w * td_err, lr=0.001)

        if "text" in observations:
            for t in str(observations["text"]).split():
                self.text_decoder.observe(t, z_top)

        introspect = {"distill": False, "revert": False}
        if self.intro is not None:
            introspect = self.intro.observe(sensory_err)

        if self.neuro is not None:
            self.neuro.report_usage([action % self.neuro.n], [abs(td_err) + 1e-3])
            self.neuro.maybe_grow()

        self._prev_z, self._prev_a = z_top, action
        self._step += 1

        v9_out = self._v9_step(
            z_top=z_top,
            action_probs=action_probs,
            uncertainty=float(uncertainty),
            sensory_err=float(sensory_err),
            td_err=float(td_err),
            attn=attn,
        )

        v10_out = self._v10_step(
            z_top=z_top,
            action_probs=action_probs,
            sensory_err=float(sensory_err),
            td_err=float(td_err),
            uncertainty=float(uncertainty),
            epistemic_mode=(v9_out or {}).get("epistemic", {}).get("mode") if v9_out else None,
            cortisol=float(self.nocic.state.cortisol),
        )

        v11_out = self._v11_step(z_top=z_top)

        do_sleep = (self._step % self.cfg.consolidation_every == 0) or self.homeo.request_sleep()
        if do_sleep:
            self.sleep()

        self.curriculum.adapt(self._ema_error)
        t_learn = time.perf_counter() - t0
        self.telemetry["learning_telemetry"].append(t_learn)

        # Full round trip
        t_total = time.perf_counter() - t_start
        self.telemetry["total_round_trip"].append(t_total)

        # Re-pack and return
        out = dict(
            step=self._step, action=action, value=self.basal.value(z_top),
            sensory_err=sensory_err, td_err=td_err, uncertainty=uncertainty,
            attn=attn,
            modulators=self.modulators.snapshot(),
            affect=self.nocic.snapshot(),
            z_top=z_top,
            action_mode=action_mode,
            cerebellum_err=cb_err,
            introspection=introspect,
            workspace_coherence=(self.workspace.coherence
                                 if self.workspace is not None else 0.0),
            anatomy_active=(anatomy_states is not None),
        )
        if self.cerebrum is not None:
            out["interhem_coherence"] = float(self.cerebrum._interhem_coherence)
            if self.cerebrum.metabolic is not None:
                out["metabolic_active_frac"] = float(
                    self.cerebrum.metabolic._last_used_frac)
        if symbiosis_stats is not None:
            out["symbiosis"] = symbiosis_stats
        if v9_out is not None:
            out.update(v9_out)
        if v10_out is not None:
            out.update(v10_out)
        if v11_out is not None:
            out.update(v11_out)
        return out


def run_benchmark():
    print("\n" + "="*70)
    print("  HYPERION PREREGISTRATION HARNESS — SENSORIMOTOR LATENCY SUITE  ")
    print("="*70)
    
    # 1. Setup the config matching cmd_aurora
    cfg = AuroraOmniaConfig(
        tier="base",
        log_every=999,
        consolidation_every=999,
        anatomy_enable=True,
        neuromod_bus_enable=True,
        metabolic_budget_enable=True,
        thalamic_router_enable=True,
    )
    
    print("[Harness] Booting AGI core...")
    org = ProfiledAuroraOmnia(cfg)
    
    # Pre-warm to load all modules cleanly
    org.step({"text": "Hello world", "reward": 0.0})
    
    # 2. Structured inputs simulating complex cognitive tasks
    test_prompts = [
        "analyze codebase and locate memory leaks",
        "predict stock market price movements",
        "ls -la",
        "fix syntax error in rust backend shim",
        "echo 'Reflex testing'",
        "calculate recursive fibonacci sequence",
        "write high-performance memory pager",
        "search wiktionary definition of homeostasis",
        "simulate Hodgkin-Huxley membrane dynamics",
        "evaluate threat landscape for cognitive network",
        "coordinate motor cortex command sequences",
        "trigger microglial synaptic pruning cycle",
        "execute sensory-motor integration loop",
        "optimize predictive coding block learning rate",
        "evaluate cortisol neuromodulation level",
        "measure interhemispheric callosal coherence",
        "read proprioceptive joint angle feedback",
        "balance default mode network daydream weight",
        "re-route thalamic reticular attention map",
        "crystallize permanent synaptic memory state"
    ]
    
    print(f"[Harness] Running {len(test_prompts)} trials in sensorimotor loop...")
    for idx, prompt in enumerate(test_prompts):
        reward = 0.5 - abs((idx % 10) - 5) / 10.0
        org.step({"text": prompt, "reward": reward})
    
    # 3. Compile statistics
    print("\n" + "-"*70)
    print("  LATENCY TELEMETRY REPORT (in milliseconds)")
    print("-"*70)
    print(f"{'Phase / Layer':<26} | {'Min':<8} | {'Mean':<8} | {'Median':<8} | {'Max':<8} | {'P95':<8}")
    print("-"*70)
    
    total_hz_sum = 0.0
    for phase, values in org.telemetry.items():
        ms_vals = np.array(values) * 1000.0
        p_min = ms_vals.min()
        p_mean = ms_vals.mean()
        p_med = np.median(ms_vals)
        p_max = ms_vals.max()
        p_95 = np.percentile(ms_vals, 95)
        
        name = phase.replace("_", " ").title()
        print(f"{name:<26} | {p_min:8.2f} | {p_mean:8.2f} | {p_med:8.2f} | {p_max:8.2f} | {p_95:8.2f}")
        
        if phase == "total_round_trip":
            total_hz_sum = 1.0 / (p_mean / 1000.0)
            
    print("-"*70)
    
    # 4. Check for mathematical integrity (NaN/Inf checks)
    nan_count = 0
    inf_count = 0
    
    # Check Cortex layers
    if hasattr(org.cortex, "layers"):
        for i, layer in enumerate(org.cortex.layers):
            if hasattr(layer, "experts"):
                for exp_idx, exp in enumerate(layer.experts):
                    if exp.W_gen is not None:
                        if np.isnan(exp.W_gen).any() or np.isnan(exp.W_rec).any():
                            nan_count += 1
                        if np.isinf(exp.W_gen).any() or np.isinf(exp.W_rec).any():
                            inf_count += 1
            elif hasattr(layer, "W_gen") and layer.W_gen is not None:
                if np.isnan(layer.W_gen).any() or np.isnan(layer.W_rec).any():
                    nan_count += 1
                if np.isinf(layer.W_gen).any() or np.isinf(layer.W_rec).any():
                    inf_count += 1

    # Check Substrate active neurons
    for nid, n in org.cerebrum.biological_substrate.active_biology.items():
        v, spiked = n.izhikevich.step(1.0, 0.0)
        if np.isnan(v):
            nan_count += 1
        if np.isinf(v):
            inf_count += 1

    print("\n" + "-"*70)
    print("  SYSTEM HEALTH & PERFORMANCE ")
    print("-"*70)
    
    ram_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    
    print(f"  Throughput      : {total_hz_sum:.2f} Hz (steps/second)")
    print(f"  Peak RAM (RSS)  : {ram_rss:.2f} MB")
    
    health_status = "HEALTHY (Pass)" if (nan_count == 0 and inf_count == 0) else "COMPROMISED (Fail)"
    print(f"  Math Integrity  : {health_status} (NaNs={nan_count}, Infs={inf_count})")
    print("="*70 + "\n")
    
    assert nan_count == 0 and inf_count == 0, "Benchmark failed: Math integrity compromised!"
    print("[Harness] Preregistration Bench completed successfully.")

if __name__ == "__main__":
    run_benchmark()
