"""
Sleep cycle (v3).

Differences from v2:
  * Pulls trauma-quarantined episodes ONLY when cortisol is below a
    threshold (graduated-exposure therapy mode).
  * Stops imagined rollouts as soon as ensemble uncertainty exceeds a
    safety threshold (avoid training on hallucinated nonsense).
  * Reports a safety_report dict so the organism can log how often
    these guards fired.
"""
from __future__ import annotations
import numpy as np


def sleep_cycle(hippocampus, cortex, world_model, basal,
                modulators, nociceptor, rumination_guard,
                cfg, policy_fn, n_steps: int):
    stats = dict(real_replay_loss=0.0, cortex_recon_loss=0.0,
                 dream_value_update=0.0, n_real=0, n_dream=0,
                 trauma_replayed=0, dream_aborted_uncertain=0,
                 ruminations_skipped=0)
    if n_steps <= 0: return stats

    # Therapy mode: include quarantined episodes only when stress is low.
    therapy = nociceptor.state.cortisol < 0.3

    for _ in range(n_steps):
        if therapy:
            batch = hippocampus.replay_in_therapy_mode(cfg.replay_batch)
        else:
            batch = hippocampus.sample_replay(cfg.replay_batch)
        if not batch: break

        before = len(batch)
        batch = rumination_guard.filter(batch)
        stats["ruminations_skipped"] += before - len(batch)
        if not batch: continue

        new_tds = []
        for ep in batch:
            if ep.quarantine: stats["trauma_replayed"] += 1
            # Train world model on the real transition.
            wm_loss = world_model.learn(ep.z, ep.a, ep.z_next, ep.r,
                                        lr=cfg.lr_world)
            stats["real_replay_loss"] += wm_loss
            stats["n_real"] += 1

            _, recon = cortex.forward(ep.z_next,
                                      lr=modulators.effective_lr(cfg.lr_cortex,
                                                                 cortisol=nociceptor.state.cortisol))
            stats["cortex_recon_loss"] += recon

            # Imagined rollout WITH uncertainty stop.
            uncertainty_stop = 0.5
            traj = world_model.rollout(
                ep.z_next, policy_fn,
                horizon=cfg.imagination_horizon,
                uncertainty_stop=uncertainty_stop,
            )
            if len(traj) < cfg.imagination_horizon:
                stats["dream_aborted_uncertain"] += 1

            for step in traj:
                # EnsembleWorldModel rollout returns (z, a, r, z_next, u)
                zd, ad, rd, zd_next = step[0], step[1], step[2], step[3]
                _, _ = cortex.forward(zd_next,
                                      lr=0.5 * modulators.effective_lr(cfg.lr_cortex,
                                                                       cortisol=nociceptor.state.cortisol))
                td = basal.learn_td(zd, ad, rd, zd_next, done=False,
                                    dopamine_gain=modulators.td_gain(),
                                    discount=modulators.discount())
                stats["dream_value_update"] += abs(td)
                stats["n_dream"] += 1

            td_real = basal.learn_td(ep.z, ep.a, ep.r, ep.z_next, done=False,
                                     dopamine_gain=modulators.td_gain(),
                                     discount=modulators.discount())
            new_tds.append(td_real)

        hippocampus.update_priorities(batch, new_tds)

    for k in ("real_replay_loss", "cortex_recon_loss"):
        if stats["n_real"]: stats[k] /= stats["n_real"]
    if stats["n_dream"]: stats["dream_value_update"] /= stats["n_dream"]
    return stats
