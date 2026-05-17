---
name: project-pulse-phase1
description: PULSE Phase 1 implementation status — 8x8 GridWorld with PPO agent
metadata:
  type: project
---

Phase 1 of PULSE is implemented and SB3-compatible. Three files created:

- `pulse_env.py` — custom `PulseGridWorld(gym.Env)`, 8x8 grid, agent (0,0)→goal (7,7), traps at (3,3),(4,5),(2,6), rewards +10/-10/-0.1
- `train_phase1.py` — PPO (stable-baselines3), 50k timesteps, saves to `models/`, TensorBoard logs to `logs/`
- `visualise_phase1.py` — loads best checkpoint, runs one greedy episode, plots path with matplotlib

**Why:** Phase 1 establishes the simplest possible RL baseline before adding pain signals, partial observability, or richer state in later phases.

**How to apply:** When Phase 3 is discussed, build on PulseGridWorld by subclassing or wrapping it rather than rewriting from scratch. The env already supports extensible reward shaping and step info dicts.

## Phase 2 — Vector Slab (complete)

- `slab.py` — `SlabNetwork(nn.Module)`: 8×8×16 `nn.Parameter`, methods: `deform()`, `deformation_depth()` (L2 norm), `reset_cell()`, `elastic_recovery()`, `pain_map()`, `summary()`
- `visualise_slab.py` — simulates trap damage, renders white→red heatmap before and after elastic recovery, demonstrates `reset_cell`
- Key design choices: L2 norm (not L1/Lmax) as pain score; elastic (decay=0.99) vs plastic (no decay) memory; vector_dim=16 as middle ground between expressiveness and interpretability
- Phase 3 goal: couple the Slab to the live PPO agent — deform on each trap visit, feed pain_map as auxiliary observation channel
