---
name: project-pulse
description: PULSE project phases, architecture, and design decisions across all phases
metadata:
  type: project
---

PULSE (Pain, Understanding, and Learning State Engine) is an 8×8 RL grid world (Gymnasium) with a biologically-inspired pain system layered on top of standard RL.

**Phase 1** — `pulse_env.py`: PulseGridWorld, basic trap/goal environment, -10 trap, +10 goal, -0.1 step penalty.

**Phase 2** — `slab.py`: SlabNetwork (PyTorch nn.Module), 8×8×16 learnable parameter grid representing the agent's "skin." deform(), deformation_depth(), elastic_recovery().

**Phase 3** — `pulse_env.py` additions: apply_spike() bridges env→slab. Trap reward = -10 - pain_score (growing penalty). elastic_recovery() called at episode boundaries. pain_history list.

**Phase 4** — `resistance_field.py` + `visualise_resistance.py` + `pulse_env.py` updates:
- ResistanceField: 8×8 float grid. BASELINE=0.1, TRAP_RESISTANCE=0.7 at traps + all 8 neighbours.
- movement_cost(x,y) returns -resistance (negative reward).
- dynamic_resistance(slab, scale=0.5) raises resistance where slab deformation is high (feedback loop).
- step() initialises reward=resistance_cost, then all branches use +=.
- Trap neighbour reward: -0.7 (resistance) + -0.1 (step) = -0.8 (pre-entry dread signal).
- Trap entry reward: -0.7 (resistance) + -10.0 (base) + -pain_score = very negative.

**Why:** A comment-heavy codebase — every line explains WHY (physics vs. learning signal, pre-entry signal concept, feedback loop for convergence). Comments are a first-class deliverable for this project.

**Phase 5** — `pain_shaped_policy.py` + `train_phase5.py` + `visualise_aversion.py` + `compare_phase5.py`:
- PainShapedPolicy wraps a trained PPO model with an aversion filter at EVALUATION time (never during training).
- `aversion_score(x,y) = deformation_depth(x,y) × resistance(x,y)` — product of memory and physics.
- `should_avoid(x,y, threshold=0.5)` — True if score > threshold (requires ~18 trap hits before firing).
- `select_action(obs, candidate_actions)` — filters blocked actions, picks best PPO-preferred safe action; cornered fallback picks least-aversion action.
- train_phase5.py: same vanilla PPO training + post-training PULSE evaluation loop.
- visualise_aversion.py: 10 eval episodes, per-step console diagnostics + matplotlib path+blocked-arrow rendering.
- compare_phase5.py: 50 vanilla vs 50 PULSE episodes, 3-panel comparison plot (reward, length, trap/goal rates).

**Key design principles:**
1. Aversion filter at eval only — training stays on-policy vanilla PPO.
2. Pain as MEMORY (aversion) vs pain as REFLEX (hard trap lookup) — the product formula requires both evidence and physics.
3. No hard-blocking — cornered fallback always provides an action.
4. All files comment every line, especially WHY comments on mechanisms.

**Phase 6** — `benchmark.py` + `plot_benchmark.py` + `results_summary.py` + rewritten `README.md`:
- benchmark.py: 3 agents (PPO, DQN, PULSE-PPO) × 5 seeds × 50 checkpoints. Chunked training via model.learn(EVAL_EVERY, reset_num_timesteps=False). PULSE-PPO reads training_inner_env.slab (via DummyVecEnv.envs[0]) for eval aversion. Saves to benchmark_results.csv.
- plot_benchmark.py: 4-panel figure with ±1std shaded bands. smooth_series() for trend lines. Blue=PPO, Orange=DQN, Purple=PULSE-PPO.
- results_summary.py: Convergence timestamp (first 80% goal rate), final metrics table, Cohen's d effect sizes, honest interpretation for all 4 possible outcome scenarios.
- README.md: Complete rewrite covering all 6 phases, file table, how to read benchmark results, statistical caveats.

**Key benchmark design:** PULSE-PPO training = vanilla PPO (identical to baseline). Only eval differs (PainShapedPolicy filter). All three agents train on SAME PulseGridWorld (Phases 3-4 active). This isolates the aversion filter's contribution from the shaped reward's contribution.

**How to apply:** All phases build on each other. Never remove prior phase comments or logic. Label new additions with ### PHASE N ### / ### END PHASE N ### markers.
