# PULSE — Pain, Understanding & Learning State Engine

A research project exploring whether an artificial agent can develop something analogous to a **bodily sense of pain** — not as a simple penalty signal, but as a persistent, spatial, time-varying internal state that shapes how it learns and behaves.

Standard reinforcement learning agents receive a reward number and move on. They have no memory of *where* it hurt, no sense that some regions of the world feel dangerous in a way that lingers. PULSE asks: what if the agent had **skin**?

---

## Core Idea

Pain is modelled not as a scalar punishment, but as a **deformation in a learnable vector field** — a grid of high-dimensional vectors that accumulate damage from harmful experiences, fade slowly over time, and can be read out as a spatial pain map. This gives the agent a body that remembers.

---

## Project Structure

```
PULSE/
├── pulse_env.py          # Phase 1 — Custom Gymnasium environment
├── train_phase1.py       # Phase 1 — PPO training script
├── visualise_phase1.py   # Phase 1 — Agent path visualisation
├── slab.py               # Phase 2 — Vector Slab (the agent's skin)
├── visualise_slab.py     # Phase 2 — Pain heatmap visualisation
├── models/               # Saved model checkpoints and output figures
└── logs/                 # TensorBoard training logs
```

---

## Phases

### Phase 1 — The World
**Status: Complete**

Built the foundation: a custom 8×8 gridworld as a [Gymnasium](https://gymnasium.farama.org/) environment, and trained a PPO agent to navigate it.

**Environment (`pulse_env.py`)**
- 8×8 discrete grid. Agent starts at `(0, 0)`, goal at `(7, 7)`.
- Three hardcoded trap cells at `(3,3)`, `(4,5)`, `(2,6)`.
- Four actions: up, down, left, right. Wall collisions keep the agent in place.
- Rewards: `+10` for reaching the goal, `-10` for entering a trap, `-0.1` per step (efficiency pressure).
- Episode terminates on goal or trap. No time limit.
- Observation: agent's `(row, col)` position as a `float32` vector.
- Fully compatible with [stable-baselines3](https://stable-baselines3.readthedocs.io/) (`check_env` validated).

**Training (`train_phase1.py`)**
- Algorithm: PPO (`MlpPolicy`) — chosen for stability on small discrete environments and as a foundation for future phases that will require an actor-critic structure.
- 50,000 timesteps. `EvalCallback` saves the best checkpoint every 2,000 steps.
- TensorBoard logging enabled. Run `tensorboard --logdir logs/` to inspect reward curves.

**Visualisation (`visualise_phase1.py`)**
- Loads the best saved checkpoint.
- Runs one greedy (deterministic) episode and records the agent's path.
- Renders the path as a matplotlib figure: traps in red, goal in green, path in blue, step numbers annotated on each cell.
- Saves output to `models/pulse_phase1_path.png`.

**To run Phase 1:**
```bash
python train_phase1.py
python visualise_phase1.py
```

---

### Phase 2 — The Skin (Vector Slab)
**Status: Complete**

Introduced the **Slab**: a learnable vector field that sits alongside the environment as the agent's body surface. Every cell in the grid has a 16-dimensional vector. Harmful experiences deform these vectors; their magnitudes are the pain scores.

**Slab Network (`slab.py`)**

The `SlabNetwork` is a `torch.nn.Module` with shape `(8, 8, 16)` — one 16-dimensional vector per grid cell.

| Method | What it does |
|---|---|
| `get_vector(x, y)` | Returns the 16-dim pain vector at cell `(x, y)` |
| `deform(x, y, force_vector)` | Adds `lr × force_vector` to the cell's vector — simulates a painful stimulus |
| `deformation_depth(x, y)` | Returns the **L2 norm** of the cell's vector — the pain score |
| `reset_cell(x, y)` | Zeroes out a cell — simulates complete local healing |
| `elastic_recovery(decay=0.99)` | Multiplies all vectors by `decay` — simulates gradual healing over time |
| `pain_map()` | Returns an `(8, 8)` numpy array of L2 pain scores across the whole grid |

**Key design decisions:**

- **L2 norm as pain score** — measures total vector displacement from the resting state (zero). Holistic: grows with both the magnitude and spread of activation across all 16 dimensions. L1 would smear signal equally regardless of which channels activate; L∞ reads only the loudest channel; the mean shrinks as dimensionality grows. L2 is the natural measure of "how far has this vector moved?"

- **vector_dim = 16** — a deliberate middle ground. Too small (e.g. 2) and the vector can only encode pain intensity with no room for texture. Too large (e.g. 256) and most dimensions never receive enough stimuli to carry meaningful signal. At 16, each cell has 16 distinct pain "channels" — enough for future phases to encode directionality, recency, proximity, and intensity separately.

- **Elastic vs plastic memory** — `elastic_recovery()` multiplies all vectors by a decay factor (`0.99` by default), so pain fades gradually when stimuli stop. This is elastic memory: the skin returns toward its resting state over time, like rubber. Plastic memory (no decay) would make deformations permanent, like a dent in clay. PULSE uses elastic memory by default so the agent's pain state reflects *recent* experience, not its entire history. At `decay=0.99`, a cell retains ~37% of its pain after 100 timesteps with no new stimuli — the ranking of damaged cells is preserved, but absolute values fade.

- **`nn.Module` base class** — gives `state_dict()` save/load, automatic parameter discovery for gradient-based optimisers, and `model.to(device)` for GPU support. All needed in Phase 3 when the Slab is trained end-to-end.

**Visualisation (`visualise_slab.py`)**
- Initialises a fresh Slab.
- Simulates 50 trap hits on each of the three trap cells (pain intensity 1.0) and minor random damage on 6 additional cells (intensity 0.3).
- Renders the pain landscape as a matplotlib heatmap: **white = no pain, deep red = high pain**. Pain scores printed inside each cell.
- Applies 100 steps of elastic recovery (`decay=0.99`) and renders the heatmap again.
- Demonstrates `reset_cell()` on one trap for targeted healing.
- Saves both figures to `models/`.

**To run Phase 2:**
```bash
python visualise_slab.py
```

---

### Phase 3 — Coupling (Planned)

Wire the Slab into the live training loop. The agent's observation will include not just its `(row, col)` position but a channel from the pain map beneath it. Every trap visit deforms the Slab in real time. The agent must learn to navigate using both environmental feedback and its own accumulated pain state.

---

### Phase 4+ — To Be Defined

Possible directions: partial observability, multiple agents sharing a Slab, pain-modulated exploration bonuses, chronic pain modelling, interpretability probes on slab vectors.

---

## Why It Matters

Most RL reward signals are stateless — they fire once and disappear. PULSE investigates whether giving an agent a **persistent, embodied representation of harm** changes what it learns and how it generalises. This touches questions relevant to:

- **Computational neuroscience** — modelling nociception and central sensitisation
- **Safe RL** — agents that carry forward memory of dangerous states rather than rediscovering danger from scratch each episode
- **Interpretability** — the pain map is a human-readable internal state, not a black-box activation

The Slab is the agent's scar tissue. The goal is to find out whether scars make it smarter.

---

## Setup

```bash
pip install gymnasium stable-baselines3 torch matplotlib numpy
```

Developed with Python 3.10+. No GPU required for Phases 1–2.
