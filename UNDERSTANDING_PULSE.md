# Understanding PULSE — A Complete Reader's Guide

This guide is for anyone who wants to truly understand the PULSE codebase: not just what each function does, but why it was built the way it was, what theoretical ideas it embodies, and how the pieces connect. You can read it front-to-back as a walkthrough, or jump to any section as a reference.

---

## Table of Contents

1. [The Big Picture — What PULSE Is Doing and Why](#1-the-big-picture)
2. [The Conceptual Stack — How the Layers Build on Each Other](#2-the-conceptual-stack)
3. [Prerequisites — What You Need to Know First](#3-prerequisites)
4. [Phase 1 — The World (`pulse_env.py`)](#4-phase-1--the-world)
5. [Phase 2 — The Skin (`slab.py`)](#5-phase-2--the-skin)
6. [Phase 3 — Coupling (Wired into `pulse_env.py`)](#6-phase-3--coupling)
7. [Phase 4 — Resistance (`resistance_field.py`)](#7-phase-4--resistance)
8. [Phase 5 — Aversion (`pain_shaped_policy.py`)](#8-phase-5--aversion)
9. [Phase 6 — Benchmark (`benchmark.py`, `plot_benchmark.py`, `results_summary.py`)](#9-phase-6--benchmark)
10. [Key Code Patterns Used Throughout](#10-key-code-patterns)
11. [The Complete Data Flow — One Training Step, End to End](#11-the-complete-data-flow)
12. [The Reward Signal — All Its Components in One Place](#12-the-reward-signal)
13. [How to Read the Visualiser Scripts](#13-how-to-read-the-visualiser-scripts)
14. [Glossary of PULSE-Specific Terms](#14-glossary)
15. [Common Misconceptions](#15-common-misconceptions)
16. [Suggested Reading Order](#16-suggested-reading-order)

---

## 1. The Big Picture

### What standard RL does

A standard RL agent navigates a world by trial and error. When it does something bad — say, walks into a trap — it receives a negative reward number (e.g. `-10`), and its policy network is nudged away from that action in that state. The negative number fires once, then disappears. The next episode, the agent starts fresh. If it walks into the same trap again, it gets the same `-10`, the same nudge. The world is stateless from the agent's body's perspective.

This has a practical consequence: **the agent rediscovers that the trap is bad every single time it encounters it.** The reward signal carries no memory of whether it has been hurt there before. A cell that caused suffering 20 episodes ago looks identical to a cell the agent has never visited.

### What PULSE adds

PULSE asks: what if the agent had **skin** — a physical body surface that accumulates damage over time?

In biology, your nociceptors (pain receptors) don't just fire when you are hurt. They sensitise. A bruised area hurts more on the second touch than the first. Regions around wounds become hypersensitive. Your body builds up a spatial map of where it has been hurt, and that map actively steers you away from those regions before you commit to the next painful action.

PULSE implements a version of this:

1. **The Slab** (`slab.py`): a grid of learnable 16-dimensional vectors — one per grid cell — representing the agent's skin. When the agent enters a trap, that cell's vector is "deformed" (pushed away from zero). The L2 norm of the vector is the pain score. Pain accumulates across episodes and fades slowly (elastic recovery), just like sensitisation and healing.

2. **The Resistance Field** (`resistance_field.py`): a grid of viscosity values. Cells near traps are "thick jelly" — the agent pays a movement cost even before reaching the trap itself. This is the spatial pre-entry warning signal.

3. **The Pain-Shaped Policy** (`pain_shaped_policy.py`): at evaluation time, before committing to an action, the agent checks whether the target cell has a high **aversion score** (slab depth × resistance). If it does, that action is vetoed and the agent picks the best remaining option.

The fundamental hypothesis PULSE is testing: **Does giving an agent an embodied, persistent, spatial memory of harm change how it learns and behaves — compared to an agent that has no such body?**

---

## 2. The Conceptual Stack

Think of PULSE as four layers sitting on top of each other. Each phase adds one layer:

```
┌─────────────────────────────────────────────────────┐
│  PHASE 5: Aversion Filter                           │
│  PainShapedPolicy reads both layers below to        │
│  veto dangerous actions at evaluation time          │
├─────────────────────────────────────────────────────┤
│  PHASE 4: Resistance Field                          │
│  Physics layer — spatial viscosity cost             │
│  pre-entry signal: danger felt before entry         │
├─────────────────────────────────────────────────────┤
│  PHASE 3: Slab Coupling                             │
│  Body layer — trap entries deform the skin          │
│  pain bonus grows with repeated visits              │
├─────────────────────────────────────────────────────┤
│  PHASE 2: Slab (Vector Field)                       │
│  Data structure: (8,8,16) learnable parameter       │
│  L2 norm = pain score per cell                      │
├─────────────────────────────────────────────────────┤
│  PHASE 1: Environment                               │
│  The world: 8×8 grid, traps, goal, rewards          │
│  Standard Gymnasium API                             │
└─────────────────────────────────────────────────────┘
```

The layers are designed so each one is **independently understandable** and can be removed without breaking the layer below it. You can run Phase 1 in isolation. You can visualise Phase 2 without ever running the environment. This is intentional separation of concerns.

---

## 3. Prerequisites

### Reinforcement Learning concepts

Before reading the code, you should understand:

**Episode**: a single run of the agent from its starting position until it reaches the goal or falls into a trap. The agent then resets and tries again.

**State / Observation**: what the agent currently sees. In PULSE, this is just `[row, col]` — the agent's position on the 8×8 grid, as a `float32` array of shape `(2,)`.

**Action**: what the agent chooses to do. In PULSE, there are 4 actions: up (0), down (1), left (2), right (3).

**Reward**: a scalar number the agent receives after each action. Positive reward = good. Negative reward = bad. The agent's goal is to maximise total reward across an episode.

**Policy**: a function that maps observations to actions (or action probabilities). PPO trains a neural network to be this function.

**Value function**: a function that estimates "how much total future reward can I expect from this state?" Used by PPO to compute policy gradient updates.

### The Gymnasium API

All RL environments in PULSE implement the Gymnasium interface. The three critical methods are:

```python
observation, info = env.reset()
# → starts a new episode; returns the first observation

observation, reward, terminated, truncated, info = env.step(action)
# → applies one action; returns:
#     observation  — new agent position
#     reward       — scalar signal for this transition
#     terminated   — True if episode ended naturally (goal or trap)
#     truncated    — True if episode hit a time limit (never in PULSE)
#     info         — dict of extra data (pain_score, resistance_cost, etc.)

env.close()
# → cleanup
```

When you see `make_vec_env()` or `DummyVecEnv([...])`, that is SB3 wrapping the environment in a container that adds batch dimensions (shape `(1, obs_dim)` instead of `(obs_dim,)`) because SB3 expects vectorised environments.

### PyTorch basics

The Slab (`slab.py`) uses PyTorch. You need to understand:

**`nn.Module`**: the base class for all neural networks in PyTorch. Subclassing it gives you `state_dict()` for saving/loading, automatic parameter discovery, and GPU support.

**`nn.Parameter`**: a tensor that is registered as a learnable parameter of an `nn.Module`. It shows up in `model.parameters()` and gets gradients during backpropagation. In PULSE, the slab vectors are `nn.Parameter`s — not because we train them via backprop right now, but to use the infrastructure.

**`torch.no_grad()`**: a context manager that tells PyTorch not to track gradients for operations inside it. Used in PULSE's manual slab updates to prevent accidental gradient graph pollution.

**`tensor.detach()`**: returns a view of a tensor disconnected from the computation graph. Used when reading slab values for inspection or scoring — we don't want those reads to affect any future gradient computation.

---

## 4. Phase 1 — The World

**File**: `pulse_env.py` (the `PulseGridWorld` class, Phase 1 parts only)  
**Concept**: Build a minimal, correct Gymnasium environment.

### What the grid looks like

```
(0,0) A . . . . . . .      A = agent (starts here)
      . . . . . . . .      G = goal  (7,7)
      . . . . . . T .      T = trap  at (2,6), (3,3), (4,5)
      . . . T . . . .
      . . . . . T . .
      . . . . . . . .
      . . . . . . . .
      . . . . . . . G
```

Matrix convention: row 0 is the top row, column 0 is the left column. Moving "down" increases the row index. This is important when reading the action map.

### The action map

```python
action_map = {
    0: (-1,  0),   # up    — row decreases (moves toward row 0)
    1: ( 1,  0),   # down  — row increases (moves toward row 7)
    2: ( 0, -1),   # left  — col decreases
    3: ( 0,  1),   # right — col increases
}
```

Row decreasing = moving *up* visually. This confuses people at first. Remember: row 0 is the top.

### Wall collision

```python
new_row = int(np.clip(new_row, 0, self.grid_size - 1))
new_col = int(np.clip(new_col, 0, self.grid_size - 1))
```

`np.clip(value, min, max)` clamps the value to `[min, max]`. If the agent tries to go up from row 0, new_row becomes -1, which clips back to 0 — the agent stays in place. This is the "wall" rule: hitting a wall costs a step but doesn't move.

### The observation space

```python
self.observation_space = spaces.Box(
    low=np.array([0, 0], dtype=np.float32),
    high=np.array([7, 7], dtype=np.float32),
    dtype=np.float32
)
```

This declares: "my observations are 2D float32 vectors where each dimension is between 0 and 7." SB3 reads this at runtime to automatically size its neural network's input layer. The Box type is used even though row/col are integers because SB3's MLP policy expects float inputs and handles them correctly.

### Phase 1 reward structure

| Situation | Reward |
|-----------|--------|
| Reach goal (7,7) | +10.0 |
| Enter trap | −10.0 |
| Any other step | −0.1 |

The `−0.1` step penalty is called the "living penalty." Without it, the agent has no reason to reach the goal quickly — it could just wander safely forever. The penalty makes time itself costly, so reaching the goal fast is better than reaching it slowly.

---

## 5. Phase 2 — The Skin

**File**: `slab.py`  
**Concept**: A grid of learnable vectors that represent accumulated pain.

### The core data structure

```python
self.slab_vectors = nn.Parameter(
    torch.zeros(grid_size, grid_size, vector_dim)   # shape: (8, 8, 16)
)
```

This is a single 3D tensor. Think of it as a grid of 64 cells, where each cell holds a 16-dimensional vector. All vectors start at zero (resting state, no pain). When a cell is "hurt," its vector is pushed away from zero. The further from zero, the more pain.

### Why 16 dimensions?

Each of the 16 dimensions is a "pain channel." In Phase 3, all 16 are set uniformly (`[intensity, intensity, ..., intensity]`) — the pain is undifferentiated. But the infrastructure is there for richer encoding: in a future phase, different channels could encode pain direction, velocity at impact, time since last hit, etc. 16 is compact enough to visualise and interpret, but large enough to carry nuanced signal.

### Pain score = L2 norm

```python
def deformation_depth(self, x: int, y: int) -> float:
    return torch.norm(self.slab_vectors[x, y]).item()
```

`torch.norm` computes the L2 norm: `||v|| = sqrt(v₁² + v₂² + ... + v₁₆²)`. This measures how far the vector has been pushed from the origin (zero = no pain). It grows as the cell accumulates more deformations and shrinks as elastic recovery pulls it back toward zero.

Why L2 and not something else? L2 is holistic — it grows with both the magnitude of each dimension and the number of dimensions activated. L1 would treat all dimensions equally regardless of scale. The maximum (L∞) would only read the single loudest channel, ignoring the others. L2 is the natural measure of "how far has this vector moved?"

### How deformation accumulates

```python
def deform(self, x: int, y: int, force_vector: torch.Tensor) -> None:
    with torch.no_grad():
        self.slab_vectors[x, y] += self.lr * force_vector
```

Each call to `deform()` adds `lr × force_vector` to the cell's existing vector. With `lr=0.01`:
- After 1 hit: vector = `0.01 × [1,...,1]` → L2 norm = `0.01 × 4.0 = 0.04`
- After 10 hits: vector = `0.10 × [1,...,1]` → L2 norm = `0.40`
- After 100 hits: vector = `1.00 × [1,...,1]` → L2 norm = `4.0`

(The L2 norm of a vector `[c, c, ..., c]` with 16 elements is `c × √16 = 4c`.)

### Elastic recovery

```python
def elastic_recovery(self, decay: float = 0.99) -> None:
    with torch.no_grad():
        self.slab_vectors *= decay
```

Multiplying all vectors by a number less than 1 makes them smaller — they shrink toward zero (the resting/healed state). This happens at every episode boundary (`reset()` in `pulse_env.py`). At `decay=0.995`:
- After 100 episodes with no new hits: `0.995^100 ≈ 0.606` — 60% of damage remains
- After 1000 episodes: `0.995^1000 ≈ 0.007` — essentially healed

Cells that are hit repeatedly accumulate damage faster than it decays. Cells that stop being hit gradually heal. This is **elastic** memory — the slab returns toward zero naturally, like a rubber band.

---

## 6. Phase 3 — Coupling

**Location**: `pulse_env.py`, marked `### PHASE 3 ###`  
**Concept**: Wire the Slab into the live training loop.

### The `apply_spike()` method

```python
def apply_spike(self, x: int, y: int, intensity: float) -> float:
    force_vector = torch.ones(self.slab.vector_dim) * intensity
    self.slab.deform(x, y, force_vector)
    pain_score = self.slab.deformation_depth(x, y)
    return pain_score
```

This is the **bridge** between the environment's event system and the slab's deformation API. `step()` knows *when* a trap entry happens (game logic); `apply_spike()` knows *what* to do about it (pain mechanics). The separation means you can change the spike shape (e.g. Gaussian spread to neighbouring cells) without touching `step()`.

`torch.ones(16) * intensity` creates a 16-element tensor where every element equals `intensity`. This is "uniform pain encoding" — all channels hurt equally. The L2 norm of `0.01 × [1,...,1]` is exactly `0.04`, so the first visit to a trap gives a pain score of `0.04`.

### The growing pain penalty

In Phase 1, entering a trap always gave `-10`. In Phase 3:

```python
reward += base_trap_reward - pain_score   # = -10.0 - pain_score
```

The `pain_score` is the L2 norm of the slab at that cell *after* this deformation. So:

| Visit number | Pain score | Trap reward |
|-------------|-----------|-------------|
| 1st | ≈ 0.04 | ≈ −10.04 |
| 5th | ≈ 0.20 | ≈ −10.20 |
| 20th | ≈ 0.80 | ≈ −10.80 |
| 50th | ≈ 2.00 | ≈ −12.00 |

The punishment *grows* with repeated visits. This is a **non-stationary reward signal** — the same state gives a worse reward the more it has been visited. This is the computational analogue of peripheral sensitisation: your nervous system makes a repeatedly-injured area more sensitive over time, not less.

### Why this helps learning

A purely stronger fixed penalty (say, `-15` instead of `-10`) would penalise every trap equally regardless of history. The growing pain bonus instead creates a **gradient across trap history**: traps you've never visited are slightly less aversive than traps you've been in 20 times. This could guide exploration — the agent is relatively more reluctant to re-enter familiar painful zones than to probe unfamiliar ones.

### Pain history

```python
self.pain_history.append({
    "episode":    self.episode_count,
    "step":       self.step_count,
    "cell":       (new_row, new_col),
    "pain_score": pain_score,
})
```

Every spike event is recorded. This is used by visualisers to reconstruct the agent's complete "pain biography" — which cells hurt it, in which episode, at which step. It is not used by the learning algorithm; it is purely for analysis and debugging.

---

## 7. Phase 4 — Resistance

**File**: `resistance_field.py`  
**Concept**: Make space physically thick near danger — the agent feels it before entering.

### The static topology

When `ResistanceField` is created, `_stamp_traps()` sets resistance values:

```
0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1
0.1  0.1  0.1  0.1  0.7  0.7  0.7  0.1    ← neighbours of (2,6)
0.1  0.1  0.7  0.7  0.7  0.7  0.7  0.7    ← (2,6) and its neighbours
0.1  0.1  0.7  0.7  0.7  0.7  0.7  0.7    ← (3,3) and its neighbours
0.1  0.1  0.7  0.7  0.7  0.7  0.7  0.1    ← (4,5) and its neighbours
0.1  0.1  0.1  0.1  0.7  0.7  0.7  0.1
0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1
0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1
```

Trap cells AND all 8 of their diagonal+orthogonal neighbours are stamped at `0.7`. Open cells stay at `0.1`. The pattern extends 1 cell in every direction from each trap.

### Why include diagonal neighbours?

The agent can approach a trap from any direction, including diagonally (by moving orthogonally into a diagonal-neighbour cell). If only orthogonal neighbours were stamped, an agent could approach trap `(3,3)` by going to `(2,2)` → `(3,3)` without ever touching a resistance-elevated cell. The 8-neighbour stamp closes all approach angles.

### The resistance cost in step()

```python
resistance_cost = self.resistance_field.movement_cost(new_row, new_col)
# ...
reward = resistance_cost     # initialise reward WITH the physics cost

if pos_tuple == self.goal_pos:
    reward += 10.0            # ADD goal bonus on top
elif pos_tuple in self.trap_positions:
    reward += base_trap_reward - pain_score   # ADD trap penalty on top
else:
    reward += -0.1            # ADD living penalty on top
```

The resistance cost is computed first and then added to by the reward branches using `+=`. This guarantees the physics cost applies to every move, including goal entry. `movement_cost()` returns a negative number (`-resistance_value`), so adding it always reduces reward.

**Reward breakdown for each cell type:**

| Cell type | Resistance | Living | Goal/Trap | Total |
|-----------|-----------|--------|-----------|-------|
| Open cell | −0.1 | −0.1 | — | **−0.2** |
| Trap neighbour | −0.7 | −0.1 | — | **−0.8** |
| Trap cell | −0.7 | — | −10 − pain | **≤ −10.7** |
| Goal cell | −0.1 | — | +10.0 | **+9.9** |

The `-0.8` for trap-neighbour cells is the **pre-entry dread signal**. The agent pays it every step it lingers near a trap — not just when it enters. This creates a spatial gradient in the reward landscape pointing away from danger zones.

### The dynamic component

```python
def dynamic_resistance(self, slab, scale: float = 0.5) -> None:
    for row in range(self.grid_size):
        for col in range(self.grid_size):
            deformation = slab.deformation_depth(row, col)
            new_resistance = self.BASELINE_RESISTANCE + scale * deformation
            new_resistance = float(np.clip(new_resistance, 0.0, 1.0))
            self.field[row, col] = max(self.field[row, col], new_resistance)
```

After every step in the environment, this is called. It raises the resistance of cells where the slab has accumulated deformation. The `max()` ensures the static trap topology is never lowered — it only adds on top.

This creates a **feedback loop**:
1. Agent hits trap → slab deformation increases at that cell
2. `dynamic_resistance()` raises resistance at that cell
3. Next visit to that area costs more movement reward
4. Agent's policy gradient nudges it away from that direction
5. The feedback makes the signal grow over training, not stay flat

---

## 8. Phase 5 — Aversion

**File**: `pain_shaped_policy.py`  
**Concept**: An evaluation-time action filter that reads both the slab and resistance field to veto dangerous moves.

### The aversion score

```python
def aversion_score(self, x: int, y: int) -> float:
    depth = self.slab.deformation_depth(x, y)
    resistance = self.resistance_field.get_resistance(x, y)
    return depth * resistance
```

The score is the **product** of two quantities:

- `depth`: slab deformation = "how much has my body been hurt at this cell?"
- `resistance`: field value = "how physically dangerous is this zone?"

The product requires both to be elevated. Neither alone is sufficient:

| depth | resistance | score | interpretation |
|-------|-----------|-------|----------------|
| 0.0 | 0.7 | 0.00 | Trap zone but never hurt there — ok to explore |
| 0.4 | 0.1 | 0.04 | Past pain in safe zone — probably fine |
| 0.4 | 0.7 | 0.28 | Pain in danger zone — approaching threshold |
| 0.8 | 0.7 | 0.56 | Chronic pain in danger zone — **blocked** |

This is the "AND gate" design: both memory (depth) and physics (resistance) must be elevated to trigger avoidance.

### The threshold

Default threshold is `0.5`. With `slab.lr=0.01`:
- Each hit adds `0.01` to depth at the trap cell
- Trap cells have `resistance=0.7`
- Score = `depth × 0.7` reaches `0.5` when `depth ≈ 0.71`, which requires ~71 hits

Wait — that's a lot. In practice, `dynamic_resistance()` raises the resistance of trap cells above `0.7` as the slab accumulates, so the threshold is reached earlier. But the key design intent is: **the agent needs real, substantial history before the filter fires.** It cannot block a direction it has barely experienced.

### The select_action algorithm

```python
def select_action(self, obs, candidate_actions=None, threshold=0.5):
    # 1. Resolve where each action would take the agent
    # 2. Score each target cell with aversion_score()
    # 3. Get PPO's action probability distribution
    # 4. Separate safe (score ≤ threshold) from blocked (score > threshold)
    # 5. If safe actions exist: pick the one with highest PPO probability
    # 6. If all blocked (cornered): pick the one with lowest aversion score
```

Step 3 is important: we get the **full probability distribution** from PPO's policy network, not just the argmax action. This lets us rank all safe actions by the policy's preference, not just pick any safe action arbitrarily.

```python
obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
with torch.no_grad():
    distribution = self.model.policy.get_distribution(obs_tensor)
    probs = distribution.distribution.probs.squeeze(0).cpu().numpy()
```

`get_distribution()` returns a `CategoricalDistribution` object (since our action space is discrete). Its `.distribution.probs` gives the softmax probabilities over the 4 actions — the policy's "vote" for each direction.

### Why apply the filter at evaluation, not training?

This is one of the most important design decisions in PULSE. The filter is applied **only at evaluation time**. Here is why training-time application would break things:

**PPO's on-policy assumption**: PPO computes policy gradients by comparing the new policy's probability for actions against the old policy's probability (the "proximal" ratio). This comparison only makes sense if the **old policy** was the one that actually took those actions. If a filter is silently redirecting actions during training, the ratio is computed against the wrong denominator — the gradient is wrong.

**Non-stationary action space**: The set of available actions changes depending on the slab state, which changes every episode. From PPO's value function's perspective, the transition dynamics appear to change randomly — it cannot build a consistent value estimate.

**Double-penalisation**: The Phase 3-4 reward already penalises trap entries (base penalty + pain bonus) and trap-neighbour cells (resistance cost). Applying the filter during training would additionally reduce the probability of taking those actions — the same signal hits the policy twice, making the gradient noisy and hard to interpret.

At evaluation, none of this matters: we are not updating any weights. The filter is a **deployment guard** — a reflex arc between the brain's planned action and the muscle's execution.

### Pain as memory vs pain as reflex

The aversion filter embodies a specific theory of pain's role in learning:

**Pain as reflex** (classical avoidance): Any cell labelled "trap" is avoided, always. This is a lookup table, not learning. It is fast but inflexible — it requires knowing trap locations in advance, and it cannot adapt if traps move.

**Pain as memory** (PULSE): The agent avoids cells where it has *accumulated suffering*, weighted by physical danger. It doesn't know trap locations — it infers them from its body's history. A cell that has never hurt it is not avoided, even if it is "a trap" by the world's rules. Conversely, if new traps appear at runtime, the agent will learn to avoid them through experience without being told.

---

## 9. Phase 6 — Benchmark

**Files**: `benchmark.py`, `plot_benchmark.py`, `results_summary.py`  
**Concept**: Measure whether PULSE actually converges faster than standard algorithms.

### The three agents

| Agent | Training | Evaluation |
|-------|---------|-----------|
| PPO | Vanilla SB3 PPO | `model.predict()` |
| DQN | Vanilla SB3 DQN | `model.predict()` |
| PULSE-PPO | Vanilla SB3 PPO | `PainShapedPolicy.select_action()` |

All three train on the **same PulseGridWorld** (Phases 3-4 active). The only difference between PULSE-PPO and vanilla PPO is what happens at evaluation checkpoints.

### Why the same environment for all three?

If vanilla PPO trained on a simpler environment (no slab, no resistance), and PULSE-PPO trained on the full environment, any difference in performance could be attributed to the different reward signals, not the aversion filter. Using the same environment **isolates the variable**: the only difference is the filter.

This also means the comparison answers a more precise question: **does the aversion filter add value on top of what the shaped reward already provides?**

### The chunked training loop

```python
while steps_trained < TOTAL_TIMESTEPS:
    chunk = min(EVAL_EVERY, TOTAL_TIMESTEPS - steps_trained)
    model.learn(
        total_timesteps=chunk,
        reset_num_timesteps=first_call,   # True only on first call
    )
    first_call = False
    steps_trained += chunk
    # ... evaluate and record metrics
```

`model.learn(1000, reset_num_timesteps=False)` tells SB3: "train for 1000 more steps, continuing from where you left off." For PPO, `n_steps=1000` means exactly one full rollout per chunk. For DQN, epsilon continues decaying from its current value rather than resetting.

The `reset_num_timesteps=True` on the first call is necessary to initialise SB3's internal step counter correctly. `False` on all subsequent calls preserves the counter so DQN's epsilon schedule runs over the full training budget.

### Why 5 seeds?

A single RL training run is one sample from a distribution of possible outcomes. Neural network weight initialisation, episode ordering, and exploration randomness all differ between runs. A single run might be lucky (unusually fast convergence) or unlucky (gets stuck in a bad local policy). 

Five seeds allows you to compute a **mean ± standard deviation** across seeds. The shaded bands in `plot_benchmark.py` show this std — wide bands mean the algorithm is unstable (sensitive to initialisation), narrow bands mean it is reliable. Two algorithms whose bands overlap substantially may not actually be different — the difference might just be noise.

With N=5, formal hypothesis tests (p-values) are unreliable — there is too little statistical power to distinguish real effects from noise. `results_summary.py` therefore reports **Cohen's d** (effect size) instead:
- `|d| < 0.2`: negligible
- `0.2–0.5`: small
- `0.5–0.8`: medium  
- `> 0.8`: large (strong evidence for a real effect even with small N)

### Four metrics and why each one matters

**Average reward**: the direct optimisation target. Easy to compute, but ambiguous — a high reward could come from reaching the goal faster, avoiding traps, or taking fewer steps. Cannot attribute it to any specific mechanism alone.

**Trap entry rate**: the PULSE-specific test. Measures directly what the aversion filter targets. If PULSE-PPO's trap rate is lower but reward is unchanged, the filter helps safety without improving efficiency. If both improve, the filter is genuinely better.

**Goal reach rate**: the primary task success metric. An agent could reduce trap rate to zero by never moving — but then goal rate would be zero too. Both must be good simultaneously.

**Episode length**: efficiency. Shorter episodes mean the agent finds the goal faster. Beware: trap-terminated episodes are also very short (the trap might be 2-3 steps from start), so low episode length in early training usually means many trap hits, not good navigation.

---

## 10. Key Code Patterns

### Pattern 1: `nn.Parameter` with manual updates

```python
# In slab.py
self.slab_vectors = nn.Parameter(torch.zeros(8, 8, 16))

# Manual (non-gradient) update:
with torch.no_grad():
    self.slab_vectors[x, y] += self.lr * force_vector
```

The `torch.no_grad()` context is mandatory when manually modifying `nn.Parameter`s. Without it, PyTorch tries to record the operation in the computation graph, causing memory leaks or incorrect gradients. The `+=` is in-place — it modifies the tensor's data without creating a new tensor, which is required to keep the Parameter registration intact.

### Pattern 2: The `## PHASE N ##` comment blocks

Throughout `pulse_env.py`, new code is wrapped in:
```python
### PHASE 3 ###
# ... new code ...
### END PHASE 3 ###
```

These delimiters let you mentally "strip out" phases to understand what the code looked like at each stage of development. If you want to understand Phase 1 alone, mentally delete all `### PHASE 3 ###` and `### PHASE 4 ###` blocks.

### Pattern 3: Gradients disabled for slab inference

```python
# In pain_shaped_policy.py
with torch.no_grad():
    distribution = self.model.policy.get_distribution(obs_tensor)
    probs = distribution.distribution.probs.squeeze(0).cpu().numpy()
```

And in `slab.py`:
```python
def get_vector(self, x: int, y: int) -> torch.Tensor:
    return self.slab_vectors[x, y].detach()
```

Both `.detach()` and `torch.no_grad()` prevent gradient tracking. They serve slightly different purposes:
- `torch.no_grad()`: tells the computation engine "don't build a graph for anything in this block"
- `.detach()`: creates a copy of a specific tensor that is permanently disconnected from the graph

Use `no_grad()` for inference blocks; use `.detach()` when you want to return a tensor to a caller who should not be able to trigger gradients through it.

### Pattern 4: DummyVecEnv for direct attribute access

```python
# In benchmark.py
inner_env = PulseGridWorld(render_mode=None)
vec_env = DummyVecEnv([lambda: inner_env])

# Direct access to inner env attributes:
slab = vec_env.envs[0].slab
resistance_field = vec_env.envs[0].resistance_field
```

SB3 requires a `VecEnv`. `DummyVecEnv` is the single-process version — it wraps environments in the same Python process. Because it runs in-process, `vec_env.envs[0]` gives you a direct Python reference to the wrapped environment, including all its custom attributes (`slab`, `resistance_field`, `pain_history`). This is the cleanest way to read the slab during evaluation without going through SB3's `get_attr()` API.

### Pattern 5: Chunked training for mid-training evaluation

```python
model.learn(1000, reset_num_timesteps=True)   # first call — initialise everything
model.learn(1000, reset_num_timesteps=False)  # second call — continue from step 1000
model.learn(1000, reset_num_timesteps=False)  # third call — continue from step 2000
```

`reset_num_timesteps=False` preserves SB3's internal step counter. This matters for:
- DQN's epsilon-greedy schedule (epsilon decays based on `num_timesteps`, so it needs to accumulate)
- PPO's learning rate scheduling (if used)
- TensorBoard logs (so the x-axis shows total steps, not per-chunk steps)

---

## 11. The Complete Data Flow

Here is what happens during **one training step** with the full PULSE stack active (Phases 3 and 4):

```
① PPO policy network receives obs = [row, col]
   ↓ forward pass through 2-layer MLP
   ↓ outputs action probabilities: [0.1, 0.6, 0.2, 0.1]
   ↓ samples action: 1 (down)

② pulse_env.step(action=1):
   ├── Compute new position: (row+1, col) clamped to [0,7]
   ├── Query resistance_field.movement_cost(new_row, new_col)
   │     → reads self.field[new_row, new_col]
   │     → returns -0.1 (baseline) or -0.7 (near trap)
   ├── Update resistance_field.dynamic_resistance(slab, scale=0.5)
   │     → reads slab.deformation_depth(r, c) for all cells
   │     → raises resistance where deformation is high
   ├── Compute reward:
   │   reward = resistance_cost               (universal physics cost)
   │   if goal:   reward += 10.0             (goal bonus)
   │   if trap:   reward += -10 - pain_score  (trap penalty + body memory)
   │              apply_spike(x, y) → slab.deform(x, y, force_vector)
   │   else:      reward += -0.1             (living penalty)
   └── Return (obs, reward, terminated, truncated, info)

③ PPO rollout buffer stores (obs, action, reward, value, log_prob)

④ After n_steps=1000 transitions:
   PPO computes advantage estimates using value function
   PPO computes policy gradient loss: L_clip
   PPO computes value function loss: L_VF
   PPO computes entropy bonus: L_entropy
   Adam optimizer updates policy + value network weights

⑤ At episode boundary (terminated=True):
   env.reset() is called
   slab.elastic_recovery(decay=0.995) → all slab vectors shrink by 0.5%
   episode_count += 1
```

During **PULSE-PPO evaluation**, step ① is replaced by:

```
① PainShapedPolicy.select_action(obs):
   For each of 4 actions:
     ├── _resolve_action() → compute target cell
     ├── slab.deformation_depth(nr, nc) → body memory
     ├── resistance_field.get_resistance(nr, nc) → physics danger
     └── aversion_score = depth × resistance
   Get PPO probs from model.policy.get_distribution(obs)
   Filter: separate safe (score ≤ 0.5) from blocked (score > 0.5)
   Choose: argmax(PPO probs) over safe set
          OR min(aversion_score) if all blocked (cornered)
```

---

## 12. The Reward Signal — All Components Together

The full reward at each step is a sum of up to four components:

```
reward = resistance_cost          (always present, Phase 4)
       + living_penalty           (−0.1, present on non-terminal steps)
       + goal_bonus               (+10, present only when goal is reached)
       + trap_base_penalty        (−10, present only when trap is entered)
       + pain_bonus_penalty       (−pain_score, present only on trap entry, Phase 3)
```

In practice, the combinations are:

```python
# Open cell (not trap, not goal):
reward = −resistance(cell)  +  (−0.1)
       = −0.1 − 0.1  =  −0.2    for baseline cells
       = −0.7 − 0.1  =  −0.8    for trap-neighbour cells

# Trap cell:
reward = −resistance(trap)  +  (−10.0)  +  (−pain_score)
       = −0.7 − 10.0 − pain_score  ≤  −10.7

# Goal cell:
reward = −resistance(goal)  +  (+10.0)
       = −0.1 + 10.0  =  +9.9
```

The `-0.8` for trap-neighbour cells is the most important novelty of Phase 4. It means the agent is already paying a penalty **before** it enters the trap. The gradient from this cost reaches all the preceding decisions that led the agent into that neighbourhood — not just the single step of trap entry.

---

## 13. How to Read the Visualiser Scripts

There are five visualiser scripts. Here is what each one tests:

### `visualise_phase1.py`
Loads a trained Phase 1 model. Runs one deterministic episode. Renders the path on the grid. Useful for quickly checking: "does my model know how to navigate?"

### `visualise_slab.py`
Does NOT train anything. Manually calls `slab.deform()` 50 times on each trap cell, then renders the pain heatmap. Then applies `elastic_recovery()` 100 times and renders again. Shows the slab mechanics in isolation without any RL training.

### `visualise_pain.py`
Connects to a training run in progress (or just completed). Reads `env.pain_history` and renders pain scores over time. Shows when and how often each trap cell was hit.

### `visualise_resistance.py`
Instantiates a fresh `ResistanceField` and renders it as a heatmap. No training, no slab needed — this just shows the static topology: which cells are 0.1 (blue) vs 0.7 (red). Run this to verify the pre-entry gradient looks correct before training.

### `visualise_aversion.py`
The most complex visualiser. Loads a trained model, runs 10 evaluation episodes with PainShapedPolicy active, and:
1. Prints per-step console diagnostics (aversion scores for all 4 directions, blocked actions)
2. Renders a figure per episode: aversion heatmap background + green path + faded red arrows for blocked actions

The faded red arrows are **counterfactual paths** — directions the agent considered but refused. They make the aversion mechanism directly visible.

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| **Slab** | The `SlabNetwork` — a `(8,8,16)` PyTorch parameter representing the agent's accumulated pain |
| **Deformation** | The state of one cell's slab vector being pushed away from zero by trap hits |
| **Deformation depth** | The L2 norm of a cell's slab vector — the scalar pain score |
| **Elastic recovery** | Multiplicative decay of all slab vectors per episode — simulates healing |
| **Spike** | A single pain event: one call to `apply_spike()`, one call to `slab.deform()` |
| **Pain score** | Synonym for deformation depth — `slab.deformation_depth(x, y)` |
| **Resistance** | The viscosity value at a cell in the `ResistanceField` — how expensive it is to enter |
| **Baseline resistance** | `0.1` — the movement cost in empty, safe cells |
| **Trap resistance** | `0.7` — the movement cost in trap cells and their 8-neighbours |
| **Pre-entry signal** | The resistance cost felt when entering a trap-neighbour cell — before the trap itself |
| **Dynamic resistance** | The component of the resistance field that grows as slab deformation grows |
| **Aversion score** | `depth × resistance` — the combined "should I avoid this cell?" signal |
| **Threshold** | `0.5` — if `aversion_score > threshold`, the action is blocked |
| **Cornered** | The state where all 4 actions are blocked; the least-aversion action is chosen |
| **Living penalty** | `−0.1` per step in open space — makes time itself costly, prevents aimless wandering |
| **Pain bonus penalty** | `−pain_score` added on top of the `−10` trap penalty — grows with repeated visits |
| **Phase** | A development stage of PULSE; each adds one conceptual layer |
| **Seed** | A fixed random initialisation for reproducibility — used to compare algorithms fairly |

---

## 15. Common Misconceptions

### "The slab is the policy"

No. The slab is a **body state** — a record of where the agent has been hurt. It does not decide actions. In Phase 3-4, the slab influences reward, and reward influences the policy network (via gradient descent). In Phase 5, the slab is read by `PainShapedPolicy` to *filter* actions at evaluation time. But the policy network (the MLP inside PPO) makes all action decisions during training — the slab only modulates the reward signal it learns from.

### "The resistance field replaces the trap reward"

No. The resistance field is a **separate, additive cost**. The trap reward (`−10`) still fires on trap entry. The resistance cost (`−0.7`) additionally fires when moving into any cell near a trap. Both happen. The resistance cost is a *pre-entry* signal; the trap reward is the *entry* signal. Together they give the policy gradient a multi-step warning rather than a single-step punishment.

### "PULSE trains a different neural network"

No. All three agents in Phase 6 (PPO, DQN, PULSE-PPO) use the same neural network architectures from SB3. PULSE-PPO trains with the same PPO algorithm as vanilla PPO. The only difference is what happens at *evaluation* — the `PainShapedPolicy` wrapper.

### "The aversion filter is applied during training"

No, and this is critical to understand. The filter is applied only at **evaluation time**. During training, all three agents make decisions purely through their policy network. The slab only influences training through the *reward signal* (the pain bonus penalty in Phase 3). The filter is a deployment-time mechanism, not a training mechanism. See [Phase 5](#8-phase-5--aversion) for the detailed explanation of why applying it during training would break PPO.

### "Higher resistance = more negative reward = better learning signal"

Not straightforward. Higher resistance does create a stronger gradient away from danger zones, but if resistance is too high (e.g., 1.0 on all trap neighbours), the agent might completely refuse to approach any region of the grid that contains a trap, even if the optimal path passes through that region. The `0.7` value is a deliberate calibration: strong enough to signal danger, low enough not to create an impassable wall.

### "The slab vectors are trained by backpropagation"

Not in any current phase. The slab uses `nn.Parameter` (which *could* receive gradients), but all updates are **manual**: `slab_vectors[x, y] += lr * force_vector` inside `torch.no_grad()`. The infrastructure is there for gradient-based training in future phases, but right now the slab updates are Hebbian — direct experience-driven, not loss-function-driven.

---

## 16. Suggested Reading Order

If you are new to the codebase, read files in this order:

### Pass 1: Understand the world

1. **`pulse_env.py`** — Read from top to bottom, ignoring `### PHASE 3 ###` and `### PHASE 4 ###` blocks. Understand: what is the grid, what are the actions, what do the rewards mean, what does `reset()` do, what does `step()` return.

2. **`train_phase1.py`** — Read to understand how SB3 PPO trains on the environment. Note the hyperparameters and understand why each was chosen (the comments explain).

3. **`visualise_phase1.py`** — Read to understand how a trained model is loaded and used.

### Pass 2: Understand the body

4. **`slab.py`** — Read entirely. Focus on `deform()`, `deformation_depth()`, and `elastic_recovery()`. Trace the math: what is the L2 norm of `[0.01, 0.01, ..., 0.01]` (16 elements)?

5. **`visualise_slab.py`** — Read to see the slab used in isolation. This is the clearest demonstration of the deformation + healing cycle.

### Pass 3: Understand the coupling

6. **`pulse_env.py`** — Re-read, now including the `### PHASE 3 ###` blocks. Focus on `apply_spike()` and the new trap reward formula.

7. **`resistance_field.py`** — Read entirely. Focus on `_stamp_traps()`, `movement_cost()`, and `dynamic_resistance()`.

8. **`pulse_env.py`** — Re-read once more, now including `### PHASE 4 ###` blocks. Trace the reward formula for one trap-neighbour step and one trap-entry step.

### Pass 4: Understand the policy layer

9. **`pain_shaped_policy.py`** — Read entirely. Focus on `aversion_score()`, `should_avoid()`, and `select_action()`. Trace what happens when all 4 actions are blocked.

10. **`train_phase5.py`** — Read to understand how PULSE evaluation is different from vanilla evaluation.

### Pass 5: Understand the experiment

11. **`benchmark.py`** — Read `train_one_seed()`, `evaluate_vanilla()`, and `evaluate_pulse()`. These contain the critical difference: `model.predict()` vs `pulse_policy.select_action()`.

12. **`results_summary.py`** — Read `interpret_result()`. This contains the honest analysis framework: what each possible outcome means and what to try next.

13. **`plot_benchmark.py`** — Read `plot_panel()` to understand how the shaded learning curves are constructed.

---

*This guide was written to be read alongside the code, not instead of it. Every function in the codebase has inline comments explaining the WHY — treat those as the authoritative source. This document is the map; the code is the territory.*
