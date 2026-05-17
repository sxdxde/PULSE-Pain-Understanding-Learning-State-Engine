# =============================================================================
# benchmark.py — Phase 6 Convergence Benchmark for PULSE
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 6
# =============================================================================
#
# WHAT this script measures:
#   Three agents are trained on the SAME PulseGridWorld environment. At every
#   EVAL_EVERY training steps, a batch of evaluation episodes is run and four
#   metrics are recorded: average reward, trap entry rate, goal reach rate, and
#   average episode length. All results are saved to benchmark_results.csv.
#
# THE THREE AGENTS:
#   PPO       — Vanilla stable-baselines3 PPO.  No aversion filter at eval time.
#   DQN       — Vanilla stable-baselines3 DQN.  No aversion filter at eval time.
#   PULSE-PPO — PPO training (identical to Vanilla PPO) + PainShapedPolicy at
#               evaluation time. The ONLY difference from vanilla PPO is that
#               during eval episodes the action is routed through the aversion
#               filter rather than directly from model.predict().
#
# DESIGN DECISION: same environment for all three.
#   All agents face PulseGridWorld with Phases 3–4 active (slab pain bonus,
#   resistance costs). This means vanilla PPO and DQN also experience the shaped
#   reward — they are NOT protected from it. The benchmark therefore answers:
#   "Does the explicit aversion filter at eval time add value BEYOND what the
#   shaped reward already teaches the vanilla policy?"
#   This is a more honest question than comparing PULSE against a plain Phase 1
#   environment, because it isolates exactly one variable: the filter.
#
# WHY 5 SEEDS (statistical validity):
#   A single RL training run is a single random trajectory through policy space.
#   The initialisation of neural network weights, the order in which episodes
#   are encountered, and the exploration randomness all differ between runs.
#   A single run might be unusually lucky (or unlucky). Publishing a single run
#   as evidence for a hypothesis is like concluding a coin is biased after one flip.
#   With N=5 seeds we can:
#     1. Compute MEAN and STANDARD DEVIATION across seeds at every timestep.
#     2. Show shaded bands in the plot — revealing where the variance is high
#        (early training = noisy) vs low (converged behaviour = stable).
#     3. Make directional claims: "PULSE mean reward was higher than vanilla PPO
#        mean reward at 40,000 steps across 4/5 seeds" is much stronger than
#        "PULSE reward was higher in one run."
#   N=5 is the minimum for RL research credibility. N=20+ would give proper
#   statistical power for formal hypothesis tests, but N=5 is standard in
#   conference publications for demonstration-level claims.
#
# WHY measure trap entry rate SEPARATELY from reward:
#   Reward is an aggregate signal. A high reward could come from:
#     (a) avoiding traps (fewer -10 spikes) — PULSE's hypothesised effect
#     (b) finding the goal faster (more +10 hits per time) — general policy quality
#     (c) fewer wasted steps (fewer -0.1 living penalties) — path efficiency
#   Trap entry rate directly measures (a) alone. If PULSE improves reward but
#   trap entry rate is unchanged, the improvement came from (b) or (c), not from
#   the aversion mechanism. If PULSE improves BOTH reward AND trap rate, we have
#   direct evidence that the aversion filter is doing its specific job.
#   Without the separate metric, we cannot attribute the reward improvement to
#   the mechanism we designed. The trap rate is the isolation probe.
# =============================================================================

# WHY os: For creating the output directory and composing save paths.
import os

# WHY csv: We build records incrementally and write to CSV at the end.
# csv is stdlib (no extra install) and produces universally readable output.
import csv

# WHY time: For printing elapsed time per training run, so the user can
# estimate how long the full benchmark will take.
import time

# WHY logging: To suppress PainShapedPolicy's per-action block messages during
# benchmark runs. We only want high-level progress, not thousands of block lines.
import logging

# WHY random: For seeding Python's built-in random module alongside numpy/torch.
# Some SB3 internals use Python's random module, so seeding all three is correct.
import random

# WHY numpy: For metric aggregation in evaluation loops.
import numpy as np

# WHY torch: For seeding the PyTorch RNG at the start of each seed run.
# Neural network weight initialisation uses PyTorch RNG, so setting it is
# essential for exact reproducibility across runs.
import torch

# WHY stable_baselines3: PPO and DQN are the two baseline RL algorithms.
from stable_baselines3 import PPO, DQN

# WHY DummyVecEnv: SB3 algorithms require a VecEnv. DummyVecEnv wraps a single
# environment. Unlike SubprocVecEnv (which spawns a subprocess), DummyVecEnv
# runs in the same process and gives us direct attribute access via .envs[0] —
# crucial for reading the PULSE-PPO training environment's slab.
from stable_baselines3.common.vec_env import DummyVecEnv

# WHY PulseGridWorld: The single shared environment for all three agents.
from pulse_env import PulseGridWorld

# WHY PainShapedPolicy: The PULSE-PPO evaluation-time filter. Only used during
# evaluation of PULSE-PPO episodes; never during training of any agent.
from pain_shaped_policy import PainShapedPolicy


# =============================================================================
# CONFIGURATION — all tuneable parameters in one block
# =============================================================================

# WHY 50,000 timesteps?
# The shortest optimal path is 14 steps. At 50k steps the agent has had ~3,500+
# episodes to learn, which is well above the convergence point for a simple 8×8
# grid. This is the same budget used in Phases 1 and 5 — ensuring fair comparison.
TOTAL_TIMESTEPS = 50_000

# WHY eval every 1,000 steps?
# With TOTAL_TIMESTEPS=50,000, this gives 50 evaluation checkpoints — enough
# resolution to see the shape of the learning curve without a prohibitive runtime.
# We set PPO's n_steps=1,000 to align rollout boundaries with eval checkpoints,
# so each learn(1000) call completes exactly one rollout before evaluating.
EVAL_EVERY = 1_000

# WHY 5 seeds? See module-level comment for the full argument.
N_SEEDS = 5

# WHY 10 eval episodes per checkpoint?
# 10 gives a meaningful estimate of mean/std at each checkpoint without dominating
# the runtime. With 50 checkpoints × 5 seeds × 3 agents = 750 eval batches,
# 10 episodes per batch = 7,500 total eval episodes. Fast on this simple env.
N_EVAL_EPISODES = 10

# WHY 0.5 as the aversion threshold? Matches pain_shaped_policy.py's default.
# Keeping it consistent means the benchmark result is comparable to Phase 5 runs.
AVERSION_THRESHOLD = 0.5

# WHY "benchmark_results.csv" as the output path?
# A flat CSV is the most portable format for research results — it can be loaded
# by pandas, R, Excel, and any plotting library without conversion.
OUTPUT_CSV = "benchmark_results.csv"

# CSV column names — defined once so both writing and reading use identical keys.
CSV_COLUMNS = [
    "agent",         # "PPO", "DQN", or "PULSE-PPO"
    "seed",          # integer 0–4
    "timesteps",     # training steps completed at this checkpoint
    "avg_reward",    # mean total reward across N_EVAL_EPISODES
    "std_reward",    # standard deviation of total reward across N_EVAL_EPISODES
    "trap_rate",     # fraction of eval episodes terminated by trap entry
    "goal_rate",     # fraction of eval episodes terminated by goal reach
    "avg_length",    # mean episode step count across N_EVAL_EPISODES
]


# =============================================================================
# SEEDING
# =============================================================================

def set_global_seed(seed: int) -> None:
    """
    Seeds all random number generators for reproducibility.

    WHY seed all three (numpy, random, torch)?
    SB3 uses all three internally:
      - numpy:  environment sampling, array ops in rollout buffers
      - random: Python stdlib used by some SB3 wrappers
      - torch:  neural network weight initialisation and dropout
    Seeding only numpy while leaving torch random would mean the neural networks
    initialise differently each run — defeating the purpose of seeding.

    WHY seed the OS environment variable too?
    Some PyTorch backends use CUBLAS or other C++ libraries with their own RNG
    state. Setting PYTHONHASHSEED makes Python string/object hashing deterministic,
    which can affect dict ordering in edge cases.

    Args:
        seed: Integer seed to apply globally.
    """
    # WHY str(seed)? PYTHONHASHSEED is an environment variable, expects a string.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)                       # Python stdlib
    np.random.seed(seed)                    # NumPy
    torch.manual_seed(seed)                 # PyTorch CPU
    # WHY torch.cuda.manual_seed_all?
    # Even if we're not using GPU here, seeding CUDA RNG is a no-op when no GPU
    # is present, and costs nothing. It future-proofs the benchmark for GPU runs.
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# ENVIRONMENT FACTORIES
# =============================================================================

def make_vanilla_vec_env(seed: int) -> DummyVecEnv:
    """
    Creates a DummyVecEnv wrapping a single fresh PulseGridWorld.
    Used for PPO and DQN training environments.

    WHY DummyVecEnv (not make_vec_env)?
    make_vec_env uses a callable factory and wraps in DummyVecEnv internally,
    but accessing the inner env requires going through vec_env.get_attr(), which
    is less convenient than vec_env.envs[0]. We build it directly so the access
    pattern is identical for vanilla and PULSE environments.

    Args:
        seed: Passed to the env's first reset() to seed its internal RNG.

    Returns:
        A DummyVecEnv ready for SB3 training.
    """
    # WHY lambda? DummyVecEnv requires a list of callables, not instances.
    # The lambda is called once internally to create the env. We capture seed
    # in the closure so the env is seeded consistently.
    env = PulseGridWorld(render_mode=None)
    env.reset(seed=seed)     # seeds the env's np_random for future resets
    return DummyVecEnv([lambda: env])


def make_pulse_vec_env(seed: int):
    """
    Creates a DummyVecEnv for PULSE-PPO training AND returns the inner env
    reference so its slab can be read by PainShapedPolicy during evaluation.

    WHY return the inner env (not just the VecEnv)?
    The PainShapedPolicy needs direct access to env.slab and env.resistance_field
    to compute aversion scores. These are Python attributes on PulseGridWorld,
    but VecEnv wraps the env and doesn't automatically expose custom attributes
    through the standard VecEnv API. Holding a direct reference is cleaner and
    more efficient than vec_env.get_attr("slab")[0] on every evaluation step.

    Args:
        seed: Env seed.

    Returns:
        (vec_env, inner_env): The VecEnv for SB3 training + the raw env for slab access.
    """
    inner_env = PulseGridWorld(render_mode=None)  # the raw env we'll inspect
    inner_env.reset(seed=seed)
    vec_env = DummyVecEnv([lambda: inner_env])   # SB3 wrapper around it
    return vec_env, inner_env


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def make_ppo(vec_env: DummyVecEnv, seed: int) -> PPO:
    """
    Creates a fresh PPO model with Phase 6 benchmark hyperparameters.

    Hyperparameter rationale:
      n_steps=1000: Aligns exactly with EVAL_EVERY=1000, so each learn(1000)
        call collects exactly one complete rollout before we pause to evaluate.
        The rollout buffer is neither overfull nor partially empty at eval time.
      batch_size=200: 1000/200 = 5 complete minibatches. No truncated minibatch
        warning. (The default batch_size=64 with n_steps=1000 would produce a
        truncated final minibatch of 40 samples — cleaner to avoid that.)
      n_epochs=10: Standard PPO setting. 10 passes over each rollout.
      learning_rate=3e-4: PPO default, well-validated on small envs.
      gamma=0.99: Strong discounting — goal 14 steps away is still valued.
      seed=seed: Seeds SB3's internal numpy/torch RNGs for reproducibility.
    """
    return PPO(
        policy="MlpPolicy",    # flat vector obs → MLP policy
        env=vec_env,
        verbose=0,             # WHY 0? Suppress SB3 training output. We print our own.
        learning_rate=3e-4,
        n_steps=1_000,         # rollout buffer size: aligned with EVAL_EVERY
        batch_size=200,        # 5 clean minibatches from each 1000-step rollout
        n_epochs=10,
        gamma=0.99,
        seed=seed,
    )


def make_dqn(vec_env: DummyVecEnv, seed: int) -> DQN:
    """
    Creates a fresh DQN model with benchmark-appropriate hyperparameters.

    Hyperparameter rationale:
      learning_rate=1e-3: DQN typically uses a higher learning rate than PPO
        because it updates from a replay buffer (not on-policy gradients).
        1e-3 is the standard DQN default from the original Atari paper, adapted.
      buffer_size=10000: Stores the last 10,000 transitions. At 50,000 total
        steps, this is a rolling window of the most recent 20% of experience.
        Large enough to break temporal correlation; small enough to not hold
        too much stale data from early random exploration.
      learning_starts=500: DQN takes random actions for the first 500 steps to
        pre-fill the replay buffer before any gradient updates. Without this,
        the buffer would be nearly empty and every batch would be repetitive.
      batch_size=32: Standard DQN minibatch size. Smaller than PPO because DQN
        updates more frequently (every train_freq steps vs. PPO's per-rollout).
      gamma=0.99: Matches PPO for fair comparison — both agents value future
        rewards identically.
      exploration_fraction=0.4: DQN explores (epsilon-greedy) for the first
        40% of training (20,000 steps), then exploits. This gives DQN enough
        exploration to discover the goal while still converging by 50,000 steps.
      exploration_final_eps=0.05: Ends at 5% random exploration. Never fully
        greedy — maintains some exploration throughout.
      train_freq=4: Update the Q-network every 4 environment steps. More
        frequent updates than DQN's default (1) cause instability; less frequent
        wastes experience. 4 is a standard choice for small discrete envs.
      target_update_interval=500: Sync the target network every 500 steps.
        Longer intervals (e.g. 1000) are more stable but converge slower.
    """
    return DQN(
        policy="MlpPolicy",
        env=vec_env,
        verbose=0,
        learning_rate=1e-3,
        buffer_size=10_000,
        learning_starts=500,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.4,
        exploration_final_eps=0.05,
        train_freq=4,
        target_update_interval=500,
        seed=seed,
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_vanilla(model, n_episodes: int) -> dict:
    """
    Runs N_EVAL_EPISODES on a fresh PulseGridWorld using model.predict().
    Used for both vanilla PPO and vanilla DQN evaluation.

    WHY a fresh eval env (not the training env)?
    The training environment's slab accumulates pain from training. Using it for
    evaluation would mean evaluation reward is influenced by the accumulated pain
    bonus penalties — mixing training state into evaluation metrics. A fresh env
    starts with a cold slab (no pain history) so every eval checkpoint is measured
    under identical physics, making checkpoints comparable to each other.

    WHY deterministic=True?
    We want to measure the agent's LEARNED POLICY, not its exploration behaviour.
    Stochastic predict() would add variance between checkpoints that comes from
    randomness rather than from learning progress — the noise would obscure trends.
    Deterministic (greedy) prediction shows the policy's best guess at each
    checkpoint, which is what we care about.

    Args:
        model:      Trained SB3 PPO or DQN model.
        n_episodes: Number of evaluation episodes.

    Returns:
        dict with keys: avg_reward, std_reward, trap_rate, goal_rate, avg_length.
    """
    eval_env = PulseGridWorld(render_mode=None)
    rewards  = []
    lengths  = []
    trap_eps = 0    # episodes that ended in a trap
    goal_eps = 0    # episodes that reached the goal

    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done   = False
        ep_r   = 0.0
        ep_len = 0

        while not done:
            # WHY obs.reshape(1,-1)?
            # SB3 model.predict() accepts either shape (obs_dim,) or (1, obs_dim).
            # Explicit reshaping avoids shape warnings in some SB3 versions.
            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            ep_r   += reward
            ep_len += 1
            done    = terminated or truncated

        # WHY check final position against goal_pos (not just "not trap")?
        # The episode terminates on BOTH goal and trap. Checking final position
        # is more explicit than inferring from the terminal condition.
        final_pos = (int(obs[0]), int(obs[1]))
        if final_pos == eval_env.goal_pos:
            goal_eps += 1
        # WHY pain_score > 0 to detect trap termination?
        # The environment sets a non-zero pain_score in info only when a trap
        # is entered (apply_spike fires). This is the canonical trap-detection
        # signal without hardcoding trap positions in this script.
        elif info.get("pain_score", 0.0) > 0.0:
            trap_eps += 1

        rewards.append(ep_r)
        lengths.append(ep_len)

    eval_env.close()

    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "trap_rate":  trap_eps / n_episodes,   # fraction of episodes ending in trap
        "goal_rate":  goal_eps / n_episodes,   # fraction of episodes reaching goal
        "avg_length": float(np.mean(lengths)),
    }


def evaluate_pulse(model, training_inner_env, n_episodes: int) -> dict:
    """
    Runs N_EVAL_EPISODES using PainShapedPolicy (aversion filter active).
    The aversion filter reads the TRAINING environment's slab, but the actual
    episodes run in a fresh PulseGridWorld.

    WHY use the TRAINING slab for aversion (not the fresh eval env's slab)?
    The training slab carries the pain history accumulated over the entire
    training run up to this checkpoint. At early checkpoints the slab is mostly
    empty (agent hasn't been hurt much) → aversion is weak. At late checkpoints
    the slab is heavily deformed at trap cells → aversion is strong.
    This creates the WARMUP EFFECT we want to observe: the filter starts nearly
    inactive and grows stronger as training progresses and the slab accumulates
    damage. Using the eval env's fresh slab would always give weak aversion
    (cold slab) — the filter would never kick in, and we'd see no benefit.

    WHY is the eval reward still measured in the FRESH eval env?
    The eval env has its own slab (fresh, cold). Its step() function computes
    rewards using its own slab state (near-zero pain bonus, baseline resistance).
    This means eval rewards are NOT inflated by the training slab's accumulated
    penalties. Every eval checkpoint is thus measured under the same physics —
    the training slab only affects ACTION SELECTION (via the filter), not the
    reward the action receives. This keeps the metrics clean and comparable.

    Args:
        model:              Trained SB3 PPO model.
        training_inner_env: The raw PulseGridWorld used for training (not VecEnv).
                            Its .slab and .resistance_field are read by the filter.
        n_episodes:         Number of evaluation episodes.

    Returns:
        dict with keys: avg_reward, std_reward, trap_rate, goal_rate, avg_length.
    """
    # WHY suppress logging here? PainShapedPolicy emits INFO-level block messages
    # for every vetoed action. During the benchmark, thousands of such messages
    # would scroll past and bury the progress output we actually want to see.
    # We re-enable at the end to avoid affecting other modules.
    pain_logger = logging.getLogger("PULSE.aversion")
    original_level = pain_logger.level
    pain_logger.setLevel(logging.WARNING)   # silence INFO blocks during benchmark

    pulse_policy = PainShapedPolicy(
        model=model,
        slab=training_inner_env.slab,
        resistance_field=training_inner_env.resistance_field,
        grid_size=training_inner_env.grid_size,
    )

    eval_env = PulseGridWorld(render_mode=None)
    rewards  = []
    lengths  = []
    trap_eps = 0
    goal_eps = 0

    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done   = False
        ep_r   = 0.0
        ep_len = 0

        while not done:
            # WHY select_action instead of model.predict?
            # This is the ONLY difference between evaluate_pulse and evaluate_vanilla.
            # The rest of the loop — env.step(), reward accumulation, termination —
            # is identical. Isolating this one substitution means any measured
            # difference in metrics is attributable to the aversion filter alone.
            action, _ = pulse_policy.select_action(obs, threshold=AVERSION_THRESHOLD)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_r   += reward
            ep_len += 1
            done    = terminated or truncated

        final_pos = (int(obs[0]), int(obs[1]))
        if final_pos == eval_env.goal_pos:
            goal_eps += 1
        elif info.get("pain_score", 0.0) > 0.0:
            trap_eps += 1

        rewards.append(ep_r)
        lengths.append(ep_len)

    eval_env.close()
    pain_logger.setLevel(original_level)   # restore logging level

    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "trap_rate":  trap_eps / n_episodes,
        "goal_rate":  goal_eps / n_episodes,
        "avg_length": float(np.mean(lengths)),
    }


# =============================================================================
# CORE TRAINING LOOP
# =============================================================================

def train_one_seed(agent_name: str, seed: int) -> list:
    """
    Trains one agent for one seed and records metrics at every EVAL_EVERY steps.

    WHY a separate function per (agent, seed)?
    Encapsulating one training run makes it easy to parallelise across seeds
    in the future (e.g. multiprocessing.Pool). It also makes error handling
    cleaner — if one seed crashes, the others continue and partial results are saved.

    Args:
        agent_name: "PPO", "DQN", or "PULSE-PPO".
        seed:       Integer seed (0–N_SEEDS-1).

    Returns:
        List of dicts, one per evaluation checkpoint. Each dict has all CSV_COLUMNS.
    """
    set_global_seed(seed)    # seed ALL RNGs before creating any objects

    records = []    # accumulates one dict per eval checkpoint
    t_start = time.time()

    print(f"  [{agent_name}] seed={seed} — starting training...")

    # --- Create training environment and model ---
    if agent_name == "PULSE-PPO":
        # WHY special handling for PULSE-PPO?
        # We need a reference to the inner (unwrapped) env so PainShapedPolicy
        # can read its slab during evaluation. Vanilla agents don't need this.
        vec_env, inner_env = make_pulse_vec_env(seed)
        model = make_ppo(vec_env, seed)
    elif agent_name == "PPO":
        vec_env    = make_vanilla_vec_env(seed)
        inner_env  = None    # not needed for vanilla evaluation
        model      = make_ppo(vec_env, seed)
    elif agent_name == "DQN":
        vec_env    = make_vanilla_vec_env(seed)
        inner_env  = None
        model      = make_dqn(vec_env, seed)
    else:
        raise ValueError(f"Unknown agent name: {agent_name!r}")

    # --- Incremental training loop ---
    # WHY this loop structure (not a single model.learn(TOTAL_TIMESTEPS))?
    # We need to pause at every EVAL_EVERY steps to run evaluation episodes and
    # record metrics. SB3 doesn't support "pause and evaluate every N steps"
    # out of the box without writing a custom Callback — and a Callback can't
    # easily inject the PULSE aversion filter for evaluation. The manual loop
    # gives us full control at the cost of slightly more code.
    steps_trained = 0
    first_call    = True    # track whether to reset SB3's internal step counter

    while steps_trained < TOTAL_TIMESTEPS:
        # WHY min? The last chunk may be smaller than EVAL_EVERY if they don't
        # divide evenly. min() ensures we never train past TOTAL_TIMESTEPS.
        chunk = min(EVAL_EVERY, TOTAL_TIMESTEPS - steps_trained)

        # WHY reset_num_timesteps only on the first call?
        # reset_num_timesteps=True (default) clears SB3's internal step counter.
        # We want the counter to accumulate across calls, so only the first call
        # resets it. Subsequent calls with False continue the counter — this
        # is essential for DQN's epsilon schedule, which uses num_timesteps to
        # decay epsilon over the full TOTAL_TIMESTEPS budget.
        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=first_call,
        )
        first_call     = False
        steps_trained += chunk

        # --- Evaluate at this checkpoint ---
        if agent_name == "PULSE-PPO":
            # WHY evaluate_pulse with inner_env?
            # The aversion filter reads the training slab, which has accumulated
            # pain from all steps_trained steps so far.
            metrics = evaluate_pulse(model, inner_env, N_EVAL_EPISODES)
        else:
            metrics = evaluate_vanilla(model, N_EVAL_EPISODES)

        # WHY build a flat dict (not a nested structure)?
        # The CSV writer expects flat key→value pairs. Flat dicts map directly
        # to CSV rows without needing serialisation of nested objects.
        record = {
            "agent":      agent_name,
            "seed":       seed,
            "timesteps":  steps_trained,
            "avg_reward": metrics["avg_reward"],
            "std_reward": metrics["std_reward"],
            "trap_rate":  metrics["trap_rate"],
            "goal_rate":  metrics["goal_rate"],
            "avg_length": metrics["avg_length"],
        }
        records.append(record)

        # WHY print progress at every checkpoint?
        # With 50 checkpoints per run and 15 total runs, the user should see
        # the benchmark progressing rather than a silent wait. Printing every
        # checkpoint also means the user can kill the run early if needed —
        # partial results are still interpretable.
        elapsed = time.time() - t_start
        print(
            f"  [{agent_name}] seed={seed} | "
            f"t={steps_trained:5d} | "
            f"reward={metrics['avg_reward']:+6.2f} | "
            f"trap={metrics['trap_rate']:.2f} | "
            f"goal={metrics['goal_rate']:.2f} | "
            f"len={metrics['avg_length']:.1f} | "
            f"{elapsed:.1f}s"
        )

    vec_env.close()
    total_time = time.time() - t_start
    print(f"  [{agent_name}] seed={seed} — done in {total_time:.1f}s")
    return records


# =============================================================================
# MAIN BENCHMARK ORCHESTRATION
# =============================================================================

def run_benchmark() -> None:
    """
    Runs all agents across all seeds and saves results to OUTPUT_CSV.

    WHY run agents sequentially (not in parallel)?
    Parallelising across seeds would speed up the benchmark on multi-core machines,
    but adds complexity (process-safe CSV writing, seed isolation across processes).
    Sequential execution is simpler, deterministic, and still completes in a few
    minutes for this environment. Adding multiprocessing is a future optimisation.
    """
    print("=" * 65)
    print("PULSE Phase 6 — Convergence Benchmark")
    print(f"Agents: PPO, DQN, PULSE-PPO  |  Seeds: {N_SEEDS}  |  "
          f"Steps: {TOTAL_TIMESTEPS:,}  |  Eval every: {EVAL_EVERY:,}")
    print("=" * 65)

    # WHY define agent order? PPO first gives us a reference point early.
    # PULSE-PPO last so we can compare against both baselines as they complete.
    agents = ["PPO", "DQN", "PULSE-PPO"]

    all_records = []    # accumulates dicts from every (agent, seed) run

    for agent_name in agents:
        print(f"\n{'─'*65}")
        print(f"Agent: {agent_name}")
        print(f"{'─'*65}")

        for seed in range(N_SEEDS):
            # WHY try/except around each seed run?
            # If one seed crashes (e.g. memory issue, numerical instability),
            # we want the other seeds to complete. Partial results are still
            # usable for analysis — better than crashing the whole benchmark.
            try:
                seed_records = train_one_seed(agent_name, seed)
                all_records.extend(seed_records)

                # WHY save after every seed (not just at the end)?
                # If the benchmark crashes midway, all completed runs are saved
                # and not lost. The CSV is overwritten with increasingly complete
                # data — the final write has everything.
                _save_csv(all_records)
                print(f"  → Saved {len(all_records)} rows to {OUTPUT_CSV}")

            except Exception as exc:
                # WHY print but not re-raise?
                # We prefer partial results over no results. The failure message
                # tells the user which run failed without stopping the others.
                print(f"  [ERROR] {agent_name} seed={seed} failed: {exc}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 65)
    print(f"Benchmark complete. {len(all_records)} rows saved to {OUTPUT_CSV}")
    print("Run plot_benchmark.py to visualise results.")
    print("Run results_summary.py for the text summary.")
    print("=" * 65)


def _save_csv(records: list) -> None:
    """
    Writes all records to OUTPUT_CSV, overwriting any previous version.

    WHY overwrite (not append)?
    Appending would produce duplicate rows if the benchmark is restarted after
    a partial run. Overwriting always produces a clean, complete file reflecting
    all records collected in the CURRENT run.

    WHY csv.DictWriter (not pandas.DataFrame.to_csv)?
    DictWriter is stdlib — no extra imports. It writes incrementally if needed.
    The overhead of loading pandas just to write a CSV is unnecessary here.
    """
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


# WHY __main__ guard?
# Prevents the benchmark from running if this module is imported by plot_benchmark.py
# or results_summary.py (which import OUTPUT_CSV and CSV_COLUMNS from here).
if __name__ == "__main__":
    run_benchmark()
