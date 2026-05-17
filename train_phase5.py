# =============================================================================
# train_phase5.py — PPO Training + PULSE Evaluation for PulseGridWorld
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 5
# =============================================================================
#
# HOW this differs from train_phase1.py:
#   TRAINING:    Identical to Phase 1 — vanilla PPO with no modifications.
#                The aversion mechanism is NOT applied during training.
#                See pain_shaped_policy.py for the WHY in detail.
#   EVALUATION:  After training, a second evaluation loop runs 20 episodes using
#                PainShapedPolicy.select_action() instead of model.predict().
#                This gives us a direct before/after comparison on the same model.
#
# WHY keep training vanilla even though we have the aversion system?
#   The short answer: training the PPO model with the filter active would break
#   the on-policy guarantee. The filter changes which actions are actually taken,
#   but PPO's gradient update assumes the POLICY generated those actions.
#   If the filter suppresses an action the policy preferred, the policy never
#   gets a gradient update reflecting "I tried to go that direction but was
#   blocked" — it just sees "I went a different direction and got this reward."
#   Over time, the policy would learn to prefer the filter-allowed actions for
#   the WRONG reason (because they got higher rewards, not because they were
#   intrinsically better). This conflation makes the policy's behaviour at
#   deployment time unpredictable when the filter is removed or changed.
#   Keeping training vanilla preserves the policy's integrity: it learns from
#   honest experience. The filter is then tested as a deployment-time hypothesis.
# =============================================================================

# WHY: Import our Phase 5 environment (includes Phases 3 and 4 mechanics).
from pulse_env import PulseGridWorld

# WHY: PPO is unchanged — vanilla algorithm, same as Phase 1.
from stable_baselines3 import PPO

# WHY: make_vec_env wraps our env in the VecEnv container SB3 requires.
from stable_baselines3.common.env_util import make_vec_env

# WHY: EvalCallback for periodic best-model saving during training.
from stable_baselines3.common.callbacks import EvalCallback

# WHY: PainShapedPolicy is the Phase 5 core — it wraps the trained model
# with aversion filtering for the post-training evaluation section.
from pain_shaped_policy import PainShapedPolicy

# WHY: logging gives structured, levelled output. We want PULSE messages
# to appear clearly in training logs without cluttering SB3's own output.
import logging

# WHY: numpy for reward/stat aggregation in the evaluation loop.
import numpy as np

# WHY: os for path handling across platforms.
import os


# --- LOGGING SETUP ---
# WHY basicConfig at INFO level?
# INFO shows blocked-action messages from PainShapedPolicy during evaluation
# while filtering out DEBUG noise from SB3 internals. This keeps the evaluation
# output focused on the PULSE mechanism we care about observing.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# WHY 50,000 timesteps? Same as Phase 1 — the grid is simple enough that
# a well-trained policy emerges in this budget. Keeping it identical to Phase 1
# ensures any performance difference in evaluation comes from the aversion filter,
# not from more training.
TOTAL_TIMESTEPS = 50_000

# WHY phase5 naming? We save separately from phase1 so both model checkpoints
# coexist and can be loaded for cross-phase comparison in compare_phase5.py.
MODEL_SAVE_PATH      = os.path.join("models", "ppo_pulse_phase5")
BEST_MODEL_SAVE_PATH = os.path.join("models", "best_ppo_pulse_phase5")
LOG_PATH             = os.path.join("logs",   "ppo_pulse_phase5")

# WHY 20 evaluation episodes (not 10)?
# Phase 5 evaluation is more informative than standard SB3 eval because we
# print per-step diagnostics. 20 episodes gives enough samples to see the
# aversion filter firing multiple times while keeping console output manageable.
N_EVAL_EPISODES = 20


def make_env():
    """
    WHY this function:
    make_vec_env needs a zero-argument callable returning a fresh env.
    Wrapping the constructor here also lets us add wrappers in one place later.
    """
    # WHY render_mode=None: No rendering during training — speed is paramount.
    return PulseGridWorld(render_mode=None)


def train() -> PPO:
    """
    Trains a vanilla PPO agent on PulseGridWorld (Phase 5 mechanics).
    Returns the trained model for use in the evaluation section below.
    """
    # WHY makedirs with exist_ok=True?
    # Safe to re-run: if directories already exist from a prior run, no error.
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    print("=" * 60)
    print("PULSE Phase 5 — Training PPO (vanilla) on PulseGridWorld")
    print("Phase 4 mechanics active: resistance field + slab pain")
    print("=" * 60)

    # WHY vec_env and eval_env are separate?
    # eval_env is a fresh instance that has not been stepped through training.
    # EvalCallback's periodic evaluations are uncontaminated by training state.
    vec_env  = make_vec_env(make_env, n_envs=1)
    eval_env = make_vec_env(make_env, n_envs=1)

    # WHY these PPO hyperparameters? Identical to Phase 1 for fair comparison.
    # Any performance difference between phases is due to the environment mechanics
    # and evaluation-time aversion, not different training hyperparameters.
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=LOG_PATH,
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=BEST_MODEL_SAVE_PATH,
        log_path=LOG_PATH,
        eval_freq=2_000,
        n_eval_episodes=10,
        deterministic=True,     # greedy policy for clean evaluation signal
        render=False,
    )

    print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")
    print(f"Best model  saved to:   {BEST_MODEL_SAVE_PATH}")
    print("-" * 60)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
    )

    print("-" * 60)
    print("Training complete.")
    model.save(MODEL_SAVE_PATH)
    print(f"Final model saved to:  {MODEL_SAVE_PATH}.zip")
    print(f"Best model  saved to:  {BEST_MODEL_SAVE_PATH}/best_model.zip")

    vec_env.close()
    eval_env.close()

    return model


def run_pulse_evaluation(model: PPO, n_episodes: int = N_EVAL_EPISODES) -> dict:
    """
    Runs evaluation episodes using PainShapedPolicy (aversion filter active).

    WHY a separate evaluation function (not a custom SB3 Callback)?
    SB3 callbacks call model.predict() internally — they don't expose a hook
    to substitute a custom action selection function. A manual evaluation loop
    gives us full control: we call select_action() ourselves and can log every
    step's aversion diagnostics.

    WHY use a FRESH environment for evaluation (not the training env)?
    The training environment's slab carries training history. Using a fresh env
    gives the evaluation a clean slab state — the aversion filter starts with
    no accumulated pain and must build it up across these evaluation episodes.
    This is the COLDSTART scenario: does the filter help even from scratch?
    In compare_phase5.py we also test WARMSTART (pre-loaded slab from training).

    Args:
        model:      Trained SB3 PPO model.
        n_episodes: Number of evaluation episodes to run.

    Returns:
        dict with keys:
          "rewards":       list of total episode rewards
          "lengths":       list of episode step counts
          "trap_entries":  total number of trap cell visits across all episodes
          "goal_reaches":  total number of successful goal arrivals
    """
    print("\n" + "=" * 60)
    print(f"PULSE Phase 5 — Evaluation with PainShapedPolicy")
    print(f"Running {n_episodes} episodes (aversion filter ACTIVE)")
    print("=" * 60)

    # WHY a fresh PulseGridWorld (not make_vec_env)?
    # VecEnv wraps the environment and intercepts reset/step calls, which
    # would complicate direct attribute access (env.slab, env.resistance_field).
    # For evaluation where we query these objects directly, a raw env is cleaner.
    env = PulseGridWorld(render_mode=None)
    obs, _ = env.reset()

    # WHY create PainShapedPolicy here (not reuse a global one)?
    # The env's slab is a fresh instance — it starts with zero pain. We pass
    # the same slab and resistance_field objects that the env owns, so any pain
    # accumulated during evaluation is immediately visible to the filter.
    pulse_policy = PainShapedPolicy(
        model=model,
        slab=env.slab,
        resistance_field=env.resistance_field,
        grid_size=env.grid_size,
    )

    # WHY collect these metrics? They are the exact same stats compare_phase5.py
    # measures, so the numbers are directly comparable.
    episode_rewards  = []   # total reward per episode
    episode_lengths  = []   # step count per episode
    total_trap_entries = 0  # trap hits across all episodes
    total_goal_reaches = 0  # successful goal arrivals

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_steps  = 0
        done = False

        print(f"\n--- Episode {ep+1}/{n_episodes} ---")

        while not done:
            # WHY select_action (not model.predict)?
            # This is the key Phase 5 difference: the aversion filter intercepts
            # the policy's preferred action and may redirect it.
            action, step_info = pulse_policy.select_action(obs, threshold=0.5)

            # WHY print per-step diagnostics?
            # The aversion mechanism is the research contribution of Phase 5.
            # Printing every blocked action makes the mechanism OBSERVABLE — we
            # can verify it is working as designed and trace the agent's reasoning.
            blocked = step_info["blocked"]
            if blocked:
                blocked_str = ", ".join(
                    f"{a}({step_info['target_cells'][a]})"
                    for a in blocked
                )
                print(
                    f"  step={episode_steps+1:3d} pos=({int(obs[0])},{int(obs[1])}) "
                    f"BLOCKED: {blocked_str} → chose action {action}"
                )

            # WHY unpack as 5 values? gymnasium v26+ step() returns
            # (obs, reward, terminated, truncated, info). Both terminated and
            # truncated must be checked to know if the episode is over.
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps  += 1

            # WHY check info for trap detection (not just terminated)?
            # Traps terminate episodes, but the info dict from pulse_env.py
            # carries pain_score > 0 only on trap entries. This lets us count
            # trap hits from the info without hardcoding trap positions here.
            if info.get("pain_score", 0.0) > 0.0:
                total_trap_entries += 1

            done = terminated or truncated

        # WHY check goal_pos directly?
        # The environment terminates on BOTH goal and trap. We need to
        # distinguish which terminal state we reached. Checking the final
        # obs against goal_pos is the cleanest way without adding env coupling.
        final_pos = (int(obs[0]), int(obs[1]))
        if final_pos == env.goal_pos:
            total_goal_reaches += 1
            print(f"  → GOAL reached in {episode_steps} steps | reward={episode_reward:.2f}")
        else:
            print(f"  → TRAP/timeout at {final_pos} in {episode_steps} steps | reward={episode_reward:.2f}")

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)

    # WHY print summary stats? They let us quickly compare PULSE vs. vanilla
    # without running compare_phase5.py — useful during development.
    print("\n" + "=" * 60)
    print(f"PULSE Evaluation Summary ({n_episodes} episodes)")
    print("=" * 60)
    print(f"  Mean reward:       {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean ep length:    {np.mean(episode_lengths):.1f} steps")
    print(f"  Goal reach rate:   {total_goal_reaches}/{n_episodes} ({100*total_goal_reaches/n_episodes:.0f}%)")
    print(f"  Trap entry count:  {total_trap_entries} total across all episodes")
    pulse_policy.print_stats()

    env.close()

    return {
        "rewards":      episode_rewards,
        "lengths":      episode_lengths,
        "trap_entries": total_trap_entries,
        "goal_reaches": total_goal_reaches,
    }


# WHY: This guard ensures train() only runs when the script is executed directly.
if __name__ == "__main__":
    # WHY: Train first, then immediately evaluate. The trained model's slab
    # carries all the pain history from training — the evaluation demonstrates
    # the filter on a WARMSTART slab, which is the realistic deployment scenario.
    trained_model = train()
    run_pulse_evaluation(trained_model, n_episodes=N_EVAL_EPISODES)
