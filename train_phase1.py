# =============================================================================
# train_phase1.py — PPO Training Script for PulseGridWorld
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 1
# =============================================================================

# WHY: We import our custom environment from the same directory.
# Python treats pulse_env.py as a module, so `from pulse_env import` works as
# long as both files are in the same folder or pulse_env is on the Python path.
from pulse_env import PulseGridWorld

# WHY: PPO (Proximal Policy Optimisation) is our chosen RL algorithm.
# We import it from stable-baselines3, which provides clean, well-tested
# implementations of modern RL algorithms that work with any gymnasium env.
#
# WHY PPO specifically?
# - PPO is the most widely used on-policy algorithm in practice (OpenAI used it
#   for GPT fine-tuning with RLHF, robotics, game agents, etc.)
# - It's stable: the "proximal" constraint prevents the policy from changing too
#   drastically in one update, which avoids catastrophic forgetting.
# - It works well on small discrete environments like this 8x8 grid, where
#   simpler methods like DQN are also viable but PPO requires less tuning.
# - DQN would be another valid choice, but PPO's actor-critic structure gives
#   us a better foundation for future PULSE phases (partial obs, pain signals, etc.)
from stable_baselines3 import PPO

# WHY: make_vec_env wraps our environment in a VecEnv (vectorised environment).
# SB3's PPO expects a VecEnv, not a raw gymnasium env. Vectorisation allows SB3
# to run multiple parallel environment instances simultaneously — even with n_envs=1
# it still wraps the env in the required VecEnv container.
from stable_baselines3.common.env_util import make_vec_env

# WHY: EvalCallback evaluates the current policy periodically during training
# (every eval_freq steps) and saves the best-performing model checkpoint.
# Without this, we'd only have the model at the very end, which might not be
# the best it ever was (RL training can degrade if learning rate is too high).
from stable_baselines3.common.callbacks import EvalCallback

# WHY: We import os to handle file paths in a cross-platform way.
# os.path.join works on both Mac/Linux and Windows, unlike hardcoded "/" slashes.
import os


# =============================================================================
# CONFIGURATION — all training hyperparameters in one place
# =============================================================================
# WHY: Grouping all numbers at the top makes it easy to experiment without
# hunting through code. This is a common pattern in ML research scripts.

# WHY 50,000 timesteps: This is a modest number sufficient for a simple 8x8 grid.
# The shortest optimal path from (0,0) to (7,7) is 14 steps (7 right + 7 down).
# 50,000 timesteps gives the agent ~3,500+ episodes to learn, which is more
# than enough. For complex environments you'd need millions of timesteps.
TOTAL_TIMESTEPS = 50_000

# WHY: Save paths use os.path.join so they work regardless of OS.
# We save the final model and the best model separately so we can compare.
MODEL_SAVE_PATH = os.path.join("models", "ppo_pulse_phase1")
BEST_MODEL_SAVE_PATH = os.path.join("models", "best_ppo_pulse_phase1")
LOG_PATH = os.path.join("logs", "ppo_pulse_phase1")


def make_env():
    """
    WHY this function exists:
    make_vec_env needs a callable that returns a fresh environment instance.
    Wrapping the constructor in a function is the cleanest way to pass it.
    This pattern also lets us add wrappers (e.g. TimeLimit, Monitor) in one place.
    """
    # WHY render_mode=None: During training we don't want to render every step —
    # rendering is thousands of times slower than a pure numerical step() call.
    # We only render during the final visualisation in visualise_phase1.py.
    return PulseGridWorld(render_mode=None)


def train():
    """
    WHY a train() function:
    Wrapping the training logic in a function (rather than bare module-level code)
    means this script can be imported by other modules without immediately
    triggering training. It also makes the entry point (`if __name__ == "__main__"`)
    cleaner and more Pythonic.
    """

    # WHY: Create the directories before training starts so SB3 doesn't crash
    # when it tries to write logs or model files. exist_ok=True means no error
    # if the directory already exists (idempotent, safe to re-run).
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("=" * 60)
    print("PULSE Phase 1 — Training PPO on PulseGridWorld")
    print("=" * 60)

    # WHY: make_vec_env creates a vectorised environment wrapper.
    # n_envs=1 means we use a single environment instance. You could increase this
    # (e.g. n_envs=4) to collect experience from 4 parallel envs simultaneously,
    # which speeds up training on multi-core machines. For this small env,
    # n_envs=1 is sufficient and keeps things simple for learning purposes.
    vec_env = make_vec_env(make_env, n_envs=1)

    # WHY: We create a separate evaluation environment.
    # We NEVER evaluate the policy on the training environment because the agent
    # might overfit to specific rollout patterns. A fresh env gives a clean signal.
    eval_env = make_vec_env(make_env, n_envs=1)

    # --- PPO MODEL DEFINITION ---
    # WHY "MlpPolicy": MLP = Multi-Layer Perceptron (a standard feedforward neural
    # network). This is the right policy class when observations are flat vectors
    # (our (row, col) pair). If observations were images, we'd use "CnnPolicy".
    #
    # Key hyperparameters explained:
    #
    # verbose=1: Print training progress to stdout. verbose=0 is silent,
    #            verbose=2 gives even more detail. During learning, seeing
    #            loss curves and episode rewards is invaluable for debugging.
    #
    # learning_rate=3e-4: This is the Adam optimiser step size. 3e-4 (0.0003)
    #   is the PPO default and widely regarded as a good starting point.
    #   Too high → unstable updates; too low → very slow learning.
    #
    # n_steps=512: How many steps of experience we collect per environment per
    #   update. With n_envs=1, we collect 512 steps before each gradient update.
    #   Larger n_steps means more data per update (more stable) but slower
    #   iteration. 512 is a reasonable default for small environments.
    #
    # batch_size=64: How many experience samples we use per gradient step.
    #   SB3 will split the n_steps=512 buffer into mini-batches of 64 to
    #   make multiple gradient updates from one collection round. This is the
    #   core of PPO's efficiency — reusing collected data multiple times.
    #
    # n_epochs=10: How many times we iterate over the collected buffer per
    #   update cycle. More epochs = more learning from each batch, but risks
    #   overfitting to the collected data. 10 is the SB3 default.
    #
    # gamma=0.99: The discount factor. Future rewards are worth 99% of present
    #   rewards. gamma=1.0 means all future rewards matter equally; gamma=0.0
    #   means only immediate rewards matter. 0.99 makes the agent care about
    #   the goal (which is ~14 steps away) while slightly preferring faster paths.
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=LOG_PATH,   # WHY: log to TensorBoard so we can visualise
                                    # reward curves, loss, entropy over training.
                                    # Run `tensorboard --logdir logs/` to view.
    )

    # --- EVALUATION CALLBACK ---
    # WHY EvalCallback:
    # During training, the agent's policy improves non-monotonically — it can
    # temporarily get worse before getting better. EvalCallback runs the current
    # policy on eval_env every eval_freq steps and saves the model if it beats
    # the previous best mean reward. This ensures we keep the best checkpoint.
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=BEST_MODEL_SAVE_PATH,
        log_path=LOG_PATH,
        eval_freq=2_000,        # WHY: evaluate every 2,000 training steps.
                                # Too frequent → slow; too rare → miss improvements.
        n_eval_episodes=10,     # WHY: average over 10 episodes for a stable estimate.
                                # One episode of 8x8 is fast, so 10 is cheap.
        deterministic=True,     # WHY: evaluate with the greedy (argmax) policy,
                                # not the stochastic training policy. This gives
                                # a clean picture of what the agent "knows" vs
                                # what it randomly explores.
        render=False,           # WHY: no rendering during eval — speed matters.
    )

    print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")
    print(f"Best model will be saved to: {BEST_MODEL_SAVE_PATH}")
    print(f"TensorBoard logs at: {LOG_PATH}")
    print("-" * 60)

    # WHY: .learn() is the core training call. It runs the PPO algorithm for
    # exactly TOTAL_TIMESTEPS environment steps. The callback fires automatically
    # at the intervals we specified above — we don't need to manage it manually.
    # progress_bar=True gives a nice tqdm bar showing training progress.
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
    )

    print("-" * 60)
    print("Training complete!")

    # WHY: Save the *final* model in addition to the best checkpoint.
    # The best checkpoint (saved by EvalCallback) might be from mid-training.
    # The final model is what the policy looks like at the very end of training.
    # Comparing them tells us if the policy improved or degraded late in training.
    model.save(MODEL_SAVE_PATH)
    print(f"Final model saved to: {MODEL_SAVE_PATH}.zip")
    print(f"Best model saved to:  {BEST_MODEL_SAVE_PATH}/best_model.zip")

    # WHY: Clean up the VecEnv properly. This closes any background processes or
    # shared memory that the vectorised environment may have opened.
    vec_env.close()
    eval_env.close()

    return model


# WHY: This guard ensures train() only runs when this script is executed directly
# (e.g. `python train_phase1.py`), NOT when it's imported by another module
# (e.g. visualise_phase1.py). Without this guard, importing train_phase1 would
# immediately start training, which is almost never what you want.
if __name__ == "__main__":
    train()
