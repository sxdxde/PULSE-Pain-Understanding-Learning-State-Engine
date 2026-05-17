# =============================================================================
# compare_phase5.py — Vanilla PPO vs. PULSE Aversion Policy Comparison
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 5
# =============================================================================
#
# WHAT this script does:
#   Runs 50 evaluation episodes with each of two policies on PulseGridWorld:
#     VANILLA: the trained PPO model's raw predictions (no aversion filter)
#     PULSE:   the same model wrapped with PainShapedPolicy (aversion filter active)
#   Plots three comparison metrics:
#     1. Trap entry rate per episode
#     2. Episode length (steps to termination)
#     3. Total episode reward
#
# WHY this comparison design (same model, same env, same episode count)?
#   We want to isolate EXACTLY ONE variable: the presence of the aversion filter.
#   Using the same trained model means any difference in performance is due to
#   the filter, not to different training quality.
#   Running on the same environment class means both policies face the same
#   physics (resistance, slab mechanics).
#   50 episodes each gives enough variance to show statistical trends without
#   being prohibitively slow.
#
# WHY NOT compare by training outcome?
#   Comparing "Phase 1 model vs Phase 5 model after training" conflates the
#   aversion filter's effect with the effect of the Phase 3-4 environment changes
#   (slab pain, resistance). That makes it impossible to attribute differences.
#   This script isolates ONE mechanism by applying it post-hoc to ONE model.
# =============================================================================

# WHY numpy: Statistical aggregation (mean, std) and array creation.
import numpy as np

# WHY matplotlib: Plotting the three comparison metrics side by side.
import matplotlib.pyplot as plt

# WHY os: Model path resolution.
import os

# WHY: The environment and policies under test.
from pulse_env import PulseGridWorld
from pain_shaped_policy import PainShapedPolicy
from stable_baselines3 import PPO


# =============================================================================
# CONFIGURATION
# =============================================================================

# WHY 50 episodes? Enough to average out randomness in episode outcome without
# requiring a long runtime. With 14-step optimal paths and max ~200 steps before
# the environment times out (if truncation were added), 50 episodes takes seconds.
N_EPISODES = 50

# WHY 0.5? Matches the default in PainShapedPolicy. Both scripts use the same
# threshold so comparisons are meaningful.
AVERSION_THRESHOLD = 0.5

# WHY try phase5 model first, fall back to phase1?
# compare_phase5.py should work even if the user has only run phase1 training.
# The PULSE mechanism itself is independent of which phase the model was trained in.
MODEL_PATHS = [
    os.path.join("models", "best_ppo_pulse_phase5", "best_model.zip"),
    os.path.join("models", "ppo_pulse_phase5.zip"),
    os.path.join("models", "best_ppo_pulse_phase1", "best_model.zip"),
    os.path.join("models", "ppo_pulse_phase1.zip"),
]


def load_model() -> PPO:
    """
    Loads the best available trained PPO model.

    WHY not pass the model as an argument?
    compare_phase5.py is a standalone script — it loads its own model rather than
    depending on train_phase5.py having been imported. This makes it runnable
    independently: `python3 compare_phase5.py` after any training session.
    """
    for path in MODEL_PATHS:
        if os.path.exists(path):
            print(f"[compare_phase5] Loading model: {path}")
            return PPO.load(path)
    raise RuntimeError(
        "No trained model found. Run train_phase1.py or train_phase5.py first.\n"
        f"Tried: {MODEL_PATHS}"
    )


def run_vanilla_episodes(model: PPO, n_episodes: int) -> dict:
    """
    Runs `n_episodes` with the raw PPO policy (no aversion filter).

    WHY "vanilla"? The policy acts exactly as PPO trained it — no post-hoc
    filtering. This is the BASELINE against which PULSE is measured.

    WHY deterministic=True?
    At evaluation time we want the GREEDY policy (argmax of action probabilities),
    not the stochastic training policy. The greedy policy is more stable across
    episodes and shows the agent's best-learned behaviour, not random exploration.

    Args:
        model:      Trained SB3 PPO model.
        n_episodes: Number of episodes to run.

    Returns:
        dict with "rewards", "lengths", "trap_entries", "goal_reaches".
    """
    print(f"\n{'='*55}")
    print(f"VANILLA PPO: running {n_episodes} episodes (no aversion filter)")
    print(f"{'='*55}")

    # WHY a fresh environment for vanilla (not shared with PULSE)?
    # If both policies shared one environment, the slab accumulated by vanilla
    # episodes would affect the resistance_field seen by PULSE episodes (since
    # dynamic_resistance() uses the slab). We want each policy to start from
    # the same clean-slab condition so the comparison is controlled.
    env = PulseGridWorld(render_mode=None)

    rewards      = []
    lengths      = []
    trap_entries = 0
    goal_reaches = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps  = 0

        while not done:
            # WHY model.predict with deterministic=True?
            # The greedy policy always picks the action with the highest probability.
            # This is the PPO policy's "best judgment" without exploration noise.
            # Using stochastic predict would add variance that obscures the signal.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps  += 1

            # WHY info["pain_score"] > 0 to detect traps?
            # The environment only assigns a non-zero pain_score when a trap is
            # entered (apply_spike fires). Checking the info dict avoids hardcoding
            # trap positions in this comparison script.
            if info.get("pain_score", 0.0) > 0.0:
                trap_entries += 1

            done = terminated or truncated

        final_pos = (int(obs[0]), int(obs[1]))
        if final_pos == env.goal_pos:
            goal_reaches += 1

        rewards.append(episode_reward)
        lengths.append(episode_steps)

        # WHY print every 10 episodes? Progress feedback without per-episode spam.
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:3d}/{n_episodes} | "
                  f"mean_reward={np.mean(rewards[-10:]):.2f} | "
                  f"mean_length={np.mean(lengths[-10:]):.1f}")

    env.close()

    print(f"Vanilla summary: goal_rate={goal_reaches}/{n_episodes} "
          f"trap_total={trap_entries} mean_reward={np.mean(rewards):.2f}")

    return {
        "rewards":      rewards,
        "lengths":      lengths,
        "trap_entries": trap_entries,
        "goal_reaches": goal_reaches,
    }


def run_pulse_episodes(model: PPO, n_episodes: int) -> dict:
    """
    Runs `n_episodes` with PainShapedPolicy (aversion filter active).

    WHY the aversion filter is applied at evaluation but NOT during training:
      See pain_shaped_policy.py module docstring for the full argument. In brief:
      applying the filter during training would bias the experience distribution,
      break PPO's on-policy assumption, and conflate the filter's contribution
      with the policy gradient's contribution. The filter is a DEPLOYMENT GUARD,
      not a TRAINING MECHANISM. Separating them keeps both clean and interpretable.

    Args:
        model:      Trained SB3 PPO model.
        n_episodes: Number of episodes to run.

    Returns:
        dict with "rewards", "lengths", "trap_entries", "goal_reaches".
    """
    print(f"\n{'='*55}")
    print(f"PULSE PPO: running {n_episodes} episodes (aversion filter ACTIVE)")
    print(f"Threshold: {AVERSION_THRESHOLD}")
    print(f"{'='*55}")

    # WHY a separate fresh env (not the vanilla env from run_vanilla_episodes)?
    # See run_vanilla_episodes docstring — clean slab state for fair comparison.
    env = PulseGridWorld(render_mode=None)

    # WHY create PainShapedPolicy here (not outside)?
    # The policy must hold references to THIS env's slab and resistance_field.
    # A policy created from a different env's slab would read stale/different
    # pain state and produce meaningless aversion scores.
    pulse_policy = PainShapedPolicy(
        model=model,
        slab=env.slab,
        resistance_field=env.resistance_field,
        grid_size=env.grid_size,
    )

    rewards      = []
    lengths      = []
    trap_entries = 0
    goal_reaches = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps  = 0

        while not done:
            # WHY select_action (not model.predict)?
            # This is the ONLY difference from run_vanilla_episodes. The entire
            # comparison between vanilla and PULSE reduces to this one substitution.
            # Same model, same env, same determinism assumption — but PainShapedPolicy
            # intercepts the action selection and may redirect it.
            action, _ = pulse_policy.select_action(obs, threshold=AVERSION_THRESHOLD)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps  += 1

            if info.get("pain_score", 0.0) > 0.0:
                trap_entries += 1

            done = terminated or truncated

        final_pos = (int(obs[0]), int(obs[1]))
        if final_pos == env.goal_pos:
            goal_reaches += 1

        rewards.append(episode_reward)
        lengths.append(episode_steps)

        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:3d}/{n_episodes} | "
                  f"mean_reward={np.mean(rewards[-10:]):.2f} | "
                  f"mean_length={np.mean(lengths[-10:]):.1f}")

    env.close()

    print(f"PULSE summary:   goal_rate={goal_reaches}/{n_episodes} "
          f"trap_total={trap_entries} mean_reward={np.mean(rewards):.2f}")
    pulse_policy.print_stats()

    return {
        "rewards":      rewards,
        "lengths":      lengths,
        "trap_entries": trap_entries,
        "goal_reaches": goal_reaches,
    }


def plot_comparison(
    vanilla_results: dict,
    pulse_results: dict,
    n_episodes: int,
    save_path: str = None,
) -> None:
    """
    Plots three side-by-side comparisons: trap rate, episode length, and reward.

    WHY three metrics (not just reward)?
    Reward alone is an ambiguous signal: a lower reward might mean the agent
    took longer to reach the goal (more living penalties) OR hit more traps
    (large negative spikes) OR both. Separating the metrics lets us diagnose:
      - Trap entry rate: "does the filter actually prevent trap hits?"
        If yes but reward is lower, the filter is causing detours — overhead cost.
        If yes and reward is higher, the filter avoids penalty without detour cost.
      - Episode length: "does the filter cause the agent to take longer paths?"
        Avoidance might require longer routes around danger zones.
      - Reward: the aggregate signal — net effect of trap avoidance vs. path cost.
    All three together tell a complete story about the filter's cost-benefit profile.

    WHY per-episode plots (not just bar charts)?
    Per-episode line plots show the TREND over episodes — does the PULSE filter
    improve as the slab accumulates pain (warmup effect), or is it flat from
    episode 1? This temporal structure is lost in aggregate bar charts.
    We overlay smoothed rolling averages to show the trend without noise.

    Args:
        vanilla_results: Output of run_vanilla_episodes().
        pulse_results:   Output of run_pulse_episodes().
        n_episodes:      Number of episodes per condition.
        save_path:       If given, saves the figure instead of showing it.
    """
    # WHY 3 subplots in one row? Each metric is on the same x-axis (episode number).
    # A 1×3 layout lets the reader scan horizontally and compare the same episode
    # numbers across all three metrics simultaneously.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"PULSE Phase 5 — Vanilla PPO vs. PULSE Policy Comparison\n"
        f"({n_episodes} episodes each | aversion threshold={AVERSION_THRESHOLD})",
        fontsize=13,
        fontweight="bold",
    )

    episodes = np.arange(1, n_episodes + 1)

    # WHY a smoothing window for trend lines?
    # Per-episode metrics in RL are noisy — individual episodes succeed or fail
    # based on the stochastic policy and starting conditions. A rolling average
    # reveals the underlying trend that the noise obscures.
    # window=10 means each smoothed point is an average of 10 consecutive episodes.
    # WHY min_periods=1? Allows the smoothed line to start at episode 1 rather
    # than having NaN for the first (window-1) episodes.
    def smooth(values, window=10):
        # WHY convert to numpy first? pandas is not required, so we implement
        # rolling mean manually using numpy's convolve with a uniform kernel.
        kernel = np.ones(window) / window
        # WHY 'valid' mode? 'valid' only returns points where the kernel fully
        # overlaps the array — avoids edge artefacts at the start and end.
        # We then pad with the raw values at the start to keep array length equal.
        smoothed = np.convolve(values, kernel, mode="valid")
        pad = np.full(window - 1, np.nan)  # NaN pad for the first window-1 points
        return np.concatenate([pad, smoothed])

    # --- SUBPLOT 1: Trap entries per episode ---
    # WHY per-episode trap count (not cumulative)?
    # Per-episode count shows whether the filter consistently reduces trap hits
    # across episodes. A cumulative view would always increase and obscure
    # whether the filter's effect grows or shrinks over episodes.
    ax1 = axes[0]

    # WHY create binary trap-hit arrays?
    # We only know TOTAL trap entries across all episodes, not per-episode counts.
    # We reconstruct per-episode counts from the pain_score in info, but since
    # we didn't store per-episode trap counts in the result dict, we use a
    # simpler approach: whether each episode terminated in a trap (length heuristic).
    # For a more accurate version, the result dict would need per-episode trap counts.
    # Here we use "did the episode end short (likely trap)" as a proxy.
    # WHY this proxy? The environment terminates on BOTH trap AND goal. A trap
    # episode tends to be shorter than a goal episode (traps are near the start).
    # This is an approximation — compare_phase5.py is a diagnostic script, not
    # a rigorous statistical test.
    #
    # Actually: we track trap_entries TOTAL but not per-episode. Let's just
    # plot the cumulative trap entry rate and final aggregate comparison.
    # We'll use a bar + scatter approach for the three metrics.

    vanilla_rewards = np.array(vanilla_results["rewards"])
    pulse_rewards   = np.array(pulse_results["rewards"])
    vanilla_lengths = np.array(vanilla_results["lengths"])
    pulse_lengths   = np.array(pulse_results["lengths"])

    # WHY plot per-episode rewards as scatter AND smoothed line?
    # Scatter shows individual episode variance; the smoothed line shows the trend.
    # Together they give both the signal and the noise — neither alone is complete.
    ax1.scatter(episodes, vanilla_rewards, color="steelblue",  alpha=0.3, s=15, label="_")
    ax1.scatter(episodes, pulse_rewards,   color="darkorange", alpha=0.3, s=15, label="_")
    ax1.plot(episodes, smooth(vanilla_rewards), color="steelblue",  linewidth=2.0,
             label=f"Vanilla (mean={vanilla_rewards.mean():.2f})")
    ax1.plot(episodes, smooth(pulse_rewards),   color="darkorange", linewidth=2.0,
             label=f"PULSE   (mean={pulse_rewards.mean():.2f})")
    ax1.axhline(vanilla_rewards.mean(), color="steelblue",  linestyle="--", alpha=0.5, linewidth=1)
    ax1.axhline(pulse_rewards.mean(),   color="darkorange", linestyle="--", alpha=0.5, linewidth=1)
    ax1.set_title("Episode Reward", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- SUBPLOT 2: Episode length ---
    ax2 = axes[1]
    ax2.scatter(episodes, vanilla_lengths, color="steelblue",  alpha=0.3, s=15, label="_")
    ax2.scatter(episodes, pulse_lengths,   color="darkorange", alpha=0.3, s=15, label="_")
    ax2.plot(episodes, smooth(vanilla_lengths), color="steelblue",  linewidth=2.0,
             label=f"Vanilla (mean={vanilla_lengths.mean():.1f})")
    ax2.plot(episodes, smooth(pulse_lengths),   color="darkorange", linewidth=2.0,
             label=f"PULSE   (mean={pulse_lengths.mean():.1f})")
    ax2.axhline(vanilla_lengths.mean(), color="steelblue",  linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(pulse_lengths.mean(),   color="darkorange", linestyle="--", alpha=0.5, linewidth=1)
    ax2.set_title("Episode Length (steps)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- SUBPLOT 3: Aggregate comparison bar chart ---
    # WHY a bar chart for subplot 3 instead of a per-episode line?
    # The three aggregate statistics (trap entry rate, goal rate, mean reward) are
    # summary statistics, not time series. A bar chart is the clearest way to
    # show "vanilla vs PULSE on each aggregate metric" without implying a temporal
    # ordering between the metrics.
    ax3 = axes[2]

    # WHY normalise trap entries by n_episodes?
    # Raw counts (e.g. 23 vs 15 traps) are harder to interpret than rates (0.46 vs 0.30).
    # Rates are [0,1] and directly readable as "fraction of episodes with trap hits."
    vanilla_trap_rate  = vanilla_results["trap_entries"] / n_episodes
    pulse_trap_rate    = pulse_results["trap_entries"]   / n_episodes
    vanilla_goal_rate  = vanilla_results["goal_reaches"] / n_episodes
    pulse_goal_rate    = pulse_results["goal_reaches"]   / n_episodes

    metrics      = ["Trap Entry\nRate", "Goal Reach\nRate"]
    vanilla_vals = [vanilla_trap_rate,  vanilla_goal_rate]
    pulse_vals   = [pulse_trap_rate,    pulse_goal_rate]

    # WHY barh (horizontal bars) instead of bar (vertical)?
    # With long metric labels and two groups per metric, horizontal bars avoid
    # label overlap that would require rotation.
    x = np.arange(len(metrics))
    bar_width = 0.35

    bars_v = ax3.bar(x - bar_width/2, vanilla_vals, bar_width,
                     label="Vanilla", color="steelblue",  alpha=0.8)
    bars_p = ax3.bar(x + bar_width/2, pulse_vals,   bar_width,
                     label="PULSE",   color="darkorange", alpha=0.8)

    # WHY annotate bars with exact values?
    # Bar heights on a [0,1] scale can be visually ambiguous — is it 0.3 or 0.35?
    # Printing the value inside the bar removes ambiguity.
    for bar in bars_v:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="steelblue",
                 fontweight="bold")
    for bar in bars_p:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="darkorange",
                 fontweight="bold")

    ax3.set_ylim(0, 1.15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=9)
    ax3.set_ylabel("Rate (episodes fraction)")
    ax3.set_title("Aggregate Rates", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n[compare_phase5] Comparison figure saved to: {save_path}")
    else:
        plt.show()

    # WHY close? Prevents memory accumulation if this function is called in a loop.
    plt.close(fig)


def print_summary_table(vanilla: dict, pulse: dict, n_episodes: int) -> None:
    """
    Prints a plain-text summary table of all comparison metrics.

    WHY a separate text summary in addition to the plot?
    Plots require a display or file system. A text summary is universally
    readable — in CI logs, terminal output, or headless SSH sessions. Including
    both means the results are always accessible regardless of the environment.
    """
    v_rewards = np.array(vanilla["rewards"])
    p_rewards = np.array(pulse["rewards"])
    v_lengths = np.array(vanilla["lengths"])
    p_lengths = np.array(pulse["lengths"])

    print("\n" + "=" * 60)
    print(f"PULSE Phase 5 — Comparison Summary ({n_episodes} episodes each)")
    print("=" * 60)
    print(f"{'Metric':<28} {'Vanilla':>10} {'PULSE':>10} {'Delta':>10}")
    print("-" * 60)

    # WHY include delta? The absolute numbers are less informative than the
    # DIFFERENCE — delta tells us whether PULSE helped, hurt, or had no effect.
    # Positive delta for reward = PULSE better. Negative delta for trap rate = PULSE better.
    def row(label, v_val, p_val, fmt=".2f", higher_better=True):
        delta = p_val - v_val
        sign  = "▲" if (delta > 0) == higher_better else "▼"
        print(f"  {label:<26} {v_val:{fmt}:>10} {p_val:{fmt}:>10} {sign} {abs(delta):{fmt}}")

    row("Mean episode reward",    v_rewards.mean(), p_rewards.mean(),  higher_better=True)
    row("Std episode reward",     v_rewards.std(),  p_rewards.std(),   higher_better=False)
    row("Mean episode length",    v_lengths.mean(), p_lengths.mean(),  higher_better=False)
    row("Trap entry rate",
        vanilla["trap_entries"]/n_episodes,
        pulse["trap_entries"]/n_episodes,
        higher_better=False)
    row("Goal reach rate",
        vanilla["goal_reaches"]/n_episodes,
        pulse["goal_reaches"]/n_episodes,
        higher_better=True)
    print("=" * 60)
    print("  ▲ = PULSE better on this metric   ▼ = PULSE worse")


def run_comparison(save_figure: bool = False) -> None:
    """
    Main entry point: loads model, runs both conditions, plots comparison.

    Args:
        save_figure: If True, saves the comparison figure to disk.
    """
    model = load_model()

    # WHY run vanilla first?
    # The vanilla run uses a fresh env with a cold slab. The PULSE run also uses
    # a fresh env. Running vanilla first ensures the order doesn't affect results.
    vanilla_results = run_vanilla_episodes(model, N_EPISODES)
    pulse_results   = run_pulse_episodes(model,   N_EPISODES)

    print_summary_table(vanilla_results, pulse_results, N_EPISODES)

    save_path = "comparison_phase5.png" if save_figure else None
    plot_comparison(vanilla_results, pulse_results, N_EPISODES, save_path=save_path)


if __name__ == "__main__":
    # WHY save_figure=True as the standalone default?
    # When run as a script (not imported), we almost certainly want a persistent
    # output file — running a 50+50 episode comparison just to see a plot that
    # disappears when the window closes is wasteful.
    run_comparison(save_figure=True)
