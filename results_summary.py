# =============================================================================
# results_summary.py — Text Summary and Honest Interpretation of Benchmark
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 6
# =============================================================================
#
# WHAT this script does:
#   Loads benchmark_results.csv and prints a clean, opinionated text summary:
#     1. Which agent reached 80% goal rate first (in training timesteps)
#     2. Which agent had the lowest trap entry rate at convergence
#     3. A statistical caveat about the reliability of the N=5 seed comparison
#     4. An honest interpretation section — including what to conclude if
#        PULSE does NOT outperform the baselines
#
# WHY include an explicit "if PULSE doesn't win" section?
#   Research that only reports its successes is not science — it is marketing.
#   PULSE is a hypothesis about whether pain-memory aversion speeds convergence.
#   A benchmark that honestly acknowledges possible null or negative results:
#     1. Provides actionable information for the next iteration (what to fix)
#     2. Prevents the researcher from over-interpreting noise as signal
#     3. Is publishable even if the result is negative — null results in RL
#        are as informative as positive results (they constrain the hypothesis space)
#   The interpretation section below explains WHAT each possible outcome means
#   and WHAT to try next — turning every result into learning.
# =============================================================================

# WHY pandas: For loading CSV and computing grouped statistics in one line.
import pandas as pd

# WHY numpy: For t-test statistic calculation (manual, without scipy).
import numpy as np

# WHY sys: To exit with a clear error if the CSV doesn't exist.
import sys

# WHY os: To check CSV existence before loading.
import os

# WHY: Import the constant from benchmark.py so both files agree on the filename.
from benchmark import OUTPUT_CSV


# =============================================================================
# CONFIGURATION
# =============================================================================

# WHY 0.80 as the convergence threshold?
# 80% goal rate means the agent reaches the goal in at least 8 out of 10
# evaluation episodes. This is a strong but achievable bar for an 8×8 grid
# with 3 traps. At 50% the agent barely beats random; at 100% the grid is
# fully solved. 80% is a standard "mostly converged" threshold used in
# small-env RL research. If no agent reaches 80%, we use the PEAK goal rate.
CONVERGENCE_THRESHOLD = 0.80

# WHY 5 last checkpoints for "at convergence" statistics?
# The last 5 evaluation checkpoints (last 5,000 steps) represent the agent's
# final, stable behaviour. Averaging over 5 points reduces noise from individual
# evaluation batches. Using only the very last checkpoint would give a single
# noisy estimate. Using all 50 checkpoints would dilute the converged behaviour
# with early-training metrics.
CONVERGENCE_WINDOW = 5


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def find_convergence_timestep(df: pd.DataFrame, agent: str) -> int | None:
    """
    Finds the first timestep at which the agent's MEAN goal rate (across seeds)
    crossed the CONVERGENCE_THRESHOLD.

    WHY the MEAN across seeds (not any individual seed)?
    A single seed might reach 80% by luck at timestep 1000 and then fall back
    to 40% by timestep 5000. That is not convergence — it is noise. The mean
    across all seeds is stable: if the mean crosses 80%, the MAJORITY of seeds
    are consistently performing at that level.

    WHY "first timestep" (not "highest sustained period")?
    "First crossing" is the most common RL convergence definition and is the
    easiest to compare across algorithms. It answers: "when does the algorithm
    FIRST know how to solve the task reliably?" — the most relevant question for
    sample efficiency comparison.

    Args:
        df:    Full results DataFrame.
        agent: Agent name string ("PPO", "DQN", "PULSE-PPO").

    Returns:
        The first timestep where mean goal_rate >= CONVERGENCE_THRESHOLD,
        or None if the threshold was never reached.
    """
    # WHY group by timestep and take mean across seeds?
    agent_df = df[df["agent"] == agent].groupby("timesteps")["goal_rate"].mean()

    # WHY sort_index? The groupby might not return sorted timesteps if the CSV
    # was written out of order (e.g. from two partial benchmark runs combined).
    agent_df = agent_df.sort_index()

    above = agent_df[agent_df >= CONVERGENCE_THRESHOLD]

    if above.empty:
        return None    # never converged within TOTAL_TIMESTEPS

    return int(above.index[0])    # first timestep above threshold


def final_metrics(df: pd.DataFrame, agent: str) -> dict:
    """
    Computes mean and std of key metrics over the last CONVERGENCE_WINDOW
    checkpoints, averaged across seeds.

    WHY these specific metrics (trap_rate, goal_rate, avg_reward)?
    Together they characterise the agent's FINAL behaviour:
      - trap_rate: is the agent SAFE at convergence? (PULSE's target)
      - goal_rate: is the agent SUCCESSFUL at convergence? (primary task metric)
      - avg_reward: is the agent EFFICIENT? (aggregate optimisation signal)
    Episode length is excluded here because it conflates short-trap episodes
    with long-successful episodes in a way that is hard to interpret in a table.

    Args:
        df:    Full results DataFrame.
        agent: Agent name string.

    Returns:
        dict with keys: trap_{mean,std}, goal_{mean,std}, reward_{mean,std}
    """
    agent_df = df[df["agent"] == agent]

    # WHY nlargest(CONVERGENCE_WINDOW * n_seeds)?
    # We want the last CONVERGENCE_WINDOW timesteps. Each timestep has N_SEEDS rows.
    # Getting the last CONVERGENCE_WINDOW unique timesteps across all seeds.
    last_ts = agent_df["timesteps"].unique()
    last_ts_sorted = np.sort(last_ts)[-CONVERGENCE_WINDOW:]    # last N timesteps
    final_df = agent_df[agent_df["timesteps"].isin(last_ts_sorted)]

    # WHY use per-episode std (not cross-seed std) for trap_rate?
    # trap_rate is already an average over N_EVAL_EPISODES per seed per checkpoint.
    # Its std across seeds reflects how much DIFFERENT SEEDS vary at convergence —
    # which is the relevant measure of algorithm stability, not within-episode noise.
    return {
        "trap_mean":   final_df["trap_rate"].mean(),
        "trap_std":    final_df["trap_rate"].std(),
        "goal_mean":   final_df["goal_rate"].mean(),
        "goal_std":    final_df["goal_rate"].std(),
        "reward_mean": final_df["avg_reward"].mean(),
        "reward_std":  final_df["avg_reward"].std(),
    }


def compute_effect_size(df: pd.DataFrame, metric: str, agent_a: str, agent_b: str) -> float:
    """
    Computes Cohen's d effect size between two agents on a metric at the final
    CONVERGENCE_WINDOW checkpoints.

    WHY Cohen's d (not a p-value)?
    A p-value answers: "is the difference unlikely to be pure chance?"
    Cohen's d answers: "how LARGE is the difference relative to the spread?"
    With N=5 seeds, p-values are almost meaningless — the test is drastically
    underpowered. We would need ~20+ seeds for reliable p-values. But Cohen's d
    gives directional information even with small N:
      d < 0.2: negligible effect
      0.2 ≤ d < 0.5: small effect
      0.5 ≤ d < 0.8: medium effect
      d ≥ 0.8: large effect
    Even with N=5, a large Cohen's d (≥ 0.8) suggests the difference is real
    and worth investigating with more seeds. A small Cohen's d means the
    experiment needs more seeds before a conclusion can be drawn.

    Args:
        df:      Full results DataFrame.
        metric:  Column name to compare (e.g. "goal_rate", "trap_rate").
        agent_a: Name of first agent.
        agent_b: Name of second agent.

    Returns:
        Cohen's d (positive means agent_a > agent_b).
    """
    def get_final_values(agent):
        agent_df = df[df["agent"] == agent]
        last_ts = np.sort(agent_df["timesteps"].unique())[-CONVERGENCE_WINDOW:]
        final_df = agent_df[agent_df["timesteps"].isin(last_ts)]
        # WHY group by seed? We want ONE value per seed (the mean over the last
        # CONVERGENCE_WINDOW checkpoints for that seed), not all checkpoint values.
        return final_df.groupby("seed")[metric].mean().values

    vals_a = get_final_values(agent_a)
    vals_b = get_final_values(agent_b)

    mean_diff = np.mean(vals_a) - np.mean(vals_b)
    # WHY pooled std? Cohen's d uses the pooled standard deviation of both groups
    # to normalise the mean difference. This is standard for two-sample d.
    pooled_std = np.sqrt((np.std(vals_a, ddof=1)**2 + np.std(vals_b, ddof=1)**2) / 2)

    if pooled_std == 0:
        return 0.0    # both groups have identical variance (fully deterministic)
    return mean_diff / pooled_std


# =============================================================================
# INTERPRETATION LOGIC
# =============================================================================

def interpret_result(
    pulse_converged: int | None,
    ppo_converged: int | None,
    dqn_converged: int | None,
    pulse_trap: float,
    ppo_trap: float,
    dqn_trap: float,
) -> str:
    """
    Generates a human-readable interpretation of the benchmark result.

    WHY hard-code the interpretation logic (not just print numbers)?
    Numbers without context are uninterpretable. A researcher looking at
    "PULSE trap_rate=0.12, PPO trap_rate=0.18" needs to know:
      - Is this difference meaningful given N=5 seeds?
      - What does it mean if PULSE is NOT better?
      - What should be tried next?
    The interpretation section below answers these questions for every possible
    outcome, making this script genuinely useful rather than just a number printer.

    HOW TO INTERPRET THE RESULTS HONESTLY:
      If PULSE-PPO converges faster AND has lower trap rate:
        → Strong evidence that the aversion filter provides a genuine benefit.
          The mechanism works: pain memory + physics danger → earlier avoidance.
          Recommended next step: test on harder environments (more traps, larger grid).

      If PULSE-PPO has lower trap rate but NOT faster convergence:
        → The filter succeeds at its SPECIFIC goal (trap avoidance) but not at
          the broader goal (sample efficiency). This may mean:
          (a) The shaped reward (Phase 3-4) already guides vanilla PPO away from
              traps well enough that the filter adds no additional benefit.
          (b) The aversion filter sometimes prevents the agent from taking the
              optimal path (near-trap routes), slowing goal-seeking. The filter's
              BENEFIT (fewer trap hits) is offset by its COST (longer detours).
          Recommended next step: lower the aversion threshold, or apply the filter
          only after N training episodes to give the policy time to learn the
          trap-adjacent optimal paths before activating avoidance.

      If PULSE-PPO is WORSE than vanilla PPO on all metrics:
        → The aversion filter is doing more harm than good. Most likely cause:
          the filter is over-active (threshold too low OR slab accumulates too
          fast), blocking the agent from taking good paths near trap neighbours
          even when those paths are actually safe. The filter is treating the
          danger GRADIENT (high resistance near traps) as a danger LABEL ("never
          go here"), which is not the intended semantics.
          Recommended next step: increase the aversion threshold to 1.0+, or
          decouple the resistance field contribution from the aversion score
          (use only slab depth, not depth × resistance).

      If DQN outperforms both PPO variants:
        → This environment may favour Q-learning over policy gradient methods.
          The small discrete state space (64 cells, 4 actions) is exactly where
          tabular or near-tabular methods excel. DQN's value function can
          memorise Q-values for all 256 state-action pairs. PPO's policy network
          may be overparameterised for this task. This is a property of the
          environment, not a criticism of PULSE — it would look the same on any
          PPO vs DQN comparison without PULSE.

      If nobody converges to 80% goal rate:
        → The training budget (50,000 steps) is insufficient, OR the environment
          is harder than expected. Check: does the optimal path at (0,0)→(7,7)
          pass near any traps? If the shaped reward (resistance near traps) is
          penalising the optimal path, NONE of the agents will converge stably.
          Recommended next step: run 200,000 steps, or check whether the trap
          layout creates a "resistance wall" blocking the optimal path.
    """
    lines = []

    def convergence_str(ts):
        return f"{ts:,} steps" if ts is not None else "did not reach 80% (within budget)"

    lines.append("")
    lines.append("RESULT INTERPRETATION")
    lines.append("─" * 55)

    # --- Convergence speed ---
    converged = {k: v for k, v in {
        "PPO": ppo_converged,
        "DQN": dqn_converged,
        "PULSE-PPO": pulse_converged,
    }.items() if v is not None}

    if not converged:
        lines.append("⚠  No agent reached the 80% goal rate threshold.")
        lines.append("   Possible reasons:")
        lines.append("   - Training budget too small (try 200k steps)")
        lines.append("   - Phase 3/4 resistance near optimal path is too costly")
        lines.append("   - Environment is harder than it looks — check trap layout")
    else:
        fastest = min(converged, key=converged.get)
        lines.append(f"Fastest convergence: {fastest} ({convergence_str(converged[fastest])})")
        for agent, ts in sorted(converged.items(), key=lambda x: x[1]):
            tag = " ← fastest" if agent == fastest else ""
            lines.append(f"  {agent:12s}: {convergence_str(ts)}{tag}")
        for agent in ["PPO", "DQN", "PULSE-PPO"]:
            if agent not in converged:
                lines.append(f"  {agent:12s}: {convergence_str(None)}")

    lines.append("")

    # --- Trap avoidance ---
    trap_rates = {"PPO": ppo_trap, "DQN": dqn_trap, "PULSE-PPO": pulse_trap}
    safest = min(trap_rates, key=trap_rates.get)
    lines.append(f"Lowest trap rate at convergence: {safest} ({trap_rates[safest]:.3f})")
    for agent, rate in sorted(trap_rates.items(), key=lambda x: x[1]):
        tag = " ← safest" if agent == safest else ""
        lines.append(f"  {agent:12s}: trap_rate={rate:.3f}{tag}")

    lines.append("")

    # --- PULSE-specific verdict ---
    pulse_won_convergence = (pulse_converged is not None and
                              (ppo_converged is None or pulse_converged < ppo_converged) and
                              (dqn_converged is None or pulse_converged < dqn_converged))
    pulse_won_traps = pulse_trap < ppo_trap and pulse_trap < dqn_trap

    lines.append("PULSE-PPO verdict:")
    if pulse_won_convergence and pulse_won_traps:
        lines.append("  ✓ PULSE converged fastest AND had the lowest trap rate.")
        lines.append("  → Strong evidence that pain-memory aversion adds genuine value.")
        lines.append("  → Recommended next step: test on a harder environment.")

    elif pulse_won_traps and not pulse_won_convergence:
        lines.append("  ◑ PULSE had lower trap rate but did NOT converge faster.")
        lines.append("  → The filter is working for its specific goal (trap avoidance)")
        lines.append("    but adds path-cost overhead that slows goal-reaching.")
        lines.append("  → Try: higher aversion threshold, or delayed filter activation.")

    elif pulse_won_convergence and not pulse_won_traps:
        lines.append("  ◑ PULSE converged faster but did NOT have the lowest trap rate.")
        lines.append("  → Faster goal-reaching, but not via the intended mechanism.")
        lines.append("  → Check: is PULSE-PPO taking riskier paths near traps?")

    else:
        lines.append("  ✗ PULSE did NOT outperform baselines on either key metric.")
        lines.append("  → This is a scientifically meaningful negative result:")
        lines.append("    The aversion filter at eval time does not reliably improve")
        lines.append("    performance on this environment and training budget.")
        lines.append("  → Possible fixes:")
        lines.append("    1. Lower aversion threshold (currently 0.5 — try 0.2)")
        lines.append("    2. Apply filter during training (accept on-policy assumption break)")
        lines.append("    3. Increase training budget (more slab accumulation needed)")
        lines.append("    4. Re-examine whether Phase 3/4 reward already captures the signal")

    return "\n".join(lines)


# =============================================================================
# MAIN SUMMARY PRINTER
# =============================================================================

def print_summary(csv_path: str = OUTPUT_CSV) -> None:
    """
    Loads the benchmark CSV and prints the complete results summary.
    """
    if not os.path.exists(csv_path):
        print(f"[results_summary] ERROR: {csv_path} not found.")
        print("Run benchmark.py first: python3 benchmark.py")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    agents = df["agent"].unique().tolist()

    print("\n" + "=" * 65)
    print("PULSE Phase 6 — Benchmark Results Summary")
    print("=" * 65)

    # --- Data overview ---
    print(f"\nData overview:")
    print(f"  Total rows:  {len(df)}")
    print(f"  Agents:      {', '.join(agents)}")
    print(f"  Seeds:       {sorted(df['seed'].unique().tolist())}")
    print(f"  Timesteps:   {df['timesteps'].min():,} → {df['timesteps'].max():,}")
    print(f"  Eval steps:  {df['timesteps'].nunique()} checkpoints")

    # --- Per-agent summary table ---
    print(f"\n{'─'*65}")
    print("Final performance (mean ± std across seeds, last 5 checkpoints):")
    print(f"{'─'*65}")
    print(f"  {'Agent':<12} {'Goal Rate':>12} {'Trap Rate':>12} {'Reward':>12}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*12}")

    agent_finals = {}
    for agent in ["PPO", "DQN", "PULSE-PPO"]:
        if agent not in agents:
            print(f"  {agent:<12} (no data)")
            continue
        m = final_metrics(df, agent)
        agent_finals[agent] = m
        print(
            f"  {agent:<12} "
            f"{m['goal_mean']:.3f}±{m['goal_std']:.3f}  "
            f"{m['trap_mean']:.3f}±{m['trap_std']:.3f}  "
            f"{m['reward_mean']:+.2f}±{m['reward_std']:.2f}"
        )

    # --- Convergence timing ---
    print(f"\n{'─'*65}")
    print(f"Convergence: first timestep where mean goal_rate ≥ "
          f"{CONVERGENCE_THRESHOLD:.0%}")
    print(f"{'─'*65}")

    conv_ts = {}
    for agent in ["PPO", "DQN", "PULSE-PPO"]:
        if agent not in agents:
            continue
        ts = find_convergence_timestep(df, agent)
        conv_ts[agent] = ts
        ts_str = f"{ts:,}" if ts is not None else "—  (never reached threshold)"
        print(f"  {agent:<12}: {ts_str}")

    # --- Effect sizes ---
    print(f"\n{'─'*65}")
    print("Effect sizes (Cohen's d, PULSE-PPO vs baselines, final performance):")
    print("  |d| < 0.2=negligible  0.2-0.5=small  0.5-0.8=medium  >0.8=large")
    print(f"{'─'*65}")

    # WHY display Cohen's d rather than p-values?
    # See compute_effect_size() docstring for the full argument.
    # With N=5, p-values are unreliable. Cohen's d gives directional evidence
    # about the SIZE of any observed effect, which is what we can report honestly.
    if "PULSE-PPO" in agents:
        for baseline in ["PPO", "DQN"]:
            if baseline not in agents:
                continue
            for metric, label in [("goal_rate", "goal_rate"), ("trap_rate", "trap_rate")]:
                d = compute_effect_size(df, metric, "PULSE-PPO", baseline)
                direction = "better" if (
                    (metric == "goal_rate" and d > 0) or
                    (metric == "trap_rate" and d < 0)
                ) else "worse"
                size_label = (
                    "negligible" if abs(d) < 0.2 else
                    "small"      if abs(d) < 0.5 else
                    "medium"     if abs(d) < 0.8 else
                    "large"
                )
                print(f"  PULSE vs {baseline:<4} on {label:<12}: "
                      f"d={d:+.3f} ({size_label}, PULSE is {direction})")

    # --- Statistical reliability note ---
    print(f"\n{'─'*65}")
    print("Statistical reliability note:")
    print(f"{'─'*65}")
    n_seeds = df["seed"].nunique()
    print(f"  Seeds used: N={n_seeds}")
    if n_seeds < 10:
        print(f"  ⚠  N={n_seeds} is too small for formal hypothesis testing.")
        print("  With N<10, p-values are unreliable (high Type I and Type II error rates).")
        print("  Cohen's d provides directional evidence but should be interpreted cautiously.")
        print("  For publication-strength claims, use N≥20 seeds.")
        print("  Current results support: hypothesis generation and further investigation.")
        print("  They do NOT support: strong causal claims about the mechanism.")
    else:
        print(f"  N={n_seeds} is sufficient for directional claims.")
        print("  Still recommend reporting effect sizes (Cohen's d) alongside p-values.")

    # --- Interpretation ---
    if all(a in agents for a in ["PPO", "DQN", "PULSE-PPO"]):
        pulse_finals = agent_finals.get("PULSE-PPO", {})
        ppo_finals   = agent_finals.get("PPO",   {})
        dqn_finals   = agent_finals.get("DQN",   {})

        interpretation = interpret_result(
            pulse_converged=conv_ts.get("PULSE-PPO"),
            ppo_converged=conv_ts.get("PPO"),
            dqn_converged=conv_ts.get("DQN"),
            pulse_trap=pulse_finals.get("trap_mean", 1.0),
            ppo_trap=ppo_finals.get("trap_mean", 1.0),
            dqn_trap=dqn_finals.get("trap_mean", 1.0),
        )
        print(f"\n{'─'*65}")
        print(interpretation)

    print("\n" + "=" * 65)
    print("Plots: run python3 plot_benchmark.py")
    print("=" * 65)


if __name__ == "__main__":
    print_summary()
