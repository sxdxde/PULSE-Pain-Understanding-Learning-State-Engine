# =============================================================================
# plot_benchmark.py — Learning Curve Visualiser for PULSE Phase 6 Benchmark
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 6
# =============================================================================
#
# WHAT this script does:
#   Loads benchmark_results.csv (produced by benchmark.py) and generates a
#   four-panel comparison figure showing all three agents across all seeds,
#   with shaded standard-deviation bands. Saves to benchmark_results.png.
#
# WHY shaded standard-deviation bands matter in RL research:
#   A single training run is a single trajectory through a high-dimensional
#   stochastic optimisation landscape. It is not representative of the algorithm's
#   typical behaviour — it is one sample from a distribution of possible outcomes.
#   Plotting a single run and claiming an algorithm "learns faster" is a category
#   error equivalent to measuring one person's height and claiming the population
#   is that tall.
#   Shaded bands show:
#     1. WHERE the mean curve is — the algorithm's expected performance
#     2. HOW WIDE the variance is — how much the algorithm's performance varies
#        across runs with different random initialisations
#   An algorithm with a higher mean but VERY wide bands (high variance) may
#   actually be WORSE in practice than one with a lower mean but narrow bands
#   (reliable). A shaded-band plot makes this trade-off immediately visible.
#   It also reveals whether two algorithms' performance regions OVERLAP — if
#   their bands overlap substantially, the observed difference may be noise,
#   not a real effect.
#
# COLOUR CONVENTION (consistent throughout):
#   PPO       → steelblue   (the familiar baseline)
#   DQN       → darkorange  (warm contrast against blue)
#   PULSE-PPO → mediumpurple (distinct from both; purple = fusion of blue+something new)
# =============================================================================

# WHY pandas: The canonical tool for loading and groupby-aggregating CSV data.
import pandas as pd

# WHY numpy: For computing rolling smoothing and generating timestep arrays.
import numpy as np

# WHY matplotlib: The standard plotting library for publication-quality figures.
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# WHY os: For checking whether the CSV exists before attempting to load it.
import os

# WHY: Import the CSV path constant from benchmark.py.
# Importing from benchmark.py means both files always agree on the filename —
# changing OUTPUT_CSV in benchmark.py automatically propagates here.
# The __main__ guard in benchmark.py prevents training from running on import.
from benchmark import OUTPUT_CSV

# =============================================================================
# CONFIGURATION
# =============================================================================

# WHY define colours here (not inline)?
# Consistent colours across all four panels require a single source of truth.
# If we wrote the colour string inline in each plot call, a typo would produce
# an inconsistent plot without any error message.
AGENT_COLOURS = {
    "PPO":       "steelblue",
    "DQN":       "darkorange",
    "PULSE-PPO": "mediumpurple",
}

# WHY linewidth=2.0? Thick enough to see clearly on a four-panel figure without
# overwhelming the shaded band.
LINE_WIDTH    = 2.0

# WHY alpha=0.2 for the band? Transparent enough to see overlapping bands;
# opaque enough to be visible on white backgrounds.
BAND_ALPHA    = 0.2

# WHY output PNG at 150 dpi? High enough for figures in presentations and papers.
# The default 72 dpi produces blurry text when zoomed in.
OUTPUT_FIGURE = "benchmark_results.png"

# WHY rolling window of 3?
# With 50 evaluation checkpoints, a window of 3 applies light smoothing —
# enough to reduce high-frequency noise in early training without distorting
# the convergence timing. A window of 10 would over-smooth and make it look
# like all algorithms converge at the same point.
SMOOTH_WINDOW = 3


# =============================================================================
# DATA LOADING
# =============================================================================

def load_results(csv_path: str) -> pd.DataFrame:
    """
    Loads benchmark_results.csv into a pandas DataFrame.

    WHY validate at load time (not just read and hope)?
    If the CSV was only partially written (e.g. benchmark crashed mid-run),
    the data is still usable but the user should know it is incomplete.
    Printing a warning with exact counts gives them the context to interpret
    the plot correctly — partial data can still show trends.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with one row per (agent, seed, timestep) triple.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Results file not found: {csv_path}\n"
            "Run benchmark.py first: python3 benchmark.py"
        )

    df = pd.read_csv(csv_path)

    # WHY print a data-quality report?
    # Gives the user immediate visibility into whether the benchmark ran
    # completely or needs to be re-run. A plot without this context might
    # be misinterpreted as complete when only 2 of 5 seeds finished.
    print(f"[plot_benchmark] Loaded {len(df)} rows from {csv_path}")
    for agent in df["agent"].unique():
        n_seeds = df[df["agent"] == agent]["seed"].nunique()
        n_steps = df[df["agent"] == agent]["timesteps"].nunique()
        print(f"  {agent}: {n_seeds} seeds × {n_steps} checkpoints "
              f"= {n_seeds * n_steps} rows")

    return df


# =============================================================================
# DATA PREPARATION
# =============================================================================

def aggregate_by_timestep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups by (agent, timesteps) and computes mean and std across seeds.

    WHY std (not SEM)?
    Standard deviation (std) shows the SPREAD of individual runs — how different
    each seed's result is from the others. Standard error of the mean (SEM) would
    show how precisely we know the true mean, which shrinks with N seeds.
    For visualisation, std is more honest: it shows the actual variability a
    practitioner would experience when running the algorithm once. SEM can make
    a noisy algorithm look deceptively stable when N is large.

    Returns:
        DataFrame with columns: agent, timesteps, mean_{metric}, std_{metric}
        for each of the four plotted metrics.
    """
    metrics = ["avg_reward", "trap_rate", "goal_rate", "avg_length"]

    # WHY agg with both mean and std in one pass?
    # GroupBy.agg with a dict computes both in one scan of the data —
    # more efficient than two separate groupby calls.
    agg_dict = {}
    for m in metrics:
        agg_dict[f"mean_{m}"] = (m, "mean")
        agg_dict[f"std_{m}"]  = (m, "std")

    grouped = (
        df.groupby(["agent", "timesteps"])
          .agg(**agg_dict)
          .reset_index()
    )

    # WHY fillna(0)?
    # If only one seed has data at a checkpoint, std is NaN (std of one sample
    # is undefined). Replacing with 0 means the shaded band disappears at that
    # point — honest representation of "no variance data here."
    grouped = grouped.fillna(0)
    return grouped


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """
    Applies a centred rolling mean to a 1D array.

    WHY centred (not trailing) rolling mean?
    A trailing mean lags behind the actual curve — the smoothed point at step N
    reflects data from steps N-(window-1) to N. This makes convergence look like
    it happens LATER than it actually did. A centred mean reflects data from
    N-(window//2) to N+(window//2), giving an unbiased estimate of the local
    trend at each point.

    WHY not use pandas rolling().mean()?
    We already have the data as numpy arrays at this point. numpy convolve is
    faster and avoids converting back and forth.

    Args:
        values: 1D numpy array of metric values.
        window: Number of points in the rolling window.

    Returns:
        Smoothed array of the same length. Endpoints use the raw values
        (not enough context for full smoothing at the edges).
    """
    if window < 2 or len(values) < window:
        return values    # no smoothing if window is trivially small

    kernel    = np.ones(window) / window
    smoothed  = np.convolve(values, kernel, mode="same")

    # WHY fix the edges?
    # Convolution in "same" mode pads with zeros at the edges, which pulls
    # the smoothed value DOWN near the start (zero-padding artefact). We
    # replace the first and last (window//2) points with the raw values to
    # avoid this artefact making early-training curves look worse than they are.
    half = window // 2
    smoothed[:half] = values[:half]
    smoothed[-half:] = values[-half:]
    return smoothed


# =============================================================================
# PLOTTING
# =============================================================================

def plot_panel(
    ax: plt.Axes,
    grouped: pd.DataFrame,
    metric_col: str,     # "avg_reward", "trap_rate", "goal_rate", "avg_length"
    ylabel: str,
    title: str,
    reference_line: float = None,   # optional horizontal reference line
    reference_label: str = None,
    invert_good: bool = False,      # if True, LOWER is better (shown in annotation)
) -> None:
    """
    Plots one metric panel with mean curves and std bands for all three agents.

    WHY one function for all four panels?
    The four panels share identical structure (mean line + std band per agent).
    A single function guarantees visual consistency — same line widths, same
    alpha, same colour scheme — and avoids copy-paste bugs.

    Args:
        ax:              The matplotlib Axes to draw on.
        grouped:         Aggregated DataFrame from aggregate_by_timestep().
        metric_col:      The metric column name (without mean_ / std_ prefix).
        ylabel:          Y-axis label string.
        title:           Panel title string.
        reference_line:  If given, draws a horizontal dashed line at this value.
        reference_label: Label for the reference line (shown in legend).
        invert_good:     If True, adds "(lower = better)" to the title.
    """
    for agent_name, colour in AGENT_COLOURS.items():
        agent_data = grouped[grouped["agent"] == agent_name].sort_values("timesteps")

        if agent_data.empty:
            # WHY warn instead of crash?
            # If the benchmark only partially completed (e.g. DQN not yet run),
            # we still want to see the curves that DID complete.
            print(f"  [plot_benchmark] No data for agent={agent_name}, skipping.")
            continue

        ts     = agent_data["timesteps"].values
        mean_v = agent_data[f"mean_{metric_col}"].values
        std_v  = agent_data[f"std_{metric_col}"].values

        # WHY smooth before plotting?
        # Raw learning curves are noisy, especially in early training. Smoothing
        # reveals the convergence trend. We smooth the MEAN, not the raw per-seed
        # values — smoothing std would misrepresent the true variance.
        smoothed_mean = smooth_series(mean_v, SMOOTH_WINDOW)

        # WHY plot both the smoothed mean line AND the raw-std band?
        # The smoothed line shows the TREND; the std band shows the SPREAD
        # around the RAW (unsmoothed) mean, which is the honest variance.
        # Using smoothed std would visually narrow the band in a misleading way.
        ax.plot(
            ts, smoothed_mean,
            color=colour,
            linewidth=LINE_WIDTH,
            label=f"{agent_name}",
            zorder=3,   # WHY zorder=3? Draw lines above bands (zorder=2).
        )

        # WHY fill_between with raw mean ± std?
        # The shaded region shows: "if you ran this algorithm once with a random
        # seed, your result would likely fall in this region."
        ax.fill_between(
            ts,
            mean_v - std_v,
            mean_v + std_v,
            color=colour,
            alpha=BAND_ALPHA,
            zorder=2,
        )

        # WHY a tiny dot at the final timestep?
        # Anchors the label position to the end of the curve, making it easier
        # to match lines to the legend without hunting for the right colour.
        ax.plot(ts[-1], smoothed_mean[-1], "o", color=colour, markersize=5, zorder=4)

    # WHY a reference line for the 80% goal rate panel?
    # 80% is the convergence threshold we define in results_summary.py.
    # Showing it on the plot lets the reader visually see when each agent
    # crosses it — no need to look up the number in the summary.
    if reference_line is not None:
        ax.axhline(
            reference_line,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            label=reference_label or f"y={reference_line}",
            zorder=1,
        )

    title_suffix = "\n(lower = better)" if invert_good else ""
    ax.set_title(title + title_suffix, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Training Timesteps", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(axis="both", labelsize=8)

    # WHY format x-axis as thousands (e.g. "10k")?
    # 50,000 written as "50000" wastes axis space and reads slowly.
    # "50k" is instantly readable.
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
    )


def plot_all(df: pd.DataFrame, save_path: str = OUTPUT_FIGURE) -> None:
    """
    Generates the four-panel comparison figure and saves it.

    Panel layout:
      [1] Average Episode Reward    [2] Trap Entry Rate
      [3] Goal Reach Rate           [4] Average Episode Length

    WHY this particular set of four metrics?
    Together they give a complete picture of the learning process:
      1. Reward: the aggregate optimisation target — overall learning quality
      2. Trap rate: the SPECIFIC mechanism test — does PULSE reduce dangerous entries?
      3. Goal rate: the PRIMARY TASK success — is the agent actually solving the problem?
      4. Episode length: the EFFICIENCY measure — is the agent finding short paths?
    Trap rate and goal rate together are more informative than reward alone:
    an agent could get a high reward by reaching the goal quickly WITHOUT avoiding
    traps (if the path to the goal never crosses them), or by avoiding traps but
    never reaching the goal (wandering safely but aimlessly). Separating them
    reveals WHICH mechanism produced the observed reward improvement.

    Args:
        df:        Full results DataFrame from load_results().
        save_path: Path to save the output figure PNG.
    """
    grouped = aggregate_by_timestep(df)

    # WHY figsize=(16, 10)? Four panels need horizontal space. 16×10 inches
    # gives each panel ~7×10 inches, enough for readable tick labels and titles.
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "PULSE Phase 6 — Convergence Benchmark\n"
        "Shaded bands = ±1 std across seeds  |  "
        "PPO (blue) vs DQN (orange) vs PULSE-PPO (purple)",
        fontsize=13,
        fontweight="bold",
        y=1.01,   # WHY y=1.01? Pushes the title above the subplot area
                  # so it doesn't overlap the top panels' titles.
    )

    # --- Panel 1: Average Episode Reward ---
    # WHY higher = better for reward (no invert_good)?
    # Reward is a maximisation objective. The best agent should have the
    # highest (least negative) mean reward — this is the standard RL criterion.
    plot_panel(
        axes[0, 0],
        grouped,
        metric_col="avg_reward",
        ylabel="Mean Episode Reward",
        title="Learning Curve (Reward)",
    )

    # --- Panel 2: Trap Entry Rate ---
    # WHY lower = better here?
    # The trap entry rate measures HARM — how often the agent walks into danger.
    # A lower rate means the agent has learned (or the filter prevents it) from
    # entering traps. PULSE's aversion mechanism should specifically reduce this.
    # The (lower = better) annotation reminds the reader of the correct direction
    # when interpreting whether a downward-trending curve is good news.
    plot_panel(
        axes[0, 1],
        grouped,
        metric_col="trap_rate",
        ylabel="Trap Entry Rate (fraction of episodes)",
        title="Trap Entry Rate",
        invert_good=True,
    )

    # --- Panel 3: Goal Reach Rate ---
    # WHY a reference line at 0.8?
    # 80% goal rate is our convergence threshold (see results_summary.py).
    # Showing it on this panel lets the reader visually identify when each agent
    # crosses the convergence threshold without reading the summary text.
    plot_panel(
        axes[1, 0],
        grouped,
        metric_col="goal_rate",
        ylabel="Goal Reach Rate (fraction of episodes)",
        title="Goal Reach Rate",
        reference_line=0.80,
        reference_label="80% convergence threshold",
    )

    # --- Panel 4: Average Episode Length ---
    # WHY lower = better for episode length?
    # Shorter episodes mean the agent is solving the task more efficiently.
    # An agent that reaches the goal in 14 steps (optimal) is better than one
    # that wanders for 80 steps. Trap-terminated episodes are very short (~1-5
    # steps if the trap is near the start), so a VERY low length can mean many
    # trap hits rather than good navigation. Read this panel alongside Panel 2.
    plot_panel(
        axes[1, 1],
        grouped,
        metric_col="avg_length",
        ylabel="Mean Episode Length (steps)",
        title="Episode Length",
        invert_good=True,
    )

    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[plot_benchmark] Figure saved to: {save_path}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    df = load_results(OUTPUT_CSV)
    plot_all(df)
    print("[plot_benchmark] Done. Open benchmark_results.png to inspect results.")
