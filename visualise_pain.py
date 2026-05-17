# =============================================================================
# visualise_pain.py — Multi-Episode Pain Accumulation Visualisation
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 3
# =============================================================================
# WHY this script exists:
# Phase 3's central claim is that the Slab "remembers" damage across episodes.
# To verify that claim visually, we need to WATCH the pain landscape build up
# over many episodes. This script runs 20 random episodes, captures the agent's
# path and the Slab's state after each one, and saves a side-by-side frame so
# we can observe the memory accumulation episode by episode.
#
# WHY random episodes (not the trained PPO agent)?
# The trained PPO agent AVOIDS traps — that's the whole point of training it.
# A policy that avoids traps would rarely spike the Slab, giving us a nearly
# blank heatmap and nothing to analyse. A random agent stumbles into traps
# frequently, which is exactly what we need to demonstrate accumulation. This
# is a diagnostic tool, not a performance benchmark.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# WHY: We import the modified PulseGridWorld which now owns the Slab internally.
# Phase 3 couples them — instantiating the env automatically instantiates the
# Slab, so we don't need to manage two separate objects.
from pulse_env import PulseGridWorld


# =============================================================================
# CONFIGURATION
# =============================================================================

# WHY 20 episodes: Enough to see clear accumulation at trap cells without
# making the script take too long. Each episode ends when the agent reaches
# the goal or a trap, so episodes are short (typically 5–50 steps).
NUM_EPISODES = 20

# WHY: Paths for saved frames and the final pain history plot.
FRAMES_DIR = os.path.join("models", "pain_frames")

# WHY: These must match pulse_env.py exactly — they're used for visual overlays.
GRID_SIZE = 8
TRAP_POSITIONS = [(3, 3), (4, 5), (2, 6)]
GOAL_POS = (7, 7)
START_POS = (0, 0)


# =============================================================================
# HELPER: Draw the grid + agent path (LEFT panel)
# =============================================================================

def draw_grid_path(ax, path: list, outcome: str, episode_num: int) -> None:
    """
    Renders the 8×8 grid with the agent's path for one episode.
    Reuses the same visual language as visualise_phase1.py for consistency.

    Args:
        ax:          matplotlib Axes to draw onto.
        path:        List of (row, col) tuples visited during the episode.
        outcome:     "goal", "trap", or "timeout" — determines end-marker colour.
        episode_num: Used in the panel title.
    """

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            # WHY: Flip row for matplotlib (row 0 → top of plot).
            cell_x = col
            cell_y = GRID_SIZE - 1 - row

            # WHY: Colour priority: trap > goal > start > empty.
            if (row, col) in TRAP_POSITIONS:
                facecolor = "#FF4444"
                edgecolor = "#CC0000"
                lw = 2.0
            elif (row, col) == GOAL_POS:
                facecolor = "#44CC44"
                edgecolor = "#008800"
                lw = 2.0
            elif (row, col) == START_POS:
                facecolor = "#AADDFF"
                edgecolor = "#5599BB"
                lw = 1.5
            else:
                facecolor = "#F5F5F5"
                edgecolor = "#CCCCCC"
                lw = 0.8

            ax.add_patch(mpatches.Rectangle(
                (cell_x, cell_y), 1, 1,
                facecolor=facecolor, edgecolor=edgecolor, linewidth=lw
            ))

    # WHY: Draw cell labels only on key cells to keep the panel uncluttered.
    for row, col in TRAP_POSITIONS:
        ax.text(col + 0.5, GRID_SIZE - 1 - row + 0.5, "T",
                ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    ax.text(GOAL_POS[1] + 0.5, GRID_SIZE - 1 - GOAL_POS[0] + 0.5, "G",
            ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    # WHY: Draw the path as a connected blue line + dots, same as Phase 1.
    if len(path) > 1:
        px = [c + 0.5 for (r, c) in path]
        py = [GRID_SIZE - 1 - r + 0.5 for (r, c) in path]

        ax.plot(px, py, color="#3366CC", linewidth=2.0, alpha=0.8, zorder=3)
        ax.scatter(px, py, color="#3366CC", s=40, zorder=4, alpha=0.9)

        # WHY: Star at start, diamond at end — same markers as Phase 1.
        ax.scatter([px[0]], [py[0]], color="#0033AA", s=150, marker="*", zorder=5)
        end_color = {"goal": "#008800", "trap": "#CC0000", "timeout": "#FF8800"}.get(outcome, "gray")
        ax.scatter([px[-1]], [py[-1]], color=end_color, s=150, marker="D", zorder=5)

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(range(GRID_SIZE + 1))
    ax.set_yticks(range(GRID_SIZE + 1))
    ax.set_xticklabels([str(c) for c in range(GRID_SIZE + 1)], fontsize=7)
    ax.set_yticklabels([str(r) for r in range(GRID_SIZE, -1, -1)], fontsize=7)
    ax.set_xlabel("Col", fontsize=9)
    ax.set_ylabel("Row", fontsize=9)
    ax.set_title(
        f"Episode {episode_num} — Path\nOutcome: {outcome.upper()} | Steps: {len(path)-1}",
        fontsize=9, fontweight="bold"
    )


# =============================================================================
# HELPER: Draw the slab heatmap (RIGHT panel)
# =============================================================================

def draw_slab_heatmap(ax, pain_map: np.ndarray, episode_num: int,
                      global_max: float) -> None:
    """
    Renders the slab's pain landscape as a white→red heatmap.

    WHY global_max instead of per-episode max?
    If we normalise each frame to its own maximum, the colour scale shifts every
    episode — a cell that was "maximum red" in episode 3 might appear pale in
    episode 10 just because new cells got hurt worse. By anchoring all frames
    to the SAME global max (computed before any episodes run), the colour
    meaning is consistent: deep red always means the same pain score across
    every frame. This makes the accumulation visually legible over time.

    Args:
        ax:          matplotlib Axes to draw onto.
        pain_map:    (8, 8) numpy array of L2 pain scores.
        episode_num: Used in panel title.
        global_max:  The maximum pain score across ALL episodes (for colour anchor).
    """
    cmap = cm.get_cmap("Reds")

    # WHY max(global_max, 0.01)? Prevents division by zero on the very first
    # frame when all scores are 0 (no traps visited yet in episode 1).
    normalizer = mcolors.Normalize(vmin=0, vmax=max(global_max, 0.01))
    scalar_map = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell_x = col
            cell_y = GRID_SIZE - 1 - row
            pain_score = pain_map[row, col]
            face_color = scalar_map.to_rgba(pain_score)

            # WHY: Keep trap/goal borders on the heatmap so the viewer can
            # cross-reference which cells ARE traps vs which cells HURT the most.
            # These should always overlap, but the visual confirmation matters.
            if (row, col) in TRAP_POSITIONS:
                edgecolor, lw = "#CC0000", 2.5
            elif (row, col) == GOAL_POS:
                edgecolor, lw = "#006600", 2.5
            else:
                edgecolor, lw = "#AAAAAA", 0.8

            ax.add_patch(mpatches.Rectangle(
                (cell_x, cell_y), 1, 1,
                facecolor=face_color, edgecolor=edgecolor, linewidth=lw
            ))

            # WHY: Print score in each cell. Dark background → white text.
            text_color = "white" if (pain_score / max(global_max, 0.01)) > 0.45 else "#555555"
            ax.text(cell_x + 0.5, cell_y + 0.5, f"{pain_score:.2f}",
                    ha="center", va="center", fontsize=6.5, color=text_color)

    plt.colorbar(scalar_map, ax=ax, label="Pain score (L2 norm)",
                 fraction=0.046, pad=0.04)

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(range(GRID_SIZE + 1))
    ax.set_yticks(range(GRID_SIZE + 1))
    ax.set_xticklabels([str(c) for c in range(GRID_SIZE + 1)], fontsize=7)
    ax.set_yticklabels([str(r) for r in range(GRID_SIZE, -1, -1)], fontsize=7)
    ax.set_xlabel("Col", fontsize=9)
    ax.set_ylabel("Row", fontsize=9)
    ax.set_title(
        f"Slab State After Episode {episode_num}\n"
        f"max={pain_map.max():.3f} | mean={pain_map.mean():.3f}",
        fontsize=9, fontweight="bold"
    )


# =============================================================================
# HELPER: Run one random episode, return path + outcome
# =============================================================================

def run_random_episode(env: PulseGridWorld, episode_num: int,
                       seed: int) -> tuple[list, str]:
    """
    Runs one episode with RANDOM actions and returns the path and outcome.

    WHY random actions here instead of the trained PPO agent?
    See the module-level comment. Short version: the trained agent avoids traps,
    so the Slab stays blank. We need trap hits to demonstrate accumulation.

    Args:
        env:        The PulseGridWorld instance (owns the Slab across episodes).
        episode_num: Used for the random seed so each episode is different but
                     reproducible.
        seed:       Base seed — combined with episode_num for per-episode RNG.

    Returns:
        path:    List of (row, col) positions visited.
        outcome: "goal", "trap", or "timeout".
    """
    # WHY seed=seed+episode_num? Each episode gets a unique but reproducible seed.
    # Without this every episode would be identical (same random action sequence).
    obs, _ = env.reset(seed=seed + episode_num)

    # WHY: Record start position before the first step.
    path = [tuple(env.agent_pos)]
    outcome = "timeout"

    # WHY max_steps=100: A hard cap so a random agent that never finds the goal
    # or a trap doesn't run forever. 100 >> the 14-step optimal path length.
    for _ in range(100):
        # WHY env.action_space.sample()? This draws uniformly from {0,1,2,3} —
        # each direction is equally likely on every step. It requires no model,
        # no policy, no gradient: just the environment's own RNG.
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        path.append(tuple(env.agent_pos))

        if terminated:
            # WHY: Determine outcome by checking current position against known
            # landmarks. We can't rely solely on reward sign because the trap reward
            # now includes a variable pain bonus (Phase 3), making exact comparison
            # unreliable.
            if tuple(env.agent_pos) == GOAL_POS:
                outcome = "goal"
            elif tuple(env.agent_pos) in [tuple(t) for t in TRAP_POSITIONS]:
                outcome = "trap"
            break

        if truncated:
            outcome = "timeout"
            break

    return path, outcome


# =============================================================================
# HELPER: Plot the full pain history as a time series
# =============================================================================

def plot_pain_history(pain_history: list) -> None:
    """
    Plots the pain score at the moment of each spike, in chronological order.

    WHY this plot?
    The heatmap frames show WHERE pain accumulates. This plot shows HOW FAST
    it accumulates and whether there is a trend. We expect:
      - Early spikes: small pain scores (Slab is fresh, barely deformed).
      - Later spikes at the SAME cell: larger pain scores (accumulated damage).
      - Cells hit for the first time late: small scores again (fresh territory).
    This is the "chronic sensitisation" pattern: repeated exposure to the same
    danger becomes progressively more painful, while new dangers start fresh.

    Args:
        pain_history: List of dicts with keys: episode, step, cell, pain_score.
                      Each entry represents one spike event.
    """
    if not pain_history:
        print("No spike events recorded — no traps were hit.")
        return

    # WHY: Extract parallel arrays for plotting. enumerate gives us a global
    # "spike index" (0, 1, 2, ...) as the x-axis, which represents chronological
    # order across all episodes regardless of episode/step boundaries.
    spike_indices = list(range(len(pain_history)))
    pain_scores   = [entry["pain_score"] for entry in pain_history]
    cells         = [entry["cell"] for entry in pain_history]
    episodes      = [entry["episode"] for entry in pain_history]

    # WHY: Assign a distinct colour to each unique trap cell so we can see
    # which cell's pain is growing fastest. trap_colors maps (row,col) → colour.
    unique_cells  = list(set(cells))
    palette       = ["#CC0000", "#FF6600", "#990099"]   # red, orange, purple
    trap_colors   = {cell: palette[i % len(palette)] for i, cell in enumerate(unique_cells)}

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8), dpi=110,
                                          gridspec_kw={"height_ratios": [3, 1]})

    # --- TOP PANEL: pain score per spike ---
    # WHY: Plot each spike as a coloured dot grouped by cell.
    # Connecting dots of the SAME cell with a line would show the growth curve
    # for that cell. We draw each cell's spikes separately for clarity.
    for cell in unique_cells:
        # WHY: Filter to only spikes at this cell.
        indices_for_cell = [i for i, c in enumerate(cells) if c == cell]
        scores_for_cell  = [pain_scores[i] for i in indices_for_cell]
        color            = trap_colors[cell]

        # WHY: Plot as both a line (connecting same-cell spikes) and scatter dots.
        # The line shows the growth trend; dots show individual events.
        ax_top.plot(indices_for_cell, scores_for_cell,
                    color=color, linewidth=1.5, alpha=0.7,
                    label=f"Cell {cell}")
        ax_top.scatter(indices_for_cell, scores_for_cell,
                       color=color, s=50, zorder=5)

    # WHY: Mark episode boundaries as vertical dashed lines so we can see
    # when elastic_recovery fires (at each boundary the pain decays slightly —
    # visible as a small dip in pain score for the first spike of each episode
    # compared to the last spike of the previous episode at the same cell).
    episode_boundaries = []
    for i in range(1, len(episodes)):
        if episodes[i] != episodes[i - 1]:
            episode_boundaries.append(i - 0.5)

    for boundary in episode_boundaries:
        ax_top.axvline(x=boundary, color="#AAAAAA", linewidth=0.7,
                       linestyle="--", alpha=0.6)

    ax_top.set_xlabel("Spike index (chronological order)", fontsize=10)
    ax_top.set_ylabel("Pain score at moment of spike (L2 norm)", fontsize=10)
    ax_top.set_title(
        "PULSE Phase 3 — Pain History: Score at Each Spike Event\n"
        "(dashed lines = episode boundaries where elastic recovery fires)",
        fontsize=11, fontweight="bold"
    )
    ax_top.legend(loc="upper left", fontsize=9)
    ax_top.grid(axis="y", alpha=0.3)

    # --- BOTTOM PANEL: which episode each spike came from ---
    # WHY: This panel makes it easy to see how many spikes occurred per episode.
    # Dense spikes in one episode = agent hit many traps; sparse = lucky run.
    ax_bot.scatter(spike_indices, episodes,
                   c=[trap_colors[c] for c in cells],
                   s=30, zorder=3, alpha=0.8)
    ax_bot.set_xlabel("Spike index", fontsize=10)
    ax_bot.set_ylabel("Episode", fontsize=10)
    ax_bot.set_yticks(range(1, NUM_EPISODES + 1))
    ax_bot.grid(axis="y", alpha=0.3)
    ax_bot.set_title("Which episode each spike occurred in", fontsize=9)

    plt.tight_layout()

    save_path = os.path.join("models", "pain_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Pain history plot saved to: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # WHY: Create the frames output directory.
    os.makedirs(FRAMES_DIR, exist_ok=True)

    print("=" * 60)
    print("PULSE Phase 3 — Multi-Episode Pain Accumulation")
    print("=" * 60)

    # --- STEP 1: Initialise one persistent environment ---
    # WHY ONE environment across all 20 episodes (not 20 separate envs)?
    # The Slab lives inside the environment. If we created a new environment
    # for each episode, the Slab would reset to zero every time — defeating
    # the entire point of Phase 3. By using ONE env and calling reset() 20 times,
    # we let the Slab persist and accumulate damage across every episode.
    env = PulseGridWorld(render_mode=None)
    print(f"Environment created. Slab initialised to all zeros.")

    # WHY: We need the global_max pain score to anchor all heatmap colour scales
    # to the same value. We collect all per-episode pain maps first, then plot.
    all_pain_maps = []   # list of (8,8) numpy arrays, one per episode
    all_paths     = []   # list of path lists
    all_outcomes  = []   # list of outcome strings

    # --- STEP 2: Run all episodes ---
    print(f"\nRunning {NUM_EPISODES} random episodes...\n")

    for ep in range(1, NUM_EPISODES + 1):
        path, outcome = run_random_episode(env, episode_num=ep, seed=0)

        # WHY: Read the slab's current pain map AFTER the episode ends (and
        # BEFORE the next reset(), which fires elastic_recovery). This captures
        # the slab state at the highest point of accumulation for this episode.
        pain_snap = env.slab.pain_map().copy()   # .copy() so later resets don't mutate it

        all_pain_maps.append(pain_snap)
        all_paths.append(path)
        all_outcomes.append(outcome)

        n_spikes = sum(1 for e in env.pain_history if e["episode"] == ep)
        print(
            f"  Episode {ep:2d}: outcome={outcome:7s} | steps={len(path)-1:3d} "
            f"| spikes={n_spikes} | slab_max={pain_snap.max():.3f}"
        )

    # WHY: Compute global max AFTER all episodes so we have the true maximum
    # across the entire run. This is used to fix the colorbar scale for all frames.
    global_max = max(pm.max() for pm in all_pain_maps)
    print(f"\nGlobal max pain score across all episodes: {global_max:.3f}")

    # --- STEP 3: Save side-by-side frame for each episode ---
    print(f"\nSaving {NUM_EPISODES} side-by-side frames to {FRAMES_DIR}/")

    for ep in range(1, NUM_EPISODES + 1):
        # WHY figsize=(14, 6)? Two square-ish panels side by side with room for
        # the colorbar on the right panel. 14 inches wide, 6 inches tall.
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6), dpi=110)

        draw_grid_path(ax_left, all_paths[ep - 1], all_outcomes[ep - 1], ep)
        draw_slab_heatmap(ax_right, all_pain_maps[ep - 1], ep, global_max)

        # WHY: Add a supertitle with cumulative stats so each frame is self-contained.
        total_spikes_so_far = sum(
            1 for e in env.pain_history if e["episode"] <= ep
        )
        fig.suptitle(
            f"PULSE Phase 3 — Episode {ep}/{NUM_EPISODES}   |   "
            f"Cumulative spikes: {total_spikes_so_far}   |   "
            f"Slab max pain: {all_pain_maps[ep-1].max():.3f}",
            fontsize=11, fontweight="bold", y=1.01
        )

        plt.tight_layout()

        # WHY zero-pad episode number? "episode_03.png" sorts correctly in
        # file explorers; "episode_3.png" would sort as 3, 30, 31, ..., 4.
        frame_path = os.path.join(FRAMES_DIR, f"episode_{ep:02d}.png")
        plt.savefig(frame_path, dpi=130, bbox_inches="tight")
        plt.close(fig)    # WHY close: prevents matplotlib from accumulating 20 open figures in memory

    print(f"All frames saved.")

    # --- STEP 4: Plot the full pain history ---
    print("\nPlotting pain history time series...")
    plot_pain_history(env.pain_history)

    # --- STEP 5: Print a final slab summary ---
    print()
    env.slab.summary()

    # WHY: Show the last frame and the pain history together at the end.
    plt.show()
